"""Preflight for real evaluation requests; repeated inside the executing worker."""
import logging
import asyncio

from app.core.evaluation_context import EVALUATION_RUNTIME, EvaluationRuntime, EvaluationCancelled
from app.evaluation.rate_limits import RedisProviderGovernor, configured_limits

from fastapi import HTTPException
from langchain_core.messages import HumanMessage
from redis.exceptions import RedisError
from rq import Worker

from app.api.llm import MAX_ANSWER_TOKENS, LLMConfigurationError, LLMProvider, LLMProviderRequestError, invoke_chat_model
from app.api.services.ingestion_runner import ingestion_runner
from app.core.redis_client import get_evaluation_queue
from app.core.safe_errors import log_safe_exception
from app.evaluation.engine import _judge_draft, EvaluationOutputError
from app.evaluation.models import JudgeDraft
from app.retrieval.config import RETRIEVAL_SETTINGS

logger = logging.getLogger(__name__)


def _validate_runtime_preflight(
    answer_providers: list[LLMProvider],
    judge_provider: LLMProvider,
    *,
    check_worker: bool = False,
) -> dict[LLMProvider, str]:
    try:
        if ingestion_runner.get_status().status != "completed":
            raise HTTPException(422, "Cannot start evaluation. Complete knowledge-base ingestion on the Data page first.")
        if RETRIEVAL_SETTINGS.number_of_chunks < 1 or RETRIEVAL_SETTINGS.number_of_candidates < RETRIEVAL_SETTINGS.number_of_chunks:
            raise HTTPException(422, "Cannot start evaluation. Retrieval candidate/chunk settings are invalid.")
        if check_worker:
            queue = get_evaluation_queue()
            if not Worker.count(queue=queue):
                raise HTTPException(503, "Cannot start evaluation. No evaluation worker is available. Start employee-rag-worker and retry.")
    except (RedisError, OSError) as error:
        raise HTTPException(503, "Cannot start evaluation. Redis is unavailable; ingestion and worker readiness could not be checked.") from error
    models = {}
    # Check both operations even when the answer and judge share a provider.
    for role, provider in [("answer", p) for p in answer_providers] + [("judge", judge_provider)]:
        try:
            response = invoke_chat_model(
                [HumanMessage(content='Preflight: return a short greeting.' if role == "answer" else
                              'Preflight: return completeness=1, relevance=1, rationale="preflight" and empty missing_facts and unsupported_claims.')],
                provider, max_output_tokens=MAX_ANSWER_TOKENS if role == "answer" else 4096,
                timeout_seconds=20, max_attempts=1,
                response_schema=JudgeDraft if role == "judge" else None,
            )
            if role == "judge":
                _judge_draft(response.text)
            models[provider] = response.model_name
        except EvaluationCancelled:
            raise
        except (LLMProviderRequestError, LLMConfigurationError, EvaluationOutputError) as error:
            log_safe_exception(logger, error, f"Evaluation preflight failed: role={role} provider={provider.value}")
            category = getattr(error, "category", "judge_response_parsing" if isinstance(error, EvaluationOutputError) else "provider_configuration")
            http_status = 503 if getattr(error, "retryable", False) else 422
            raise HTTPException(http_status, f"Cannot start evaluation. {role.capitalize()} provider {provider.value} [{category}]: {error}") from None
        except Exception as error:
            log_safe_exception(logger, error, f"Evaluation preflight failed: role={role} provider={provider.value}")
            raise HTTPException(503, f"Cannot start evaluation. {role.capitalize()} provider {provider.value} could not initialize. Check the model configuration and server logs.") from None
    return models


def validate_runtime_preflight(answer_providers, judge_provider, *, check_worker=False, should_cancel=None):
    """Probes share the same Redis budgets as evaluation jobs."""
    if should_cancel and should_cancel():
        raise EvaluationCancelled()
    try:
        governor = RedisProviderGovernor(limits=configured_limits())
    except (ValueError, TypeError):
        raise HTTPException(422, "Cannot start evaluation. EVALUATION_PROVIDER_LIMITS is invalid; check the documented provider limits.") from None

    async def run():
        runtime = EvaluationRuntime(governor)
        token = EVALUATION_RUNTIME.set(runtime)
        async def watch():
            while True:
                await asyncio.sleep(.2)
                try:
                    if await asyncio.to_thread(should_cancel):
                        runtime.cancel()
                        return
                except Exception:
                    runtime.cancel()
                    raise
        watcher = asyncio.create_task(watch()) if should_cancel else None
        try:
            return await asyncio.to_thread(_validate_runtime_preflight, answer_providers, judge_provider, check_worker=check_worker)
        finally:
            runtime.cancel()
            watcher_error = None
            if watcher:
                if not watcher.done():
                    watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
                if not watcher.cancelled():
                    watcher_error = watcher.exception()
            await asyncio.gather(*tuple(runtime.tasks), return_exceptions=True)
            EVALUATION_RUNTIME.reset(token)
            if watcher_error is not None:
                raise watcher_error
    return asyncio.run(run())
