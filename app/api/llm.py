"""Selectable LangChain chat-model integrations for answer generation."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import asyncio
import inspect
import json
import logging
import math
from time import sleep
from typing import Any, Callable, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core.evaluation_context import EVALUATION_RUNTIME, CASE_TRACE
from app.api import config
from app.api.llm_usage import LLMUsage, calculate_llm_usage
from app.api.prompts import (
    build_answer_messages,
    build_tool_selection_messages,
    format_retrieval_context,
)
from app.core.safe_errors import provider_failure, log_safe_exception
from app.core.credential_store import (
    CredentialStoreConfigurationError,
    CredentialStoreUnavailable,
    get_credential_store,
)
from app.retrieval.models import RetrievedChunk


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GROK_BASE_URL = config.GROK_BASE_URL
MAX_ANSWER_TOKENS = 512
MODEL_TIMEOUT_SECONDS = 120


class LLMProvider(StrEnum):
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROK = "grok"


class LLMConfigurationError(ValueError):
    """Raised when the selected provider is missing required configuration."""


class LLMProviderRequestError(RuntimeError):
    """Raised with a safe explanation of an upstream provider failure."""

    def __init__(
        self,
        provider: LLMProvider,
        detail: str,
        *,
        category: str = "provider_request",
        retryable: bool = False,
        model: str | None = None,
    ) -> None:
        self.provider = provider
        self.detail = detail
        self.category = category
        self.retryable = retryable
        self.model = model
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class LLMSettings:
    gemini_api_key: str
    gemini_model: str = DEFAULT_GEMINI_MODEL
    openai_api_key: str = ""
    openai_model: str = ""
    anthropic_api_key: str = ""
    anthropic_model: str = ""
    grok_api_key: str = ""
    grok_model: str = ""
    grok_base_url: str = DEFAULT_GROK_BASE_URL

    def model_for(self, provider: LLMProvider) -> str:
        models = {
            LLMProvider.GEMINI: self.gemini_model,
            LLMProvider.OPENAI: self.openai_model,
            LLMProvider.ANTHROPIC: self.anthropic_model,
            LLMProvider.GROK: self.grok_model,
        }
        return models[provider]

    def validate_provider(self, provider: LLMProvider) -> None:
        credentials = {
            LLMProvider.GEMINI: self.gemini_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.GROK: self.grok_api_key,
        }
        api_key = credentials[provider]
        if not api_key:
            raise LLMConfigurationError(
                f"Configure an active {provider.value} API key on the Settings page."
            )

        model = self.model_for(provider)
        if not model:
            raise LLMConfigurationError(
                f"Configure a model name for {provider.value} on the Settings page."
            )


@dataclass(frozen=True, slots=True)
class ChatModelClient:
    provider: LLMProvider
    model_name: str
    model: BaseChatModel


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    answer: str
    provider: LLMProvider
    model_name: str
    usage: LLMUsage | None = None


@dataclass(frozen=True, slots=True)
class GeneratedText:
    text: str
    provider: LLMProvider
    model_name: str
    usage: LLMUsage | None = None


@dataclass(frozen=True, slots=True)
class ToolDrivenAnswer:
    """A chat answer plus any evidence retrieved through the model's tool."""

    answer: str
    provider: LLMProvider
    model_name: str
    usage: LLMUsage | None = None
    chunks: tuple[RetrievedChunk, ...] = ()
    retrieval_query: str | None = None

    @property
    def retrieval_used(self) -> bool:
        return self.retrieval_query is not None


def select_provider(
    *,
    gemini: bool = False,
    openai: bool = False,
    anthropic: bool = False,
    grok: bool = False,
) -> LLMProvider:
    """Select one provider, defaulting to Gemini when none is enabled."""
    enabled = [
        provider
        for provider, selected in (
            (LLMProvider.GEMINI, gemini),
            (LLMProvider.OPENAI, openai),
            (LLMProvider.ANTHROPIC, anthropic),
            (LLMProvider.GROK, grok),
        )
        if selected
    ]
    if len(enabled) > 1:
        raise ValueError("only one LLM provider can be enabled")
    return enabled[0] if enabled else LLMProvider.GEMINI


def create_chat_model_client(
    provider: LLMProvider,
    settings: LLMSettings | None = None,
    *,
    max_output_tokens: int = MAX_ANSWER_TOKENS,
    timeout_seconds: int = MODEL_TIMEOUT_SECONDS,
) -> ChatModelClient:
    """Create the LangChain chat model configured for one provider."""
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be at least 1")

    settings = settings or _stored_llm_settings(provider)
    settings.validate_provider(provider)
    model_name = settings.model_for(provider)

    if provider is LLMProvider.GEMINI:
        model: BaseChatModel = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=settings.gemini_api_key,
            max_output_tokens=max_output_tokens,
            timeout=timeout_seconds,
            max_retries=0,
        )
    elif provider is LLMProvider.OPENAI:
        model = ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
            max_completion_tokens=max_output_tokens,
            timeout=timeout_seconds,
            max_retries=0,
        )
    elif provider is LLMProvider.ANTHROPIC:
        model = ChatAnthropic(
            model_name=model_name,
            api_key=settings.anthropic_api_key,
            max_tokens_to_sample=max_output_tokens,
            timeout=timeout_seconds,
            max_retries=0,
        )
    else:
        model = ChatOpenAI(
            model=model_name,
            api_key=settings.grok_api_key,
            base_url=settings.grok_base_url,
            max_completion_tokens=max_output_tokens,
            timeout=timeout_seconds,
            max_retries=0,
            use_responses_api=False,
        )

    return ChatModelClient(provider, model_name, model)


def get_chat_model_client(
    provider: LLMProvider,
    *,
    max_output_tokens: int = MAX_ANSWER_TOKENS,
    timeout_seconds: int = MODEL_TIMEOUT_SECONDS,
) -> ChatModelClient:
    """Build a client from the active, unexpired database credential."""
    return create_chat_model_client(
        provider,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )


def _stored_llm_settings(provider: LLMProvider) -> LLMSettings:
    try:
        credential = get_credential_store().get_active(provider.value)
    except (CredentialStoreConfigurationError, CredentialStoreUnavailable) as error:
        raise LLMConfigurationError(str(error)) from error
    if credential is None:
        raise LLMConfigurationError(
            f"Configure an active {provider.value} API key on the Settings page."
        )

    values = {
        "gemini_api_key": "",
        "openai_api_key": "",
        "anthropic_api_key": "",
        "grok_api_key": "",
    }
    models = {
        "gemini_model": DEFAULT_GEMINI_MODEL,
        "openai_model": "",
        "anthropic_model": "",
        "grok_model": "",
    }
    key_field, model_field = {
        LLMProvider.GEMINI: ("gemini_api_key", "gemini_model"),
        LLMProvider.OPENAI: ("openai_api_key", "openai_model"),
        LLMProvider.ANTHROPIC: ("anthropic_api_key", "anthropic_model"),
        LLMProvider.GROK: ("grok_api_key", "grok_model"),
    }[provider]
    values[key_field] = credential.api_key
    models[model_field] = credential.model_name
    return LLMSettings(**values, **models)


def _clear_client_cache() -> None:
    """Compatibility hook retained now that clients are intentionally uncached."""


get_chat_model_client.cache_clear = _clear_client_cache  # type: ignore[attr-defined]


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("output_text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts).strip()
    return str(content).strip()


def _provider_error_detail(
    provider: LLMProvider,
    error: Exception,
) -> str:
    return f"{provider.value} {provider_failure(error)[1]}"


def generate_grounded_answer(
    question: str,
    chunks: list[RetrievedChunk],
    provider: LLMProvider,
    *,
    history: Sequence[tuple[str, str]] = (),
) -> GeneratedAnswer:
    """Generate one answer grounded in the supplied retrieved chunks."""
    generated = invoke_chat_model(
        build_answer_messages(question, chunks, history),
        provider,
    )
    return GeneratedAnswer(
        generated.text,
        generated.provider,
        generated.model_name,
        generated.usage,
    )


def _combined_usage(*usages: LLMUsage | None) -> LLMUsage | None:
    present = [usage for usage in usages if usage is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]

    costs = [usage.estimated_cost_usd for usage in present]
    first = present[0]
    return LLMUsage(
        input_tokens=sum(usage.input_tokens for usage in present),
        cached_input_tokens=sum(
            usage.cached_input_tokens for usage in present
        ),
        cache_creation_input_tokens=sum(
            usage.cache_creation_input_tokens for usage in present
        ),
        output_tokens=sum(usage.output_tokens for usage in present),
        total_tokens=sum(usage.total_tokens for usage in present),
        token_count_source=(
            "provider"
            if all(usage.token_count_source == "provider" for usage in present)
            else "estimated"
        ),
        input_cost_per_million_usd=first.input_cost_per_million_usd,
        cached_input_cost_per_million_usd=(
            first.cached_input_cost_per_million_usd
        ),
        cache_creation_cost_per_million_usd=(
            first.cache_creation_cost_per_million_usd
        ),
        output_cost_per_million_usd=first.output_cost_per_million_usd,
        estimated_cost_usd=(
            round(sum(cost for cost in costs if cost is not None), 10)
            if all(cost is not None for cost in costs)
            else None
        ),
        pricing_source=(
            "built_in"
            if all(usage.pricing_source == "built_in" for usage in present)
            else "unavailable"
        ),
    )


def _invoke_client(
    client: ChatModelClient,
    messages: Sequence[BaseMessage],
    runnable: Any | None = None,
    response_schema: type[BaseModel] | None = None,
    max_attempts: int = 3,
) -> tuple[BaseMessage, LLMUsage]:
    try:
        selected_runnable = runnable if runnable is not None else client.model
        uses_native_structure = response_schema is not None and runnable is None
        if uses_native_structure:
            selected_runnable = client.model.with_structured_output(
                response_schema,
                method="json_schema",
                include_raw=True,
            )
        for attempt in range(max_attempts):
            try:
                result = selected_runnable.invoke(messages)
                break
            except Exception as error:
                if not provider_failure(error)[2] or attempt == max_attempts - 1:
                    raise
                sleep(0.5 * (2 ** attempt))
        return _normalize_response(client, messages, result, response_schema, uses_native_structure)
    except LLMProviderRequestError:
        raise
    except Exception as error:
        raise LLMProviderRequestError(
            client.provider,
            _provider_error_detail(client.provider, error),
            category=provider_failure(error)[0],
            retryable=provider_failure(error)[2],
            model=client.model_name,
        ) from error


def _normalize_response(client, messages, result, response_schema, uses_native_structure):
    parsed = None
    if uses_native_structure:
        if not isinstance(result, Mapping):
            raise TypeError(
                "Structured output did not include its raw response."
            )
        response = result.get("raw")
        parsed = result.get("parsed")
        if result.get("parsing_error") is not None:
            raise LLMProviderRequestError(
                client.provider, "The judge returned invalid structured output.",
                category="judge_response_parsing", model=client.model_name,
            )
        if not isinstance(response, BaseMessage):
            raise TypeError(
                "Structured output did not include a model message."
            )
    else:
        response = result
        if not isinstance(response, BaseMessage):
            raise TypeError("The model did not return a chat message.")
    usage = calculate_llm_usage(
        messages,
        response,
        client.provider.value,
        client.model_name,
    )
    if parsed is not None:
        if isinstance(parsed, BaseModel):
            structured_text = parsed.model_dump_json()
        elif isinstance(parsed, Mapping):
            structured_text = json.dumps(
                dict(parsed),
                ensure_ascii=False,
                default=str,
            )
        else:
            structured_text = response_schema.model_validate(
                parsed
            ).model_dump_json()
        response = response.model_copy(update={"content": structured_text})
    return response, usage


def generate_tool_driven_answer(
    question: str,
    provider: LLMProvider,
    search: Callable[[str], list[RetrievedChunk]],
    *,
    history: Sequence[tuple[str, str]] = (),
) -> ToolDrivenAnswer:
    """Let the chat model decide whether to search employee knowledge."""
    retrieved_chunks: list[RetrievedChunk] = []
    retrieval_query: str | None = None

    @tool("search_employee_knowledge")
    def search_employee_knowledge(query: str) -> str:
        """Search uploaded employee documents and company-specific knowledge."""
        nonlocal retrieval_query
        normalized_query = query.strip() or question.strip()
        retrieval_query = normalized_query
        retrieved_chunks.extend(search(normalized_query))
        if not retrieved_chunks:
            return "No relevant employee documents were found."
        return format_retrieval_context(retrieved_chunks)

    client = get_chat_model_client(provider)
    selection_messages = build_tool_selection_messages(question, history)
    try:
        tool_model = client.model.bind_tools(
            [search_employee_knowledge],
            tool_choice="auto",
        )
    except Exception as error:
        raise LLMProviderRequestError(
            provider,
            (
                f"The configured {provider.value} model does not support "
                "tool calling. Choose a tool-capable model in Settings."
            ),
        ) from error

    selection, selection_usage = _invoke_client(
        client,
        selection_messages,
        tool_model,
    )
    tool_calls = getattr(selection, "tool_calls", None) or []
    matching_call = next(
        (
            call
            for call in tool_calls
            if call.get("name") == search_employee_knowledge.name
        ),
        None,
    )

    if matching_call is None:
        direct_answer = _message_text(selection.content)
        if not direct_answer:
            raise LLMProviderRequestError(
                provider,
                f"{provider.value} returned neither an answer nor a tool call.",
            )
        return ToolDrivenAnswer(
            direct_answer,
            client.provider,
            client.model_name,
            selection_usage,
        )

    arguments = matching_call.get("args")
    query = arguments.get("query") if isinstance(arguments, dict) else None
    search_employee_knowledge.invoke(
        {"query": query if isinstance(query, str) else question}
    )

    if not retrieved_chunks:
        return ToolDrivenAnswer(
            (
                "I searched the uploaded employee documents but could not find "
                "relevant information for that question."
            ),
            client.provider,
            client.model_name,
            selection_usage,
            retrieval_query=retrieval_query,
        )

    grounded_messages = build_answer_messages(
        question,
        retrieved_chunks,
        history,
    )
    grounded_response, grounded_usage = _invoke_client(
        client,
        grounded_messages,
    )
    grounded_answer = _message_text(grounded_response.content)
    if not grounded_answer:
        raise LLMProviderRequestError(
            provider,
            f"{provider.value} returned an empty grounded answer.",
        )
    return ToolDrivenAnswer(
        grounded_answer,
        client.provider,
        client.model_name,
        _combined_usage(selection_usage, grounded_usage),
        tuple(retrieved_chunks),
        retrieval_query,
    )


def invoke_chat_model(
    messages: Sequence[BaseMessage],
    provider: LLMProvider,
    *,
    max_output_tokens: int = MAX_ANSWER_TOKENS,
    response_schema: type[BaseModel] | None = None,
    timeout_seconds: int = MODEL_TIMEOUT_SECONDS,
    max_attempts: int = 3,
) -> GeneratedText:
    """Invoke one provider and optionally constrain it to a JSON schema."""
    runtime = EVALUATION_RUNTIME.get()
    if runtime is not None:
        return runtime.invoke(
            messages, provider, max_output_tokens=max_output_tokens,
            response_schema=response_schema, timeout_seconds=timeout_seconds, max_attempts=max_attempts,
        )
    if not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts must be between 1 and 3")
    client = get_chat_model_client(
        provider,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
    response, usage = _invoke_client(
        client,
        messages,
        response_schema=response_schema,
        max_attempts=max_attempts,
    )

    text = _message_text(response.content)
    if not text:
        raise LLMProviderRequestError(
            provider,
            f"{provider.value} returned an empty response. Check the model output budget and capabilities.",
            category="provider_empty_response", model=client.model_name,
        )
    return GeneratedText(text, client.provider, client.model_name, usage)


async def ainvoke_chat_model(
    messages: Sequence[BaseMessage],
    provider: LLMProvider,
    *,
    governor,
    max_output_tokens: int = MAX_ANSWER_TOKENS,
    response_schema: type[BaseModel] | None = None,
    timeout_seconds: int = MODEL_TIMEOUT_SECONDS,
    max_attempts: int = 3,
) -> GeneratedText:
    """Native async model I/O with shared evaluation budgets and bounded retries."""
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts must be between 1 and 3")
    # Each invocation owns its client; credential-store operations open and close
    # their own psycopg connections in the executor, never shared sessions.
    client = await asyncio.to_thread(
        get_chat_model_client, provider, max_output_tokens=max_output_tokens, timeout_seconds=timeout_seconds,
    )
    try:
        return await _ainvoke_owned_client(
            client, messages, provider, governor=governor, max_output_tokens=max_output_tokens,
            response_schema=response_schema, timeout_seconds=timeout_seconds, max_attempts=max_attempts,
        )
    finally:
        # Dispose of each owned async HTTP client on its event loop. This also
        # runs on cancellation, avoiding leaked connection pools across RQ runs.
        root = getattr(client.model, "root_async_client", None)
        if provider is LLMProvider.ANTHROPIC:
            root = getattr(getattr(client.model, "_async_client", None), "_client", None)
        elif provider is LLMProvider.GEMINI:
            root = getattr(client.model, "async_client", None)
        close = getattr(root, "close", None)
        if not inspect.iscoroutinefunction(close):
            close = getattr(root, "aclose", None)
        if inspect.iscoroutinefunction(close):
            try:
                await close()
            except Exception as error:
                log_safe_exception(logging.getLogger(__name__), error, "Evaluation provider client cleanup failed")


async def _ainvoke_owned_client(client, messages, provider, *, governor, max_output_tokens,
                                response_schema, timeout_seconds, max_attempts):
    from app.evaluation.rate_limits import ProviderBudgetError, ProviderBudgetUnavailable, retry_delay
    role = "judge" if response_schema is not None else "answer"
    schema_text = json.dumps(response_schema.model_json_schema()) if response_schema else ""
    # Conservative reservation, reconciled with actual provider usage. Exact
    # tokenization is provider/model-specific; 429 remains authoritative.
    reserved = math.ceil((sum(len(str(m.content).encode("utf-8")) for m in messages) + len(schema_text.encode("utf-8"))) / 2) + max_output_tokens + 256
    try:
        runnable = client.model.with_structured_output(response_schema, method="json_schema", include_raw=True) if response_schema else client.model
        for attempt in range(max_attempts):
            try:
                async with governor.slot(provider.value, role, reserved, timeout_seconds) as usage:
                    # wait_for also bounds native clients that do not apply their
                    # timeout consistently, so leases cannot expire mid-request.
                    result = await asyncio.wait_for(runnable.ainvoke(messages), timeout_seconds)
                    response, measured = _normalize_response(client, messages, result, response_schema, response_schema is not None)
                    usage["tokens"] = measured.total_tokens
                break
            except (LLMProviderRequestError, ProviderBudgetError):
                raise
            except Exception as error:
                if not provider_failure(error)[2]:
                    raise
                delay = retry_delay(error, attempt)
                # Publish cooldown even on the final attempt or a long
                # Retry-After, so other jobs do not hammer the same provider.
                await governor.cooldown(provider.value, delay)
                if attempt == max_attempts - 1 or delay > 300:
                    raise
                trace = CASE_TRACE.get()
                if trace is not None:
                    trace.retries += 1
                # All jobs back off together; smooth admission staggers the next
                # attempts, and per-attempt jitter avoids synchronized retries.
                await asyncio.sleep(delay)
    except LLMProviderRequestError:
        raise
    except ProviderBudgetError as error:
        raise LLMProviderRequestError(
            provider, str(error), category="provider_unavailable" if isinstance(error, ProviderBudgetUnavailable) else "provider_rate_configuration",
            retryable=isinstance(error, ProviderBudgetUnavailable), model=client.model_name,
        ) from error
    except Exception as error:
        raise LLMProviderRequestError(
            provider, _provider_error_detail(provider, error), category=provider_failure(error)[0],
            retryable=provider_failure(error)[2], model=client.model_name,
        ) from error
    text = _message_text(response.content)
    if not text:
        raise LLMProviderRequestError(provider, f"{provider.value} returned an empty response. Check the model output budget and capabilities.", category="provider_empty_response", model=client.model_name)
    return GeneratedText(text, provider, client.model_name, measured)
