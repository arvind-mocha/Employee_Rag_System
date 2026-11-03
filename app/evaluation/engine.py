"""On-demand RAG evaluation of user-provided reference datasets."""

from collections.abc import Callable, Iterable
from datetime import datetime, timezone
import json
import logging
from statistics import fmean
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from app.api.answer_metrics import (
    citation_validity,
    lexical_similarity,
    semantic_similarity,
    overall_answer_score,
)
from app.api.llm import (
    LLMConfigurationError,
    LLMProvider,
    LLMProviderRequestError,
    generate_grounded_answer,
    invoke_chat_model,
)
from app.api.llm_usage import LLMUsage, summarize_llm_usage
from app.core.safe_errors import log_safe_exception
from app.evaluation.dataset import dataset_sha256, ground_truth_status, text_sha256
from app.evaluation.models import (
    AnswerScores,
    EvaluationCaseResult,
    EvaluationRun,
    EvaluationTiming,
    GroundTruthRecord,
    GroundTruthStatus,
    JudgeDraft,
    ProviderEvaluationSummary,
    RetrievalConfiguration,
    RetrievalScores,
    ReviewStatus,
    ScoreAverages,
)
from app.evaluation.prompts import build_judge_messages
from app.retrieval.config import RETRIEVAL_SETTINGS
from app.retrieval.models import RetrievedChunk
from app.retrieval.query_router import QueryRouter, RetrievalOptions, get_query_router
from app.retrieval.retrieval_metrics import evaluate_retrieval


logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.80
JUDGE_GENERATION_MAX_OUTPUT_TOKENS = 4_096


class EvaluationOutputError(ValueError):
    """Raised when a judge returns invalid structured output."""


def _retrieval_options(
    configuration: RetrievalConfiguration,
) -> RetrievalOptions:
    return RetrievalOptions(**configuration.model_dump())


def _parse_json_object(text: str) -> dict[str, Any]:
    normalized = text.strip()
    if normalized.startswith("```"):
        normalized = normalized.removeprefix("```json").removeprefix("```")
        normalized = normalized.removesuffix("```").strip()
    start = normalized.find("{")
    end = normalized.rfind("}")
    if start < 0 or end < start:
        raise EvaluationOutputError("the model did not return a JSON object")
    try:
        value = json.loads(normalized[start : end + 1])
    except json.JSONDecodeError as error:
        raise EvaluationOutputError(
            "the model returned malformed evaluation JSON"
        ) from error
    if not isinstance(value, dict):
        raise EvaluationOutputError("the model evaluation output must be an object")
    return value


def _judge_draft(text: str) -> JudgeDraft:
    try:
        return JudgeDraft.model_validate(_parse_json_object(text))
    except ValidationError as error:
        raise EvaluationOutputError(
            "the judge returned an invalid JSON schema"
        ) from error


def _retrieval_scores(
    record: GroundTruthRecord,
    chunks: list[RetrievedChunk],
) -> RetrievalScores | None:
    relevant_hashes = {
        evidence.content_sha256 or text_sha256(evidence.content) for evidence in record.relevant_evidence
    }
    if not relevant_hashes:
        return None
    retrieved_hashes = [text_sha256(chunk.content) for chunk in chunks]
    metrics = evaluate_retrieval(
        retrieved_hashes,
        relevant_hashes,
        RETRIEVAL_SETTINGS.number_of_chunks,
    )
    relevant_sources = {
        evidence.source_file
        for evidence in record.relevant_evidence
        if evidence.source_file
    }
    source_metrics = (
        evaluate_retrieval(
            [chunk.source_file for chunk in chunks if chunk.source_file],
            relevant_sources,
            RETRIEVAL_SETTINGS.number_of_chunks,
        )
        if relevant_sources
        else None
    )
    return RetrievalScores(
        precision_at_k=metrics.precision,
        recall_at_k=metrics.recall,
        reciprocal_rank=metrics.reciprocal_rank,
        average_precision_at_k=metrics.average_precision,
        ndcg_at_k=metrics.ndcg,
        source_precision_at_k=(
            source_metrics.precision if source_metrics is not None else None
        ),
        source_recall_at_k=(
            source_metrics.recall if source_metrics is not None else None
        ),
    )


def _safe_error(error: Exception, stage: str) -> str:
    if isinstance(error, LLMProviderRequestError):
        return error.detail
    if isinstance(error, (LLMConfigurationError, EvaluationOutputError)):
        return str(error)
    messages = {
        "retrieval": "Retrieval failed. Check knowledge-base ingestion and database availability.",
        "reference_validation": "Reference validation failed. Check the dataset evidence and source files.",
        "answer_generation": "Answer generation failed. Check the answer model configuration.",
        "judge_invocation": "Judge invocation failed. Check the judge model configuration.",
        "metrics": "Metric calculation failed. Check the backend logs for this case.",
        "serialization": "Evaluation result serialization failed. Check the backend logs for this case.",
    }
    return f"{messages.get(stage, 'Internal evaluation error. Check the backend logs.')} ({type(error).__name__})"


def _failed_case(
    record: GroundTruthRecord,
    answer_provider: LLMProvider,
    judge_provider: LLMProvider,
    status: GroundTruthStatus,
    started_at: float,
    *,
    chunks: list[RetrievedChunk] | None = None,
    retrieval_ms: float = 0.0,
    answer_generation_ms: float = 0.0,
    grading_ms: float = 0.0,
    answer: str | None = None,
    answer_model: str | None = None,
    answer_usage: LLMUsage | None = None,
    judge_model: str | None = None,
    judge_usage: LLMUsage | None = None,
    warnings: list[str] | None = None,
    retrieval_scores: RetrievalScores | None = None,
    error: Exception,
    stage: str,
) -> EvaluationCaseResult:
    chunks = chunks or []
    category = {
        "retrieval": "retrieval_failure", "judge_parsing": "judge_response_parsing",
        "metrics": "metric_calculation", "reference_validation": "jsonl_validation",
    }.get(stage, "internal_evaluation")
    provider = judge_provider if stage.startswith("judge") else answer_provider
    model = judge_model if stage.startswith("judge") else answer_model
    if isinstance(error, LLMProviderRequestError):
        category = error.category
        provider = error.provider
        model = error.model or model
    elif isinstance(error, LLMConfigurationError):
        category = "provider_configuration"
        stage = "judge_initialization" if stage.startswith("judge") else "answer_initialization"
    if category == "judge_response_parsing":
        stage = "judge_parsing"
    log_safe_exception(logger, error, f"Evaluation case failed: record={record.id!r} stage={stage} category={category} provider={provider.value}")
    total_ms = (perf_counter() - started_at) * 1000
    return EvaluationCaseResult(
        record_id=record.id,
        question=record.question,
        category=record.category,
        reference_answer=record.reference_answer,
        required_facts=record.required_facts,
        review_status=record.review_status,
        ground_truth_status=status,
        answer_provider=answer_provider,
        answer_model=answer_model,
        judge_provider=judge_provider,
        judge_model=judge_model,
        answer=answer,
        answer_usage=answer_usage,
        judge_usage=judge_usage,
        usage=summarize_llm_usage([answer_usage, judge_usage]),
        retrieved_chunk_ids=[
            chunk.chunk_id or text_sha256(chunk.content) for chunk in chunks
        ],
        retrieved_sources=list(
            dict.fromkeys(
                chunk.source_file for chunk in chunks if chunk.source_file
            )
        ),
        retrieval_scores=retrieval_scores,
        timing=EvaluationTiming(
            retrieval_ms=round(retrieval_ms, 2),
            answer_generation_ms=round(answer_generation_ms, 2),
            grading_ms=round(grading_ms, 2),
            total_ms=round(total_ms, 2),
        ),
        warnings=warnings or [],
        error=_safe_error(error, stage),
        error_category=category, failed_stage=stage,
        error_provider=provider, error_model=model,
    )


def evaluate_record(
    record: GroundTruthRecord,
    answer_provider: LLMProvider,
    judge_provider: LLMProvider,
    retrieval: RetrievalConfiguration,
    *,
    router: QueryRouter | None = None,
) -> EvaluationCaseResult:
    """Run retrieval, answer generation, and grading for one record."""
    started_at = perf_counter()
    status = GroundTruthStatus.UNVERIFIABLE
    warnings = []
    chunks = []
    generated_answer = judged = None
    retrieval_scores = None
    retrieval_ms = answer_ms = grading_ms = 0.0
    stage = "reference_validation"
    stage_started = started_at
    try:
        status = ground_truth_status(record)
        if not record.relevant_evidence:
            warnings.append("Retrieval metrics unavailable: no reference evidence was supplied.")
        if answer_provider is judge_provider:
            warnings.append("The answer and judge use the same provider.")
        stage = "retrieval"
        stage_started = perf_counter()
        query_router = router or get_query_router()
        chunks = query_router.search(record.question, _retrieval_options(retrieval))
        retrieval_ms = (perf_counter() - stage_started) * 1000
        stage = "metrics"
        stage_started = perf_counter()
        retrieval_scores = _retrieval_scores(record, chunks)
        stage = "answer_generation"
        stage_started = perf_counter()
        generated_answer = generate_grounded_answer(record.question, chunks, answer_provider)
        answer_ms = (perf_counter() - stage_started) * 1000
        stage = "judge_invocation"
        stage_started = perf_counter()
        judged = invoke_chat_model(
            build_judge_messages(record, generated_answer.answer, chunks), judge_provider,
            max_output_tokens=JUDGE_GENERATION_MAX_OUTPUT_TOKENS, response_schema=JudgeDraft,
        )
        stage = "judge_parsing"
        judge = _judge_draft(judged.text)
        stage = "metrics"
        overall = overall_answer_score(completeness=judge.completeness, relevance=judge.relevance)
        similarity = None
        try:
            similarity = semantic_similarity(record.reference_answer, generated_answer.answer)
        except Exception:
            warnings.append("Semantic similarity unavailable: the embedding model could not score this answer.")
        answer_scores = AnswerScores(
            correctness=judge.correctness, groundedness=judge.groundedness,
            abstention=judge.abstention if not record.answerable else None,
            citation_validity=citation_validity(generated_answer.answer, len(chunks)),
            semantic_similarity=similarity,
            lexical_similarity=lexical_similarity(record.reference_answer, generated_answer.answer),
            completeness=judge.completeness, relevance=judge.relevance, overall=overall,
        )
        grading_ms = (perf_counter() - stage_started) * 1000
        stage = "serialization"
        result = EvaluationCaseResult(
            record_id=record.id, question=record.question, category=record.category,
            reference_answer=record.reference_answer, required_facts=record.required_facts,
            review_status=record.review_status, ground_truth_status=status,
            answer_provider=generated_answer.provider, answer_model=generated_answer.model_name,
            judge_provider=judged.provider, judge_model=judged.model_name,
            answer=generated_answer.answer, answer_usage=generated_answer.usage, judge_usage=judged.usage,
            usage=summarize_llm_usage([generated_answer.usage, judged.usage]),
            retrieved_chunk_ids=[chunk.chunk_id or text_sha256(chunk.content) for chunk in chunks],
            retrieved_sources=list(dict.fromkeys(chunk.source_file for chunk in chunks if chunk.source_file)),
            retrieval_scores=retrieval_scores, answer_scores=answer_scores,
            unsupported_claims=judge.unsupported_claims, missing_facts=judge.missing_facts,
            rationale=judge.rationale,
            passed=judge.completeness >= PASS_THRESHOLD and judge.relevance >= PASS_THRESHOLD,
            timing=EvaluationTiming(retrieval_ms=round(retrieval_ms, 2), answer_generation_ms=round(answer_ms, 2),
                                    grading_ms=round(grading_ms, 2), total_ms=round((perf_counter()-started_at)*1000, 2)),
            warnings=warnings,
        )
        result.model_dump(mode="json")
        return result
    except Exception as error:
        elapsed = (perf_counter() - stage_started) * 1000
        if stage == "retrieval":
            retrieval_ms = elapsed
        elif stage == "answer_generation":
            answer_ms = elapsed
        elif stage.startswith("judge") or stage == "metrics":
            grading_ms = elapsed
        return _failed_case(
            record, answer_provider, judge_provider, status, started_at,
            chunks=chunks, retrieval_ms=retrieval_ms, answer_generation_ms=answer_ms,
            grading_ms=grading_ms, answer=generated_answer.answer if generated_answer else None,
            answer_model=generated_answer.model_name if generated_answer else None,
            answer_usage=generated_answer.usage if generated_answer else None,
            judge_model=judged.model_name if judged else None, judge_usage=judged.usage if judged else None,
            warnings=warnings, retrieval_scores=retrieval_scores, error=error, stage=stage,
        )


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return fmean(present) if present else None


def _summary(
    provider: LLMProvider,
    results: list[EvaluationCaseResult],
) -> ProviderEvaluationSummary:
    provider_results = [
        result for result in results if result.answer_provider is provider
    ]
    eligible = [
        result
        for result in provider_results
        if result.answer_scores is not None
    ]
    passed = sum(result.passed is True for result in eligible)
    answer_scores = [result.answer_scores for result in eligible]
    retrieval_scores = [
        result.retrieval_scores
        for result in eligible
        if result.retrieval_scores is not None
    ]
    averages = ScoreAverages(
        precision_at_k=_mean(
            score.precision_at_k for score in retrieval_scores
        ),
        recall_at_k=_mean(score.recall_at_k for score in retrieval_scores),
        reciprocal_rank=_mean(
            score.reciprocal_rank for score in retrieval_scores
        ),
        average_precision_at_k=_mean(
            score.average_precision_at_k for score in retrieval_scores
        ),
        ndcg_at_k=_mean(score.ndcg_at_k for score in retrieval_scores),
        source_precision_at_k=_mean(
            score.source_precision_at_k for score in retrieval_scores
        ),
        source_recall_at_k=_mean(
            score.source_recall_at_k for score in retrieval_scores
        ),
        completeness=_mean(score.completeness for score in answer_scores),
        relevance=_mean(score.relevance for score in answer_scores),
        overall=_mean(score.overall for score in answer_scores),
        **{
            name: _mean(getattr(score, name) for score in answer_scores)
            for name in (
                "correctness", "groundedness", "citation_validity",
                "semantic_similarity", "lexical_similarity", "abstention",
            )
        },
    )
    return ProviderEvaluationSummary(
        provider=provider,
        evaluated=len(eligible),
        passed=passed,
        failed=len(eligible) - passed,
        errors=sum(result.error is not None for result in provider_results),
        stale_ground_truth=sum(
            result.ground_truth_status is GroundTruthStatus.STALE
            for result in provider_results
        ),
        unreviewed_ground_truth=sum(
            result.review_status is ReviewStatus.MACHINE_GENERATED
            for result in provider_results
        ),
        pass_rate=passed / len(eligible) if eligible else None,
        average_scores=averages,
        average_latency_ms=_mean(
            result.timing.total_ms for result in provider_results
        ),
        usage=summarize_llm_usage(
            usage
            for result in provider_results
            for usage in (result.answer_usage, result.judge_usage)
        ),
    )


def run_evaluation(
    dataset_name: str,
    records: list[GroundTruthRecord],
    answer_providers: list[LLMProvider],
    judge_provider: LLMProvider,
    retrieval: RetrievalConfiguration,
    *,
    router: QueryRouter | None = None,
    content_sha256: str | None = None,
    on_case_complete: Callable[[], None] | None = None,
) -> EvaluationRun:
    """Run a dataset now; this function is never called automatically."""
    run_id = str(uuid4())
    query_router = router
    results = []
    for answer_provider in answer_providers:
        for record in records:
            started_at = perf_counter()
            try:
                result = evaluate_record(
                    record, answer_provider, judge_provider, retrieval, router=query_router,
                )
            except Exception as error:
                # Last-resort isolation if even case error construction fails.
                result = _failed_case(
                    record, answer_provider, judge_provider, GroundTruthStatus.UNVERIFIABLE,
                    started_at, error=error, stage="internal_evaluation",
                )
            results.append(result)
            if on_case_complete is not None:
                on_case_complete()
    for result in results:
        result.run_id = run_id
        result.dataset_id = dataset_name
    return EvaluationRun(
        run_id=run_id,
        status="failed" if all(result.error for result in results) else "partial" if any(result.error for result in results) else "completed",
        error="No cases could be evaluated. Expand the case details for errors." if all(result.error for result in results) else None,
        dataset_name=dataset_name,
        dataset_sha256=content_sha256 or dataset_sha256(dataset_name),
        generated_at=datetime.now(timezone.utc),
        judge_provider=judge_provider,
        usage=summarize_llm_usage(
            usage
            for result in results
            for usage in (result.answer_usage, result.judge_usage)
        ),
        summaries=[
            _summary(provider, results) for provider in answer_providers
        ],
        results=results,
    )
