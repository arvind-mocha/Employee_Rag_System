"""Ground-truth generation and on-demand RAG evaluation engine."""

from collections.abc import Iterable
from datetime import datetime, timezone
import json
from statistics import fmean
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from app.api.answer_metrics import (
    citation_validity,
    lexical_similarity,
    overall_answer_score,
    semantic_similarity,
)
from app.api.llm import (
    LLMConfigurationError,
    LLMProvider,
    LLMProviderRequestError,
    generate_grounded_answer,
    invoke_chat_model,
)
from app.api.llm_usage import LLMUsage, summarize_llm_usage
from app.evaluation.dataset import (
    dataset_sha256,
    evidence_fingerprint,
    ground_truth_status,
    indexed_corpus_fingerprint,
    snapshot_evidence,
    text_sha256,
)
from app.evaluation.models import (
    AnswerScores,
    EvaluationCaseResult,
    EvaluationRun,
    EvaluationTiming,
    GroundTruthRecord,
    GroundTruthStatus,
    JudgeDraft,
    ProviderEvaluationSummary,
    ReferenceDraft,
    RetrievalConfiguration,
    RetrievalScores,
    ReviewStatus,
    ScoreAverages,
)
from app.evaluation.prompts import (
    build_judge_messages,
    build_reference_messages,
)
from app.retrieval.config import RETRIEVAL_SETTINGS
from app.retrieval.models import RetrievedChunk
from app.retrieval.query_router import QueryRouter, RetrievalOptions, get_query_router
from app.retrieval.retrieval_metrics import evaluate_retrieval


PASS_THRESHOLD = 0.80


class EvaluationOutputError(ValueError):
    """Raised when a generator or judge returns invalid structured output."""


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


def _reference_draft(text: str) -> ReferenceDraft:
    try:
        draft = ReferenceDraft.model_validate(_parse_json_object(text))
    except ValidationError as error:
        raise EvaluationOutputError(
            "the reference model returned an invalid JSON schema"
        ) from error
    if not draft.reference_answer.strip():
        raise EvaluationOutputError("the reference answer must not be empty")
    if draft.answerable and not draft.required_facts:
        raise EvaluationOutputError(
            "an answerable reference must contain required facts"
        )
    return draft


def _judge_draft(text: str) -> JudgeDraft:
    try:
        return JudgeDraft.model_validate(_parse_json_object(text))
    except ValidationError as error:
        raise EvaluationOutputError(
            "the judge returned an invalid JSON schema"
        ) from error


def _normalized_questions(questions: Iterable[str]) -> list[str]:
    normalized = []
    seen = set()
    for question in questions:
        candidate = " ".join(question.split())
        if not candidate or candidate.casefold() in seen:
            continue
        seen.add(candidate.casefold())
        normalized.append(candidate)
    if not normalized:
        raise ValueError("at least one non-empty question is required")
    return normalized


def generate_ground_truth_records(
    questions: Iterable[str],
    reference_provider: LLMProvider,
    retrieval: RetrievalConfiguration,
    *,
    router: QueryRouter | None = None,
) -> list[GroundTruthRecord]:
    """Create machine-generated, reviewable ground truth from current evidence."""
    query_router = router or get_query_router()
    options = _retrieval_options(retrieval)
    records = []
    corpus_fingerprint = indexed_corpus_fingerprint()

    for question in _normalized_questions(questions):
        chunks = query_router.search(question, options)
        generated = invoke_chat_model(
            build_reference_messages(question, chunks),
            reference_provider,
        )
        draft = _reference_draft(generated.text)
        evidence_numbers = list(dict.fromkeys(draft.relevant_evidence_numbers))
        if any(number < 1 or number > len(chunks) for number in evidence_numbers):
            raise EvaluationOutputError(
                "the reference model returned an invalid evidence number"
            )
        if draft.answerable and not evidence_numbers:
            raise EvaluationOutputError(
                "an answerable reference must identify supporting evidence"
            )

        retrieved_evidence = [
            snapshot_evidence(chunk, position)
            for position, chunk in enumerate(chunks, start=1)
        ]
        relevant_evidence = (
            [
                retrieved_evidence[number - 1]
                for number in evidence_numbers
            ]
            if draft.answerable
            else []
        )
        required_facts = draft.required_facts if draft.answerable else []
        records.append(
            GroundTruthRecord(
                id=text_sha256(question.casefold())[:16],
                question=question,
                answerable=draft.answerable,
                reference_answer=draft.reference_answer.strip(),
                required_facts=required_facts,
                retrieved_evidence=retrieved_evidence,
                relevant_evidence=relevant_evidence,
                evidence_fingerprint=evidence_fingerprint(retrieved_evidence),
                corpus_fingerprint=corpus_fingerprint,
                generated_at=datetime.now(timezone.utc),
                reference_provider=generated.provider,
                reference_model=generated.model_name,
                reference_usage=generated.usage,
                retrieval=retrieval,
                review_status=ReviewStatus.MACHINE_GENERATED,
            )
        )
    return records


def _retrieval_scores(
    record: GroundTruthRecord,
    chunks: list[RetrievedChunk],
) -> RetrievalScores | None:
    relevant_hashes = {
        evidence.content_sha256 for evidence in record.relevant_evidence
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


def _safe_error(error: Exception) -> str:
    if isinstance(error, LLMProviderRequestError):
        return error.detail
    if isinstance(error, (LLMConfigurationError, EvaluationOutputError)):
        return str(error)
    return f"evaluation failed ({type(error).__name__})"


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
    error: Exception,
) -> EvaluationCaseResult:
    chunks = chunks or []
    total_ms = (perf_counter() - started_at) * 1000
    return EvaluationCaseResult(
        record_id=record.id,
        question=record.question,
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
        retrieval_scores=_retrieval_scores(record, chunks) if chunks else None,
        timing=EvaluationTiming(
            retrieval_ms=round(retrieval_ms, 2),
            answer_generation_ms=round(answer_generation_ms, 2),
            grading_ms=round(grading_ms, 2),
            total_ms=round(total_ms, 2),
        ),
        warnings=warnings or [],
        error=_safe_error(error),
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
    status = ground_truth_status(record)
    warnings = []
    if status is GroundTruthStatus.STALE:
        warnings.append(
            "Ground-truth source content changed; regenerate this dataset."
        )
    elif status is GroundTruthStatus.UNVERIFIABLE:
        warnings.append("Ground-truth source freshness could not be verified.")
    if record.review_status is ReviewStatus.MACHINE_GENERATED:
        warnings.append("Ground truth is machine-generated and needs human review.")
    if answer_provider is judge_provider:
        warnings.append("The answer and judge use the same provider.")

    query_router = router or get_query_router()
    retrieval_started = perf_counter()
    try:
        chunks = query_router.search(
            record.question,
            _retrieval_options(retrieval),
        )
    except Exception as error:
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        return _failed_case(
            record,
            answer_provider,
            judge_provider,
            status,
            started_at,
            retrieval_ms=retrieval_ms,
            warnings=warnings,
            error=error,
        )
    retrieval_ms = (perf_counter() - retrieval_started) * 1000

    answer_started = perf_counter()
    try:
        generated_answer = generate_grounded_answer(
            record.question,
            chunks,
            answer_provider,
        )
    except Exception as error:
        answer_ms = (perf_counter() - answer_started) * 1000
        return _failed_case(
            record,
            answer_provider,
            judge_provider,
            status,
            started_at,
            chunks=chunks,
            retrieval_ms=retrieval_ms,
            answer_generation_ms=answer_ms,
            warnings=warnings,
            error=error,
        )
    answer_ms = (perf_counter() - answer_started) * 1000

    grading_started = perf_counter()
    judged = None
    try:
        judged = invoke_chat_model(
            build_judge_messages(record, generated_answer.answer, chunks),
            judge_provider,
        )
        judge = _judge_draft(judged.text)
    except Exception as error:
        grading_ms = (perf_counter() - grading_started) * 1000
        return _failed_case(
            record,
            answer_provider,
            judge_provider,
            status,
            started_at,
            chunks=chunks,
            retrieval_ms=retrieval_ms,
            answer_generation_ms=answer_ms,
            grading_ms=grading_ms,
            answer=generated_answer.answer,
            answer_model=generated_answer.model_name,
            answer_usage=generated_answer.usage,
            judge_model=judged.model_name if judged is not None else None,
            judge_usage=judged.usage if judged is not None else None,
            warnings=warnings,
            error=error,
        )
    try:
        semantic_score = semantic_similarity(
            record.reference_answer,
            generated_answer.answer,
        )
    except Exception:
        semantic_score = None
        warnings.append("Semantic similarity could not be calculated.")
    grading_ms = (perf_counter() - grading_started) * 1000

    overall = overall_answer_score(
        correctness=judge.correctness,
        groundedness=judge.groundedness,
        completeness=judge.completeness,
        relevance=judge.relevance,
        citation_accuracy=judge.citation_accuracy,
        citation_coverage=judge.citation_coverage,
        abstention_correctness=judge.abstention_correctness,
    )
    answer_scores = AnswerScores(
        correctness=judge.correctness,
        groundedness=judge.groundedness,
        completeness=judge.completeness,
        relevance=judge.relevance,
        citation_accuracy=judge.citation_accuracy,
        citation_coverage=judge.citation_coverage,
        citation_validity=citation_validity(
            generated_answer.answer,
            len(chunks),
        ),
        abstention_correctness=judge.abstention_correctness,
        lexical_similarity=lexical_similarity(
            record.reference_answer,
            generated_answer.answer,
        ),
        semantic_similarity=semantic_score,
        overall=overall,
    )
    passed = (
        overall >= PASS_THRESHOLD
        and judge.correctness >= PASS_THRESHOLD
        and judge.groundedness >= PASS_THRESHOLD
    )
    if status is GroundTruthStatus.STALE:
        passed = None

    total_ms = (perf_counter() - started_at) * 1000
    return EvaluationCaseResult(
        record_id=record.id,
        question=record.question,
        reference_answer=record.reference_answer,
        required_facts=record.required_facts,
        review_status=record.review_status,
        ground_truth_status=status,
        answer_provider=generated_answer.provider,
        answer_model=generated_answer.model_name,
        judge_provider=judged.provider,
        judge_model=judged.model_name,
        answer=generated_answer.answer,
        answer_usage=generated_answer.usage,
        judge_usage=judged.usage,
        usage=summarize_llm_usage(
            [generated_answer.usage, judged.usage]
        ),
        retrieved_chunk_ids=[
            chunk.chunk_id or text_sha256(chunk.content) for chunk in chunks
        ],
        retrieved_sources=list(
            dict.fromkeys(
                chunk.source_file for chunk in chunks if chunk.source_file
            )
        ),
        retrieval_scores=_retrieval_scores(record, chunks),
        answer_scores=answer_scores,
        unsupported_claims=judge.unsupported_claims,
        missing_facts=judge.missing_facts,
        rationale=judge.rationale,
        passed=passed,
        timing=EvaluationTiming(
            retrieval_ms=round(retrieval_ms, 2),
            answer_generation_ms=round(answer_ms, 2),
            grading_ms=round(grading_ms, 2),
            total_ms=round(total_ms, 2),
        ),
        warnings=warnings,
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
        and result.ground_truth_status is not GroundTruthStatus.STALE
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
        correctness=_mean(score.correctness for score in answer_scores),
        groundedness=_mean(score.groundedness for score in answer_scores),
        completeness=_mean(score.completeness for score in answer_scores),
        relevance=_mean(score.relevance for score in answer_scores),
        citation_accuracy=_mean(
            score.citation_accuracy for score in answer_scores
        ),
        citation_coverage=_mean(
            score.citation_coverage for score in answer_scores
        ),
        citation_validity=_mean(
            score.citation_validity for score in answer_scores
        ),
        abstention_correctness=_mean(
            score.abstention_correctness for score in answer_scores
        ),
        lexical_similarity=_mean(
            score.lexical_similarity for score in answer_scores
        ),
        semantic_similarity=_mean(
            score.semantic_similarity for score in answer_scores
        ),
        overall=_mean(score.overall for score in answer_scores),
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
) -> EvaluationRun:
    """Run a dataset now; this function is never called automatically."""
    query_router = router or get_query_router()
    results = [
        evaluate_record(
            record,
            answer_provider,
            judge_provider,
            retrieval,
            router=query_router,
        )
        for answer_provider in answer_providers
        for record in records
    ]
    return EvaluationRun(
        run_id=str(uuid4()),
        dataset_name=dataset_name,
        dataset_sha256=dataset_sha256(dataset_name),
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
