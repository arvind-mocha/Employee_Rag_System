"""Ground-truth generation and on-demand RAG evaluation engine."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import json
from statistics import fmean
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from app.api.answer_metrics import (
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
from app.api import config
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
    EvaluationCategory,
    EvaluationCaseResult,
    EvaluationRun,
    EvaluationTiming,
    GroundTruthRecord,
    GroundTruthStatus,
    GeneratedEvaluationQuestion,
    JudgeDraft,
    ProviderEvaluationSummary,
    ReferenceDraft,
    QuestionGenerationDraft,
    RetrievalConfiguration,
    RetrievalScores,
    ReviewStatus,
    ScoreAverages,
)
from app.evaluation.prompts import (
    build_judge_messages,
    build_question_generation_messages,
    build_reference_messages,
)
from app.ingestion.chunker import (
    DocumentChunk,
    chunk_documents,
    chunk_structured_data,
)
from app.ingestion.ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKING_BATCH_SIZE,
    MAX_WORKERS,
)
from app.ingestion.evaluation_snapshot import (
    read_evaluation_corpus_snapshot,
    write_evaluation_corpus_snapshot,
)
from app.retrieval.config import RETRIEVAL_SETTINGS
from app.retrieval.models import RetrievedChunk
from app.retrieval.query_router import QueryRouter, RetrievalOptions, get_query_router
from app.retrieval.retrieval_metrics import evaluate_retrieval


PASS_THRESHOLD = 0.80
QUESTION_CONTEXT_MAX_CHUNKS = 24
QUESTION_CONTEXT_MAX_CHARACTERS = 16_000
QUESTION_GENERATION_CATEGORY_BATCH_SIZE = 3
QUESTION_GENERATION_BASE_OUTPUT_TOKENS = 512
QUESTION_GENERATION_OUTPUT_TOKENS_PER_QUESTION = 128
QUESTION_GENERATION_MAX_OUTPUT_TOKENS = 8_192
REFERENCE_GENERATION_MAX_OUTPUT_TOKENS = 4_096
JUDGE_GENERATION_MAX_OUTPUT_TOKENS = 4_096


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
    if draft.answerable and not draft.keywords:
        raise EvaluationOutputError(
            "an answerable reference must contain keywords"
        )
    if draft.answerable and draft.category is EvaluationCategory.UNANSWERABLE:
        raise EvaluationOutputError(
            "an answerable reference cannot use the unanswerable category"
        )
    return draft


def _judge_draft(text: str) -> JudgeDraft:
    try:
        return JudgeDraft.model_validate(_parse_json_object(text))
    except ValidationError as error:
        raise EvaluationOutputError(
            "the judge returned an invalid JSON schema"
        ) from error


def _question_generation_draft(text: str) -> QuestionGenerationDraft:
    try:
        return QuestionGenerationDraft.model_validate(_parse_json_object(text))
    except ValidationError as error:
        raise EvaluationOutputError(
            "the question generator returned an invalid JSON schema"
        ) from error


def _ingested_chunks() -> list[DocumentChunk]:
    """Recreate chunks from artifacts already produced by ingestion."""
    if config.EVALUATION_CORPUS_SNAPSHOT_FILE.is_file():
        return read_evaluation_corpus_snapshot()

    chunks: list[DocumentChunk] = []
    if config.TEMP_DATA_DIRECTORY.is_dir():
        chunks.extend(
            chunk_documents(
                input_directory=config.TEMP_DATA_DIRECTORY,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                batch_size=CHUNKING_BATCH_SIZE,
                max_workers=MAX_WORKERS,
            )
        )
    if config.STRUCTURED_DATA_DIRECTORY.is_dir():
        chunks.extend(
            chunk_structured_data(
                input_directory=config.STRUCTURED_DATA_DIRECTORY,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                batch_size=CHUNKING_BATCH_SIZE,
                max_workers=MAX_WORKERS,
            )
        )
    if not chunks:
        raise ValueError(
            "No ingested content is available. Complete ingestion before "
            "generating an evaluation dataset."
        )
    write_evaluation_corpus_snapshot(chunks)
    return chunks


def _diverse_question_context(
    chunks: list[DocumentChunk],
) -> list[DocumentChunk]:
    """Select bounded, source-diverse excerpts for question generation."""
    by_source: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source_file].append(chunk)

    interleaved: list[DocumentChunk] = []
    position = 0
    sources = sorted(by_source)

    while True:
        added = False
        for source in sources:
            source_chunks = by_source[source]
            if position >= len(source_chunks):
                continue
            interleaved.append(source_chunks[position])
            added = True
        if not added:
            break
        position += 1

    candidates = interleaved
    if len(interleaved) > QUESTION_CONTEXT_MAX_CHUNKS:
        last_index = len(interleaved) - 1
        candidates = [
            interleaved[
                round(
                    index
                    * last_index
                    / (QUESTION_CONTEXT_MAX_CHUNKS - 1)
                )
            ]
            for index in range(QUESTION_CONTEXT_MAX_CHUNKS)
        ]

    selected: list[DocumentChunk] = []
    selected_characters = 0
    for chunk in candidates:
        character_count = min(len(chunk.content), 1_200)
        if (
            selected
            and selected_characters + character_count
            > QUESTION_CONTEXT_MAX_CHARACTERS
        ):
            break
        selected.append(chunk)
        selected_characters += character_count
    return selected


def generate_evaluation_questions(
    reference_provider: LLMProvider,
    questions_per_category: int,
    *,
    chunks: list[DocumentChunk] | None = None,
) -> tuple[
    list[GeneratedEvaluationQuestion],
    list[LLMUsage | None],
    dict[str, list[DocumentChunk]],
]:
    """Generate a balanced question set from ingested corpus excerpts."""
    if questions_per_category < 1 or questions_per_category > 5:
        raise ValueError("questions_per_category must be between 1 and 5")

    available_chunks = _ingested_chunks() if chunks is None else chunks
    if not available_chunks:
        raise ValueError(
            "No ingested content is available. Complete ingestion before "
            "generating an evaluation dataset."
        )
    context = _diverse_question_context(available_chunks)
    questions: list[GeneratedEvaluationQuestion] = []
    usages: list[LLMUsage | None] = []
    categories = tuple(EvaluationCategory)
    for start in range(
        0,
        len(categories),
        QUESTION_GENERATION_CATEGORY_BATCH_SIZE,
    ):
        category_batch = categories[
            start : start + QUESTION_GENERATION_CATEGORY_BATCH_SIZE
        ]
        expected_question_count = len(category_batch) * questions_per_category
        max_output_tokens = min(
            QUESTION_GENERATION_MAX_OUTPUT_TOKENS,
            QUESTION_GENERATION_BASE_OUTPUT_TOKENS
            + expected_question_count
            * QUESTION_GENERATION_OUTPUT_TOKENS_PER_QUESTION,
        )
        generated = invoke_chat_model(
            build_question_generation_messages(
                context,
                questions_per_category,
                category_batch,
            ),
            reference_provider,
            max_output_tokens=max_output_tokens,
            response_schema=QuestionGenerationDraft,
        )
        draft = _question_generation_draft(generated.text)
        counts = Counter(question.category for question in draft.questions)
        expected_categories = set(category_batch)
        if set(counts) != expected_categories or any(
            counts[category] != questions_per_category
            for category in expected_categories
        ):
            raise EvaluationOutputError(
                "the question generator did not return the requested number of "
                "questions for every category"
            )
        for question in draft.questions:
            evidence_numbers = list(
                dict.fromkeys(question.relevant_evidence_numbers)
            )
            if any(
                number < 1 or number > len(context)
                for number in evidence_numbers
            ):
                raise EvaluationOutputError(
                    "the question generator returned an invalid evidence number"
                )
            if (
                question.category is EvaluationCategory.UNANSWERABLE
                and evidence_numbers
            ):
                raise EvaluationOutputError(
                    "an unanswerable question cannot identify supporting evidence"
                )
            if (
                question.category is not EvaluationCategory.UNANSWERABLE
                and not evidence_numbers
            ):
                raise EvaluationOutputError(
                    "an answerable generated question must identify supporting "
                    "evidence"
                )
            question.relevant_evidence_numbers = evidence_numbers
        questions.extend(draft.questions)
        usages.append(generated.usage)

    normalized_questions = [
        " ".join(question.question.split()) for question in questions
    ]
    if any(not question for question in normalized_questions):
        raise EvaluationOutputError(
            "the question generator returned a blank question"
        )
    if len({question.casefold() for question in normalized_questions}) != len(
        normalized_questions
    ):
        raise EvaluationOutputError(
            "the question generator returned duplicate questions"
        )

    normalized = [
        question.model_copy(update={"question": normalized})
        for question, normalized in zip(questions, normalized_questions)
    ]
    evidence_hints = {
        question.question.casefold(): [
            context[number - 1]
            for number in question.relevant_evidence_numbers
        ]
        for question in normalized
    }
    return normalized, usages, evidence_hints


def _retrieved_document_chunk(chunk: DocumentChunk) -> RetrievedChunk:
    return RetrievedChunk(
        content=chunk.content,
        relevance_score=1.0,
        chunk_id=chunk.chunk_id,
        source_file=chunk.source_file,
        source_type=chunk.source_type,
        section=chunk.section,
        chunk_index=chunk.chunk_index,
    )


def _reference_evidence(
    retrieved: list[RetrievedChunk],
    anchored: Iterable[DocumentChunk],
) -> list[RetrievedChunk]:
    """Put source anchors first, followed by deduplicated retrieval results."""
    combined = [*map(_retrieved_document_chunk, anchored), *retrieved]
    selected = []
    seen_content = set()
    for chunk in combined:
        content_hash = text_sha256(chunk.content)
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        selected.append(chunk)
    return selected


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
    category_hints: Mapping[str, EvaluationCategory] | None = None,
    evidence_hints: Mapping[str, Iterable[DocumentChunk]] | None = None,
) -> list[GroundTruthRecord]:
    """Create machine-generated, reviewable ground truth from current evidence."""
    query_router = router or get_query_router()
    options = _retrieval_options(retrieval)
    records = []
    corpus_fingerprint = indexed_corpus_fingerprint()

    for question in _normalized_questions(questions):
        category_hint = (category_hints or {}).get(question.casefold())
        retrieved_chunks = query_router.search(question, options)
        chunks = _reference_evidence(
            retrieved_chunks,
            (evidence_hints or {}).get(question.casefold(), ()),
        )
        generated = invoke_chat_model(
            build_reference_messages(question, chunks),
            reference_provider,
            max_output_tokens=REFERENCE_GENERATION_MAX_OUTPUT_TOKENS,
            response_schema=ReferenceDraft,
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
                category=(
                    category_hint
                    if draft.answerable
                    and category_hint is not None
                    and category_hint is not EvaluationCategory.UNANSWERABLE
                    else (
                        draft.category
                        if draft.answerable
                        else EvaluationCategory.UNANSWERABLE
                    )
                ),
                keywords=draft.keywords if draft.answerable else [],
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
            max_output_tokens=JUDGE_GENERATION_MAX_OUTPUT_TOKENS,
            response_schema=JudgeDraft,
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
    grading_ms = (perf_counter() - grading_started) * 1000

    overall = overall_answer_score(
        completeness=judge.completeness,
        relevance=judge.relevance,
    )
    answer_scores = AnswerScores(
        completeness=judge.completeness,
        relevance=judge.relevance,
        overall=overall,
    )
    passed = (
        judge.completeness >= PASS_THRESHOLD
        and judge.relevance >= PASS_THRESHOLD
    )

    total_ms = (perf_counter() - started_at) * 1000
    return EvaluationCaseResult(
        record_id=record.id,
        question=record.question,
        category=record.category,
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
