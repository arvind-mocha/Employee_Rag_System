"""Validated models stored in evaluation JSONL files and API results."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.api.llm import LLMProvider
from app.api.llm_usage import LLMUsage, LLMUsageSummary, summarize_llm_usage


class ReviewStatus(StrEnum):
    MACHINE_GENERATED = "machine_generated"
    APPROVED = "approved"


class GroundTruthStatus(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    UNVERIFIABLE = "unverifiable"


class EvaluationCategory(StrEnum):
    """Primary reasoning or retrieval pattern exercised by a test case."""

    DIRECT_FACT = "direct_fact"
    TEMPORAL = "temporal"
    SPANNING = "spanning"
    RELATIONSHIP = "relationship"
    NUMERICAL = "numerical"
    COMPARATIVE = "comparative"
    HOLISTIC = "holistic"
    PROCEDURAL = "procedural"
    UNANSWERABLE = "unanswerable"


class RetrievalConfiguration(BaseModel):
    semantic_search: bool = True
    keyword_search: bool = True
    query_rewriting: bool = False
    reranking: bool = False
    source_diversity: bool = False

    @model_validator(mode="after")
    def require_search_strategy(self) -> "RetrievalConfiguration":
        if not self.semantic_search and not self.keyword_search:
            raise ValueError("at least one retrieval strategy must be enabled")
        return self


class EvidenceSnapshot(BaseModel):
    position: int = Field(ge=1)
    chunk_id: str | None = None
    content: str
    content_sha256: str
    source_file: str | None = None
    source_type: str | None = None
    source_sha256: str | None = None
    section: str | None = None
    chunk_index: int | None = None


class GroundTruthRecord(BaseModel):
    schema_version: Literal[1, 2] = 2
    id: str
    question: str
    answerable: bool
    reference_answer: str
    category: EvaluationCategory = EvaluationCategory.DIRECT_FACT
    keywords: list[str] = Field(default_factory=list, max_length=8)
    required_facts: list[str]
    retrieved_evidence: list[EvidenceSnapshot] = Field(default_factory=list)
    relevant_evidence: list[EvidenceSnapshot]
    evidence_fingerprint: str
    corpus_fingerprint: str | None = None
    generated_at: datetime
    reference_provider: LLMProvider
    reference_model: str
    reference_usage: LLMUsage | None = None
    retrieval: RetrievalConfiguration
    review_status: ReviewStatus = ReviewStatus.MACHINE_GENERATED

    @model_validator(mode="after")
    def align_unanswerable_category(self) -> "GroundTruthRecord":
        if not self.answerable:
            self.category = EvaluationCategory.UNANSWERABLE
            self.keywords = []
        elif self.category is EvaluationCategory.UNANSWERABLE:
            raise ValueError(
                "an answerable record cannot use the unanswerable category"
            )
        return self


class ReferenceDraft(BaseModel):
    answerable: bool
    reference_answer: str
    category: EvaluationCategory
    keywords: list[str] = Field(default_factory=list, max_length=8)
    required_facts: list[str] = Field(default_factory=list)
    relevant_evidence_numbers: list[int] = Field(default_factory=list)


class GeneratedEvaluationQuestion(BaseModel):
    """One corpus-derived question requested for a specific test category."""

    question: str = Field(min_length=1, max_length=2_000)
    category: EvaluationCategory
    relevant_evidence_numbers: list[int] = Field(
        default_factory=list,
        max_length=8,
    )


class QuestionGenerationDraft(BaseModel):
    """Structured question-generation response returned by the reference model."""

    questions: list[GeneratedEvaluationQuestion]


class JudgeDraft(BaseModel):
    completeness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    missing_facts: list[str] = Field(default_factory=list)
    rationale: str


class RetrievalScores(BaseModel):
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    average_precision_at_k: float
    ndcg_at_k: float
    source_precision_at_k: float | None = None
    source_recall_at_k: float | None = None


class AnswerScores(BaseModel):
    completeness: float
    relevance: float
    overall: float


class EvaluationTiming(BaseModel):
    retrieval_ms: float
    answer_generation_ms: float
    grading_ms: float
    total_ms: float


class EvaluationCaseResult(BaseModel):
    record_id: str
    question: str
    category: EvaluationCategory
    reference_answer: str
    required_facts: list[str]
    review_status: ReviewStatus
    ground_truth_status: GroundTruthStatus
    answer_provider: LLMProvider
    answer_model: str | None = None
    judge_provider: LLMProvider
    judge_model: str | None = None
    answer: str | None = None
    answer_usage: LLMUsage | None = None
    judge_usage: LLMUsage | None = None
    usage: LLMUsageSummary = Field(
        default_factory=lambda: summarize_llm_usage([])
    )
    retrieved_chunk_ids: list[str]
    retrieved_sources: list[str]
    retrieval_scores: RetrievalScores | None = None
    answer_scores: AnswerScores | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    rationale: str | None = None
    passed: bool | None = None
    timing: EvaluationTiming
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class ScoreAverages(BaseModel):
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    reciprocal_rank: float | None = None
    average_precision_at_k: float | None = None
    ndcg_at_k: float | None = None
    source_precision_at_k: float | None = None
    source_recall_at_k: float | None = None
    completeness: float | None = None
    relevance: float | None = None
    overall: float | None = None


class ProviderEvaluationSummary(BaseModel):
    provider: LLMProvider
    evaluated: int
    passed: int
    failed: int
    errors: int
    stale_ground_truth: int
    unreviewed_ground_truth: int
    pass_rate: float | None
    average_scores: ScoreAverages
    average_latency_ms: float | None
    usage: LLMUsageSummary = Field(
        default_factory=lambda: summarize_llm_usage([])
    )


class EvaluationRun(BaseModel):
    run_id: str
    dataset_name: str
    dataset_sha256: str
    generated_at: datetime
    judge_provider: LLMProvider
    usage: LLMUsageSummary = Field(
        default_factory=lambda: summarize_llm_usage([])
    )
    summaries: list[ProviderEvaluationSummary]
    results: list[EvaluationCaseResult]
