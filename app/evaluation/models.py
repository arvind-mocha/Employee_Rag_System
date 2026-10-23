"""Validated models stored in evaluation JSONL files and API results."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.api.llm import LLMProvider


class ReviewStatus(StrEnum):
    MACHINE_GENERATED = "machine_generated"
    APPROVED = "approved"


class GroundTruthStatus(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    UNVERIFIABLE = "unverifiable"


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
    schema_version: Literal[1] = 1
    id: str
    question: str
    answerable: bool
    reference_answer: str
    required_facts: list[str]
    retrieved_evidence: list[EvidenceSnapshot] = Field(default_factory=list)
    relevant_evidence: list[EvidenceSnapshot]
    evidence_fingerprint: str
    corpus_fingerprint: str | None = None
    generated_at: datetime
    reference_provider: LLMProvider
    reference_model: str
    retrieval: RetrievalConfiguration
    review_status: ReviewStatus = ReviewStatus.MACHINE_GENERATED


class ReferenceDraft(BaseModel):
    answerable: bool
    reference_answer: str
    required_facts: list[str] = Field(default_factory=list)
    relevant_evidence_numbers: list[int] = Field(default_factory=list)


class JudgeDraft(BaseModel):
    correctness: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    citation_accuracy: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    abstention_correctness: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
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
    correctness: float
    groundedness: float
    completeness: float
    relevance: float
    citation_accuracy: float
    citation_coverage: float
    citation_validity: float
    abstention_correctness: float
    lexical_similarity: float
    semantic_similarity: float | None
    overall: float


class EvaluationTiming(BaseModel):
    retrieval_ms: float
    answer_generation_ms: float
    grading_ms: float
    total_ms: float


class EvaluationCaseResult(BaseModel):
    record_id: str
    question: str
    reference_answer: str
    required_facts: list[str]
    review_status: ReviewStatus
    ground_truth_status: GroundTruthStatus
    answer_provider: LLMProvider
    answer_model: str | None = None
    judge_provider: LLMProvider
    judge_model: str | None = None
    answer: str | None = None
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
    correctness: float | None = None
    groundedness: float | None = None
    completeness: float | None = None
    relevance: float | None = None
    citation_accuracy: float | None = None
    citation_coverage: float | None = None
    citation_validity: float | None = None
    abstention_correctness: float | None = None
    lexical_similarity: float | None = None
    semantic_similarity: float | None = None
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


class EvaluationRun(BaseModel):
    run_id: str
    dataset_name: str
    dataset_sha256: str
    generated_at: datetime
    judge_provider: LLMProvider
    summaries: list[ProviderEvaluationSummary]
    results: list[EvaluationCaseResult]
