"""Request and response schemas for on-demand RAG evaluations."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.api.llm import LLMProvider
from app.api.llm_usage import LLMUsageSummary
from app.evaluation.models import EvaluationRun, RetrievalConfiguration
from app.retrieval.config import RETRIEVAL_SETTINGS


class EvaluationRetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_search: bool = True
    keyword_search: bool = True
    query_rewriting: bool = False
    reranking: bool = False
    source_diversity: bool = False

    @model_validator(mode="after")
    def require_search_strategy(self) -> "EvaluationRetrievalRequest":
        if not self.semantic_search and not self.keyword_search:
            raise ValueError("at least one retrieval strategy must be enabled")
        return self

    def retrieval_configuration(self) -> RetrievalConfiguration:
        return RetrievalConfiguration(
            semantic_search=self.semantic_search,
            keyword_search=self.keyword_search,
            query_rewriting=self.query_rewriting,
            reranking=self.reranking,
            source_diversity=self.source_diversity,
        )


class GenerateEvaluationDatasetRequest(EvaluationRetrievalRequest):
    dataset_name: str
    questions: list[str] = Field(min_length=1, max_length=100)
    reference_provider: LLMProvider
    overwrite: bool = False

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, questions: list[str]) -> list[str]:
        normalized = [" ".join(question.split()) for question in questions]
        if any(not question for question in normalized):
            raise ValueError("questions must not contain blank values")
        if any(
            len(question) > RETRIEVAL_SETTINGS.maximum_question_length
            for question in normalized
        ):
            raise ValueError(
                "questions must not exceed the maximum question length"
            )
        return normalized


class GenerateEvaluationDatasetResponse(BaseModel):
    dataset_name: str
    dataset_file: str
    dataset_sha256: str
    record_count: int
    reference_provider: LLMProvider
    reference_model: str
    usage: LLMUsageSummary
    review_required: bool


class UploadEvaluationDatasetResponse(BaseModel):
    dataset_name: str
    dataset_file: str
    dataset_sha256: str
    record_count: int
    review_required: bool


class RunEvaluationRequest(EvaluationRetrievalRequest):
    dataset_name: str
    answer_providers: list[LLMProvider] = Field(min_length=1, max_length=4)
    judge_provider: LLMProvider
    record_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    @field_validator("answer_providers")
    @classmethod
    def unique_answer_providers(
        cls,
        providers: list[LLMProvider],
    ) -> list[LLMProvider]:
        if len(set(providers)) != len(providers):
            raise ValueError("answer_providers must not contain duplicates")
        return providers


class RunEvaluationResponse(EvaluationRun):
    pass
