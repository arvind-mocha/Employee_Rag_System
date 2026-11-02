"""Request and response schemas for uploaded evaluation datasets."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.api.llm import LLMProvider
from app.evaluation.dataset import validate_dataset_name
from app.evaluation.models import (
    EvaluationBatchRun,
    GroundTruthRecord,
    RetrievalConfiguration,
)


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


class EvaluationDatasetMetadata(BaseModel):
    dataset_id: str
    filename: str
    record_count: int
    size_bytes: int
    uploaded_at: datetime
    validation_status: Literal["valid", "invalid"]
    validation_error: str | None = None
    dataset_sha256: str


class EvaluationDatasetList(BaseModel):
    datasets: list[EvaluationDatasetMetadata]


class EvaluationDatasetPreview(EvaluationDatasetMetadata):
    records: list[GroundTruthRecord]
    offset: int
    limit: int


class DatasetUploadError(BaseModel):
    filename: str
    error: str


class UploadEvaluationDatasetResponse(EvaluationDatasetList):
    errors: list[DatasetUploadError] = Field(default_factory=list)


class DeleteEvaluationDatasetResponse(BaseModel):
    dataset_id: str
    message: str


class RunEvaluationRequest(EvaluationRetrievalRequest):
    dataset_ids: list[str] = Field(min_length=1, max_length=100)
    answer_providers: list[LLMProvider] = Field(min_length=1, max_length=4)
    judge_provider: LLMProvider
    record_ids: list[str] | None = Field(
        default=None, min_length=1, max_length=10_000,
    )

    @field_validator("dataset_ids")
    @classmethod
    def valid_dataset_ids(cls, ids: list[str]) -> list[str]:
        ids = [validate_dataset_name(dataset_id) for dataset_id in ids]
        if len(set(ids)) != len(ids):
            raise ValueError("dataset_ids must not contain duplicates")
        return ids

    @field_validator("answer_providers")
    @classmethod
    def unique_answer_providers(cls, providers: list[LLMProvider]) -> list[LLMProvider]:
        if len(set(providers)) != len(providers):
            raise ValueError("answer_providers must not contain duplicates")
        return providers


class RunEvaluationResponse(EvaluationBatchRun):
    pass
