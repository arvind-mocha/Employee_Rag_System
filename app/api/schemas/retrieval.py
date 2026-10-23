"""Schemas used by retrieval APIs."""

from pydantic import BaseModel, Field, field_validator, model_validator

from app.retrieval.config import RETRIEVAL_SETTINGS


class RetrievalRequest(BaseModel):
    """Question and retrieval options submitted as query parameters."""

    question: str = Field(
        min_length=1,
        max_length=RETRIEVAL_SETTINGS.maximum_question_length,
    )
    semantic_search: bool = Field(
        default=True,
        description="Use Hugging Face embedding similarity search.",
    )
    keyword_search: bool = Field(
        default=True,
        description="Use PostgreSQL full-text keyword search.",
    )
    query_rewriting: bool = Field(
        default=False,
        description="Normalize the question into a concise search query.",
    )
    reranking: bool = Field(
        default=False,
        description="Rerank candidates with local Hugging Face embeddings.",
    )
    source_diversity: bool = Field(
        default=False,
        description="Spread results across distinct source files.",
    )

    @field_validator("question")
    @classmethod
    def normalize_question(cls, question: str) -> str:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        return normalized_question

    @model_validator(mode="after")
    def require_retrieval_strategy(self) -> "RetrievalRequest":
        if not self.semantic_search and not self.keyword_search:
            raise ValueError("at least one retrieval strategy must be enabled")
        return self


class RetrievedChunkResponse(BaseModel):
    """One retrieved chunk with its score and source metadata."""

    content: str
    relevance_score: float
    chunk_id: str | None
    source_file: str | None
    source_type: str | None
    section: str | None
    chunk_index: int | None


class RetrievalResponse(BaseModel):
    """Document retrieval results for one question."""

    question: str
    effective_query: str
    applied_strategies: list[str]
    requested_chunks: int
    returned_chunks: int
    chunks: list[RetrievedChunkResponse]
