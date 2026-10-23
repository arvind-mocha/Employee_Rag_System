"""Schemas used by retrieval APIs."""

from pydantic import BaseModel, Field, field_validator

from app.retrieval.config import RETRIEVAL_SETTINGS


class RetrievalRequest(BaseModel):
    """Question and retrieval options submitted as query parameters."""

    question: str = Field(
        min_length=1,
        max_length=RETRIEVAL_SETTINGS.maximum_question_length,
    )
    semantic_search: bool = True
    keyword_search: bool = True
    query_rewriting: bool = False
    reranking: bool = False
    source_diversity: bool = False

    @field_validator("question")
    @classmethod
    def normalize_question(cls, question: str) -> str:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        return normalized_question


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
    """Semantic retrieval results for one question."""

    question: str
    requested_chunks: int
    returned_chunks: int
    chunks: list[RetrievedChunkResponse]
