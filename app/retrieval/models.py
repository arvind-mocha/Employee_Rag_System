"""Models produced by the retrieval layer."""

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One document chunk returned by a retrieval strategy."""

    content: str
    relevance_score: float
    chunk_id: str | None
    source_file: str | None
    source_type: str | None
    section: str | None
    chunk_index: int | None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_integer(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def document_to_retrieved_chunk(
    document: Document,
    relevance_score: float,
    *,
    decimal_places: int,
) -> RetrievedChunk:
    """Convert a LangChain document and score into the public model."""
    metadata = document.metadata
    bounded_score = max(0.0, min(1.0, float(relevance_score)))
    return RetrievedChunk(
        content=document.page_content,
        relevance_score=round(bounded_score, decimal_places),
        chunk_id=_optional_text(document.id),
        source_file=_optional_text(metadata.get("source_file")),
        source_type=_optional_text(metadata.get("source_type")),
        section=_optional_text(metadata.get("section")),
        chunk_index=_optional_integer(metadata.get("chunk_index")),
    )
