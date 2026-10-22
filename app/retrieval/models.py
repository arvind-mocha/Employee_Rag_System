"""Models produced by the retrieval layer."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One document chunk returned by semantic search."""

    content: str
    relevance_score: float
    chunk_id: str | None
    source_file: str | None
    source_type: str | None
    section: str | None
    chunk_index: int | None
