"""Basic LangChain semantic retrieval using Hugging Face embeddings."""

from functools import lru_cache
from typing import Any

from langchain_postgres import PGVectorStore

from app.core.embeddings import create_embedding_model
from app.core.vector_store import connect_vector_store
from app.retrieval.config import RETRIEVAL_SETTINGS
from app.retrieval.models import RetrievedChunk


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


class SemanticSearch:
    """Retrieve the most relevant chunks from an existing PGVector table."""

    def __init__(self, vector_store: PGVectorStore) -> None:
        self._vector_store = vector_store

    def search(self, question: str) -> list[RetrievedChunk]:
        """Embed a question and return the configured number of chunks."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        matches = self._vector_store.similarity_search_with_relevance_scores(
            query=normalized_question,
            k=RETRIEVAL_SETTINGS.number_of_chunks,
        )

        chunks: list[RetrievedChunk] = []
        for document, score in matches:
            metadata = document.metadata
            chunks.append(
                RetrievedChunk(
                    content=document.page_content,
                    relevance_score=round(
                        float(score),
                        RETRIEVAL_SETTINGS.score_decimal_places,
                    ),
                    chunk_id=_optional_text(document.id),
                    source_file=_optional_text(metadata.get("source_file")),
                    source_type=_optional_text(metadata.get("source_type")),
                    section=_optional_text(metadata.get("section")),
                    chunk_index=_optional_integer(metadata.get("chunk_index")),
                )
            )
        return chunks


@lru_cache(maxsize=1)
def get_semantic_search() -> SemanticSearch:
    """Create one reusable embedding model and PGVector connection."""
    embedding_model = create_embedding_model()
    vector_store = connect_vector_store(embedding_model)
    return SemanticSearch(vector_store)
