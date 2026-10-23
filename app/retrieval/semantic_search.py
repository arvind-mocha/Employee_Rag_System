"""Basic LangChain semantic retrieval using Hugging Face embeddings."""

from functools import lru_cache

from langchain_postgres import PGVectorStore

from app.core.embeddings import create_embedding_model
from app.core.vector_store import connect_vector_store
from app.retrieval.config import RETRIEVAL_SETTINGS
from app.retrieval.models import RetrievedChunk, document_to_retrieved_chunk


class SemanticSearch:
    """Retrieve the most relevant chunks from an existing PGVector table."""

    def __init__(self, vector_store: PGVectorStore) -> None:
        self._vector_store = vector_store

    def search(
        self,
        question: str,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        """Embed a question and return the most similar chunks."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        result_limit = (
            RETRIEVAL_SETTINGS.number_of_chunks if limit is None else limit
        )
        if result_limit < 1:
            raise ValueError("limit must be at least 1")

        matches = self._vector_store.similarity_search_with_relevance_scores(
            query=normalized_question,
            k=result_limit,
        )
        return [
            document_to_retrieved_chunk(
                document,
                score,
                decimal_places=RETRIEVAL_SETTINGS.score_decimal_places,
            )
            for document, score in matches
        ]


@lru_cache(maxsize=1)
def get_semantic_search() -> SemanticSearch:
    """Create one reusable embedding model and PGVector connection."""
    embedding_model = create_embedding_model()
    vector_store = connect_vector_store(embedding_model)
    return SemanticSearch(vector_store)
