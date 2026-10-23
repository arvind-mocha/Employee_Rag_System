"""PostgreSQL full-text keyword retrieval."""

from app.retrieval.config import RETRIEVAL_SETTINGS
from app.retrieval.models import RetrievedChunk, document_to_retrieved_chunk
from app.retrieval.sql_retrieval import SQLRetrieval


class KeywordSearch:
    """Retrieve chunks using PostgreSQL's ranked full-text search."""

    def __init__(self, sql_retrieval: SQLRetrieval) -> None:
        self._sql_retrieval = sql_retrieval

    def search(
        self,
        question: str,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        result_limit = (
            RETRIEVAL_SETTINGS.number_of_chunks if limit is None else limit
        )
        if result_limit < 1:
            raise ValueError("limit must be at least 1")
        matches = self._sql_retrieval.search(normalized_question, result_limit)
        return [
            document_to_retrieved_chunk(
                document,
                score,
                decimal_places=RETRIEVAL_SETTINGS.score_decimal_places,
            )
            for document, score in matches
        ]
