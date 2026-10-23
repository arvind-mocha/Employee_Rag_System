"""Offline embedding-based reranking."""

from dataclasses import replace
from math import sqrt

from langchain_core.embeddings import Embeddings

from app.retrieval.config import RETRIEVAL_SETTINGS
from app.retrieval.models import RetrievedChunk


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have the same dimension")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class EmbeddingReranker:
    """Rerank candidates using local Hugging Face embedding similarity."""

    def __init__(
        self,
        embedding_model: Embeddings,
        *,
        embedding_weight: float = RETRIEVAL_SETTINGS.reranker_embedding_weight,
    ) -> None:
        if not 0.0 <= embedding_weight <= 1.0:
            raise ValueError("embedding_weight must be between 0 and 1")
        self._embedding_model = embedding_model
        self._embedding_weight = embedding_weight

    def rerank(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if not chunks:
            return []

        question_embedding = self._embedding_model.embed_query(normalized_question)
        content_embeddings = self._embedding_model.embed_documents(
            [chunk.content for chunk in chunks]
        )
        if len(content_embeddings) != len(chunks):
            raise ValueError("embedding model returned an unexpected result count")

        retrieval_weight = 1.0 - self._embedding_weight
        reranked = []
        for chunk, content_embedding in zip(chunks, content_embeddings):
            cosine_score = _cosine_similarity(
                question_embedding,
                content_embedding,
            )
            embedding_score = max(0.0, min(1.0, (cosine_score + 1.0) / 2.0))
            combined_score = (
                self._embedding_weight * embedding_score
                + retrieval_weight * chunk.relevance_score
            )
            reranked.append(
                replace(
                    chunk,
                    relevance_score=round(
                        combined_score,
                        RETRIEVAL_SETTINGS.score_decimal_places,
                    ),
                )
            )
        return sorted(
            reranked,
            key=lambda chunk: chunk.relevance_score,
            reverse=True,
        )
