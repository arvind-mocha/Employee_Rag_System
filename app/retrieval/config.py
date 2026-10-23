"""Central configuration for document retrieval."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """Important settings shared by all retrieval requests."""

    number_of_chunks: int = 5
    number_of_candidates: int = 20
    maximum_question_length: int = 2_000
    score_decimal_places: int = 4
    reciprocal_rank_constant: int = 60
    reranker_embedding_weight: float = 0.7


RETRIEVAL_SETTINGS = RetrievalSettings()
