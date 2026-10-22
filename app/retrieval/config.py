"""Central configuration for semantic retrieval."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """Important settings shared by all retrieval requests."""

    number_of_chunks: int = 5
    maximum_question_length: int = 2_000
    score_decimal_places: int = 4


RETRIEVAL_SETTINGS = RetrievalSettings()
