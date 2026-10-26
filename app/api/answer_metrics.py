"""Deterministic metrics for evaluating generated answers."""

from difflib import SequenceMatcher
from functools import lru_cache
from math import sqrt
import re

from langchain_core.embeddings import Embeddings

from app.core.embeddings import create_embedding_model


CITATION_PATTERN = re.compile(r"\[(\d+)\]")
WORD_PATTERN = re.compile(r"\w+")


def _normalized_text(text: str) -> str:
    return " ".join(WORD_PATTERN.findall(text.casefold()))


def lexical_similarity(reference: str, answer: str) -> float:
    """Return wording similarity without requiring an exact answer match."""
    normalized_reference = _normalized_text(reference)
    normalized_answer = _normalized_text(answer)
    if not normalized_reference or not normalized_answer:
        return 0.0
    return SequenceMatcher(
        None,
        normalized_reference,
        normalized_answer,
    ).ratio()


def citation_validity(answer: str, evidence_count: int) -> float:
    """Return the share of citations that point to supplied evidence."""
    citations = [int(value) for value in CITATION_PATTERN.findall(answer)]
    if not citations:
        return 0.0
    valid = sum(1 <= citation <= evidence_count for citation in citations)
    return valid / len(citations)


@lru_cache(maxsize=1)
def get_evaluation_embedding_model() -> Embeddings:
    return create_embedding_model()


def semantic_similarity(
    reference: str,
    answer: str,
    embedding_model: Embeddings | None = None,
) -> float:
    """Return cosine similarity using the configured local embedding model."""
    model = embedding_model or get_evaluation_embedding_model()
    vectors = model.embed_documents([reference, answer])
    if len(vectors) != 2 or not vectors[0] or not vectors[1]:
        raise ValueError("the embedding model returned invalid vectors")
    reference_vector, answer_vector = vectors
    if len(reference_vector) != len(answer_vector):
        raise ValueError("the embedding vectors have different dimensions")
    dot_product = sum(
        left * right for left, right in zip(reference_vector, answer_vector)
    )
    reference_norm = sqrt(sum(value * value for value in reference_vector))
    answer_norm = sqrt(sum(value * value for value in answer_vector))
    if reference_norm == 0 or answer_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot_product / (reference_norm * answer_norm)))


def overall_answer_score(
    *,
    completeness: float,
    relevance: float,
) -> float:
    """Give relevance and completeness equal weight in answer quality."""
    return (completeness + relevance) / 2
