"""Metrics for evaluating retrieval quality."""

from dataclasses import dataclass
from math import log2
from typing import Iterable, Sequence


def _validate_limit(limit: int) -> None:
    if limit < 1:
        raise ValueError("limit must be at least 1")


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    limit: int,
) -> float:
    _validate_limit(limit)
    relevant = set(relevant_ids)
    retrieved = _unique(retrieved_ids)[:limit]
    return sum(identifier in relevant for identifier in retrieved) / limit


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    limit: int,
) -> float:
    _validate_limit(limit)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    retrieved = set(_unique(retrieved_ids)[:limit])
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    limit: int,
) -> float:
    _validate_limit(limit)
    relevant = set(relevant_ids)
    for rank, identifier in enumerate(_unique(retrieved_ids)[:limit], start=1):
        if identifier in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    limit: int,
) -> float:
    _validate_limit(limit)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, identifier in enumerate(_unique(retrieved_ids)[:limit], start=1):
        if identifier in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant), limit)


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    limit: int,
) -> float:
    _validate_limit(limit)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    retrieved = _unique(retrieved_ids)[:limit]
    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, identifier in enumerate(retrieved, start=1)
        if identifier in relevant
    )
    ideal_hits = min(len(relevant), limit)
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    precision: float
    recall: float
    reciprocal_rank: float
    average_precision: float
    ndcg: float


def evaluate_retrieval(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    limit: int,
) -> RetrievalEvaluation:
    """Calculate a standard binary-relevance metric set for one query."""
    relevant = set(relevant_ids)
    return RetrievalEvaluation(
        precision=precision_at_k(retrieved_ids, relevant, limit),
        recall=recall_at_k(retrieved_ids, relevant, limit),
        reciprocal_rank=reciprocal_rank(retrieved_ids, relevant, limit),
        average_precision=average_precision_at_k(
            retrieved_ids,
            relevant,
            limit,
        ),
        ndcg=ndcg_at_k(retrieved_ids, relevant, limit),
    )
