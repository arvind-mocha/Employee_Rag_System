"""Hybrid retrieval and result diversification."""

from collections import OrderedDict
from dataclasses import replace

from app.retrieval.config import RETRIEVAL_SETTINGS
from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.models import RetrievedChunk
from app.retrieval.semantic_search import SemanticSearch


def _chunk_identity(chunk: RetrievedChunk) -> tuple[object, ...]:
    normalized_content = " ".join(chunk.content.split()).casefold()
    if normalized_content:
        return ("content", normalized_content)
    return (
        "metadata",
        chunk.chunk_id,
        chunk.source_file,
        chunk.section,
        chunk.chunk_index,
    )


def deduplicate_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Remove exact-content duplicates while preserving the best-ranked copy."""
    deduplicated: list[RetrievedChunk] = []
    seen: set[tuple[object, ...]] = set()
    for chunk in chunks:
        identity = _chunk_identity(chunk)
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(chunk)
    return deduplicated


def reciprocal_rank_fusion(
    result_sets: list[list[RetrievedChunk]],
    *,
    limit: int,
    rank_constant: int = RETRIEVAL_SETTINGS.reciprocal_rank_constant,
) -> list[RetrievedChunk]:
    """Deduplicate and merge ranked lists with reciprocal-rank fusion."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if rank_constant < 1:
        raise ValueError("rank_constant must be at least 1")

    nonempty_sets = [results for results in result_sets if results]
    if not nonempty_sets:
        return []

    scores: dict[tuple[object, ...], float] = {}
    chunks: dict[tuple[object, ...], RetrievedChunk] = {}
    for results in nonempty_sets:
        seen_in_result: set[tuple[object, ...]] = set()
        for rank, chunk in enumerate(results, start=1):
            identity = _chunk_identity(chunk)
            if identity in seen_in_result:
                continue
            seen_in_result.add(identity)
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (
                rank_constant + rank
            )
            current = chunks.get(identity)
            if current is None or chunk.relevance_score > current.relevance_score:
                chunks[identity] = chunk

    maximum_score = len(nonempty_sets) / (rank_constant + 1)
    ranked_identities = sorted(
        scores,
        key=lambda identity: scores[identity],
        reverse=True,
    )[:limit]
    return [
        replace(
            chunks[identity],
            relevance_score=round(
                min(1.0, scores[identity] / maximum_score),
                RETRIEVAL_SETTINGS.score_decimal_places,
            ),
        )
        for identity in ranked_identities
    ]


def diversify_by_source(
    chunks: list[RetrievedChunk],
    *,
    limit: int,
) -> list[RetrievedChunk]:
    """Select results round-robin across source files while preserving rank."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    groups: OrderedDict[str, list[RetrievedChunk]] = OrderedDict()
    for index, chunk in enumerate(deduplicate_chunks(chunks)):
        source = chunk.source_file or f"__unknown_source_{index}"
        groups.setdefault(source, []).append(chunk)

    diversified: list[RetrievedChunk] = []
    depth = 0
    while len(diversified) < limit:
        added = False
        for source_chunks in groups.values():
            if depth < len(source_chunks):
                diversified.append(source_chunks[depth])
                added = True
                if len(diversified) == limit:
                    break
        if not added:
            break
        depth += 1
    return diversified


class HybridSearch:
    """Route to semantic, keyword, or fused hybrid retrieval."""

    def __init__(
        self,
        semantic_search: SemanticSearch,
        keyword_search: KeywordSearch,
    ) -> None:
        self._semantic_search = semantic_search
        self._keyword_search = keyword_search

    def search(
        self,
        question: str,
        *,
        use_semantic: bool,
        use_keyword: bool,
        limit: int,
    ) -> list[RetrievedChunk]:
        if not use_semantic and not use_keyword:
            raise ValueError("at least one retrieval strategy must be enabled")

        result_sets = []
        if use_semantic:
            result_sets.append(self._semantic_search.search(question, limit))
        if use_keyword:
            result_sets.append(self._keyword_search.search(question, limit))

        if len(result_sets) == 1:
            return deduplicate_chunks(result_sets[0])[:limit]
        return reciprocal_rank_fusion(result_sets, limit=limit)
