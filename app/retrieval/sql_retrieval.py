"""Low-level PostgreSQL full-text retrieval for document content."""

from collections.abc import Sequence
import re

from langchain_core.documents import Document
from langchain_postgres import PGVectorStore
from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig


WORD_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)
QUESTION_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "many",
        "me",
        "of",
        "on",
        "or",
        "please",
        "should",
        "show",
        "tell",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
    }
)
KEYWORD_RANK_CONSTANT = 60
MAX_KEYWORD_TERMS = 8


def extract_keyword_terms(query: str) -> list[str]:
    """Extract meaningful, unique terms from a natural-language question."""
    normalized_query = " ".join(query.split()).strip()
    if not normalized_query:
        raise ValueError("query must not be empty")

    terms = []
    for match in WORD_PATTERN.finditer(normalized_query.casefold()):
        term = match.group(0).strip("'-")
        if not term or term in QUESTION_STOP_WORDS or term in terms:
            continue
        terms.append(term)
        if len(terms) == MAX_KEYWORD_TERMS:
            break

    return terms


def keyword_queries(query: str) -> list[str]:
    """Return a precise phrase plus meaningful terms for OR-like retrieval."""
    normalized_query = " ".join(query.split()).strip()
    terms = extract_keyword_terms(normalized_query)

    if len(terms) <= 1:
        return [terms[0] if terms else normalized_query]
    return [normalized_query, *terms]


def _document_identity(document: Document) -> tuple[str, str]:
    if document.id is not None:
        return ("id", str(document.id))
    return ("content", " ".join(document.page_content.split()).casefold())


def _fuse_keyword_results(
    result_sets: Sequence[list[tuple[Document, float]]],
    limit: int,
) -> list[tuple[Document, float]]:
    scores: dict[tuple[str, str], float] = {}
    documents: dict[tuple[str, str], Document] = {}
    for results in result_sets:
        seen: set[tuple[str, str]] = set()
        for rank, (document, _) in enumerate(results, start=1):
            identity = _document_identity(document)
            if identity in seen:
                continue
            seen.add(identity)
            documents.setdefault(identity, document)
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (
                KEYWORD_RANK_CONSTANT + rank
            )

    if not scores:
        return []

    maximum_score = max(scores.values())
    ranked = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [
        (documents[identity], scores[identity] / maximum_score)
        for identity in ranked
    ]


class SQLRetrieval:
    """Run sparse keyword searches against the PGVector content column."""

    def __init__(self, vector_store: PGVectorStore) -> None:
        self._vector_store = vector_store

    def search(
        self,
        query: str,
        limit: int,
    ) -> list[tuple[Document, float]]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        searches = [
            self._search_once(keyword_query, limit)
            for keyword_query in keyword_queries(normalized_query)
        ]
        return _fuse_keyword_results(searches, limit)

    def _search_once(
        self,
        query: str,
        limit: int,
    ) -> list[tuple[Document, float]]:
        # A zero-sized dense branch makes LangChain Postgres execute only its
        # parameterized PostgreSQL full-text-search branch. Combining one
        # phrase query with individual terms provides OR-like recall while
        # reciprocal-rank fusion rewards documents matching several terms.
        search_config = HybridSearchConfig(
            fts_query=query,
            primary_top_k=0,
            secondary_top_k=limit,
            fusion_function_parameters={
                "primary_results_weight": 0.0,
                "secondary_results_weight": 1.0,
            },
        )
        return self._vector_store.similarity_search_with_score(
            query=query,
            k=limit,
            hybrid_search_config=search_config,
        )
