"""Request-time routing across the available retrieval strategies."""

from dataclasses import dataclass
from functools import lru_cache
import re

from app.core.embeddings import create_embedding_model
from app.core.vector_store import connect_vector_store
from app.retrieval.config import RETRIEVAL_SETTINGS
from app.retrieval.hybrid_search import HybridSearch, diversify_by_source
from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import EmbeddingReranker
from app.retrieval.semantic_search import SemanticSearch
from app.retrieval.sql_retrieval import SQLRetrieval, extract_keyword_terms


WHITESPACE_PATTERN = re.compile(r"\s+")
POLITE_PREFIX_PATTERNS = (
    re.compile(
        r"^(?:please\s+)?(?:can|could|would|will)\s+you\s+"
        r"(?:please\s+)?(?:tell|show|explain|find)\s+me\s+(?:about\s+)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^please\s+(?:tell|show|explain|find)\s+me\s+(?:about\s+)?",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True, slots=True)
class RetrievalOptions:
    semantic_search: bool = True
    keyword_search: bool = True
    query_rewriting: bool = False
    reranking: bool = False
    source_diversity: bool = False

    def __post_init__(self) -> None:
        if not self.semantic_search and not self.keyword_search:
            raise ValueError("at least one retrieval strategy must be enabled")


def rewrite_query(question: str) -> str:
    """Create a concise search query without calling a generative model."""
    rewritten = WHITESPACE_PATTERN.sub(" ", question).strip()
    if not rewritten:
        raise ValueError("question must not be empty")

    for pattern in POLITE_PREFIX_PATTERNS:
        candidate, replacements = pattern.subn("", rewritten, count=1)
        candidate = candidate.strip()
        if replacements and candidate:
            rewritten = candidate
            break
    rewritten = rewritten.rstrip(" ?!.").strip() or rewritten
    terms = extract_keyword_terms(rewritten)
    return " ".join(terms) if terms else rewritten


class QueryRouter:
    """Execute the retrieval stages selected for a request."""

    def __init__(
        self,
        hybrid_search: HybridSearch,
        reranker: EmbeddingReranker,
    ) -> None:
        self._hybrid_search = hybrid_search
        self._reranker = reranker

    def search(
        self,
        question: str,
        options: RetrievalOptions | None = None,
    ) -> list[RetrievedChunk]:
        options = options or RetrievalOptions()
        normalized_question = WHITESPACE_PATTERN.sub(" ", question).strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        search_query = (
            rewrite_query(normalized_question)
            if options.query_rewriting
            else normalized_question
        )
        use_candidates = options.reranking or options.source_diversity
        search_limit = (
            RETRIEVAL_SETTINGS.number_of_candidates
            if use_candidates
            else RETRIEVAL_SETTINGS.number_of_chunks
        )
        chunks = self._hybrid_search.search(
            search_query,
            use_semantic=options.semantic_search,
            use_keyword=options.keyword_search,
            limit=search_limit,
        )
        if options.reranking:
            chunks = self._reranker.rerank(search_query, chunks)
        if options.source_diversity:
            chunks = diversify_by_source(
                chunks,
                limit=RETRIEVAL_SETTINGS.number_of_chunks,
            )
        return chunks[: RETRIEVAL_SETTINGS.number_of_chunks]


@lru_cache(maxsize=1)
def get_query_router() -> QueryRouter:
    """Build and cache the local retrieval pipeline dependencies."""
    embedding_model = create_embedding_model()
    vector_store = connect_vector_store(embedding_model)
    semantic_search = SemanticSearch(vector_store)
    keyword_search = KeywordSearch(SQLRetrieval(vector_store))
    return QueryRouter(
        HybridSearch(semantic_search, keyword_search),
        EmbeddingReranker(embedding_model),
    )
