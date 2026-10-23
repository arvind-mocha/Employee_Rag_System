import unittest
from unittest.mock import Mock, patch

from app.retrieval.models import RetrievedChunk
from app.retrieval.query_router import (
    QueryRouter,
    RetrievalOptions,
    get_query_router,
    rewrite_query,
)


def chunk(identifier: str, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        content=f"content {identifier}",
        relevance_score=0.8,
        chunk_id=identifier,
        source_file=source,
        source_type="markdown",
        section="Policy",
        chunk_index=0,
    )


class QueryRouterTests(unittest.TestCase):
    def test_rewrites_routes_reranks_and_diversifies(self) -> None:
        hybrid_search = Mock()
        reranker = Mock()
        candidates = [chunk("a-1", "a.md"), chunk("b-1", "b.md")]
        hybrid_search.search.return_value = candidates
        reranker.rerank.return_value = list(reversed(candidates))
        router = QueryRouter(hybrid_search, reranker)

        results = router.search(
            " Could you tell me about the vacation policy? ",
            RetrievalOptions(
                semantic_search=True,
                keyword_search=True,
                query_rewriting=True,
                reranking=True,
                source_diversity=True,
            ),
        )

        hybrid_search.search.assert_called_once_with(
            "vacation policy",
            use_semantic=True,
            use_keyword=True,
            limit=20,
        )
        reranker.rerank.assert_called_once_with(
            "vacation policy",
            candidates,
        )
        self.assertEqual([result.chunk_id for result in results], ["b-1", "a-1"])

    def test_rejects_disabling_all_search_strategies(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            RetrievalOptions(semantic_search=False, keyword_search=False)

    def test_rewrite_query_is_deterministic_and_offline(self) -> None:
        self.assertEqual(
            rewrite_query("Please tell me about   vacation benefits?"),
            "vacation benefits",
        )

    @patch("app.retrieval.query_router.connect_vector_store")
    @patch("app.retrieval.query_router.create_embedding_model")
    def test_builds_and_caches_pipeline(
        self,
        create_embedding_model,
        connect_vector_store,
    ) -> None:
        get_query_router.cache_clear()

        first = get_query_router()
        second = get_query_router()

        self.assertIs(first, second)
        create_embedding_model.assert_called_once_with()
        connect_vector_store.assert_called_once_with(
            create_embedding_model.return_value
        )
        get_query_router.cache_clear()


if __name__ == "__main__":
    unittest.main()
