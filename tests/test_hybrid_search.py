import unittest
from unittest.mock import Mock

from app.retrieval.hybrid_search import (
    HybridSearch,
    diversify_by_source,
    reciprocal_rank_fusion,
)
from app.retrieval.models import RetrievedChunk


def chunk(identifier: str, score: float, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        content=f"content {identifier}",
        relevance_score=score,
        chunk_id=identifier,
        source_file=source,
        source_type="markdown",
        section="Policy",
        chunk_index=0,
    )


class HybridSearchTests(unittest.TestCase):
    def test_fusion_rewards_chunks_found_by_both_strategies(self) -> None:
        semantic_results = [
            chunk("semantic-only", 0.9, "a.md"),
            chunk("shared", 0.8, "b.md"),
        ]
        keyword_results = [
            chunk("shared", 0.7, "b.md"),
            chunk("keyword-only", 0.6, "c.md"),
        ]

        results = reciprocal_rank_fusion(
            [semantic_results, keyword_results],
            limit=3,
        )

        self.assertEqual([result.chunk_id for result in results], [
            "shared",
            "semantic-only",
            "keyword-only",
        ])
        self.assertEqual(len({result.chunk_id for result in results}), 3)

    def test_routes_to_only_the_enabled_strategy(self) -> None:
        semantic_search = Mock()
        keyword_search = Mock()
        semantic_search.search.return_value = [chunk("semantic", 0.9, "a.md")]

        results = HybridSearch(semantic_search, keyword_search).search(
            "vacation",
            use_semantic=True,
            use_keyword=False,
            limit=5,
        )

        self.assertEqual(results[0].chunk_id, "semantic")
        semantic_search.search.assert_called_once_with("vacation", 5)
        keyword_search.search.assert_not_called()

    def test_diversifies_results_across_source_files(self) -> None:
        results = diversify_by_source(
            [
                chunk("a-1", 0.9, "a.md"),
                chunk("a-2", 0.8, "a.md"),
                chunk("b-1", 0.7, "b.md"),
                chunk("c-1", 0.6, "c.md"),
            ],
            limit=4,
        )

        self.assertEqual(
            [result.chunk_id for result in results],
            ["a-1", "b-1", "c-1", "a-2"],
        )


if __name__ == "__main__":
    unittest.main()
