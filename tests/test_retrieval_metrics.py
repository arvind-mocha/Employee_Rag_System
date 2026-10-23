import unittest

from app.retrieval.retrieval_metrics import evaluate_retrieval


class RetrievalMetricsTests(unittest.TestCase):
    def test_calculates_standard_binary_relevance_metrics(self) -> None:
        metrics = evaluate_retrieval(
            ["irrelevant", "relevant-1", "relevant-2"],
            {"relevant-1", "relevant-2"},
            limit=3,
        )

        self.assertAlmostEqual(metrics.precision, 2 / 3)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.reciprocal_rank, 0.5)
        self.assertAlmostEqual(metrics.average_precision, (0.5 + 2 / 3) / 2)
        self.assertGreater(metrics.ndcg, 0.0)
        self.assertLess(metrics.ndcg, 1.0)

    def test_handles_queries_without_relevant_documents(self) -> None:
        metrics = evaluate_retrieval(["chunk-1"], set(), limit=5)

        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.reciprocal_rank, 0.0)
        self.assertEqual(metrics.ndcg, 0.0)


if __name__ == "__main__":
    unittest.main()
