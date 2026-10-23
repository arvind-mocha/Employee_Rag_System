import unittest

from app.api.answer_metrics import (
    citation_validity,
    lexical_similarity,
    overall_answer_score,
    semantic_similarity,
)


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0], [0.8, 0.6]]


class AnswerMetricTests(unittest.TestCase):
    def test_scores_wording_without_requiring_an_exact_match(self) -> None:
        exact = lexical_similarity("15 vacation days", "15 vacation days")
        paraphrase = lexical_similarity(
            "Employees receive 15 vacation days annually.",
            "Each employee gets 15 days of vacation every year.",
        )

        self.assertEqual(exact, 1.0)
        self.assertGreater(paraphrase, 0.0)

    def test_checks_that_citations_reference_supplied_evidence(self) -> None:
        self.assertEqual(citation_validity("Answer [1] and [3].", 2), 0.5)
        self.assertEqual(citation_validity("Answer without citation.", 2), 0.0)

    def test_calculates_cosine_semantic_similarity(self) -> None:
        score = semantic_similarity("reference", "answer", FakeEmbeddings())

        self.assertAlmostEqual(score, 0.8)

    def test_weights_correctness_and_groundedness_most_heavily(self) -> None:
        score = overall_answer_score(
            correctness=1.0,
            groundedness=1.0,
            completeness=0.0,
            relevance=0.0,
            citation_accuracy=0.0,
            citation_coverage=0.0,
            abstention_correctness=0.0,
        )

        self.assertEqual(score, 0.55)


if __name__ == "__main__":
    unittest.main()
