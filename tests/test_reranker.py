import unittest
from unittest.mock import Mock

from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import EmbeddingReranker


def chunk(identifier: str, content: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        relevance_score=score,
        chunk_id=identifier,
        source_file="handbook.md",
        source_type="markdown",
        section="Policy",
        chunk_index=0,
    )


class EmbeddingRerankerTests(unittest.TestCase):
    def test_promotes_content_closest_to_the_question(self) -> None:
        embedding_model = Mock()
        embedding_model.embed_query.return_value = [1.0, 0.0]
        embedding_model.embed_documents.return_value = [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        candidates = [
            chunk("initial-first", "unrelated", 0.9),
            chunk("best-match", "vacation policy", 0.5),
        ]

        results = EmbeddingReranker(
            embedding_model,
            embedding_weight=0.8,
        ).rerank("vacation", candidates)

        self.assertEqual(results[0].chunk_id, "best-match")
        embedding_model.embed_documents.assert_called_once_with(
            ["unrelated", "vacation policy"]
        )


if __name__ == "__main__":
    unittest.main()
