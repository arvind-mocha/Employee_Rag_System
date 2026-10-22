import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from app.retrieval.semantic_search import SemanticSearch, get_semantic_search


class SemanticSearchTests(unittest.TestCase):
    def test_returns_chunks_with_scores_and_metadata(self) -> None:
        vector_store = Mock()
        vector_store.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    id="chunk-1",
                    page_content="Employees receive 15 vacation days.",
                    metadata={
                        "source_file": "handbook.pdf",
                        "source_type": "pdf",
                        "section": "Vacation",
                        "chunk_index": 3,
                    },
                ),
                0.912345,
            )
        ]

        chunks = SemanticSearch(vector_store).search(" vacation policy ")

        vector_store.similarity_search_with_relevance_scores.assert_called_once_with(
            query="vacation policy",
            k=5,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "chunk-1")
        self.assertEqual(chunks[0].relevance_score, 0.9123)
        self.assertEqual(chunks[0].source_file, "handbook.pdf")
        self.assertEqual(chunks[0].chunk_index, 3)

    def test_rejects_an_empty_question(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            SemanticSearch(Mock()).search("   ")

    @patch("app.retrieval.semantic_search.connect_vector_store")
    @patch("app.retrieval.semantic_search.create_embedding_model")
    def test_builds_and_caches_retrieval_dependencies(
        self,
        create_embedding_model,
        connect_vector_store,
    ) -> None:
        get_semantic_search.cache_clear()

        first = get_semantic_search()
        second = get_semantic_search()

        self.assertIs(first, second)
        create_embedding_model.assert_called_once_with()
        connect_vector_store.assert_called_once_with(
            create_embedding_model.return_value
        )
        get_semantic_search.cache_clear()


if __name__ == "__main__":
    unittest.main()
