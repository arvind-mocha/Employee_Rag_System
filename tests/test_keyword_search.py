import unittest
from unittest.mock import Mock

from langchain_core.documents import Document

from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.sql_retrieval import SQLRetrieval


class SQLRetrievalTests(unittest.TestCase):
    def test_uses_postgres_full_text_search_without_dense_results(self) -> None:
        vector_store = Mock()
        vector_store.similarity_search_with_score.return_value = []

        SQLRetrieval(vector_store).search(" vacation policy ", 8)

        self.assertEqual(vector_store.similarity_search_with_score.call_count, 3)
        queries = []
        for call in vector_store.similarity_search_with_score.call_args_list:
            self.assertEqual(call.kwargs["k"], 8)
            config = call.kwargs["hybrid_search_config"]
            queries.append(config.fts_query)
            self.assertEqual(config.primary_top_k, 0)
            self.assertEqual(config.secondary_top_k, 8)
        self.assertEqual(queries, ["vacation policy", "vacation", "policy"])


class KeywordSearchTests(unittest.TestCase):
    def test_converts_ranked_documents_to_chunks(self) -> None:
        sql_retrieval = Mock()
        sql_retrieval.search.return_value = [
            (
                Document(
                    id="chunk-1",
                    page_content="Employees receive 15 vacation days.",
                    metadata={
                        "source_file": "handbook.md",
                        "source_type": "markdown",
                        "section": "Vacation",
                        "chunk_index": 2,
                    },
                ),
                0.875432,
            )
        ]

        chunks = KeywordSearch(sql_retrieval).search("vacation", 5)

        sql_retrieval.search.assert_called_once_with("vacation", 5)
        self.assertEqual(chunks[0].chunk_id, "chunk-1")
        self.assertEqual(chunks[0].relevance_score, 0.8754)
        self.assertEqual(chunks[0].source_file, "handbook.md")


if __name__ == "__main__":
    unittest.main()
