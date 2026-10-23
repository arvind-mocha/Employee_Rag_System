import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.api.main import app
from app.retrieval.models import RetrievedChunk


class RetrievalApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("app.api.controllers.retrieval.get_semantic_search")
    def test_returns_retrieved_chunks(self, get_semantic_search) -> None:
        retriever = Mock()
        retriever.search.return_value = [
            RetrievedChunk(
                content="Employees receive 15 vacation days.",
                relevance_score=0.9123,
                chunk_id="chunk-1",
                source_file="handbook.pdf",
                source_type="pdf",
                section="Vacation",
                chunk_index=3,
            )
        ]
        get_semantic_search.return_value = retriever

        response = self.client.get(
            "/api/v1/retrieval/search",
            params={
                "question": "What is the vacation policy?",
                "semantic_search": "true",
                "keyword_search": "true",
                "query_rewriting": "false",
                "reranking": "false",
                "source_diversity": "false",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["requested_chunks"], 5)
        self.assertEqual(response.json()["returned_chunks"], 1)
        self.assertEqual(response.json()["chunks"][0]["chunk_id"], "chunk-1")
        retriever.search.assert_called_once_with("What is the vacation policy?")

    def test_documents_retrieval_options_as_query_parameters(self) -> None:
        operation = app.openapi()["paths"]["/api/v1/retrieval/search"]["get"]
        parameters = {
            parameter["name"]: parameter for parameter in operation["parameters"]
        }

        self.assertEqual(
            set(parameters),
            {
                "question",
                "semantic_search",
                "keyword_search",
                "query_rewriting",
                "reranking",
                "source_diversity",
            },
        )
        self.assertNotIn("requestBody", operation)

    @patch("app.api.controllers.retrieval.get_semantic_search")
    def test_returns_service_unavailable_when_retrieval_fails(
        self,
        get_semantic_search,
    ) -> None:
        get_semantic_search.side_effect = ConnectionError("PostgreSQL is down")

        with self.assertLogs("app.api.controllers.retrieval", level="ERROR"):
            response = self.client.get(
                "/api/v1/retrieval/search",
                params={"question": "What is the vacation policy?"},
            )

        self.assertEqual(response.status_code, 503)

    def test_rejects_a_blank_question(self) -> None:
        response = self.client.get(
            "/api/v1/retrieval/search",
            params={"question": "   "},
        )

        self.assertEqual(response.status_code, 422)

    def test_post_is_not_allowed(self) -> None:
        response = self.client.post(
            "/api/v1/retrieval/search",
            json={"question": "What is the vacation policy?"},
        )

        self.assertEqual(response.status_code, 405)


if __name__ == "__main__":
    unittest.main()
