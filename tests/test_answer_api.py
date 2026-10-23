import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.api.llm import (
    GeneratedAnswer,
    LLMConfigurationError,
    LLMProvider,
    LLMProviderRequestError,
)
from app.api.main import app
from app.retrieval.models import RetrievedChunk
from app.retrieval.query_router import RetrievalOptions


def retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        content="Employees receive 15 vacation days.",
        relevance_score=0.91,
        chunk_id="chunk-1",
        source_file="handbook.md",
        source_type="markdown",
        section="Vacation",
        chunk_index=1,
    )


class AnswerApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("app.api.controllers.answer.generate_grounded_answer")
    @patch("app.api.controllers.answer.get_query_router")
    def test_defaults_to_huggingface_and_returns_sources(
        self,
        get_query_router,
        generate_answer,
    ) -> None:
        router = Mock()
        router.search.return_value = [retrieved_chunk()]
        get_query_router.return_value = router
        generate_answer.return_value = GeneratedAnswer(
            "Employees receive 15 vacation days [1].",
            LLMProvider.HUGGINGFACE,
            "google/gemma-2-2b-it",
        )

        response = self.client.post(
            "/api/v1/answers/generate",
            json={
                "question": "How many vacation days do employees receive?",
                "query_rewriting": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["provider"], "huggingface")
        self.assertEqual(body["model"], "google/gemma-2-2b-it")
        self.assertEqual(body["effective_query"], "vacation days employees receive")
        self.assertEqual(body["sources"][0]["chunk_id"], "chunk-1")
        options = RetrievalOptions(query_rewriting=True)
        router.search.assert_called_once_with(
            "How many vacation days do employees receive?",
            options,
        )
        generate_answer.assert_called_once_with(
            "How many vacation days do employees receive?",
            router.search.return_value,
            LLMProvider.HUGGINGFACE,
        )

    @patch("app.api.controllers.answer.generate_grounded_answer")
    @patch("app.api.controllers.answer.get_query_router")
    def test_selects_openai_when_enabled(
        self,
        get_query_router,
        generate_answer,
    ) -> None:
        get_query_router.return_value.search.return_value = [retrieved_chunk()]
        generate_answer.return_value = GeneratedAnswer(
            "Answer [1].",
            LLMProvider.OPENAI,
            "configured-openai-model",
        )

        response = self.client.post(
            "/api/v1/answers/generate",
            json={"question": "Vacation policy?", "openai": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["provider"], "openai")
        self.assertEqual(
            generate_answer.call_args.args[2],
            LLMProvider.OPENAI,
        )

    def test_rejects_multiple_llm_providers(self) -> None:
        response = self.client.post(
            "/api/v1/answers/generate",
            json={
                "question": "Vacation policy?",
                "hf": True,
                "openai": True,
            },
        )

        self.assertEqual(response.status_code, 422)

    @patch("app.api.controllers.answer.generate_grounded_answer")
    @patch("app.api.controllers.answer.get_query_router")
    def test_returns_not_found_without_context(
        self,
        get_query_router,
        generate_answer,
    ) -> None:
        get_query_router.return_value.search.return_value = []

        response = self.client.post(
            "/api/v1/answers/generate",
            json={"question": "Unknown policy?"},
        )

        self.assertEqual(response.status_code, 404)
        generate_answer.assert_not_called()

    @patch("app.api.controllers.answer.generate_grounded_answer")
    @patch("app.api.controllers.answer.get_query_router")
    def test_reports_missing_provider_configuration(
        self,
        get_query_router,
        generate_answer,
    ) -> None:
        get_query_router.return_value.search.return_value = [retrieved_chunk()]
        generate_answer.side_effect = LLMConfigurationError(
            "OPENAI_API_KEY must be set when using openai"
        )

        response = self.client.post(
            "/api/v1/answers/generate",
            json={"question": "Vacation policy?", "openai": True},
        )

        self.assertEqual(response.status_code, 503)
        self.assertIn("OPENAI_API_KEY", response.json()["detail"])

    @patch("app.api.controllers.answer.generate_grounded_answer")
    @patch("app.api.controllers.answer.get_query_router")
    def test_reports_actionable_provider_failures(
        self,
        get_query_router,
        generate_answer,
    ) -> None:
        get_query_router.return_value.search.return_value = [retrieved_chunk()]
        generate_answer.side_effect = LLMProviderRequestError(
            LLMProvider.HUGGINGFACE,
            "Hugging Face token needs Inference Providers permission.",
        )

        with self.assertLogs("app.api.controllers.answer", level="ERROR"):
            response = self.client.post(
                "/api/v1/answers/generate",
                json={"question": "Vacation policy?"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("Inference Providers", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
