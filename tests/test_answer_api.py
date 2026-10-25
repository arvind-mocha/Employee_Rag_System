import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.api.llm import (
    LLMConfigurationError,
    LLMProvider,
    LLMProviderRequestError,
    ToolDrivenAnswer,
)
from app.api.llm_usage import LLMUsage
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

    @patch("app.api.controllers.answer.generate_tool_driven_answer")
    @patch("app.api.controllers.answer.get_query_router")
    def test_defaults_to_gemini_and_returns_sources(
        self,
        get_query_router,
        generate_answer,
    ) -> None:
        router = Mock()
        router.search.return_value = [retrieved_chunk()]
        get_query_router.return_value = router

        def generate(question, provider, search, *, history):
            self.assertEqual(history, [])
            chunks = search(question)
            return ToolDrivenAnswer(
                "Employees receive 15 vacation days [1].",
                provider,
                "gemini-3.5-flash-lite",
                LLMUsage(
                    input_tokens=100,
                    output_tokens=10,
                    total_tokens=110,
                    token_count_source="provider",
                    estimated_cost_usd=None,
                    pricing_source="unavailable",
                ),
                tuple(chunks),
                question,
            )

        generate_answer.side_effect = generate

        response = self.client.post(
            "/api/v1/answers/generate",
            json={
                "question": "How many vacation days do employees receive?",
                "query_rewriting": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["provider"], "gemini")
        self.assertEqual(body["model"], "gemini-3.5-flash-lite")
        self.assertEqual(body["effective_query"], "vacation days employees receive")
        self.assertTrue(body["retrieval_used"])
        self.assertEqual(body["sources"][0]["chunk_id"], "chunk-1")
        self.assertEqual(body["usage"]["total_tokens"], 110)
        self.assertIsNone(body["usage"]["estimated_cost_usd"])
        options = RetrievalOptions(query_rewriting=True)
        router.search.assert_called_once_with(
            "How many vacation days do employees receive?",
            options,
        )
        generate_answer.assert_called_once()
        self.assertEqual(
            generate_answer.call_args.args[:2],
            (
                "How many vacation days do employees receive?",
                LLMProvider.GEMINI,
            ),
        )
        self.assertTrue(callable(generate_answer.call_args.args[2]))

    @patch("app.api.controllers.answer.generate_tool_driven_answer")
    @patch("app.api.controllers.answer.get_query_router")
    def test_skips_retrieval_for_a_direct_openai_answer(
        self,
        get_query_router,
        generate_answer,
    ) -> None:
        generate_answer.return_value = ToolDrivenAnswer(
            "Hello! How can I help?",
            LLMProvider.OPENAI,
            "configured-openai-model",
        )

        response = self.client.post(
            "/api/v1/answers/generate",
            json={"question": "Hello", "openai": True},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["provider"], "openai")
        self.assertFalse(body["retrieval_used"])
        self.assertEqual(body["sources"], [])
        self.assertEqual(body["applied_strategies"], [])
        get_query_router.assert_not_called()
        self.assertEqual(
            generate_answer.call_args.args[1],
            LLMProvider.OPENAI,
        )

    def test_rejects_multiple_llm_providers(self) -> None:
        response = self.client.post(
            "/api/v1/answers/generate",
            json={
                "question": "Vacation policy?",
                "gemini": True,
                "openai": True,
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_rejects_more_than_fifty_history_turns(self) -> None:
        response = self.client.post(
            "/api/v1/answers/generate",
            json={
                "question": "What about the policy?",
                "history": [
                    {"question": f"Question {index}", "answer": "Answer"}
                    for index in range(51)
                ],
            },
        )

        self.assertEqual(response.status_code, 422)

    @patch("app.api.controllers.answer.generate_tool_driven_answer")
    @patch("app.api.controllers.answer.get_query_router")
    def test_returns_a_clear_answer_when_the_tool_finds_no_context(
        self,
        get_query_router,
        generate_answer,
    ) -> None:
        get_query_router.return_value.search.return_value = []

        def generate(question, provider, search, *, history):
            self.assertEqual(history, [])
            chunks = search(question)
            return ToolDrivenAnswer(
                "I could not find relevant employee information.",
                provider,
                "test-model",
                chunks=tuple(chunks),
                retrieval_query=question,
            )

        generate_answer.side_effect = generate

        response = self.client.post(
            "/api/v1/answers/generate",
            json={"question": "Unknown policy?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["retrieval_used"])
        self.assertEqual(response.json()["sources"], [])

    @patch("app.api.controllers.answer.generate_tool_driven_answer")
    def test_forwards_browser_session_history_to_the_model(
        self,
        generate_answer,
    ) -> None:
        generate_answer.return_value = ToolDrivenAnswer(
            "It refers to the vacation policy.",
            LLMProvider.GEMINI,
            "test-model",
        )

        response = self.client.post(
            "/api/v1/answers/generate",
            json={
                "question": "What about part-time employees?",
                "history": [
                    {
                        "question": "What is the vacation policy?",
                        "answer": "Full-time employees receive 15 days.",
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            generate_answer.call_args.kwargs["history"],
            [
                (
                    "What is the vacation policy?",
                    "Full-time employees receive 15 days.",
                )
            ],
        )

    @patch("app.api.controllers.answer.generate_tool_driven_answer")
    @patch("app.api.controllers.answer.get_query_router")
    def test_reports_missing_provider_configuration(
        self,
        get_query_router,
        generate_answer,
    ) -> None:
        get_query_router.return_value.search.return_value = [retrieved_chunk()]
        generate_answer.side_effect = LLMConfigurationError(
            "Configure an active openai API key on the Settings page."
        )

        response = self.client.post(
            "/api/v1/answers/generate",
            json={"question": "Vacation policy?", "openai": True},
        )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Settings page", response.json()["detail"])

    @patch("app.api.controllers.answer.generate_tool_driven_answer")
    @patch("app.api.controllers.answer.get_query_router")
    def test_reports_actionable_provider_failures(
        self,
        get_query_router,
        generate_answer,
    ) -> None:
        get_query_router.return_value.search.return_value = [retrieved_chunk()]
        generate_answer.side_effect = LLMProviderRequestError(
            LLMProvider.GEMINI,
            "Gemini rejected its configured API key.",
        )

        with self.assertLogs("app.api.controllers.answer", level="WARNING"):
            response = self.client.post(
                "/api/v1/answers/generate",
                json={"question": "Vacation policy?"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("configured API key", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
