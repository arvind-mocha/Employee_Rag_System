from datetime import datetime, timedelta, timezone
import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from app.api.llm import (
    DEFAULT_GEMINI_MODEL,
    ChatModelClient,
    LLMConfigurationError,
    LLMProvider,
    LLMProviderRequestError,
    LLMSettings,
    create_chat_model_client,
    generate_grounded_answer,
    generate_tool_driven_answer,
    get_chat_model_client,
    invoke_chat_model,
    select_provider,
)
from app.core.credential_store import ActiveCredential
from app.retrieval.models import RetrievedChunk


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


class StructuredPayload(BaseModel):
    answer: str


class ProviderSelectionTests(unittest.TestCase):
    def test_defaults_to_gemini(self) -> None:
        self.assertEqual(select_provider(), LLMProvider.GEMINI)

    def test_selects_each_enabled_provider(self) -> None:
        self.assertEqual(select_provider(gemini=True), LLMProvider.GEMINI)
        self.assertEqual(select_provider(openai=True), LLMProvider.OPENAI)
        self.assertEqual(select_provider(anthropic=True), LLMProvider.ANTHROPIC)
        self.assertEqual(select_provider(grok=True), LLMProvider.GROK)

    def test_rejects_multiple_providers(self) -> None:
        with self.assertRaisesRegex(ValueError, "only one"):
            select_provider(gemini=True, openai=True)


class LLMSettingsTests(unittest.TestCase):
    def test_uses_the_flash_lite_gemini_model_by_default(self) -> None:
        settings = LLMSettings(gemini_api_key="test-key")

        self.assertEqual(settings.gemini_model, DEFAULT_GEMINI_MODEL)

    def test_requires_credentials_for_the_selected_provider(self) -> None:
        settings = LLMSettings(gemini_api_key="")

        with self.assertRaisesRegex(LLMConfigurationError, "Settings page"):
            settings.validate_provider(LLMProvider.GEMINI)


class ChatModelFactoryTests(unittest.TestCase):
    @patch("app.api.llm.ChatOpenAI")
    @patch("app.api.llm.get_credential_store")
    def test_loads_runtime_credentials_from_postgres(
        self,
        get_store,
        chat_class,
    ) -> None:
        get_store.return_value.get_active.return_value = ActiveCredential(
            provider="openai",
            model_name="database-model",
            api_key="database-secret",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        client = create_chat_model_client(LLMProvider.OPENAI)

        self.assertEqual(client.model_name, "database-model")
        self.assertEqual(chat_class.call_args.kwargs["api_key"], "database-secret")

    @patch("app.api.llm.ChatGoogleGenerativeAI")
    def test_builds_the_gemini_chat_model(self, chat_class) -> None:
        settings = LLMSettings(
            gemini_api_key="gemini-secret",
            gemini_model="gemini-3.5-flash-lite",
        )

        client = create_chat_model_client(
            LLMProvider.GEMINI,
            settings,
        )

        self.assertEqual(client.provider, LLMProvider.GEMINI)
        self.assertEqual(client.model_name, "gemini-3.5-flash-lite")
        chat_class.assert_called_once_with(
            model="gemini-3.5-flash-lite",
            api_key="gemini-secret",
            max_output_tokens=512,
            timeout=120,
        )

    @patch("app.api.llm.ChatOpenAI")
    def test_builds_the_openai_chat_model(self, chat_class) -> None:
        settings = LLMSettings(
            gemini_api_key="gemini-secret",
            openai_api_key="openai-secret",
            openai_model="configured-openai-model",
        )

        client = create_chat_model_client(
            LLMProvider.OPENAI,
            settings,
            max_output_tokens=2_048,
        )

        self.assertIs(client.model, chat_class.return_value)
        self.assertEqual(
            chat_class.call_args.kwargs["model"],
            "configured-openai-model",
        )
        self.assertEqual(
            chat_class.call_args.kwargs["api_key"],
            "openai-secret",
        )
        self.assertEqual(
            chat_class.call_args.kwargs["max_completion_tokens"],
            2_048,
        )

    @patch("app.api.llm.ChatAnthropic")
    def test_builds_the_anthropic_chat_model(self, chat_class) -> None:
        settings = LLMSettings(
            gemini_api_key="gemini-secret",
            anthropic_api_key="anthropic-secret",
            anthropic_model="configured-anthropic-model",
        )

        client = create_chat_model_client(LLMProvider.ANTHROPIC, settings)

        self.assertIs(client.model, chat_class.return_value)
        self.assertEqual(
            chat_class.call_args.kwargs["model_name"],
            "configured-anthropic-model",
        )
        self.assertEqual(
            chat_class.call_args.kwargs["api_key"],
            "anthropic-secret",
        )

    @patch("app.api.llm.ChatOpenAI")
    def test_builds_grok_with_the_xai_base_url(self, chat_class) -> None:
        settings = LLMSettings(
            gemini_api_key="gemini-secret",
            grok_api_key="xai-secret",
            grok_model="grok-4.7",
        )

        client = create_chat_model_client(LLMProvider.GROK, settings)

        self.assertIs(client.model, chat_class.return_value)
        self.assertEqual(
            chat_class.call_args.kwargs["base_url"],
            "https://api.x.ai/v1",
        )
        self.assertFalse(chat_class.call_args.kwargs["use_responses_api"])

    @patch("app.api.llm.get_chat_model_client")
    def test_uses_provider_native_structured_output(self, get_client) -> None:
        model = Mock()
        structured_model = Mock()
        model.with_structured_output.return_value = structured_model
        structured_model.invoke.return_value = {
            "raw": AIMessage(content="This text is not JSON."),
            "parsed": StructuredPayload(answer="Structured answer"),
            "parsing_error": None,
        }
        get_client.return_value = ChatModelClient(
            LLMProvider.ANTHROPIC,
            "claude-opus-5-5",
            model,
        )

        result = invoke_chat_model(
            [HumanMessage(content="Return structured output")],
            LLMProvider.ANTHROPIC,
            response_schema=StructuredPayload,
        )

        self.assertEqual(
            json.loads(result.text),
            {"answer": "Structured answer"},
        )
        model.with_structured_output.assert_called_once_with(
            StructuredPayload,
            method="json_schema",
            include_raw=True,
        )


class AnswerGenerationTests(unittest.TestCase):
    @patch("app.api.llm.get_chat_model_client")
    def test_answers_directly_without_searching(self, get_client) -> None:
        model = Mock()
        tool_model = Mock()
        model.bind_tools.return_value = tool_model
        tool_model.invoke.return_value = AIMessage(
            content="Hello! How can I help?"
        )
        get_client.return_value = ChatModelClient(
            LLMProvider.OPENAI,
            "test-model",
            model,
        )
        search = Mock()

        result = generate_tool_driven_answer(
            "Hello",
            LLMProvider.OPENAI,
            search,
        )

        self.assertEqual(result.answer, "Hello! How can I help?")
        self.assertFalse(result.retrieval_used)
        self.assertEqual(result.chunks, ())
        search.assert_not_called()
        model.invoke.assert_not_called()

    @patch("app.api.llm.get_chat_model_client")
    def test_includes_session_history_when_selecting_a_tool(
        self,
        get_client,
    ) -> None:
        model = Mock()
        tool_model = Mock()
        model.bind_tools.return_value = tool_model
        tool_model.invoke.return_value = AIMessage(content="A direct follow-up.")
        get_client.return_value = ChatModelClient(
            LLMProvider.OPENAI,
            "test-model",
            model,
        )

        generate_tool_driven_answer(
            "What about part-time employees?",
            LLMProvider.OPENAI,
            Mock(),
            history=[
                (
                    "What is the vacation policy?",
                    "Full-time employees receive 15 days.",
                )
            ],
        )

        messages = tool_model.invoke.call_args.args[0]
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertEqual(messages[1].content, "What is the vacation policy?")
        self.assertIsInstance(messages[2], AIMessage)
        self.assertIn("15 days", messages[2].content)
        self.assertEqual(messages[-1].content, "What about part-time employees?")

    @patch("app.api.llm.get_chat_model_client")
    def test_executes_the_search_tool_and_generates_grounded_answer(
        self,
        get_client,
    ) -> None:
        model = Mock()
        tool_model = Mock()
        model.bind_tools.return_value = tool_model
        tool_model.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_employee_knowledge",
                    "args": {"query": "vacation policy"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
        model.invoke.return_value = AIMessage(
            content="Employees receive 15 vacation days [1]."
        )
        get_client.return_value = ChatModelClient(
            LLMProvider.OPENAI,
            "test-model",
            model,
        )
        search = Mock(return_value=[retrieved_chunk()])

        result = generate_tool_driven_answer(
            "How much vacation do employees receive?",
            LLMProvider.OPENAI,
            search,
        )

        self.assertTrue(result.retrieval_used)
        self.assertEqual(result.retrieval_query, "vacation policy")
        self.assertEqual(result.chunks, (retrieved_chunk(),))
        self.assertIn("15 vacation days", result.answer)
        search.assert_called_once_with("vacation policy")
        grounded_messages = model.invoke.call_args.args[0]
        self.assertIn("handbook.md", grounded_messages[1].content)
        self.assertIsNotNone(result.usage)

    @patch("app.api.llm.get_chat_model_client")
    def test_explains_when_the_model_cannot_call_tools(self, get_client) -> None:
        model = Mock()
        tool_model = Mock()
        model.bind_tools.return_value = tool_model
        tool_model.invoke.side_effect = RuntimeError(
            "tools are not supported by this model"
        )
        get_client.return_value = ChatModelClient(
            LLMProvider.GEMINI,
            "model-without-tools",
            model,
        )

        with self.assertRaisesRegex(
            LLMProviderRequestError,
            "tool-capable model",
        ):
            generate_tool_driven_answer(
                "What is our vacation policy?",
                LLMProvider.GEMINI,
                Mock(),
            )

    @patch("app.api.llm.get_chat_model_client")
    def test_generates_an_answer_from_retrieved_context(self, get_client) -> None:
        model = Mock()
        model.invoke.return_value = AIMessage(
            content="Employees receive 15 vacation days [1]."
        )
        get_client.return_value = ChatModelClient(
            LLMProvider.GEMINI,
            "gemini-3.5-flash-lite",
            model,
        )

        result = generate_grounded_answer(
            "How many vacation days do employees receive?",
            [retrieved_chunk()],
            LLMProvider.GEMINI,
        )

        self.assertEqual(result.provider, LLMProvider.GEMINI)
        self.assertIn("15 vacation days", result.answer)
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.token_count_source, "estimated")
        messages = model.invoke.call_args.args[0]
        self.assertIn("handbook.md", messages[1].content)
        self.assertIn("Employees receive 15", messages[1].content)
        get_chat_model_client.cache_clear()

    @patch("app.api.llm.get_chat_model_client")
    def test_explains_invalid_gemini_api_keys(self, get_client) -> None:
        model = Mock()
        model.invoke.side_effect = RuntimeError(
            "API key not valid. Please pass a valid API key."
        )
        get_client.return_value = ChatModelClient(
            LLMProvider.GEMINI,
            "gemini-3.5-flash-lite",
            model,
        )

        with self.assertRaisesRegex(
            LLMProviderRequestError,
            "rejected its configured API key",
        ):
            generate_grounded_answer(
                "How many vacation days?",
                [retrieved_chunk()],
                LLMProvider.GEMINI,
            )


if __name__ == "__main__":
    unittest.main()
