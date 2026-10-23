import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from app.api.llm import (
    DEFAULT_HF_MODEL,
    ChatModelClient,
    LLMConfigurationError,
    LLMProvider,
    LLMProviderRequestError,
    LLMSettings,
    create_chat_model_client,
    generate_grounded_answer,
    get_chat_model_client,
    select_provider,
)
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


class ProviderSelectionTests(unittest.TestCase):
    def test_defaults_to_huggingface(self) -> None:
        self.assertEqual(select_provider(), LLMProvider.HUGGINGFACE)

    def test_selects_each_enabled_provider(self) -> None:
        self.assertEqual(select_provider(openai=True), LLMProvider.OPENAI)
        self.assertEqual(select_provider(anthropic=True), LLMProvider.ANTHROPIC)
        self.assertEqual(select_provider(grok=True), LLMProvider.GROK)

    def test_rejects_multiple_providers(self) -> None:
        with self.assertRaisesRegex(ValueError, "only one"):
            select_provider(hf=True, openai=True)


class LLMSettingsTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_uses_the_cheap_huggingface_model_by_default(self) -> None:
        settings = LLMSettings.from_env()

        self.assertEqual(settings.hf_model, DEFAULT_HF_MODEL)
        self.assertEqual(settings.hf_provider, "auto")

    def test_requires_credentials_for_the_selected_provider(self) -> None:
        settings = LLMSettings(hf_api_key="")

        with self.assertRaisesRegex(LLMConfigurationError, "HF_API_KEY"):
            settings.validate_provider(LLMProvider.HUGGINGFACE)


class ChatModelFactoryTests(unittest.TestCase):
    @patch("app.api.llm.ChatHuggingFace")
    @patch("app.api.llm.HuggingFaceEndpoint")
    def test_builds_the_huggingface_chat_model(
        self,
        endpoint_class,
        chat_class,
    ) -> None:
        settings = LLMSettings(
            hf_api_key="hf-secret",
            hf_model="google/gemma-2-2b-it",
        )

        client = create_chat_model_client(
            LLMProvider.HUGGINGFACE,
            settings,
        )

        self.assertEqual(client.provider, LLMProvider.HUGGINGFACE)
        self.assertEqual(client.model_name, "google/gemma-2-2b-it")
        endpoint_class.assert_called_once()
        chat_class.assert_called_once_with(
            llm=endpoint_class.return_value,
            model_id="google/gemma-2-2b-it",
        )

    @patch("app.api.llm.ChatOpenAI")
    def test_builds_the_openai_chat_model(self, chat_class) -> None:
        settings = LLMSettings(
            hf_api_key="hf-secret",
            openai_api_key="openai-secret",
            openai_model="configured-openai-model",
        )

        client = create_chat_model_client(LLMProvider.OPENAI, settings)

        self.assertIs(client.model, chat_class.return_value)
        self.assertEqual(
            chat_class.call_args.kwargs["model"],
            "configured-openai-model",
        )
        self.assertEqual(
            chat_class.call_args.kwargs["api_key"],
            "openai-secret",
        )

    @patch("app.api.llm.ChatAnthropic")
    def test_builds_the_anthropic_chat_model(self, chat_class) -> None:
        settings = LLMSettings(
            hf_api_key="hf-secret",
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
            hf_api_key="hf-secret",
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


class AnswerGenerationTests(unittest.TestCase):
    @patch("app.api.llm.get_chat_model_client")
    def test_generates_an_answer_from_retrieved_context(self, get_client) -> None:
        model = Mock()
        model.invoke.return_value = AIMessage(
            content="Employees receive 15 vacation days [1]."
        )
        get_client.return_value = ChatModelClient(
            LLMProvider.HUGGINGFACE,
            "google/gemma-2-2b-it",
            model,
        )

        result = generate_grounded_answer(
            "How many vacation days do employees receive?",
            [retrieved_chunk()],
            LLMProvider.HUGGINGFACE,
        )

        self.assertEqual(result.provider, LLMProvider.HUGGINGFACE)
        self.assertIn("15 vacation days", result.answer)
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.token_count_source, "estimated")
        messages = model.invoke.call_args.args[0]
        self.assertIn("handbook.md", messages[1].content)
        self.assertIn("Employees receive 15", messages[1].content)
        get_chat_model_client.cache_clear()

    @patch("app.api.llm.get_chat_model_client")
    def test_explains_huggingface_permission_failures(self, get_client) -> None:
        model = Mock()
        model.invoke.side_effect = RuntimeError(
            "403 Forbidden: authentication method does not have sufficient permissions"
        )
        get_client.return_value = ChatModelClient(
            LLMProvider.HUGGINGFACE,
            "google/gemma-2-2b-it",
            model,
        )

        with self.assertRaisesRegex(
            LLMProviderRequestError,
            "Inference Providers",
        ):
            generate_grounded_answer(
                "How many vacation days?",
                [retrieved_chunk()],
                LLMProvider.HUGGINGFACE,
            )


if __name__ == "__main__":
    unittest.main()
