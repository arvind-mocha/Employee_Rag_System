"""Selectable LangChain chat-model integrations for answer generation."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Sequence

from langchain_core.messages import BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI

from app.api import config
from app.api.llm_usage import LLMUsage, calculate_llm_usage
from app.api.prompts import build_answer_messages
from app.core.credential_store import (
    CredentialStoreConfigurationError,
    CredentialStoreUnavailable,
    get_credential_store,
)
from app.retrieval.models import RetrievedChunk


DEFAULT_HF_MODEL = "google/gemma-2-2b-it"
DEFAULT_HF_PROVIDER = config.HUGGINGFACE_INFERENCE_PROVIDER
DEFAULT_GROK_BASE_URL = config.GROK_BASE_URL
MAX_ANSWER_TOKENS = 512
MODEL_TIMEOUT_SECONDS = 120


class LLMProvider(StrEnum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROK = "grok"


class LLMConfigurationError(ValueError):
    """Raised when the selected provider is missing required configuration."""


class LLMProviderRequestError(RuntimeError):
    """Raised with a safe explanation of an upstream provider failure."""

    def __init__(self, provider: LLMProvider, detail: str) -> None:
        self.provider = provider
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class LLMSettings:
    hf_api_key: str
    hf_model: str = DEFAULT_HF_MODEL
    hf_provider: str = DEFAULT_HF_PROVIDER
    openai_api_key: str = ""
    openai_model: str = ""
    anthropic_api_key: str = ""
    anthropic_model: str = ""
    grok_api_key: str = ""
    grok_model: str = ""
    grok_base_url: str = DEFAULT_GROK_BASE_URL

    def model_for(self, provider: LLMProvider) -> str:
        models = {
            LLMProvider.HUGGINGFACE: self.hf_model,
            LLMProvider.OPENAI: self.openai_model,
            LLMProvider.ANTHROPIC: self.anthropic_model,
            LLMProvider.GROK: self.grok_model,
        }
        return models[provider]

    def validate_provider(self, provider: LLMProvider) -> None:
        credentials = {
            LLMProvider.HUGGINGFACE: self.hf_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.GROK: self.grok_api_key,
        }
        api_key = credentials[provider]
        if not api_key:
            raise LLMConfigurationError(
                f"Configure an active {provider.value} API key on the Settings page."
            )

        model = self.model_for(provider)
        if not model:
            raise LLMConfigurationError(
                f"Configure a model name for {provider.value} on the Settings page."
            )


@dataclass(frozen=True, slots=True)
class ChatModelClient:
    provider: LLMProvider
    model_name: str
    model: BaseChatModel


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    answer: str
    provider: LLMProvider
    model_name: str
    usage: LLMUsage | None = None


@dataclass(frozen=True, slots=True)
class GeneratedText:
    text: str
    provider: LLMProvider
    model_name: str
    usage: LLMUsage | None = None


def select_provider(
    *,
    hf: bool = False,
    openai: bool = False,
    anthropic: bool = False,
    grok: bool = False,
) -> LLMProvider:
    """Select one provider, defaulting to Hugging Face when none is enabled."""
    enabled = [
        provider
        for provider, selected in (
            (LLMProvider.HUGGINGFACE, hf),
            (LLMProvider.OPENAI, openai),
            (LLMProvider.ANTHROPIC, anthropic),
            (LLMProvider.GROK, grok),
        )
        if selected
    ]
    if len(enabled) > 1:
        raise ValueError("only one LLM provider can be enabled")
    return enabled[0] if enabled else LLMProvider.HUGGINGFACE


def create_chat_model_client(
    provider: LLMProvider,
    settings: LLMSettings | None = None,
) -> ChatModelClient:
    """Create the LangChain chat model configured for one provider."""
    settings = settings or _stored_llm_settings(provider)
    settings.validate_provider(provider)
    model_name = settings.model_for(provider)

    if provider is LLMProvider.HUGGINGFACE:
        endpoint = HuggingFaceEndpoint(
            repo_id=model_name,
            provider=settings.hf_provider,
            huggingfacehub_api_token=settings.hf_api_key,
            max_new_tokens=MAX_ANSWER_TOKENS,
            do_sample=False,
            timeout=MODEL_TIMEOUT_SECONDS,
        )
        model: BaseChatModel = ChatHuggingFace(
            llm=endpoint,
            model_id=model_name,
        )
    elif provider is LLMProvider.OPENAI:
        model = ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
            max_completion_tokens=MAX_ANSWER_TOKENS,
            timeout=MODEL_TIMEOUT_SECONDS,
        )
    elif provider is LLMProvider.ANTHROPIC:
        model = ChatAnthropic(
            model_name=model_name,
            api_key=settings.anthropic_api_key,
            max_tokens_to_sample=MAX_ANSWER_TOKENS,
            timeout=MODEL_TIMEOUT_SECONDS,
        )
    else:
        model = ChatOpenAI(
            model=model_name,
            api_key=settings.grok_api_key,
            base_url=settings.grok_base_url,
            max_completion_tokens=MAX_ANSWER_TOKENS,
            timeout=MODEL_TIMEOUT_SECONDS,
            use_responses_api=False,
        )

    return ChatModelClient(provider, model_name, model)


def get_chat_model_client(provider: LLMProvider) -> ChatModelClient:
    """Build a client from the active, unexpired database credential."""
    return create_chat_model_client(provider)


def _stored_llm_settings(provider: LLMProvider) -> LLMSettings:
    try:
        credential = get_credential_store().get_active(provider.value)
    except (CredentialStoreConfigurationError, CredentialStoreUnavailable) as error:
        raise LLMConfigurationError(str(error)) from error
    if credential is None:
        raise LLMConfigurationError(
            f"Configure an active {provider.value} API key on the Settings page."
        )

    values = {
        "hf_api_key": "",
        "openai_api_key": "",
        "anthropic_api_key": "",
        "grok_api_key": "",
    }
    models = {
        "hf_model": DEFAULT_HF_MODEL,
        "openai_model": "",
        "anthropic_model": "",
        "grok_model": "",
    }
    key_field, model_field = {
        LLMProvider.HUGGINGFACE: ("hf_api_key", "hf_model"),
        LLMProvider.OPENAI: ("openai_api_key", "openai_model"),
        LLMProvider.ANTHROPIC: ("anthropic_api_key", "anthropic_model"),
        LLMProvider.GROK: ("grok_api_key", "grok_model"),
    }[provider]
    values[key_field] = credential.api_key
    models[model_field] = credential.model_name
    return LLMSettings(**values, **models)


def _clear_client_cache() -> None:
    """Compatibility hook retained now that clients are intentionally uncached."""


get_chat_model_client.cache_clear = _clear_client_cache  # type: ignore[attr-defined]


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("output_text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts).strip()
    return str(content).strip()


def _provider_error_detail(
    provider: LLMProvider,
    error: Exception,
) -> str:
    message = str(error).casefold()
    if provider is LLMProvider.HUGGINGFACE and (
        "sufficient permissions" in message or "403 forbidden" in message
    ):
        return (
            "Hugging Face rejected its configured API key. Create a fine-grained token "
            "with permission to make calls to Inference Providers."
        )
    if "authentication" in message or "unauthorized" in message or "401" in message:
        return f"{provider.value} rejected its configured API key."
    if "rate limit" in message or "quota" in message or "429" in message:
        return f"{provider.value} rate limit or account quota was exceeded."
    if "model_not_found" in message or "not found" in message or "404" in message:
        return (
            f"{provider.value} could not access the configured model. "
            "Verify the model ID and account permissions."
        )
    if "temperature" in message:
        return (
            f"{provider.value} rejected the configured generation parameters."
        )
    return (
        f"{provider.value} request failed ({type(error).__name__}). "
        "Check the server logs for the upstream error."
    )


def generate_grounded_answer(
    question: str,
    chunks: list[RetrievedChunk],
    provider: LLMProvider,
) -> GeneratedAnswer:
    """Generate one answer grounded in the supplied retrieved chunks."""
    if not chunks:
        raise ValueError("at least one retrieved chunk is required")

    generated = invoke_chat_model(
        build_answer_messages(question, chunks),
        provider,
    )
    return GeneratedAnswer(
        generated.text,
        generated.provider,
        generated.model_name,
        generated.usage,
    )


def invoke_chat_model(
    messages: Sequence[BaseMessage],
    provider: LLMProvider,
) -> GeneratedText:
    """Invoke one configured provider and return normalized text safely."""
    try:
        client = get_chat_model_client(provider)
        response = client.model.invoke(messages)
    except LLMConfigurationError:
        raise
    except Exception as error:
        raise LLMProviderRequestError(
            provider,
            _provider_error_detail(provider, error),
        ) from error

    text = _message_text(response.content)
    if not text:
        raise LLMProviderRequestError(
            provider,
            f"{provider.value} returned an empty response.",
        )
    usage = calculate_llm_usage(
        messages,
        response,
        client.provider.value,
        client.model_name,
    )
    return GeneratedText(text, client.provider, client.model_name, usage)
