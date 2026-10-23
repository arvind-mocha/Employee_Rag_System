"""Selectable LangChain chat-model integrations for answer generation."""

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
import os
from typing import Any, Sequence

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI

from app.api.llm_usage import LLMUsage, calculate_llm_usage
from app.api.prompts import build_answer_messages
from app.retrieval.models import RetrievedChunk


DEFAULT_HF_MODEL = "google/gemma-2-2b-it"
DEFAULT_HF_PROVIDER = "auto"
DEFAULT_GROK_BASE_URL = "https://api.x.ai/v1"
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

    @classmethod
    def from_env(cls) -> "LLMSettings":
        load_dotenv()
        return cls(
            hf_api_key=os.getenv("HF_API_KEY", "").strip(),
            hf_model=(
                os.getenv("HF_MODEL", DEFAULT_HF_MODEL).strip()
                or DEFAULT_HF_MODEL
            ),
            hf_provider=(
                os.getenv("HF_PROVIDER", DEFAULT_HF_PROVIDER).strip()
                or DEFAULT_HF_PROVIDER
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_model=os.getenv("OPENAI_MODEL", "").strip(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "").strip(),
            grok_api_key=os.getenv("GROK_API_KEY", "").strip(),
            grok_model=os.getenv("GROK_MODEL", "").strip(),
            grok_base_url=(
                os.getenv("GROK_BASE_URL", DEFAULT_GROK_BASE_URL).strip()
                or DEFAULT_GROK_BASE_URL
            ),
        )

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
            LLMProvider.HUGGINGFACE: ("HF_API_KEY", self.hf_api_key),
            LLMProvider.OPENAI: ("OPENAI_API_KEY", self.openai_api_key),
            LLMProvider.ANTHROPIC: (
                "ANTHROPIC_API_KEY",
                self.anthropic_api_key,
            ),
            LLMProvider.GROK: ("GROK_API_KEY", self.grok_api_key),
        }
        key_name, api_key = credentials[provider]
        if not api_key:
            raise LLMConfigurationError(
                f"{key_name} must be set when using {provider.value}"
            )

        model = self.model_for(provider)
        if not model:
            model_name = {
                LLMProvider.OPENAI: "OPENAI_MODEL",
                LLMProvider.ANTHROPIC: "ANTHROPIC_MODEL",
                LLMProvider.GROK: "GROK_MODEL",
            }.get(provider, "HF_MODEL")
            raise LLMConfigurationError(
                f"{model_name} must be set when using {provider.value}"
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
    settings = settings or LLMSettings.from_env()
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


@lru_cache(maxsize=4)
def get_chat_model_client(provider: LLMProvider) -> ChatModelClient:
    return create_chat_model_client(provider)


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
            "Hugging Face rejected HF_API_KEY. Create a fine-grained token "
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
