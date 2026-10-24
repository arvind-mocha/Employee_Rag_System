"""Selectable LangChain chat-model integrations for answer generation."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI

from app.api import config
from app.api.llm_usage import LLMUsage, calculate_llm_usage
from app.api.prompts import (
    build_answer_messages,
    build_tool_selection_messages,
    format_retrieval_context,
)
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


@dataclass(frozen=True, slots=True)
class ToolDrivenAnswer:
    """A chat answer plus any evidence retrieved through the model's tool."""

    answer: str
    provider: LLMProvider
    model_name: str
    usage: LLMUsage | None = None
    chunks: tuple[RetrievedChunk, ...] = ()
    retrieval_query: str | None = None

    @property
    def retrieval_used(self) -> bool:
        return self.retrieval_query is not None


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
    *,
    max_output_tokens: int = MAX_ANSWER_TOKENS,
) -> ChatModelClient:
    """Create the LangChain chat model configured for one provider."""
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be at least 1")

    settings = settings or _stored_llm_settings(provider)
    settings.validate_provider(provider)
    model_name = settings.model_for(provider)

    if provider is LLMProvider.HUGGINGFACE:
        endpoint = HuggingFaceEndpoint(
            repo_id=model_name,
            provider=settings.hf_provider,
            huggingfacehub_api_token=settings.hf_api_key,
            max_new_tokens=max_output_tokens,
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
            max_completion_tokens=max_output_tokens,
            timeout=MODEL_TIMEOUT_SECONDS,
        )
    elif provider is LLMProvider.ANTHROPIC:
        model = ChatAnthropic(
            model_name=model_name,
            api_key=settings.anthropic_api_key,
            max_tokens_to_sample=max_output_tokens,
            timeout=MODEL_TIMEOUT_SECONDS,
        )
    else:
        model = ChatOpenAI(
            model=model_name,
            api_key=settings.grok_api_key,
            base_url=settings.grok_base_url,
            max_completion_tokens=max_output_tokens,
            timeout=MODEL_TIMEOUT_SECONDS,
            use_responses_api=False,
        )

    return ChatModelClient(provider, model_name, model)


def get_chat_model_client(
    provider: LLMProvider,
    *,
    max_output_tokens: int = MAX_ANSWER_TOKENS,
) -> ChatModelClient:
    """Build a client from the active, unexpired database credential."""
    return create_chat_model_client(
        provider,
        max_output_tokens=max_output_tokens,
    )


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
    if "tool" in message and (
        "not support" in message
        or "unsupported" in message
        or "unexpected" in message
        or "invalid" in message
    ):
        return (
            f"The configured {provider.value} model does not support tool "
            "calling. Choose a tool-capable model in Settings."
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


def _combined_usage(*usages: LLMUsage | None) -> LLMUsage | None:
    present = [usage for usage in usages if usage is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]

    costs = [usage.estimated_cost_usd for usage in present]
    first = present[0]
    return LLMUsage(
        input_tokens=sum(usage.input_tokens for usage in present),
        cached_input_tokens=sum(
            usage.cached_input_tokens for usage in present
        ),
        cache_creation_input_tokens=sum(
            usage.cache_creation_input_tokens for usage in present
        ),
        output_tokens=sum(usage.output_tokens for usage in present),
        total_tokens=sum(usage.total_tokens for usage in present),
        token_count_source=(
            "provider"
            if all(usage.token_count_source == "provider" for usage in present)
            else "estimated"
        ),
        input_cost_per_million_usd=first.input_cost_per_million_usd,
        cached_input_cost_per_million_usd=(
            first.cached_input_cost_per_million_usd
        ),
        cache_creation_cost_per_million_usd=(
            first.cache_creation_cost_per_million_usd
        ),
        output_cost_per_million_usd=first.output_cost_per_million_usd,
        estimated_cost_usd=(
            round(sum(cost for cost in costs if cost is not None), 10)
            if all(cost is not None for cost in costs)
            else None
        ),
        pricing_source=(
            "built_in"
            if all(usage.pricing_source == "built_in" for usage in present)
            else "unavailable"
        ),
    )


def _invoke_client(
    client: ChatModelClient,
    messages: Sequence[BaseMessage],
    runnable: Any | None = None,
) -> tuple[BaseMessage, LLMUsage]:
    try:
        selected_runnable = runnable if runnable is not None else client.model
        response = selected_runnable.invoke(messages)
    except Exception as error:
        raise LLMProviderRequestError(
            client.provider,
            _provider_error_detail(client.provider, error),
        ) from error
    usage = calculate_llm_usage(
        messages,
        response,
        client.provider.value,
        client.model_name,
    )
    return response, usage


def generate_tool_driven_answer(
    question: str,
    provider: LLMProvider,
    search: Callable[[str], list[RetrievedChunk]],
) -> ToolDrivenAnswer:
    """Let the chat model decide whether to search employee knowledge."""
    retrieved_chunks: list[RetrievedChunk] = []
    retrieval_query: str | None = None

    @tool("search_employee_knowledge")
    def search_employee_knowledge(query: str) -> str:
        """Search uploaded employee documents and company-specific knowledge."""
        nonlocal retrieval_query
        normalized_query = query.strip() or question.strip()
        retrieval_query = normalized_query
        retrieved_chunks.extend(search(normalized_query))
        if not retrieved_chunks:
            return "No relevant employee documents were found."
        return format_retrieval_context(retrieved_chunks)

    client = get_chat_model_client(provider)
    selection_messages = build_tool_selection_messages(question)
    try:
        tool_model = client.model.bind_tools(
            [search_employee_knowledge],
            tool_choice="auto",
        )
    except Exception as error:
        raise LLMProviderRequestError(
            provider,
            (
                f"The configured {provider.value} model does not support "
                "tool calling. Choose a tool-capable model in Settings."
            ),
        ) from error

    selection, selection_usage = _invoke_client(
        client,
        selection_messages,
        tool_model,
    )
    tool_calls = getattr(selection, "tool_calls", None) or []
    matching_call = next(
        (
            call
            for call in tool_calls
            if call.get("name") == search_employee_knowledge.name
        ),
        None,
    )

    if matching_call is None:
        direct_answer = _message_text(selection.content)
        if not direct_answer:
            raise LLMProviderRequestError(
                provider,
                f"{provider.value} returned neither an answer nor a tool call.",
            )
        return ToolDrivenAnswer(
            direct_answer,
            client.provider,
            client.model_name,
            selection_usage,
        )

    arguments = matching_call.get("args")
    query = arguments.get("query") if isinstance(arguments, dict) else None
    search_employee_knowledge.invoke(
        {"query": query if isinstance(query, str) else question}
    )

    if not retrieved_chunks:
        return ToolDrivenAnswer(
            (
                "I searched the uploaded employee documents but could not find "
                "relevant information for that question."
            ),
            client.provider,
            client.model_name,
            selection_usage,
            retrieval_query=retrieval_query,
        )

    grounded_messages = build_answer_messages(question, retrieved_chunks)
    grounded_response, grounded_usage = _invoke_client(
        client,
        grounded_messages,
    )
    grounded_answer = _message_text(grounded_response.content)
    if not grounded_answer:
        raise LLMProviderRequestError(
            provider,
            f"{provider.value} returned an empty grounded answer.",
        )
    return ToolDrivenAnswer(
        grounded_answer,
        client.provider,
        client.model_name,
        _combined_usage(selection_usage, grounded_usage),
        tuple(retrieved_chunks),
        retrieval_query,
    )


def invoke_chat_model(
    messages: Sequence[BaseMessage],
    provider: LLMProvider,
    *,
    max_output_tokens: int = MAX_ANSWER_TOKENS,
) -> GeneratedText:
    """Invoke one configured provider and return normalized text safely."""
    client = get_chat_model_client(
        provider,
        max_output_tokens=max_output_tokens,
    )
    response, usage = _invoke_client(client, messages)

    text = _message_text(response.content)
    if not text:
        raise LLMProviderRequestError(
            provider,
            f"{provider.value} returned an empty response.",
        )
    return GeneratedText(text, client.provider, client.model_name, usage)
