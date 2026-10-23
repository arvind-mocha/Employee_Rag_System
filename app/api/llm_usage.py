"""Token-usage normalization and approximate LLM cost calculation."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from app.api import config

MILLION_TOKENS = 1_000_000
APPROXIMATE_CHARACTERS_PER_TOKEN = 4


class LLMUsage(BaseModel):
    """Usage and estimated price for one model invocation."""

    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    token_count_source: Literal["provider", "estimated"]
    input_cost_per_million_usd: float | None = Field(default=None, ge=0)
    cached_input_cost_per_million_usd: float | None = Field(
        default=None,
        ge=0,
    )
    cache_creation_cost_per_million_usd: float | None = Field(
        default=None,
        ge=0,
    )
    output_cost_per_million_usd: float | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    pricing_source: Literal["built_in", "unavailable"]
    currency: Literal["USD"] = "USD"


class LLMUsageSummary(BaseModel):
    """Aggregate token usage and known estimated costs across calls."""

    llm_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    priced_calls: int = Field(ge=0)
    unpriced_calls: int = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    currency: Literal["USD"] = "USD"


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_million_usd: float
    output_per_million_usd: float
    cached_input_per_million_usd: float
    cache_creation_per_million_usd: float


# Standard API rates for exact model IDs. Unknown models remain explicitly
# unpriced rather than requiring customer configuration.
KNOWN_MODEL_PRICING: dict[tuple[str, str], ModelPricing] = {
    key: ModelPricing(*rates)
    for key, rates in config.MODEL_PRICING_USD_PER_MILLION.items()
}

def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _nonnegative_int(value)
        if parsed is not None:
            return parsed
    return None


def _provider_token_counts(
    response: BaseMessage,
) -> tuple[int, int, int, int] | None:
    usage = _mapping(getattr(response, "usage_metadata", None))
    response_metadata = _mapping(getattr(response, "response_metadata", None))
    raw_usage = _mapping(
        response_metadata.get("token_usage") or response_metadata.get("usage")
    )

    input_tokens = _first_int(
        usage.get("input_tokens"),
        raw_usage.get("prompt_tokens"),
        raw_usage.get("input_tokens"),
    )
    output_tokens = _first_int(
        usage.get("output_tokens"),
        raw_usage.get("completion_tokens"),
        raw_usage.get("output_tokens"),
    )
    if input_tokens is None or output_tokens is None:
        return None

    input_details = _mapping(usage.get("input_token_details"))
    prompt_details = _mapping(raw_usage.get("prompt_tokens_details"))
    cached_tokens = _first_int(
        input_details.get("cache_read"),
        input_details.get("cached_tokens"),
        prompt_details.get("cached_tokens"),
        raw_usage.get("cache_read_input_tokens"),
        raw_usage.get("cached_input_tokens"),
        0,
    )
    cache_creation_tokens = _first_int(
        input_details.get("cache_creation"),
        input_details.get("cache_creation_input_tokens"),
        raw_usage.get("cache_creation_input_tokens"),
        0,
    )

    # LangChain's standardized input_tokens includes cache details. Some raw
    # Anthropic responses instead report base input and cache tokens separately.
    if (
        not usage
        and raw_usage.get("prompt_tokens") is None
        and raw_usage.get("input_tokens") is not None
        and (cached_tokens or cache_creation_tokens)
    ):
        input_tokens += cached_tokens + cache_creation_tokens

    cached_tokens = min(cached_tokens, input_tokens)
    cache_creation_tokens = min(
        cache_creation_tokens,
        input_tokens - cached_tokens,
    )
    return input_tokens, cached_tokens, cache_creation_tokens, output_tokens


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


def _estimated_tokens(text: str) -> int:
    return (
        math.ceil(len(text) / APPROXIMATE_CHARACTERS_PER_TOKEN)
        if text
        else 0
    )


def _pricing_for(
    provider: str,
    model_name: str,
) -> tuple[ModelPricing | None, Literal["built_in", "unavailable"]]:
    known = KNOWN_MODEL_PRICING.get(
        (provider.casefold(), model_name.casefold())
    )
    return known, "built_in" if known else "unavailable"


def calculate_llm_usage(
    messages: Sequence[BaseMessage],
    response: BaseMessage,
    provider: str,
    model_name: str,
) -> LLMUsage:
    """Normalize provider tokens and calculate an approximate USD cost."""
    counts = _provider_token_counts(response)
    if counts is None:
        input_text = "\n".join(_content_text(message.content) for message in messages)
        output_text = _content_text(response.content)
        input_tokens = _estimated_tokens(input_text)
        cached_tokens = 0
        cache_creation_tokens = 0
        output_tokens = _estimated_tokens(output_text)
        token_source: Literal["provider", "estimated"] = "estimated"
    else:
        (
            input_tokens,
            cached_tokens,
            cache_creation_tokens,
            output_tokens,
        ) = counts
        token_source = "provider"

    pricing, pricing_source = _pricing_for(provider, model_name)
    estimated_cost = None
    if pricing is not None:
        special_input_tokens = min(
            input_tokens,
            cached_tokens + cache_creation_tokens,
        )
        uncached_input_tokens = input_tokens - special_input_tokens
        estimated_cost = (
            uncached_input_tokens * pricing.input_per_million_usd
            + cached_tokens * pricing.cached_input_per_million_usd
            + cache_creation_tokens
            * pricing.cache_creation_per_million_usd
            + output_tokens * pricing.output_per_million_usd
        ) / MILLION_TOKENS

    return LLMUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        cache_creation_input_tokens=cache_creation_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        token_count_source=token_source,
        input_cost_per_million_usd=(
            pricing.input_per_million_usd if pricing else None
        ),
        cached_input_cost_per_million_usd=(
            pricing.cached_input_per_million_usd if pricing else None
        ),
        cache_creation_cost_per_million_usd=(
            pricing.cache_creation_per_million_usd if pricing else None
        ),
        output_cost_per_million_usd=(
            pricing.output_per_million_usd if pricing else None
        ),
        estimated_cost_usd=(
            round(estimated_cost, 10) if estimated_cost is not None else None
        ),
        pricing_source=pricing_source,
    )


def summarize_llm_usage(usages: Iterable[LLMUsage | None]) -> LLMUsageSummary:
    """Aggregate calls while making incomplete pricing visible."""
    present = [usage for usage in usages if usage is not None]
    known_costs = [
        usage.estimated_cost_usd
        for usage in present
        if usage.estimated_cost_usd is not None
    ]
    return LLMUsageSummary(
        llm_calls=len(present),
        input_tokens=sum(usage.input_tokens for usage in present),
        cached_input_tokens=sum(
            usage.cached_input_tokens for usage in present
        ),
        cache_creation_input_tokens=sum(
            usage.cache_creation_input_tokens for usage in present
        ),
        output_tokens=sum(usage.output_tokens for usage in present),
        total_tokens=sum(usage.total_tokens for usage in present),
        priced_calls=len(known_costs),
        unpriced_calls=len(present) - len(known_costs),
        estimated_cost_usd=(
            round(sum(known_costs), 10) if known_costs else None
        ),
    )
