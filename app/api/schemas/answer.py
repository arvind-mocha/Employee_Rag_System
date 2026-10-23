"""Schemas for retrieval-grounded answer generation."""

from pydantic import Field, model_validator

from app.api.llm_usage import LLMUsage
from app.api.schemas.retrieval import RetrievalRequest, RetrievedChunkResponse


class AnswerRequest(RetrievalRequest):
    """Question, retrieval controls, and one optional LLM provider switch."""

    hf: bool = Field(
        default=False,
        description="Use Hugging Face; this is also the default provider.",
    )
    openai: bool = Field(default=False, description="Use the OpenAI provider.")
    anthropic: bool = Field(
        default=False,
        description="Use the Anthropic provider.",
    )
    grok: bool = Field(default=False, description="Use the xAI Grok provider.")

    @model_validator(mode="after")
    def allow_at_most_one_provider(self) -> "AnswerRequest":
        selected_count = sum((self.hf, self.openai, self.anthropic, self.grok))
        if selected_count > 1:
            raise ValueError("only one LLM provider can be enabled")
        return self


class AnswerResponse(RetrievalRequest):
    """A grounded answer and the retrieved chunks supplied to its model."""

    effective_query: str
    applied_strategies: list[str]
    provider: str
    model: str
    answer: str
    usage: LLMUsage | None
    sources: list[RetrievedChunkResponse]
