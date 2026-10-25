"""Schemas for retrieval-grounded answer generation."""

from pydantic import BaseModel, Field, field_validator, model_validator

from app.api.llm_usage import LLMUsage
from app.api.schemas.retrieval import RetrievalRequest, RetrievedChunkResponse
from app.retrieval.config import RETRIEVAL_SETTINGS


MAX_SESSION_HISTORY_TURNS = 50
MAX_SESSION_ANSWER_LENGTH = 10_000


class ConversationTurn(BaseModel):
    """One completed user/assistant exchange from the browser session."""

    question: str = Field(
        min_length=1,
        max_length=RETRIEVAL_SETTINGS.maximum_question_length,
    )
    answer: str = Field(min_length=1, max_length=MAX_SESSION_ANSWER_LENGTH)

    @field_validator("question", "answer")
    @classmethod
    def normalize_content(cls, content: str) -> str:
        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("conversation content must not be empty")
        return normalized_content


class AnswerRequest(RetrievalRequest):
    """Question, retrieval controls, and one optional LLM provider switch."""

    history: list[ConversationTurn] = Field(
        default_factory=list,
        max_length=MAX_SESSION_HISTORY_TURNS,
        description="Completed turns from the current browser chat session.",
    )
    gemini: bool = Field(
        default=False,
        description="Use Google Gemini; this is also the default provider.",
    )
    openai: bool = Field(default=False, description="Use the OpenAI provider.")
    anthropic: bool = Field(
        default=False,
        description="Use the Anthropic provider.",
    )
    grok: bool = Field(default=False, description="Use the xAI Grok provider.")

    @model_validator(mode="after")
    def allow_at_most_one_provider(self) -> "AnswerRequest":
        selected_count = sum(
            (self.gemini, self.openai, self.anthropic, self.grok)
        )
        if selected_count > 1:
            raise ValueError("only one LLM provider can be enabled")
        return self


class AnswerResponse(RetrievalRequest):
    """A direct or retrieval-grounded answer produced by the chat model."""

    effective_query: str
    applied_strategies: list[str]
    retrieval_used: bool
    provider: str
    model: str
    answer: str
    usage: LLMUsage | None
    sources: list[RetrievedChunkResponse]
