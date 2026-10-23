"""Schemas for encrypted LLM provider settings."""

from datetime import datetime

from pydantic import BaseModel, Field, SecretStr, field_validator

from app.api.llm import LLMProvider


class ProviderCredentialRequest(BaseModel):
    provider: LLMProvider
    model_name: str = Field(min_length=1, max_length=255)
    api_key: SecretStr
    expires_in_hours: float = Field(ge=0.25, le=8_760)

    @field_validator("model_name")
    @classmethod
    def normalize_model_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("model name cannot be blank")
        return normalized


class ProviderCredentialResponse(BaseModel):
    provider: LLMProvider
    model_name: str
    key_hint: str
    expires_at: datetime
    created_at: datetime
    updated_at: datetime


class ProviderCredentialListResponse(BaseModel):
    credentials: list[ProviderCredentialResponse]
