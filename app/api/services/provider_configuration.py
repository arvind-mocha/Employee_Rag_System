"""Secret-free provider configuration reporting and evaluation preflight checks."""

from datetime import datetime, timezone

from fastapi import HTTPException

from app.api.llm import LLMProvider
from app.api.schemas.settings import ProviderConfigurationStatus
from app.core.credential_store import (
    CredentialStoreError,
    StoredCredential,
    get_credential_store,
)


PROVIDER_NAMES = {
    LLMProvider.GEMINI: "Google Gemini",
    LLMProvider.OPENAI: "OpenAI",
    LLMProvider.ANTHROPIC: "Anthropic",
    LLMProvider.GROK: "Grok (xAI)",
}
MISSING_KEY_MESSAGE = (
    "API key not configured. Configure this provider in Settings before running evaluations."
)
EXPIRED_KEY_MESSAGE = "API key expired. Update your credentials in Settings."
MISSING_MODEL_MESSAGE = "Model not configured. Configure a model for this provider in Settings."
UNAVAILABLE_MESSAGE = "Configuration unavailable. Check Settings and retry."


def configuration_statuses(
    credentials: list[StoredCredential],
) -> list[ProviderConfigurationStatus]:
    by_provider = {credential.provider: credential for credential in credentials}
    now = datetime.now(timezone.utc)
    statuses = []
    for provider in LLMProvider:
        credential = by_provider.get(provider.value)
        state = "not_configured"
        message = MISSING_KEY_MESSAGE
        if credential is not None:
            if credential.expires_at <= now:
                state, message = "expired", EXPIRED_KEY_MESSAGE
            elif not credential.has_api_key:
                message = MISSING_KEY_MESSAGE
            elif not credential.model_name.strip():
                message = MISSING_MODEL_MESSAGE
            else:
                state, message = "configured", "Saved API key and model configured."
        statuses.append(ProviderConfigurationStatus(
            provider=provider,
            model_name=credential.model_name.strip() or None if credential else None,
            expires_at=credential.expires_at if credential else None,
            status=state,
            message=message,
        ))
    return statuses


def validate_evaluation_providers(
    answer_providers: list[LLMProvider],
    judge_provider: LLMProvider,
) -> None:
    """Check saved state and decryptability without contacting external providers."""
    selected = list(dict.fromkeys([*answer_providers, judge_provider]))
    try:
        store = get_credential_store()
        statuses = {
            item.provider: item
            for item in configuration_statuses(store.list_active(include_expired=True))
        }
        for provider in selected:
            name = PROVIDER_NAMES[provider]
            configuration = statuses[provider]
            if configuration.status != "configured":
                raise HTTPException(
                    status_code=422,
                    detail=f"Cannot start evaluation. {name}: {configuration.message}",
                )
            # The credential may have expired, been deleted, or been replaced
            # since the metadata query. Check the actual credential again.
            active = store.get_active(provider.value)
            if active is None or not active.api_key.strip():
                raise HTTPException(
                    status_code=422,
                    detail=f"Cannot start evaluation. {name}: {MISSING_KEY_MESSAGE}",
                )
            if active.expires_at <= datetime.now(timezone.utc):
                raise HTTPException(
                    status_code=422,
                    detail=f"Cannot start evaluation. {name}: {EXPIRED_KEY_MESSAGE}",
                )
            if not active.model_name.strip():
                raise HTTPException(
                    status_code=422,
                    detail=f"Cannot start evaluation. {name}: {MISSING_MODEL_MESSAGE}",
                )
    except CredentialStoreError:
        # Store/provider exceptions can contain internal context. Return only
        # fixed, secret-free messages and never log a credential or exception.
        names = ", ".join(PROVIDER_NAMES[provider] for provider in selected)
        raise HTTPException(
            status_code=503,
            detail=f"Cannot start evaluation. {names}: {UNAVAILABLE_MESSAGE}",
        ) from None
