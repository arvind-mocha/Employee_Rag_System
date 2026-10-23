"""HTTP handlers for encrypted LLM provider settings."""

from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Response, status

from app.api.llm import LLMProvider
from app.api.schemas.settings import (
    ProviderCredentialListResponse,
    ProviderCredentialRequest,
    ProviderCredentialResponse,
)
from app.core.credential_store import (
    CredentialStoreConfigurationError,
    CredentialStoreUnavailable,
    StoredCredential,
    get_credential_store,
)


def _response(credential: StoredCredential) -> ProviderCredentialResponse:
    return ProviderCredentialResponse(
        provider=LLMProvider(credential.provider),
        model_name=credential.model_name,
        key_hint=credential.key_hint,
        expires_at=credential.expires_at,
        created_at=credential.created_at,
        updated_at=credential.updated_at,
    )


def _service_unavailable(error: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=str(error),
    )


def list_provider_credentials() -> ProviderCredentialListResponse:
    try:
        credentials = get_credential_store().list_active()
    except (CredentialStoreConfigurationError, CredentialStoreUnavailable) as error:
        raise _service_unavailable(error) from error
    return ProviderCredentialListResponse(
        credentials=[_response(credential) for credential in credentials]
    )


def save_provider_credential(
    request: ProviderCredentialRequest,
) -> ProviderCredentialResponse:
    api_key = request.api_key.get_secret_value().strip()
    if not 8 <= len(api_key) <= 4_096:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="API key must contain between 8 and 4096 characters.",
        )
    expires_at = datetime.now(timezone.utc) + timedelta(
        hours=request.expires_in_hours
    )
    try:
        credential = get_credential_store().save(
            provider=request.provider.value,
            model_name=request.model_name,
            api_key=api_key,
            expires_at=expires_at,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except (CredentialStoreConfigurationError, CredentialStoreUnavailable) as error:
        raise _service_unavailable(error) from error
    return _response(credential)


def delete_provider_credential(provider: LLMProvider) -> Response:
    try:
        deleted = get_credential_store().delete(provider.value)
    except (CredentialStoreConfigurationError, CredentialStoreUnavailable) as error:
        raise _service_unavailable(error) from error
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active {provider.value} credential was found.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
