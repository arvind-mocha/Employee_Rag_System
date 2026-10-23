"""DELETE API definitions."""

from http import HTTPStatus

from app.api.controllers.settings import delete_provider_credential
from app.api.routes.definition import ApiRoute


DELETE_ROUTES = (
    ApiRoute(
        name="delete_provider_credential",
        path="/settings/providers/{provider}",
        endpoint=delete_provider_credential,
        response_model=None,
        status_code=HTTPStatus.NO_CONTENT,
        tags=("settings",),
    ),
)
