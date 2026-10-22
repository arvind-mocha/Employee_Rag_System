"""GET API definitions."""

from http import HTTPStatus

from app.api.controllers.ingestion import get_ingestion_status
from app.api.routes.definition import ApiRoute
from app.api.schemas.ingestion import IngestionStatusResponse


GET_ROUTES = (
    ApiRoute(
        name="get_ingestion_status",
        path="/ingestion/status",
        endpoint=get_ingestion_status,
        response_model=IngestionStatusResponse,
        status_code=HTTPStatus.OK,
        tags=("ingestion",),
    ),
)
