"""POST API definitions."""

from http import HTTPStatus

from app.api.controllers.data import upload_data
from app.api.controllers.ingestion import start_ingestion
from app.api.controllers.retrieval import retrieve_chunks
from app.api.routes.definition import ApiRoute
from app.api.schemas.data import DataUploadResponse
from app.api.schemas.ingestion import IngestionStartResponse
from app.api.schemas.retrieval import RetrievalResponse


POST_ROUTES = (
    ApiRoute(
        name="upload_data",
        path="/data/upload",
        endpoint=upload_data,
        response_model=DataUploadResponse,
        status_code=HTTPStatus.CREATED,
        tags=("data",),
    ),
    ApiRoute(
        name="start_ingestion",
        path="/ingestion/start",
        endpoint=start_ingestion,
        response_model=IngestionStartResponse,
        status_code=HTTPStatus.ACCEPTED,
        tags=("ingestion",),
    ),
    ApiRoute(
        name="retrieve_chunks",
        path="/retrieval/search",
        endpoint=retrieve_chunks,
        response_model=RetrievalResponse,
        status_code=HTTPStatus.OK,
        tags=("retrieval",),
    ),
)
