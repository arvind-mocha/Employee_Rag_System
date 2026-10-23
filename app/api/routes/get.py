"""GET API definitions."""

from http import HTTPStatus

from app.api.controllers.ingestion import get_ingestion_status
from app.api.controllers.evaluation_jobs import get_evaluation_job
from app.api.controllers.retrieval import retrieve_chunks
from app.api.controllers.settings import list_provider_credentials
from app.api.routes.definition import ApiRoute
from app.api.schemas.ingestion import IngestionStatusResponse
from app.api.schemas.jobs import EvaluationJobStatus
from app.api.schemas.retrieval import RetrievalResponse
from app.api.schemas.settings import ProviderCredentialListResponse


GET_ROUTES = (
    ApiRoute(
        name="list_provider_credentials",
        path="/settings/providers",
        endpoint=list_provider_credentials,
        response_model=ProviderCredentialListResponse,
        status_code=HTTPStatus.OK,
        tags=("settings",),
    ),
    ApiRoute(
        name="get_ingestion_status",
        path="/ingestion/status",
        endpoint=get_ingestion_status,
        response_model=IngestionStatusResponse,
        status_code=HTTPStatus.OK,
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
    ApiRoute(
        name="get_evaluation_job",
        path="/evaluations/jobs/{job_id}",
        endpoint=get_evaluation_job,
        response_model=EvaluationJobStatus,
        status_code=HTTPStatus.OK,
        tags=("evaluations",),
    ),
)
