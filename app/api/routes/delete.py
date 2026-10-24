"""DELETE API definitions."""

from http import HTTPStatus

from app.api.controllers.data import erase_vector_database
from app.api.controllers.evaluation import delete_evaluation_dataset
from app.api.controllers.settings import delete_provider_credential
from app.api.routes.definition import ApiRoute
from app.api.schemas.data import VectorDatabaseEraseResponse
from app.api.schemas.evaluation import DeleteEvaluationDatasetResponse


DELETE_ROUTES = (
    ApiRoute(
        name="delete_evaluation_dataset",
        path="/evaluations/datasets/{dataset_name}",
        endpoint=delete_evaluation_dataset,
        response_model=DeleteEvaluationDatasetResponse,
        status_code=HTTPStatus.OK,
        tags=("evaluations",),
    ),
    ApiRoute(
        name="erase_vector_database",
        path="/data/vector-database",
        endpoint=erase_vector_database,
        response_model=VectorDatabaseEraseResponse,
        status_code=HTTPStatus.OK,
        tags=("data",),
    ),
    ApiRoute(
        name="delete_provider_credential",
        path="/settings/providers/{provider}",
        endpoint=delete_provider_credential,
        response_model=None,
        status_code=HTTPStatus.NO_CONTENT,
        tags=("settings",),
    ),
)
