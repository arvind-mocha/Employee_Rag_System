"""POST API definitions."""

from http import HTTPStatus

from app.api.controllers.answer import answer_question
from app.api.controllers.data import upload_data
from app.api.controllers.evaluation import (
    generate_evaluation_dataset,
    run_evaluation_on_demand,
)
from app.api.controllers.evaluation_jobs import (
    enqueue_dataset_generation,
    enqueue_evaluation_run,
)
from app.api.controllers.ingestion import start_ingestion
from app.api.routes.definition import ApiRoute
from app.api.schemas.answer import AnswerResponse
from app.api.schemas.data import DataUploadResponse
from app.api.schemas.evaluation import (
    GenerateEvaluationDatasetResponse,
    RunEvaluationResponse,
)
from app.api.schemas.ingestion import IngestionStartResponse
from app.api.schemas.jobs import EvaluationJobAccepted


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
        name="answer_question",
        path="/answers/generate",
        endpoint=answer_question,
        response_model=AnswerResponse,
        status_code=HTTPStatus.OK,
        tags=("answers",),
    ),
    ApiRoute(
        name="generate_evaluation_dataset",
        path="/evaluations/datasets/generate",
        endpoint=generate_evaluation_dataset,
        response_model=GenerateEvaluationDatasetResponse,
        status_code=HTTPStatus.CREATED,
        tags=("evaluations",),
    ),
    ApiRoute(
        name="run_evaluation_on_demand",
        path="/evaluations/run",
        endpoint=run_evaluation_on_demand,
        response_model=RunEvaluationResponse,
        status_code=HTTPStatus.OK,
        tags=("evaluations",),
    ),
    ApiRoute(
        name="enqueue_dataset_generation",
        path="/evaluations/datasets/generate/async",
        endpoint=enqueue_dataset_generation,
        response_model=EvaluationJobAccepted,
        status_code=HTTPStatus.ACCEPTED,
        tags=("evaluations",),
    ),
    ApiRoute(
        name="enqueue_evaluation_run",
        path="/evaluations/run/async",
        endpoint=enqueue_evaluation_run,
        response_model=EvaluationJobAccepted,
        status_code=HTTPStatus.ACCEPTED,
        tags=("evaluations",),
    ),
)
