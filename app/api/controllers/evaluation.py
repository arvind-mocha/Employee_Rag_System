"""HTTP handlers for explicitly requested RAG evaluation operations."""

import logging

from fastapi import HTTPException, status

from app.api import config
from app.api.llm import LLMConfigurationError, LLMProviderRequestError
from app.api.schemas.evaluation import (
    GenerateEvaluationDatasetRequest,
    GenerateEvaluationDatasetResponse,
    RunEvaluationRequest,
    RunEvaluationResponse,
)
from app.evaluation.dataset import (
    EvaluationDatasetError,
    dataset_sha256,
    read_dataset,
    validate_dataset_name,
    write_dataset,
)
from app.evaluation.engine import (
    EvaluationOutputError,
    generate_ground_truth_records,
    run_evaluation,
)


logger = logging.getLogger(__name__)


def _dataset_request_error(error: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=str(error),
    )


def generate_evaluation_dataset(
    request: GenerateEvaluationDatasetRequest,
) -> GenerateEvaluationDatasetResponse:
    """Generate a JSONL ground-truth snapshot only when explicitly called."""
    try:
        dataset_name = validate_dataset_name(request.dataset_name)
        records = generate_ground_truth_records(
            request.questions,
            request.reference_provider,
            request.retrieval_configuration(),
        )
        destination = write_dataset(
            dataset_name,
            records,
            overwrite=request.overwrite,
        )
    except FileExistsError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except LLMConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except LLMProviderRequestError as error:
        logger.exception("Evaluation reference provider request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=error.detail,
        ) from error
    except EvaluationOutputError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error
    except (EvaluationDatasetError, ValueError) as error:
        raise _dataset_request_error(error) from error
    except Exception as error:
        logger.exception("Evaluation dataset generation failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluation dataset generation failed.",
        ) from error

    return GenerateEvaluationDatasetResponse(
        dataset_name=dataset_name,
        dataset_file=str(destination.relative_to(config.PROJECT_ROOT)),
        dataset_sha256=dataset_sha256(dataset_name),
        record_count=len(records),
        reference_provider=records[0].reference_provider,
        reference_model=records[0].reference_model,
        review_required=True,
    )


def run_evaluation_on_demand(
    request: RunEvaluationRequest,
) -> RunEvaluationResponse:
    """Run all requested scores synchronously only when explicitly called."""
    try:
        dataset_name = validate_dataset_name(request.dataset_name)
        records = read_dataset(dataset_name)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except EvaluationDatasetError as error:
        raise _dataset_request_error(error) from error

    if request.record_ids is not None:
        requested_ids = set(request.record_ids)
        records = [record for record in records if record.id in requested_ids]
        found_ids = {record.id for record in records}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"evaluation record IDs not found: {', '.join(missing_ids)}",
            )

    try:
        evaluation = run_evaluation(
            dataset_name,
            records,
            request.answer_providers,
            request.judge_provider,
            request.retrieval_configuration(),
        )
    except Exception as error:
        logger.exception("On-demand evaluation failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The on-demand evaluation could not be started.",
        ) from error
    return RunEvaluationResponse.model_validate(evaluation.model_dump())
