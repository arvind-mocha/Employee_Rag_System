"""RQ worker functions for long-running evaluation operations."""

import logging
from typing import Any, Callable

from fastapi import HTTPException
from pydantic import BaseModel

from app.api.controllers.evaluation import (
    generate_evaluation_dataset,
    run_evaluation_on_demand,
)
from app.api.schemas.evaluation import (
    GenerateEvaluationDatasetRequest,
    RunEvaluationRequest,
)


logger = logging.getLogger(__name__)


def _run_job(
    operation: Callable[[Any], BaseModel],
    request: BaseModel,
) -> dict[str, Any]:
    try:
        response = operation(request)
    except HTTPException as error:
        logger.warning("Background evaluation operation rejected: %s", error.detail)
        return {"ok": False, "error": str(error.detail)}
    except Exception as error:
        logger.exception("Background evaluation operation failed")
        return {
            "ok": False,
            "error": f"Background evaluation failed ({type(error).__name__}).",
        }
    return {
        "ok": True,
        "result": response.model_dump(mode="json"),
    }


def generate_dataset_job(payload: dict[str, Any]) -> dict[str, Any]:
    request = GenerateEvaluationDatasetRequest.model_validate(payload)
    return _run_job(generate_evaluation_dataset, request)


def run_evaluation_job(payload: dict[str, Any]) -> dict[str, Any]:
    request = RunEvaluationRequest.model_validate(payload)
    return _run_job(run_evaluation_on_demand, request)
