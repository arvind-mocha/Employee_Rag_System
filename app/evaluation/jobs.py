"""RQ worker for explicitly selected uploaded evaluation datasets."""

import logging
from typing import Any

from fastapi import HTTPException
from rq import get_current_job

from app.api.controllers.evaluation import execute_selected_evaluation
from app.api.schemas.evaluation import RunEvaluationRequest


logger = logging.getLogger(__name__)


def run_evaluation_job(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    expected_hashes = payload.pop("expected_hashes", None)
    job = get_current_job()

    def progress(completed: int, total: int, dataset_id: str) -> None:
        if job is not None:
            job.meta.update(completed_cases=completed, total_cases=total,
                            current_dataset_id=dataset_id)
            job.save_meta()

    try:
        request = RunEvaluationRequest.model_validate(payload)
        response = execute_selected_evaluation(request, expected_hashes=expected_hashes,
                                               progress=progress)
    except HTTPException as error:
        logger.warning("Background evaluation rejected: %s", error.detail)
        return {"ok": False, "error": str(error.detail)}
    except Exception as error:
        logger.exception("Background evaluation failed")
        return {"ok": False, "error": f"Background evaluation failed ({type(error).__name__})."}
    return {"ok": True, "result": response.model_dump(mode="json")}
