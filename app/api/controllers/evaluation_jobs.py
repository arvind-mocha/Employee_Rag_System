"""HTTP handlers for Redis-backed evaluation jobs."""

from typing import Any

from fastapi import HTTPException, status
from redis.exceptions import RedisError
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app.api.schemas.evaluation import (
    GenerateEvaluationDatasetRequest,
    RunEvaluationRequest,
)
from app.api.schemas.jobs import (
    EvaluationJobAccepted,
    EvaluationJobState,
    EvaluationJobStatus,
    EvaluationJobType,
)
from app.core.redis_client import (
    get_evaluation_queue,
    get_redis_connection,
    get_redis_settings,
)


_STATUS_MAP: dict[str, EvaluationJobState] = {
    "queued": "queued",
    "started": "running",
    "finished": "completed",
    "failed": "failed",
    "deferred": "deferred",
    "scheduled": "scheduled",
    "stopped": "stopped",
    "canceled": "canceled",
}


def _enqueue(
    function_name: str,
    payload: dict[str, Any],
    job_type: EvaluationJobType,
) -> EvaluationJobAccepted:
    settings = get_redis_settings()
    try:
        job = get_evaluation_queue().enqueue_call(
            func=function_name,
            args=(payload,),
            meta={"job_type": job_type},
            result_ttl=settings.result_ttl_seconds,
            failure_ttl=settings.result_ttl_seconds,
        )
    except (RedisError, OSError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is unavailable; the evaluation job was not queued.",
        ) from error
    return EvaluationJobAccepted(
        job_id=job.id,
        job_type=job_type,
        status_url=f"/api/v1/evaluations/jobs/{job.id}",
    )


def enqueue_dataset_generation(
    request: GenerateEvaluationDatasetRequest,
) -> EvaluationJobAccepted:
    return _enqueue(
        "app.evaluation.jobs.generate_dataset_job",
        request.model_dump(mode="json"),
        "dataset_generation",
    )


def enqueue_evaluation_run(
    request: RunEvaluationRequest,
) -> EvaluationJobAccepted:
    return _enqueue(
        "app.evaluation.jobs.run_evaluation_job",
        request.model_dump(mode="json"),
        "evaluation_run",
    )


def get_evaluation_job(job_id: str) -> EvaluationJobStatus:
    try:
        job = Job.fetch(
            job_id,
            connection=get_redis_connection(),
            serializer=get_evaluation_queue().serializer,
        )
        raw_status = job.get_status(refresh=True)
        metadata = job.get_meta(refresh=True)
    except NoSuchJobError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation job was not found or has expired.",
        ) from error
    except (RedisError, OSError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is unavailable; job status cannot be read.",
        ) from error

    raw_status_value = getattr(raw_status, "value", str(raw_status))
    mapped_status = _STATUS_MAP.get(raw_status_value, "failed")
    result = job.result if isinstance(job.result, dict) else None
    error = None
    response_result = None
    if mapped_status == "completed" and result is not None:
        if result.get("ok") is True and isinstance(result.get("result"), dict):
            response_result = result["result"]
        else:
            mapped_status = "failed"
            error = str(result.get("error") or "Evaluation job failed.")
    elif mapped_status == "failed":
        error = "Evaluation worker failed. Check the worker logs."

    job_type = metadata.get("job_type")
    if job_type not in {"dataset_generation", "evaluation_run"}:
        job_type = "evaluation_run"
    return EvaluationJobStatus(
        job_id=job.id,
        job_type=job_type,
        status=mapped_status,
        enqueued_at=job.enqueued_at,
        started_at=job.started_at,
        ended_at=job.ended_at,
        result=response_result,
        error=error,
    )
