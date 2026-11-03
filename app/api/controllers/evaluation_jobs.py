"""HTTP handlers for Redis-backed evaluation jobs."""

from typing import Any
from datetime import datetime, timezone

from fastapi import HTTPException, status
from redis.exceptions import RedisError
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app.api.controllers.evaluation import load_selected_datasets
from app.evaluation.preflight import validate_runtime_preflight
from app.api.services.provider_configuration import validate_evaluation_providers
from app.api.schemas.evaluation import RunEvaluationRequest
from app.api.schemas.jobs import (
    EvaluationJobAccepted,
    EvaluationCancellationResponse,
    EvaluationJobState,
    EvaluationJobStatus,
    EvaluationJobType,
)
from app.evaluation.persistence import cancellation_key, checkpoint_response
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


def enqueue_evaluation_run(
    request: RunEvaluationRequest,
) -> EvaluationJobAccepted:
    selected = load_selected_datasets(request)
    validate_evaluation_providers(request.answer_providers, request.judge_provider)
    validate_runtime_preflight(request.answer_providers, request.judge_provider, check_worker=True)
    payload = request.model_dump(mode="json")
    payload["expected_hashes"] = {
        meta["dataset_id"]: meta["dataset_sha256"] for meta, _ in selected
    }
    return _enqueue(
        "app.evaluation.jobs.run_evaluation_job",
        payload,
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
        return _describe_evaluation_job(job, raw_status, metadata)
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



def _describe_evaluation_job(job, raw_status, metadata):
    raw_status_value = getattr(raw_status, "value", str(raw_status))
    mapped_status = _STATUS_MAP.get(raw_status_value, "failed")
    result = job.result if isinstance(job.result, dict) else None
    error = None
    response_result = None
    if mapped_status == "completed" and result is not None:
        if result.get("ok") is True and isinstance(result.get("result"), dict):
            response_result = result["result"]
        else:
            mapped_status = "canceled" if result.get("canceled") else "failed"
            response_result = result.get("result")
            error = str(result.get("error") or "Evaluation job failed.")
    elif mapped_status == "failed":
        error = "Evaluation worker failed. Check the worker logs."

    now = datetime.now(timezone.utc)
    elapsed = max(0, ((job.ended_at or now) - job.started_at).total_seconds()) if job.started_at else 0.0
    if response_result is None:
        response_result = checkpoint_response(job, metadata, mapped_status, elapsed)
    completed = metadata.get("completed_cases", 0)
    successful = metadata.get("successful_cases", 0)
    failures = metadata.get("failed_cases", 0)
    low_scores = metadata.get("low_scoring_cases", 0)
    retries = metadata.get("retry_count", 0)
    if response_result and metadata.get("checkpoint_version") == 1:
        saved = response_result.get("aggregate", {}).get("results", [])
        completed = len(saved)
        successful = sum(not r.get("error") for r in saved)
        failures = completed - successful
        low_scores = sum(r.get("passed") is False for r in saved)
        retries = sum(r.get("retry_count", 0) for r in saved)
    total = metadata.get("total_cases", 0)
    terminal = mapped_status in {"completed", "failed", "canceled", "stopped"}
    return EvaluationJobStatus(
        job_id=job.id,
        job_type="evaluation_run",
        completed_cases=completed,
        running_cases=0 if terminal else min(metadata.get("running_cases", 0), max(0, total-completed)),
        successful_cases=successful, failed_cases=failures,
        low_scoring_cases=low_scores,
        progress_percent=round(100*completed/total, 1) if total else 0.0,
        elapsed_seconds=round(elapsed, 3),
        maximum_concurrent_cases=metadata.get("maximum_concurrent_cases", 0),
        retry_count=retries,
        cancellation_requested=bool(job.connection.exists(cancellation_key(job.id))) if metadata.get("checkpoint_version") == 1 else False,
        total_cases=metadata.get("total_cases", 0),
        current_dataset_id=metadata.get("current_dataset_id"),
        status=mapped_status,
        enqueued_at=job.enqueued_at,
        started_at=job.started_at,
        ended_at=job.ended_at,
        result=response_result,
        error=error,
    )


def cancel_evaluation_job(job_id: str) -> EvaluationCancellationResponse:
    """Cooperatively stop running cases; keep their completed checkpoints."""
    try:
        job = Job.fetch(job_id, connection=get_redis_connection(), serializer=get_evaluation_queue().serializer)
        if job.func_name != "app.evaluation.jobs.run_evaluation_job":
            raise HTTPException(422, "Only evaluation jobs can be canceled through this endpoint.")
        raw_status = job.get_status(refresh=True)
        state = getattr(raw_status, "value", str(raw_status))
        if state in {"finished", "failed", "stopped", "canceled"}:
            return EvaluationCancellationResponse(job_id=job.id, status="already_finished")
        job.connection.set(cancellation_key(job.id), "1", ex=get_redis_settings().result_ttl_seconds)
        if state in {"queued", "deferred", "scheduled"}:
            job.cancel()
            return EvaluationCancellationResponse(job_id=job.id, status="canceled")
        return EvaluationCancellationResponse(job_id=job.id, status="requested")
    except NoSuchJobError as error:
        raise HTTPException(404, "Evaluation job was not found or has expired.") from error
    except (RedisError, OSError) as error:
        raise HTTPException(503, "Redis is unavailable; cancellation could not be requested.") from error
