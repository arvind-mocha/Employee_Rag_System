"""Schemas for Redis-backed background evaluation jobs."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


EvaluationJobType = Literal["evaluation_run"]
EvaluationJobState = Literal[
    "queued",
    "running",
    "completed",
    "failed",
    "deferred",
    "scheduled",
    "stopped",
    "canceled",
]


class EvaluationJobAccepted(BaseModel):
    job_id: str
    job_type: EvaluationJobType
    status: Literal["queued"] = "queued"
    status_url: str


class EvaluationJobStatus(BaseModel):
    job_id: str
    job_type: EvaluationJobType
    status: EvaluationJobState
    enqueued_at: datetime | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    running_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    low_scoring_cases: int = 0
    progress_percent: float = 0.0
    elapsed_seconds: float = 0.0
    maximum_concurrent_cases: int = 0
    retry_count: int = 0
    cancellation_requested: bool = False
    completed_cases: int = 0
    total_cases: int = 0
    current_dataset_id: str | None = None


class EvaluationCancellationResponse(BaseModel):
    job_id: str
    status: Literal["requested", "canceled", "already_finished"]
