"""Schemas for Redis-backed background evaluation jobs."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


EvaluationJobType = Literal["dataset_generation", "evaluation_run"]
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
