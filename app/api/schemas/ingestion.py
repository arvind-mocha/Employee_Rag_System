"""Schemas returned by ingestion routes."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class IngestionStartResponse(BaseModel):
    """Confirmation that an ingestion run was accepted."""

    message: str
    status: Literal["accepted"]
    status_url: str


class IngestionStatusResponse(BaseModel):
    """Current state of the latest ingestion run."""

    status: Literal["idle", "pending", "running", "completed", "failed"]
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
