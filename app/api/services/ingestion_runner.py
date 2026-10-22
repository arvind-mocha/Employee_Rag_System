"""Concurrency-safe runner for the ingestion pipeline."""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Literal

from app.ingestion.ingest import main as run_ingestion_pipeline


logger = logging.getLogger(__name__)

IngestionState = Literal["idle", "pending", "running", "completed", "failed"]


@dataclass(frozen=True, slots=True)
class IngestionSnapshot:
    """Thread-safe snapshot of the latest ingestion run."""

    status: IngestionState
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None


def _safe_error_message(error: Exception) -> str:
    """Create a concise error message without exposing the database URL."""
    message = " ".join(str(error).split()) or "No error details were provided."
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        message = message.replace(database_url, "[DATABASE_URL hidden]")
    return f"{type(error).__name__}: {message}"[:1_000]


class IngestionRunner:
    """Allow only one ingestion run at a time in this API process."""

    def __init__(self) -> None:
        self._state_lock = Lock()
        self._status: IngestionState = "idle"
        self._error: str | None = None
        self._started_at: datetime | None = None
        self._finished_at: datetime | None = None

    def reserve(self) -> bool:
        """Reserve the runner if no ingestion process is active."""
        with self._state_lock:
            if self._status in {"pending", "running"}:
                return False
            self._status = "pending"
            self._error = None
            self._started_at = None
            self._finished_at = None
            return True

    def run(self) -> None:
        """Run the ingestion pipeline and record its final status."""
        with self._state_lock:
            self._status = "running"
            self._started_at = datetime.now(timezone.utc)

        try:
            run_ingestion_pipeline()
        except Exception as error:
            logger.exception("Background ingestion process failed")
            with self._state_lock:
                self._status = "failed"
                self._error = _safe_error_message(error)
                self._finished_at = datetime.now(timezone.utc)
        else:
            with self._state_lock:
                self._status = "completed"
                self._finished_at = datetime.now(timezone.utc)

    def get_status(self) -> IngestionSnapshot:
        """Return an immutable snapshot of the current ingestion state."""
        with self._state_lock:
            return IngestionSnapshot(
                status=self._status,
                error=self._error,
                started_at=self._started_at,
                finished_at=self._finished_at,
            )


ingestion_runner = IngestionRunner()
