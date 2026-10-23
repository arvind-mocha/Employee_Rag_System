"""Concurrency-safe runner for the ingestion pipeline."""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal, cast

from redis import Redis
from redis.exceptions import LockError

from app.core.redis_client import get_redis_connection
from app.ingestion.ingest import main as run_ingestion_pipeline


logger = logging.getLogger(__name__)

IngestionState = Literal["idle", "pending", "running", "completed", "failed"]
INGESTION_STATE_KEY = "employee-rag:ingestion:state"
INGESTION_LOCK_KEY = "employee-rag:ingestion:lock"
INGESTION_STATE_TTL_SECONDS = 7 * 24 * 60 * 60
INGESTION_LOCK_TTL_SECONDS = 6 * 60 * 60


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

    def __init__(self, redis_client: Redis | None = None) -> None:
        self._redis = redis_client
        self._distributed_lock: Any | None = None
        self._state_lock = Lock()
        self._status: IngestionState = "idle"
        self._error: str | None = None
        self._started_at: datetime | None = None
        self._finished_at: datetime | None = None

    def reserve(self) -> bool:
        """Reserve the runner if no ingestion process is active."""
        if self._redis is not None:
            distributed_lock = self._redis.lock(
                INGESTION_LOCK_KEY,
                timeout=INGESTION_LOCK_TTL_SECONDS,
                blocking=False,
                thread_local=False,
            )
            if not distributed_lock.acquire(blocking=False):
                return False
            try:
                self._write_redis_state("pending")
            except Exception:
                distributed_lock.release()
                raise
            self._distributed_lock = distributed_lock
            return True

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
        if self._redis is not None:
            self._run_with_redis()
            return

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
        if self._redis is not None:
            values = self._redis.hgetall(INGESTION_STATE_KEY)
            if not values:
                return IngestionSnapshot("idle", None, None, None)
            decoded = {
                _decode(key): _decode(value) for key, value in values.items()
            }
            return IngestionSnapshot(
                status=_validated_state(decoded.get("status")),
                error=decoded.get("error") or None,
                started_at=_parse_datetime(decoded.get("started_at")),
                finished_at=_parse_datetime(decoded.get("finished_at")),
            )

        with self._state_lock:
            return IngestionSnapshot(
                status=self._status,
                error=self._error,
                started_at=self._started_at,
                finished_at=self._finished_at,
            )

    def _run_with_redis(self) -> None:
        started_at = datetime.now(timezone.utc)
        self._write_redis_state("running", started_at=started_at)
        try:
            run_ingestion_pipeline()
        except Exception as error:
            logger.exception("Background ingestion process failed")
            self._write_redis_state(
                "failed",
                error=_safe_error_message(error),
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
        else:
            self._write_redis_state(
                "completed",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
        finally:
            if self._distributed_lock is not None:
                try:
                    self._distributed_lock.release()
                except LockError:
                    logger.warning("The Redis ingestion lock expired before release")
                self._distributed_lock = None

    def _write_redis_state(
        self,
        state: IngestionState,
        *,
        error: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> None:
        if self._redis is None:
            raise RuntimeError("Redis state storage is not configured")
        self._redis.hset(
            INGESTION_STATE_KEY,
            mapping={
                "status": state,
                "error": error or "",
                "started_at": started_at.isoformat() if started_at else "",
                "finished_at": finished_at.isoformat() if finished_at else "",
            },
        )
        self._redis.expire(
            INGESTION_STATE_KEY,
            INGESTION_STATE_TTL_SECONDS,
        )


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _parse_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _validated_state(value: str | None) -> IngestionState:
    if value in {"idle", "pending", "running", "completed", "failed"}:
        return cast(IngestionState, value)
    return "idle"


ingestion_runner = IngestionRunner(get_redis_connection())
