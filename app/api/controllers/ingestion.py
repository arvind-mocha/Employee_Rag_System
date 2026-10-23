"""HTTP handlers for ingestion operations."""

from fastapi import BackgroundTasks, HTTPException, status
from redis.exceptions import RedisError

from app.api.schemas.ingestion import (
    IngestionStartResponse,
    IngestionStatusResponse,
)
from app.api.services.ingestion_runner import ingestion_runner


async def start_ingestion(
    background_tasks: BackgroundTasks,
) -> IngestionStartResponse:
    """Reserve and start one ingestion run in the background."""
    try:
        reserved = ingestion_runner.reserve()
    except (RedisError, OSError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is unavailable; ingestion could not be started.",
        ) from error
    if not reserved:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An ingestion process is already running.",
        )

    background_tasks.add_task(ingestion_runner.run)
    return IngestionStartResponse(
        message="Ingestion process started.",
        status="accepted",
        status_url="/api/v1/ingestion/status",
    )


def get_ingestion_status() -> IngestionStatusResponse:
    """Return the state of the latest ingestion run."""
    try:
        snapshot = ingestion_runner.get_status()
    except (RedisError, OSError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is unavailable; ingestion status cannot be read.",
        ) from error
    return IngestionStatusResponse(
        status=snapshot.status,
        error=snapshot.error,
        started_at=snapshot.started_at,
        finished_at=snapshot.finished_at,
    )
