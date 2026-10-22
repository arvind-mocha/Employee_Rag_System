"""HTTP handlers for ingestion operations."""

from fastapi import BackgroundTasks, HTTPException, status

from app.api.schemas.ingestion import (
    IngestionStartResponse,
    IngestionStatusResponse,
)
from app.api.services.ingestion_runner import ingestion_runner


async def start_ingestion(
    background_tasks: BackgroundTasks,
) -> IngestionStartResponse:
    """Reserve and start one ingestion run in the background."""
    if not ingestion_runner.reserve():
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
    snapshot = ingestion_runner.get_status()
    return IngestionStatusResponse(
        status=snapshot.status,
        error=snapshot.error,
        started_at=snapshot.started_at,
        finished_at=snapshot.finished_at,
    )
