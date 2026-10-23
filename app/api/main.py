"""FastAPI application entry point."""

import asyncio
from contextlib import asynccontextmanager
import logging

import uvicorn
from fastapi import FastAPI

from app.api.router import api_router
from app.core.credential_store import CredentialStoreError, get_credential_store


logger = logging.getLogger(__name__)
EXPIRED_CREDENTIAL_CLEANUP_SECONDS = 60


async def _delete_expired_credentials() -> None:
    while True:
        await asyncio.sleep(EXPIRED_CREDENTIAL_CLEANUP_SECONDS)
        try:
            await asyncio.to_thread(get_credential_store().delete_expired)
        except CredentialStoreError:
            logger.warning(
                "Expired provider credential cleanup could not reach its store",
                exc_info=True,
            )


@asynccontextmanager
async def lifespan(_application: FastAPI):
    cleanup_task = asyncio.create_task(_delete_expired_credentials())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    """Create and configure the API application."""
    application = FastAPI(
        title="Employee RAG Data API",
        lifespan=lifespan,
    )
    application.include_router(api_router)
    return application


app = create_app()


def main() -> None:
    """Run the API using the project's command-line script."""
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
