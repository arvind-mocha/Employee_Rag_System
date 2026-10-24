"""HTTP handlers for data operations."""

import logging
from typing import Annotated

from fastapi import File, HTTPException, UploadFile, status
from redis.exceptions import RedisError

from app.api import config
from app.api.schemas.data import (
    DataUploadResponse,
    VectorDatabaseEraseResponse,
)
from app.api.services.ingestion_runner import ingestion_runner
from app.api.services.zip_storage import ZipStorageError, store_zip_archive
from app.core.vector_store import drop_vector_store_table
from app.retrieval.query_router import get_query_router
from app.retrieval.semantic_search import get_semantic_search


logger = logging.getLogger(__name__)


async def upload_data(
    zip_file: Annotated[
        UploadFile,
        File(description="ZIP containing the documents to store"),
    ],
) -> DataUploadResponse:
    """Receive a ZIP and store its supported files by data type."""
    try:
        archive_bytes = await zip_file.read(config.MAX_ZIP_SIZE + 1)
    finally:
        await zip_file.close()

    try:
        result = store_zip_archive(archive_bytes)
    except ZipStorageError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
        ) from error

    try:
        ingestion_runner.mark_data_changed()
    except (RedisError, OSError):
        # The upload is already committed. Keep the successful response and let
        # the status endpoint continue to report Redis availability separately.
        logger.warning(
            "Uploaded data was stored but ingestion readiness could not be reset",
            exc_info=True,
        )

    return DataUploadResponse(
        message="ZIP uploaded and extracted successfully.",
        total_files=result.total_files,
        structured_files=result.structured_files,
        unstructured_files=result.unstructured_files,
        ignored_files=result.ignored_files,
        renamed_files=result.renamed_files,
    )


def erase_vector_database() -> VectorDatabaseEraseResponse:
    """Erase indexed chunks while preserving source files and credentials."""
    try:
        reserved = ingestion_runner.reserve()
    except (RedisError, OSError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is unavailable; vector data cannot be erased safely.",
        ) from error
    if not reserved:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Wait for the active ingestion process to finish before erasing vectors.",
        )

    try:
        drop_vector_store_table()
    except Exception as error:
        logger.exception("Vector database erase failed")
        try:
            ingestion_runner.finish_maintenance(error)
        except (RedisError, OSError):
            logger.error(
                "Vector erase failure state could not be recorded",
                exc_info=True,
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PostgreSQL is unavailable; indexed vectors were not erased.",
        ) from error

    get_query_router.cache_clear()
    get_semantic_search.cache_clear()
    try:
        ingestion_runner.finish_maintenance()
    except (RedisError, OSError) as error:
        logger.error(
            "Vectors were erased but ingestion readiness could not be reset",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Vectors were erased, but readiness could not be reset. "
                "Retry after Redis is available."
            ),
        ) from error

    return VectorDatabaseEraseResponse(
        message=(
            "Indexed vectors were erased. Uploaded source files were preserved."
        ),
        status="erased",
    )
