"""HTTP handlers for data operations."""

import logging
from typing import Annotated

from fastapi import File, HTTPException, UploadFile
from redis.exceptions import RedisError

from app.api import config
from app.api.schemas.data import DataUploadResponse
from app.api.services.ingestion_runner import ingestion_runner
from app.api.services.zip_storage import ZipStorageError, store_zip_archive


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
    )
