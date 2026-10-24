"""Schemas returned by data routes."""

from typing import Literal

from pydantic import BaseModel


class DataUploadResponse(BaseModel):
    """Summary of files stored from an uploaded ZIP."""

    message: str
    total_files: int
    structured_files: list[str]
    unstructured_files: list[str]
    ignored_files: list[str]
    renamed_files: dict[str, str]


class VectorDatabaseEraseResponse(BaseModel):
    """Confirmation that indexed vectors were erased."""

    message: str
    status: Literal["erased"]
