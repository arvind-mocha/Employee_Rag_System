"""Schemas returned by data routes."""

from pydantic import BaseModel


class DataUploadResponse(BaseModel):
    """Summary of files stored from an uploaded ZIP."""

    message: str
    total_files: int
    structured_files: list[str]
    unstructured_files: list[str]
