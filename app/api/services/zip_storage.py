"""ZIP validation, extraction, and storage service."""

from dataclasses import dataclass
from http import HTTPStatus
from io import BytesIO
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import Literal
from zipfile import BadZipFile, ZipFile, ZipInfo, is_zipfile

from app.api import config


FileCategory = Literal["structured", "unstructured"]
PlannedFile = tuple[ZipInfo, str, FileCategory]


class ZipStorageError(Exception):
    """A safe validation or storage error that can be returned by the API."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class StoredArchive:
    """Summary of files stored from one archive."""

    structured_files: list[str]
    unstructured_files: list[str]

    @property
    def total_files(self) -> int:
        return len(self.structured_files) + len(self.unstructured_files)


def _stored_names() -> set[str]:
    """Return existing filenames so an upload cannot silently overwrite data."""
    names: set[str] = set()
    for directory in (config.DATA_DIRECTORY, config.STRUCTURED_DATA_DIRECTORY):
        if directory.exists():
            names.update(
                path.name.casefold()
                for path in directory.rglob("*")
                if path.is_file()
            )
    return names


def _safe_flat_filename(member_name: str) -> str:
    """Validate a ZIP member path and return only its filename."""
    member_path = PurePosixPath(member_name.replace("\\", "/"))

    if member_path.is_absolute() or ".." in member_path.parts:
        raise ZipStorageError(
            HTTPStatus.BAD_REQUEST,
            f"Unsafe path in ZIP: {member_name}",
        )

    filename = member_path.name
    if not filename:
        raise ZipStorageError(
            HTTPStatus.BAD_REQUEST,
            f"Invalid filename in ZIP: {member_name}",
        )
    return filename


def _category_for(filename: str) -> FileCategory:
    extension = Path(filename).suffix.lower()
    if extension in config.STRUCTURED_EXTENSIONS:
        return "structured"
    if extension in config.UNSTRUCTURED_EXTENSIONS:
        return "unstructured"
    raise ZipStorageError(
        HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
        f"Unsupported file type: {filename}",
    )


def _plan_files(archive: ZipFile) -> list[PlannedFile]:
    planned_files: list[PlannedFile] = []
    upload_names: set[str] = set()
    uncompressed_size = 0

    for member in archive.infolist():
        if member.is_dir():
            continue

        filename = _safe_flat_filename(member.filename)
        member_parts = PurePosixPath(member.filename.replace("\\", "/")).parts

        if "__MACOSX" in member_parts or filename == ".DS_Store":
            continue

        normalized_name = filename.casefold()
        if normalized_name in upload_names:
            raise ZipStorageError(
                HTTPStatus.CONFLICT,
                f"Duplicate filename after flattening ZIP folders: {filename}",
            )

        upload_names.add(normalized_name)
        uncompressed_size += member.file_size
        planned_files.append((member, filename, _category_for(filename)))

    if not planned_files:
        raise ZipStorageError(
            HTTPStatus.BAD_REQUEST,
            "The ZIP does not contain any supported files.",
        )

    if len(planned_files) > config.MAX_FILE_COUNT:
        raise ZipStorageError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            f"The ZIP contains more than {config.MAX_FILE_COUNT} files.",
        )

    if uncompressed_size > config.MAX_UNCOMPRESSED_SIZE:
        raise ZipStorageError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            "The uncompressed files are larger than 200 MB.",
        )

    conflicts = sorted(upload_names & _stored_names())
    if conflicts:
        raise ZipStorageError(
            HTTPStatus.CONFLICT,
            f"Files already exist: {', '.join(conflicts)}",
        )

    return planned_files


def _extract_files(archive: ZipFile, planned_files: list[PlannedFile]) -> None:
    config.DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config.STRUCTURED_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".data-upload-", dir=config.PROJECT_ROOT) as temp_dir:
        staging_directory = Path(temp_dir)

        for member, filename, _ in planned_files:
            staged_file = staging_directory / filename
            with archive.open(member) as source, staged_file.open("xb") as destination:
                copyfileobj(source, destination)

        for _, filename, category in planned_files:
            destination_directory = (
                config.STRUCTURED_DATA_DIRECTORY
                if category == "structured"
                else config.DATA_DIRECTORY
            )
            (staging_directory / filename).replace(destination_directory / filename)


def store_zip_archive(archive_bytes: bytes) -> StoredArchive:
    """Validate and store all supported files from a ZIP archive."""
    if len(archive_bytes) > config.MAX_ZIP_SIZE:
        raise ZipStorageError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            "ZIP file is larger than 50 MB.",
        )

    archive_stream = BytesIO(archive_bytes)
    if not is_zipfile(archive_stream):
        raise ZipStorageError(
            HTTPStatus.BAD_REQUEST,
            "The uploaded file is not a valid ZIP archive.",
        )

    archive_stream.seek(0)
    try:
        with ZipFile(archive_stream) as archive:
            planned_files = _plan_files(archive)

            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise ZipStorageError(
                    HTTPStatus.BAD_REQUEST,
                    f"Corrupt file in ZIP: {corrupt_member}",
                )

            _extract_files(archive, planned_files)
    except BadZipFile as error:
        raise ZipStorageError(
            HTTPStatus.BAD_REQUEST,
            "The uploaded file is not a valid ZIP archive.",
        ) from error

    return StoredArchive(
        structured_files=[
            filename
            for _, filename, category in planned_files
            if category == "structured"
        ],
        unstructured_files=[
            filename
            for _, filename, category in planned_files
            if category == "unstructured"
        ],
    )
