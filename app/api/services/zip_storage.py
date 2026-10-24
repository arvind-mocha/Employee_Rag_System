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
    ignored_files: list[str]
    renamed_files: dict[str, str]

    @property
    def total_files(self) -> int:
        return len(self.structured_files) + len(self.unstructured_files)


@dataclass(frozen=True, slots=True)
class PlannedArchive:
    """Validated archive members and the decisions made for each one."""

    files: list[PlannedFile]
    ignored_files: list[str]
    renamed_files: dict[str, str]


def _stored_names() -> set[str]:
    """Return existing filenames reserved across both source directories."""
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


def _category_for(filename: str) -> FileCategory | None:
    extension = Path(filename).suffix.lower()
    if extension in config.STRUCTURED_EXTENSIONS:
        return "structured"
    if extension in config.UNSTRUCTURED_EXTENSIONS:
        return "unstructured"
    return None


def _available_filename(filename: str, reserved_names: set[str]) -> str:
    """Reserve a case-insensitively unique flattened filename."""
    if filename.casefold() not in reserved_names:
        reserved_names.add(filename.casefold())
        return filename

    path = Path(filename)
    counter = 1
    while True:
        candidate = f"{path.stem}_{counter}{path.suffix}"
        if candidate.casefold() not in reserved_names:
            reserved_names.add(candidate.casefold())
            return candidate
        counter += 1


def _plan_files(archive: ZipFile) -> PlannedArchive:
    planned_files: list[PlannedFile] = []
    ignored_files: list[str] = []
    renamed_files: dict[str, str] = {}
    reserved_names = _stored_names()
    uncompressed_size = 0
    file_count = 0

    for member in archive.infolist():
        if member.is_dir():
            continue

        filename = _safe_flat_filename(member.filename)
        member_parts = PurePosixPath(member.filename.replace("\\", "/")).parts

        if "__MACOSX" in member_parts or filename == ".DS_Store":
            continue

        file_count += 1
        uncompressed_size += member.file_size

        category = _category_for(filename)
        if category is None:
            ignored_files.append(member.filename)
            continue

        stored_filename = _available_filename(filename, reserved_names)
        if stored_filename != filename:
            renamed_files[member.filename] = stored_filename
        planned_files.append((member, stored_filename, category))

    if file_count > config.MAX_FILE_COUNT:
        raise ZipStorageError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            f"The ZIP contains more than {config.MAX_FILE_COUNT} files.",
        )

    if uncompressed_size > config.MAX_UNCOMPRESSED_SIZE:
        raise ZipStorageError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            "The uncompressed files are larger than 200 MB.",
        )

    if not planned_files:
        raise ZipStorageError(
            HTTPStatus.BAD_REQUEST,
            "The ZIP does not contain any supported files.",
        )

    return PlannedArchive(
        files=planned_files,
        ignored_files=ignored_files,
        renamed_files=renamed_files,
    )


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
            plan = _plan_files(archive)

            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise ZipStorageError(
                    HTTPStatus.BAD_REQUEST,
                    f"Corrupt file in ZIP: {corrupt_member}",
                )

            _extract_files(archive, plan.files)
    except BadZipFile as error:
        raise ZipStorageError(
            HTTPStatus.BAD_REQUEST,
            "The uploaded file is not a valid ZIP archive.",
        ) from error

    return StoredArchive(
        structured_files=[
            filename
            for _, filename, category in plan.files
            if category == "structured"
        ],
        unstructured_files=[
            filename
            for _, filename, category in plan.files
            if category == "unstructured"
        ],
        ignored_files=plan.ignored_files,
        renamed_files=plan.renamed_files,
    )
