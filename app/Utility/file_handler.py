from pathlib import Path
import shutil


TEMP_DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "temp_data"


def create_file(
    file_name: str,
    extension: str,
    content: str = "",
    output_directory: str | Path = "temp_data",
    overwrite: bool = False,
) -> Path:
    """Create a UTF-8 text file and return its path."""
    if not file_name or file_name in {".", ".."}:
        raise ValueError("file_name must be a non-empty file name")

    if Path(file_name).name != file_name:
        raise ValueError("file_name must not contain a directory path")

    normalized_extension = extension.strip().lstrip(".").lower()

    if not normalized_extension or not normalized_extension.isalnum():
        raise ValueError("extension must contain only letters and numbers")

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    file_path = destination / f"{file_name}.{normalized_extension}"
    write_mode = "w" if overwrite else "x"

    with file_path.open(write_mode, encoding="utf-8") as file:
        file.write(content)

    return file_path


def remove_temp_directory() -> bool:
    """Remove the project's temp_data directory if it exists."""
    if not TEMP_DATA_DIRECTORY.exists():
        return False

    if not TEMP_DATA_DIRECTORY.is_dir():
        raise NotADirectoryError(
            f"Temporary data path is not a directory: {TEMP_DATA_DIRECTORY}"
        )

    shutil.rmtree(TEMP_DATA_DIRECTORY)
    return True
