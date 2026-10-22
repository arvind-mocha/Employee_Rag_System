from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import unicodedata

from app.Utility.file_handler import create_file
from app.ingestion.extractor import extract_content


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
}
PROGRESS_BAR_WIDTH = 40


def _display_progress(label: str, completed: int, total: int) -> None:
    filled_width = round(PROGRESS_BAR_WIDTH * completed / total)
    bar = "█" * filled_width + "░" * (PROGRESS_BAR_WIDTH - filled_width)
    line_ending = "\n" if completed == total else ""
    print(f"\r{label} [{bar}]", end=line_ending, flush=True)


def clean_text(content: str) -> str:
    """Remove formatting noise without rewriting the document's meaning."""
    content = unicodedata.normalize("NFC", content)
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = content.replace("\u00a0", " ")
    content = content.replace("\ufeff", "")
    content = re.sub(r"[\u200b\u200c\u200d\u2060]", "", content)
    content = re.sub(r"(?<=\w)-\n(?=\w)", "", content)

    cleaned_lines = []

    for line in content.splitlines():
        line = line.rstrip(" \f\v")
        cleaned_lines.append(line)

    content = "\n".join(cleaned_lines)
    content = re.sub(r"\n{3,}", "\n\n", content)

    return content.strip()


def _clean_document(
    source_file: Path,
    source_directory: Path,
    destination_directory: Path,
    overwrite: bool,
) -> Path:
    extracted_content = extract_content(source_file)
    cleaned_content = clean_text(extracted_content)
    relative_path = source_file.relative_to(source_directory)

    return create_file(
        file_name=source_file.stem,
        extension="md",
        content=cleaned_content,
        output_directory=destination_directory / relative_path.parent,
        overwrite=overwrite,
    )


def clean_documents(
    input_directory: str | Path,
    output_directory: str | Path,
    batch_size: int,
    max_workers: int,
    overwrite: bool = False,
) -> list[Path]:
    """Clean supported documents in parallel batches and store them as Markdown."""
    source_directory = Path(input_directory)
    destination_directory = Path(output_directory)

    if not source_directory.is_dir():
        raise NotADirectoryError(f"Data directory not found: {source_directory}")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    source_root = source_directory.resolve()
    destination_root = destination_directory.resolve()

    if source_root == destination_root:
        raise ValueError("output_directory must be different from input_directory")

    source_files = [
        path
        for path in sorted(source_directory.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and destination_root not in path.resolve().parents
    ]
    cleaned_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for start in range(0, len(source_files), batch_size):
            batch = source_files[start : start + batch_size]
            batch_number = (start // batch_size) + 1
            batch_label = f"Cleaning documents - Batch {batch_number}"
            completed_in_batch = 0
            _display_progress(batch_label, 0, len(batch))
            futures = {
                executor.submit(
                    _clean_document,
                    source_file,
                    source_directory,
                    destination_directory,
                    overwrite,
                )
                for source_file in batch
            }

            for future in as_completed(futures):
                cleaned_files.append(future.result())
                completed_in_batch += 1
                _display_progress(
                    batch_label,
                    completed_in_batch,
                    len(batch),
                )

    cleaned_files.sort()
    return cleaned_files
