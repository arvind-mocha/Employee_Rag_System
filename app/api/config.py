"""Configuration for the API layer."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data"
STRUCTURED_DATA_DIRECTORY = PROJECT_ROOT / "structured_data"
TEMP_DATA_DIRECTORY = PROJECT_ROOT / "temp_data"
EVALUATION_DATA_DIRECTORY = PROJECT_ROOT / "evaluation_data"

STRUCTURED_EXTENSIONS = frozenset({".csv", ".json", ".xlsx"})
UNSTRUCTURED_EXTENSIONS = frozenset(
    {".docx", ".md", ".markdown", ".pdf", ".txt"}
)

MAX_ZIP_SIZE = 50 * 1024 * 1024
MAX_UNCOMPRESSED_SIZE = 200 * 1024 * 1024
MAX_FILE_COUNT = 1_000
