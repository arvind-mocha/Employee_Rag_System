"""Single source of truth for application configuration.

These defaults support both local development and the standard Docker service
names without requiring an ``.env`` file. Customer-supplied model credentials
remain in the encrypted PostgreSQL credential store and are not defined here.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNING_IN_DOCKER = Path("/.dockerenv").exists()

# PostgreSQL and PGVector
POSTGRES_DATABASE = "employee_rag"
POSTGRES_USER = "employee_rag"
POSTGRES_PASSWORD = "employee_rag_dev"
POSTGRES_HOST = "pgvector" if RUNNING_IN_DOCKER else "localhost"
POSTGRES_PORT = 5432 if RUNNING_IN_DOCKER else 6024
DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"
)
PGVECTOR_TABLE = "employee_documents"
PGVECTOR_SCHEMA = "public"

# Redis and background jobs
REDIS_HOST = "redis" if RUNNING_IN_DOCKER else "localhost"
REDIS_PORT = 6379
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
REDIS_EVALUATION_QUEUE = "employee-rag-evaluations"
REDIS_JOB_TIMEOUT_SECONDS = 30 * 60
REDIS_RESULT_TTL_SECONDS = 24 * 60 * 60

# Local embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DEVICE = "cpu"

# Provider integration defaults. API keys and selected model IDs are managed
# through the Settings page instead of source configuration.
HUGGINGFACE_INFERENCE_PROVIDER = "auto"
GROK_BASE_URL = "https://api.x.ai/v1"

# input, output, cached input, and cache-creation USD per million tokens.
MODEL_PRICING_USD_PER_MILLION: dict[
    tuple[str, str],
    tuple[float, float, float, float],
] = {
    ("openai", "gpt-6-astra"): (10.0, 50.0, 1.0, 12.5),
    ("anthropic", "claude-opus-5-5"): (4.0, 20.0, 0.2, 5.0),
    ("grok", "grok-4.7"): (2.0, 6.0, 0.5, 2.0),
}

# Credential encryption. A mounted Docker secret is selected automatically;
# local runs generate a private installation key under .secrets/.
DOCKER_CREDENTIAL_KEY_FILE = Path(
    "/run/secrets/credential_encryption_key"
)
USE_DOCKER_CREDENTIAL_SECRET = DOCKER_CREDENTIAL_KEY_FILE.is_file()
CREDENTIAL_ENCRYPTION_KEY_FILE = (
    DOCKER_CREDENTIAL_KEY_FILE
    if USE_DOCKER_CREDENTIAL_SECRET
    else PROJECT_ROOT / ".secrets" / "credential_encryption_key"
)

# Data paths and upload limits
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
MAX_EVALUATION_DATASET_SIZE = 10 * 1024 * 1024
MAX_EVALUATION_RECORDS = 100
