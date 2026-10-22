from dataclasses import dataclass
import os
import re

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore


DEFAULT_TABLE_NAME = "employee_documents"
DEFAULT_SCHEMA_NAME = "public"
ID_COLUMN = "chunk_id"
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class PGVectorSettings:
    database_url: str
    table_name: str = DEFAULT_TABLE_NAME
    schema_name: str = DEFAULT_SCHEMA_NAME

    @classmethod
    def from_env(cls) -> "PGVectorSettings":
        load_dotenv()

        database_url = os.getenv("DATABASE_URL", "").strip()
        if not database_url:
            raise ValueError("DATABASE_URL must be set in the environment")

        table_name = os.getenv("PGVECTOR_TABLE", DEFAULT_TABLE_NAME).strip()
        schema_name = os.getenv("PGVECTOR_SCHEMA", DEFAULT_SCHEMA_NAME).strip()
        _validate_identifier("PGVECTOR_TABLE", table_name)
        _validate_identifier("PGVECTOR_SCHEMA", schema_name)

        return cls(
            database_url=database_url,
            table_name=table_name,
            schema_name=schema_name,
        )


def _validate_identifier(setting_name: str, value: str) -> None:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{setting_name} must start with a letter or underscore and contain "
            "only letters, numbers, and underscores"
        )


def _to_pg_engine_url(database_url: str) -> str:
    """Convert a standard PostgreSQL URL to an async psycopg URL."""
    if database_url.startswith("postgresql+psycopg://"):
        return database_url
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url
    if database_url.startswith("postgresql://"):
        return database_url.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )
    if database_url.startswith("postgres://"):
        return database_url.replace(
            "postgres://",
            "postgresql+psycopg://",
            1,
        )

    raise ValueError(
        "DATABASE_URL must use postgres://, postgresql://, "
        "postgresql+psycopg://, or postgresql+asyncpg://"
    )


def create_pg_engine(settings: PGVectorSettings) -> PGEngine:
    return PGEngine.from_connection_string(
        url=_to_pg_engine_url(settings.database_url)
    )


def open_vector_store(
    engine: PGEngine,
    embedding_model: Embeddings,
    settings: PGVectorSettings,
) -> PGVectorStore:
    """Open an existing PGVector table."""
    return PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embedding_model,
        table_name=settings.table_name,
        schema_name=settings.schema_name,
        id_column=ID_COLUMN,
    )


def connect_vector_store(
    embedding_model: Embeddings,
    settings: PGVectorSettings | None = None,
) -> PGVectorStore:
    """Connect to an existing PGVector table without modifying it."""
    settings = settings or PGVectorSettings.from_env()
    engine = create_pg_engine(settings)
    return open_vector_store(engine, embedding_model, settings)
