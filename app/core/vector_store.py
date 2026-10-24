from dataclasses import dataclass
import re

from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore

from app.api import config


DEFAULT_TABLE_NAME = config.PGVECTOR_TABLE
DEFAULT_SCHEMA_NAME = config.PGVECTOR_SCHEMA
ID_COLUMN = "chunk_id"
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class PGVectorSettings:
    database_url: str = config.DATABASE_URL
    table_name: str = config.PGVECTOR_TABLE
    schema_name: str = config.PGVECTOR_SCHEMA

    @classmethod
    def from_config(cls) -> "PGVectorSettings":
        settings = cls()
        _validate_identifier("PGVECTOR_TABLE", settings.table_name)
        _validate_identifier("PGVECTOR_SCHEMA", settings.schema_name)
        return settings


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
    settings = settings or PGVectorSettings.from_config()
    engine = create_pg_engine(settings)
    return open_vector_store(engine, embedding_model, settings)


def drop_vector_store_table(
    settings: PGVectorSettings | None = None,
) -> None:
    """Erase the configured vector table without touching other database data."""
    settings = settings or PGVectorSettings.from_config()
    engine = create_pg_engine(settings)
    engine.drop_table(
        table_name=settings.table_name,
        schema_name=settings.schema_name,
    )
