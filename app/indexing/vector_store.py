from dataclasses import dataclass
import os
import re

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGEngine, PGVectorStore

from app.indexing.embeddings import get_embedding_dimension
from app.indexing.metadata import chunks_to_documents
from app.ingestion.chunker import DocumentChunk


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


def _open_vector_store(
    engine: PGEngine,
    embedding_model: Embeddings,
    settings: PGVectorSettings,
) -> PGVectorStore:
    return PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embedding_model,
        table_name=settings.table_name,
        schema_name=settings.schema_name,
        id_column=ID_COLUMN,
    )


def create_vector_store(
    embedding_model: Embeddings,
    settings: PGVectorSettings | None = None,
    *,
    recreate_table: bool = False,
) -> PGVectorStore:
    """Connect to the configured PGVector table, creating it when absent."""
    settings = settings or PGVectorSettings.from_env()
    engine = PGEngine.from_connection_string(
        url=_to_pg_engine_url(settings.database_url)
    )

    if not recreate_table:
        try:
            return _open_vector_store(engine, embedding_model, settings)
        except ValueError as error:
            missing_table_error = f"Id column, {ID_COLUMN}, does not exist."
            if str(error) != missing_table_error:
                raise

    vector_size = get_embedding_dimension(embedding_model)
    engine.init_vectorstore_table(
        table_name=settings.table_name,
        vector_size=vector_size,
        schema_name=settings.schema_name,
        id_column=Column(ID_COLUMN, "TEXT", nullable=False),
        overwrite_existing=recreate_table,
    )
    return _open_vector_store(engine, embedding_model, settings)


def index_chunks(
    vector_store: PGVectorStore,
    chunks: list[DocumentChunk],
    batch_size: int,
) -> list[str]:
    """Embed and upsert document chunks into PGVector in batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    indexed_ids: list[str] = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        documents = chunks_to_documents(batch)
        chunk_ids = [chunk.chunk_id for chunk in batch]
        indexed_ids.extend(
            vector_store.add_documents(documents=documents, ids=chunk_ids)
        )

    return indexed_ids
