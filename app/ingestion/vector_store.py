from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGVectorStore

from app.core.embeddings import get_embedding_dimension
from app.core.vector_store import (
    ID_COLUMN,
    PGVectorSettings,
    create_pg_engine,
    open_vector_store,
)
from app.ingestion.chunker import DocumentChunk
from app.ingestion.metadata import chunks_to_documents


def create_vector_store(
    embedding_model: Embeddings,
    settings: PGVectorSettings | None = None,
    *,
    recreate_table: bool = False,
) -> PGVectorStore:
    """Connect to the configured PGVector table, creating it when absent."""
    settings = settings or PGVectorSettings.from_config()
    engine = create_pg_engine(settings)

    if not recreate_table:
        try:
            return open_vector_store(engine, embedding_model, settings)
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
    return open_vector_store(engine, embedding_model, settings)


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
