import os
import unittest
from unittest.mock import Mock, patch

from app.core.vector_store import (
    ID_COLUMN,
    PGVectorSettings,
    _to_pg_engine_url,
)
from app.ingestion.vector_store import (
    create_vector_store,
    index_chunks,
)
from app.ingestion.chunker import DocumentChunk


class PGVectorSettingsTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "DATABASE_URL": "postgresql://user:password@localhost/database",
            "PGVECTOR_TABLE": "documents",
            "PGVECTOR_SCHEMA": "rag",
        },
    )
    def test_loads_pgvector_settings_from_environment(self) -> None:
        settings = PGVectorSettings.from_env()

        self.assertEqual(settings.table_name, "documents")
        self.assertEqual(settings.schema_name, "rag")

    def test_normalizes_standard_postgres_url(self) -> None:
        self.assertEqual(
            _to_pg_engine_url("postgresql://user:password@localhost/database"),
            "postgresql+psycopg://user:password@localhost/database",
        )


class VectorStoreCreationTests(unittest.TestCase):
    @patch("app.ingestion.vector_store.open_vector_store")
    @patch("app.ingestion.vector_store.create_pg_engine")
    def test_uses_existing_vector_table(
        self,
        engine_factory: Mock,
        open_vector_store: Mock,
    ) -> None:
        expected_store = Mock()
        open_vector_store.return_value = expected_store
        settings = PGVectorSettings(
            database_url="postgresql://localhost/database"
        )

        actual_store = create_vector_store(Mock(), settings)

        self.assertIs(actual_store, expected_store)
        engine_factory.return_value.init_vectorstore_table.assert_not_called()

    @patch("app.ingestion.vector_store.get_embedding_dimension", return_value=384)
    @patch("app.ingestion.vector_store.open_vector_store")
    @patch("app.ingestion.vector_store.create_pg_engine")
    def test_creates_missing_vector_table(
        self,
        engine_factory: Mock,
        open_vector_store: Mock,
        get_dimension: Mock,
    ) -> None:
        expected_store = Mock()
        open_vector_store.side_effect = [
            ValueError(f"Id column, {ID_COLUMN}, does not exist."),
            expected_store,
        ]
        settings = PGVectorSettings(
            database_url="postgresql://localhost/database"
        )

        actual_store = create_vector_store(Mock(), settings)

        self.assertIs(actual_store, expected_store)
        get_dimension.assert_called_once()
        initialization = engine_factory.return_value.init_vectorstore_table
        initialization.assert_called_once()
        id_column = initialization.call_args.kwargs["id_column"]
        self.assertEqual(id_column.name, ID_COLUMN)
        self.assertEqual(id_column.data_type, "TEXT")


class ChunkIndexingTests(unittest.TestCase):
    def test_indexes_chunks_in_batches(self) -> None:
        chunks = [
            DocumentChunk(
                chunk_id=f"chunk-{index}",
                content=f"content {index}",
                source_file="employees/example.md",
                source_type="markdown",
                section="Summary",
                chunk_index=index,
            )
            for index in range(3)
        ]
        vector_store = Mock()
        vector_store.add_documents.side_effect = (
            lambda documents, ids: ids
        )

        indexed_ids = index_chunks(vector_store, chunks, batch_size=2)

        self.assertEqual(indexed_ids, ["chunk-0", "chunk-1", "chunk-2"])
        self.assertEqual(vector_store.add_documents.call_count, 2)


if __name__ == "__main__":
    unittest.main()
