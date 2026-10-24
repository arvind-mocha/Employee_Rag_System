import unittest
from unittest.mock import patch

from app.api import config
from app.ingestion import ingest


class IngestionPathTests(unittest.TestCase):
    @patch("app.ingestion.ingest.clean_documents", return_value=[])
    def test_cleaning_uses_central_absolute_paths(self, clean_documents) -> None:
        ingest.process_1()

        clean_documents.assert_called_once_with(
            input_directory=config.DATA_DIRECTORY,
            output_directory=config.TEMP_DATA_DIRECTORY,
            batch_size=ingest.CLEANING_BATCH_SIZE,
            max_workers=ingest.MAX_WORKERS,
            overwrite=True,
        )

    @patch("app.ingestion.ingest.chunk_all_data", return_value=[])
    def test_chunking_uses_central_absolute_paths(self, chunk_all_data) -> None:
        ingest.process_2()

        chunk_all_data.assert_called_once_with(
            markdown_directory=config.TEMP_DATA_DIRECTORY,
            structured_directory=config.STRUCTURED_DATA_DIRECTORY,
            chunk_size=ingest.CHUNK_SIZE,
            chunk_overlap=ingest.CHUNK_OVERLAP,
            batch_size=ingest.CHUNKING_BATCH_SIZE,
            max_workers=ingest.MAX_WORKERS,
        )

    @patch("app.ingestion.ingest.write_evaluation_corpus_snapshot")
    @patch(
        "app.ingestion.ingest.process_3",
        side_effect=RuntimeError("index failed"),
    )
    @patch("app.ingestion.ingest.process_2", return_value=[])
    @patch("app.ingestion.ingest.process_1", return_value=[])
    def test_does_not_publish_a_snapshot_when_indexing_fails(
        self,
        _process_1,
        _process_2,
        _process_3,
        write_snapshot,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "index failed"):
            ingest.main()

        write_snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
