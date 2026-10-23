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


if __name__ == "__main__":
    unittest.main()
