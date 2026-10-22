import unittest

from app.indexing.metadata import chunk_to_document
from app.ingestion.chunker import DocumentChunk


class ChunkMetadataTests(unittest.TestCase):
    def test_converts_chunk_to_langchain_document(self) -> None:
        chunk = DocumentChunk(
            chunk_id="abc123",
            content="Section: Summary\n\nEmployee information",
            source_file="employees/Alex Chen.md",
            source_type="markdown",
            section="Summary",
            chunk_index=2,
        )

        document = chunk_to_document(chunk)

        self.assertEqual(document.id, "abc123")
        self.assertEqual(document.page_content, chunk.content)
        self.assertEqual(
            document.metadata,
            {
                "source_file": "employees/Alex Chen.md",
                "source_type": "markdown",
                "section": "Summary",
                "chunk_index": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
