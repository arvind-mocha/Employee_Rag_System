import logging

from app.indexing.embeddings import EmbeddingSettings, create_embedding_model
from app.indexing.vector_store import create_vector_store, index_chunks
from app.ingestion.cleaner import clean_documents
from app.ingestion.chunker import DocumentChunk, chunk_all_data


DOCUMENTS_DIRECTORY = "data"
STRUCTURED_DATA_DIRECTORY = "structured_data"
TEMP_DIRECTORY = "temp_data"

# Chunk size and overlap are measured in words.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
CLEANING_BATCH_SIZE = 8
CHUNKING_BATCH_SIZE = 32
MAX_WORKERS = 4


# Process 1: Clean documents
def process_1():
    return clean_documents(
        input_directory=DOCUMENTS_DIRECTORY,
        output_directory=TEMP_DIRECTORY,
        batch_size=CLEANING_BATCH_SIZE,
        max_workers=MAX_WORKERS,
        overwrite=True,
    )


# Process 2: Chunk documents
def process_2():
    return chunk_all_data(
        markdown_directory=TEMP_DIRECTORY,
        structured_directory=STRUCTURED_DATA_DIRECTORY,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        batch_size=CHUNKING_BATCH_SIZE,
        max_workers=MAX_WORKERS,
    )


# Process 3: Embed and index chunks in PGVector
def process_3(chunks: list[DocumentChunk]) -> list[str]:
    embedding_settings = EmbeddingSettings.from_env()
    embedding_model = create_embedding_model(embedding_settings)
    vector_store = create_vector_store(embedding_model)
    return index_chunks(
        vector_store=vector_store,
        chunks=chunks,
        batch_size=embedding_settings.batch_size,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        logging.info("Process 1: extracting and cleaning documents")
        cleaned_files = process_1()
        logging.info("Process 1 complete: %d files cleaned", len(cleaned_files))

        logging.info("Process 2: chunking cleaned documents")
        chunks = process_2()
        logging.info("Process 2 complete: %d chunks created", len(chunks))

        logging.info("Process 3: embedding and indexing chunks in PGVector")
        indexed_ids = process_3(chunks)
        logging.info(
            "Process 3 complete: %d chunks indexed",
            len(indexed_ids),
        )
    except Exception:
        logging.exception("Ingestion pipeline failed")
        raise


if __name__ == "__main__":
    main()
