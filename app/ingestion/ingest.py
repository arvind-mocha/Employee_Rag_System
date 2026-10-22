import logging

from app.ingestion.cleaner import clean_documents
from app.ingestion.chunker import chunk_all_data


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

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        logging.info("Process 1: extracting and cleaning documents")
        cleaned_files = process_1()
        logging.info("Process 1 complete: %d files cleaned", len(cleaned_files))

        logging.info("Process 2: chunking cleaned documents")
        chunks = process_2()
        logging.info("Process 2 complete: %d chunks created", len(chunks))
    except Exception:
        logging.exception("Ingestion pipeline failed")
        raise


if __name__ == "__main__":
    main()
