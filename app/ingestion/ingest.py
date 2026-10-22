import logging

from app.ingestion.cleaner import clean_documents


DATA_DIRECTORY = "data"
TEMP_DIRECTORY = "temp_data"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
CLEANING_BATCH_SIZE = 8
CHUNKING_BATCH_SIZE = 32
MAX_WORKERS = 4


def process_1():
    return clean_documents(
        input_directory=DATA_DIRECTORY,
        output_directory=TEMP_DIRECTORY,
        batch_size=CLEANING_BATCH_SIZE,
        max_workers=MAX_WORKERS,
        overwrite=True,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        logging.info("Process 1: extracting and cleaning documents")
        cleaned_files = process_1()
        logging.info("Process 1 complete: %d files cleaned", len(cleaned_files))
    except Exception:
        logging.exception("Ingestion pipeline failed")
        raise


if __name__ == "__main__":
    main()
