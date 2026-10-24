"""Persist a compact corpus sample for on-demand evaluation generation."""

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from app.api import config
from app.ingestion.chunker import DocumentChunk


SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_MAX_CHUNKS = 96


def _source_diverse_chunks(
    chunks: list[DocumentChunk],
    maximum: int = SNAPSHOT_MAX_CHUNKS,
) -> list[DocumentChunk]:
    by_source: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source_file].append(chunk)

    interleaved: list[DocumentChunk] = []
    position = 0
    sources = sorted(by_source)
    while True:
        added = False
        for source in sources:
            source_chunks = by_source[source]
            if position < len(source_chunks):
                interleaved.append(source_chunks[position])
                added = True
        if not added:
            break
        position += 1

    if len(interleaved) <= maximum:
        return interleaved

    last_index = len(interleaved) - 1
    return [
        interleaved[round(index * last_index / (maximum - 1))]
        for index in range(maximum)
    ]


def write_evaluation_corpus_snapshot(
    chunks: list[DocumentChunk],
    destination: Path | None = None,
) -> Path:
    """Atomically store representative chunks already created by ingestion."""
    destination = destination or config.EVALUATION_CORPUS_SNAPSHOT_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected = _source_diverse_chunks(chunks)
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "chunks": [asdict(chunk) for chunk in selected],
    }
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def read_evaluation_corpus_snapshot(
    source: Path | None = None,
) -> list[DocumentChunk]:
    """Read and validate the compact ingestion snapshot."""
    source = source or config.EVALUATION_CORPUS_SNAPSHOT_FILE
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
        or not isinstance(payload.get("chunks"), list)
    ):
        raise ValueError("The evaluation corpus snapshot is invalid.")
    try:
        chunks = [DocumentChunk(**item) for item in payload["chunks"]]
    except (TypeError, KeyError) as error:
        raise ValueError("The evaluation corpus snapshot is invalid.") from error
    if not chunks:
        raise ValueError("The evaluation corpus snapshot is empty.")
    return chunks
