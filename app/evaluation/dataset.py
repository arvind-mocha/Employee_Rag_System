"""Safe JSONL persistence and content-version helpers for evaluations."""

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable

from pydantic import ValidationError

from app.api import config
from app.evaluation.models import (
    EvidenceSnapshot,
    GroundTruthRecord,
    GroundTruthStatus,
)
from app.retrieval.models import RetrievedChunk


DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class EvaluationDatasetError(ValueError):
    """A safe validation error for an evaluation dataset."""


def validate_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip()
    if not DATASET_NAME_PATTERN.fullmatch(normalized):
        raise EvaluationDatasetError(
            "dataset_name must contain only letters, numbers, underscores, "
            "or hyphens and be at most 64 characters"
        )
    return normalized


def dataset_path(
    dataset_name: str,
    directory: Path | None = None,
) -> Path:
    name = validate_dataset_name(dataset_name)
    root = directory or config.EVALUATION_DATA_DIRECTORY
    return root / f"{name}.jsonl"


def text_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_fingerprint(evidence: Iterable[EvidenceSnapshot]) -> str:
    hashes = sorted(item.content_sha256 for item in evidence)
    return text_sha256("\n".join(hashes))


def indexed_corpus_fingerprint() -> str:
    """Fingerprint every file that can supply the currently indexed corpus."""
    entries = []
    roots = (
        ("markdown", config.TEMP_DATA_DIRECTORY),
        ("structured", config.STRUCTURED_DATA_DIRECTORY),
    )
    for label, root in roots:
        if not root.is_dir():
            continue
        snapshot_file = config.EVALUATION_CORPUS_SNAPSHOT_FILE.resolve()
        snapshot_temporary = snapshot_file.with_suffix(
            f"{snapshot_file.suffix}.tmp"
        )
        files = sorted(
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file()
            and candidate.resolve() not in {snapshot_file, snapshot_temporary}
        )
        for path in files:
            entries.append(
                f"{label}:{path.relative_to(root).as_posix()}:{file_sha256(path)}"
            )
    return text_sha256("\n".join(entries))


def _indexed_source_path(chunk: RetrievedChunk) -> Path | None:
    if not chunk.source_file:
        return None
    root = (
        config.STRUCTURED_DATA_DIRECTORY
        if chunk.source_type in {"csv", "json", "xlsx"}
        else config.TEMP_DATA_DIRECTORY
    ).resolve()
    candidate = (root / chunk.source_file).resolve()
    if candidate != root and root not in candidate.parents:
        return None
    return candidate


def snapshot_evidence(
    chunk: RetrievedChunk,
    position: int,
) -> EvidenceSnapshot:
    source_path = _indexed_source_path(chunk)
    source_hash = (
        file_sha256(source_path)
        if source_path is not None and source_path.is_file()
        else None
    )
    return EvidenceSnapshot(
        position=position,
        chunk_id=chunk.chunk_id,
        content=chunk.content,
        content_sha256=text_sha256(chunk.content),
        source_file=chunk.source_file,
        source_type=chunk.source_type,
        source_sha256=source_hash,
        section=chunk.section,
        chunk_index=chunk.chunk_index,
    )


def ground_truth_status(record: GroundTruthRecord) -> GroundTruthStatus:
    if record.corpus_fingerprint:
        if indexed_corpus_fingerprint() != record.corpus_fingerprint:
            return GroundTruthStatus.STALE
        return GroundTruthStatus.CURRENT

    verifiable = [
        evidence
        for evidence in (record.retrieved_evidence or record.relevant_evidence)
        if evidence.source_file and evidence.source_sha256
    ]
    if not verifiable:
        return GroundTruthStatus.UNVERIFIABLE

    for evidence in verifiable:
        chunk = RetrievedChunk(
            content=evidence.content,
            relevance_score=0.0,
            chunk_id=evidence.chunk_id,
            source_file=evidence.source_file,
            source_type=evidence.source_type,
            section=evidence.section,
            chunk_index=evidence.chunk_index,
        )
        source_path = _indexed_source_path(chunk)
        if source_path is None or not source_path.is_file():
            return GroundTruthStatus.STALE
        if file_sha256(source_path) != evidence.source_sha256:
            return GroundTruthStatus.STALE
    return GroundTruthStatus.CURRENT


def _validate_record_integrity(
    record: GroundTruthRecord,
    line_number: int,
) -> None:
    if record.answerable and (
        not record.required_facts or not record.relevant_evidence
    ):
        raise EvaluationDatasetError(
            f"answerable record lacks facts or evidence at line {line_number}"
        )
    if not record.answerable and (
        record.required_facts or record.relevant_evidence
    ):
        raise EvaluationDatasetError(
            f"unanswerable record contains answer evidence at line {line_number}"
        )

    snapshots = record.retrieved_evidence or record.relevant_evidence
    for evidence in [*record.retrieved_evidence, *record.relevant_evidence]:
        if text_sha256(evidence.content) != evidence.content_sha256:
            raise EvaluationDatasetError(
                f"evidence content hash mismatch at line {line_number}"
            )
    if evidence_fingerprint(snapshots) != record.evidence_fingerprint:
        raise EvaluationDatasetError(
            f"evidence fingerprint mismatch at line {line_number}"
        )
    if record.retrieved_evidence:
        retrieved_hashes = {
            evidence.content_sha256 for evidence in record.retrieved_evidence
        }
        if any(
            evidence.content_sha256 not in retrieved_hashes
            for evidence in record.relevant_evidence
        ):
            raise EvaluationDatasetError(
                f"relevant evidence was not retrieved at line {line_number}"
            )


def write_dataset(
    dataset_name: str,
    records: list[GroundTruthRecord],
    *,
    directory: Path | None = None,
    overwrite: bool = False,
) -> Path:
    if not records:
        raise EvaluationDatasetError("an evaluation dataset cannot be empty")

    destination = dataset_path(dataset_name, directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"evaluation dataset already exists: {destination.name}"
        )

    temporary = destination.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(record.model_dump_json())
            output.write("\n")
    temporary.replace(destination)
    return destination


def parse_dataset_jsonl(
    content: bytes,
    *,
    maximum_records: int | None = None,
) -> list[GroundTruthRecord]:
    """Validate uploaded JSONL before it is allowed into dataset storage."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise EvaluationDatasetError(
            "the evaluation dataset must be UTF-8 encoded"
        ) from error

    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if maximum_records is not None and len(records) >= maximum_records:
            raise EvaluationDatasetError(
                f"the evaluation dataset cannot contain more than "
                f"{maximum_records} records"
            )
        try:
            record = GroundTruthRecord.model_validate_json(line)
            _validate_record_integrity(record, line_number)
            records.append(record)
        except (ValidationError, json.JSONDecodeError) as error:
            raise EvaluationDatasetError(
                f"invalid JSONL record at line {line_number}: {error}"
            ) from error
    if not records:
        raise EvaluationDatasetError("the evaluation dataset is empty")
    return records


def read_dataset(
    dataset_name: str,
    *,
    directory: Path | None = None,
) -> list[GroundTruthRecord]:
    source = dataset_path(dataset_name, directory)
    if not source.is_file():
        raise FileNotFoundError(f"evaluation dataset not found: {source.name}")
    return parse_dataset_jsonl(source.read_bytes())


def dataset_sha256(
    dataset_name: str,
    *,
    directory: Path | None = None,
) -> str:
    return file_sha256(dataset_path(dataset_name, directory))
