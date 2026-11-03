"""Safe JSONL persistence and content-version helpers for evaluations."""

from hashlib import sha256
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Iterable
from tempfile import TemporaryDirectory
from uuid import uuid4

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
    uploaded = root / name / "dataset.jsonl"
    candidate = uploaded if uploaded.is_file() else root / f"{name}.jsonl"
    if root.resolve() not in candidate.resolve().parents:
        raise EvaluationDatasetError("dataset path is outside dataset storage")
    return candidate


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
        # Ignore the derived snapshot left by older installations.
        snapshot_file = (config.TEMP_DATA_DIRECTORY / ".evaluation_corpus.json").resolve()
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
    if not record.answerable and (
        record.required_facts or record.relevant_evidence
    ):
        raise EvaluationDatasetError(
            f"unanswerable record contains answer evidence at line {line_number}"
        )

    snapshots = record.retrieved_evidence or record.relevant_evidence
    for evidence in [*record.retrieved_evidence, *record.relevant_evidence]:
        digest = text_sha256(evidence.content)
        if evidence.content_sha256 is not None and digest != evidence.content_sha256:
            raise EvaluationDatasetError(
                f"evidence content hash mismatch at line {line_number}"
            )
        evidence.content_sha256 = digest
    if record.evidence_fingerprint is not None and evidence_fingerprint(snapshots) != record.evidence_fingerprint:
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
        line_number = content[:error.start].count(b"\n") + 1
        raise EvaluationDatasetError(
            f"the evaluation dataset must be UTF-8 encoded (line {line_number})"
        ) from error

    records = []
    record_ids = set()
    # Split physical JSONL lines only; U+2028/U+2029 may occur inside JSON strings.
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        if maximum_records is not None and len(records) >= maximum_records:
            raise EvaluationDatasetError(
                f"the evaluation dataset cannot contain more than "
                f"{maximum_records} records (line {line_number})"
            )
        try:
            payload = json.loads(line, parse_constant=_reject_json_constant)
            if not isinstance(payload, dict):
                raise ValueError("each record must be a JSON object")
            payload.setdefault("id", f"line-{line_number}")
            record = GroundTruthRecord.model_validate(payload)
            _validate_record_integrity(record, line_number)
            if record.id in record_ids:
                raise ValueError(f"duplicate record id: {record.id}")
            record_ids.add(record.id)
            records.append(record)
        except (ValidationError, ValueError) as error:
            raise EvaluationDatasetError(
                f"invalid JSONL record at line {line_number}: {error}"
            ) from error
    if not records:
        raise EvaluationDatasetError("the evaluation dataset is empty")
    return records


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def validate_upload_filename(filename: str) -> str:
    if (
        not filename or len(filename.encode("utf-8")) > 255
        or filename != filename.strip()
        or any(character in filename for character in ("/", "\\"))
        or any(ord(character) < 32 or ord(character) == 127 for character in filename)
        or filename in {".jsonl", "..jsonl"}
    ):
        raise EvaluationDatasetError("invalid filename; paths are not allowed")
    if Path(filename).suffix.lower() != ".jsonl":
        raise EvaluationDatasetError("the evaluation dataset must be a .jsonl file")
    return filename


def store_uploaded_dataset(filename: str, content: bytes) -> dict:
    """Publish the original file and its metadata together, after full validation."""
    validate_upload_filename(filename)
    if len(content) > config.MAX_EVALUATION_DATASET_SIZE:
        raise EvaluationDatasetError(
            f"file exceeds the {config.MAX_EVALUATION_DATASET_SIZE} byte upload limit"
        )
    records = parse_dataset_jsonl(content, maximum_records=config.MAX_EVALUATION_RECORDS)
    dataset_id = str(uuid4())
    metadata = {
        "dataset_id": dataset_id,
        "filename": filename,
        "record_count": len(records),
        "size_bytes": len(content),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "validation_status": "valid",
        "validation_error": None,
        "dataset_sha256": sha256(content).hexdigest(),
    }
    root = config.EVALUATION_DATA_DIRECTORY
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".upload-", dir=root) as temporary:
        staging = Path(temporary) / dataset_id
        staging.mkdir()
        (staging / "dataset.jsonl").write_bytes(content)
        (staging / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        staging.rename(root / dataset_id)
    return metadata


def dataset_metadata(dataset_id: str) -> dict:
    source = dataset_path(dataset_id)
    if not source.is_file():
        raise FileNotFoundError(f"evaluation dataset not found: {dataset_id}")
    stat = source.stat()
    metadata = {
        "dataset_id": dataset_id,
        "filename": source.name,
        "uploaded_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    sidecar = source.parent / "metadata.json"
    if source.parent == config.EVALUATION_DATA_DIRECTORY / dataset_id and sidecar.is_file():
        try:
            metadata.update(json.loads(sidecar.read_text(encoding="utf-8")))
        except (ValueError, TypeError):
            # A damaged sidecar must not hide the file or break the library.
            pass
    metadata.update(
        dataset_id=dataset_id, size_bytes=stat.st_size,
        dataset_sha256=file_sha256(source), record_count=0,
        validation_status="valid", validation_error=None,
    )
    try:
        if stat.st_size > config.MAX_EVALUATION_DATASET_SIZE:
            raise EvaluationDatasetError("file exceeds the configured upload limit")
        metadata["record_count"] = len(parse_dataset_jsonl(
            source.read_bytes(), maximum_records=config.MAX_EVALUATION_RECORDS,
        ))
    except EvaluationDatasetError as error:
        metadata.update(validation_status="invalid", validation_error=str(error))
    return metadata


def list_datasets() -> list[dict]:
    root = config.EVALUATION_DATA_DIRECTORY
    if not root.is_dir():
        return []
    ids = {
        path.stem for path in root.glob("*.jsonl")
        if DATASET_NAME_PATTERN.fullmatch(path.stem) and not path.is_symlink()
    }
    ids.update(
        path.name for path in root.iterdir()
        if DATASET_NAME_PATTERN.fullmatch(path.name) and path.is_dir()
        and not path.is_symlink() and (path / "dataset.jsonl").is_file()
    )
    return sorted((dataset_metadata(dataset_id) for dataset_id in ids),
                  key=lambda item: item["uploaded_at"], reverse=True)


def delete_dataset(dataset_id: str) -> None:
    source = dataset_path(dataset_id)
    source.unlink()
    if source.parent == config.EVALUATION_DATA_DIRECTORY / dataset_id:
        (source.parent / "metadata.json").unlink(missing_ok=True)
        source.parent.rmdir()


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
