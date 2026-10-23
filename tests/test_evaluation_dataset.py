from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app.api.llm import LLMProvider
from app.evaluation.dataset import (
    dataset_sha256,
    evidence_fingerprint,
    ground_truth_status,
    parse_dataset_jsonl,
    read_dataset,
    snapshot_evidence,
    text_sha256,
    validate_dataset_name,
    write_dataset,
)
from app.evaluation.models import (
    EvidenceSnapshot,
    GroundTruthRecord,
    GroundTruthStatus,
    RetrievalConfiguration,
)
from app.retrieval.models import RetrievedChunk


def ground_truth_record(evidence=None) -> GroundTruthRecord:
    if evidence is None:
        content = "Employees receive 15 vacation days."
        evidence = [
            EvidenceSnapshot(
                position=1,
                chunk_id="chunk-1",
                content=content,
                content_sha256=text_sha256(content),
                source_file=None,
                source_type="markdown",
                source_sha256=None,
                section="Vacation",
                chunk_index=0,
            )
        ]
    return GroundTruthRecord(
        id="case-1",
        question="How many vacation days?",
        answerable=True,
        reference_answer="Employees receive 15 vacation days.",
        required_facts=["15 vacation days"],
        retrieved_evidence=evidence,
        relevant_evidence=evidence,
        evidence_fingerprint=evidence_fingerprint(evidence),
        generated_at=datetime.now(timezone.utc),
        reference_provider=LLMProvider.OPENAI,
        reference_model="reference-model",
        retrieval=RetrievalConfiguration(),
    )


class EvaluationDatasetTests(unittest.TestCase):
    def test_rejects_unsafe_dataset_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "dataset_name"):
            validate_dataset_name("../secrets")

    def test_round_trips_jsonl_and_prevents_accidental_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            destination = write_dataset(
                "employee-policies",
                [ground_truth_record()],
                directory=root,
            )

            records = read_dataset("employee-policies", directory=root)

            self.assertEqual(records[0].question, "How many vacation days?")
            digest = dataset_sha256("employee-policies", directory=root)
            self.assertEqual(len(digest), 64)
            self.assertEqual(destination.name, "employee-policies.jsonl")
            with self.assertRaises(FileExistsError):
                write_dataset(
                    "employee-policies",
                    records,
                    directory=root,
                )

    def test_rejects_tampered_evidence_content(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            destination = write_dataset(
                "employee-policies",
                [ground_truth_record()],
                directory=root,
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))
            payload["retrieved_evidence"][0]["content"] = "Tampered content"
            destination.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_dataset("employee-policies", directory=root)

    def test_uploaded_jsonl_must_be_utf8_and_respect_the_record_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            parse_dataset_jsonl(b"\xff")

        record = ground_truth_record().model_dump_json().encode("utf-8")
        with self.assertRaisesRegex(ValueError, "more than 1 records"):
            parse_dataset_jsonl(record + b"\n" + record, maximum_records=1)

    def test_marks_ground_truth_stale_when_indexed_source_changes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "handbook.md"
            source.write_text("15 vacation days", encoding="utf-8")
            chunk = RetrievedChunk(
                content="Employees receive 15 vacation days.",
                relevance_score=1.0,
                chunk_id="chunk-1",
                source_file="handbook.md",
                source_type="markdown",
                section="Vacation",
                chunk_index=0,
            )
            with patch(
                "app.evaluation.dataset.config.TEMP_DATA_DIRECTORY",
                root,
            ):
                evidence = snapshot_evidence(chunk, 1)
                record = ground_truth_record([evidence])
                self.assertEqual(
                    ground_truth_status(record),
                    GroundTruthStatus.CURRENT,
                )

                source.write_text("20 vacation days", encoding="utf-8")

                self.assertEqual(
                    ground_truth_status(record),
                    GroundTruthStatus.STALE,
                )


if __name__ == "__main__":
    unittest.main()
