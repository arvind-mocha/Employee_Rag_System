from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import config
from app.api.llm import LLMProvider
from app.api.llm_usage import LLMUsage
from app.api.main import app
from app.evaluation.models import (
    EvaluationCategory,
    EvaluationRun,
    GeneratedEvaluationQuestion,
    GroundTruthRecord,
    RetrievalConfiguration,
)
from app.evaluation.dataset import evidence_fingerprint


def ground_truth_record() -> GroundTruthRecord:
    return GroundTruthRecord(
        id="case-1",
        question="How many vacation days?",
        answerable=False,
        reference_answer="The documents do not contain enough information.",
        required_facts=[],
        relevant_evidence=[],
        evidence_fingerprint=evidence_fingerprint([]),
        generated_at=datetime.now(timezone.utc),
        reference_provider=LLMProvider.OPENAI,
        reference_model="reference-model",
        reference_usage=LLMUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            token_count_source="provider",
            estimated_cost_usd=0.004,
            pricing_source="built_in",
        ),
        retrieval=RetrievalConfiguration(),
    )


class EvaluationApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_deletes_the_named_evaluation_dataset(self) -> None:
        with TemporaryDirectory() as directory:
            evaluation_directory = Path(directory)
            dataset = evaluation_directory / "obsolete-evaluation.jsonl"
            dataset.write_text("generated ground truth", encoding="utf-8")
            with patch(
                "app.api.config.EVALUATION_DATA_DIRECTORY",
                evaluation_directory,
            ):
                response = self.client.delete(
                    "/api/v1/evaluations/datasets/obsolete-evaluation"
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["dataset_name"],
            "obsolete-evaluation",
        )
        self.assertFalse(dataset.exists())

    def test_returns_not_found_when_deleting_a_missing_dataset(self) -> None:
        with TemporaryDirectory() as directory:
            with patch(
                "app.api.config.EVALUATION_DATA_DIRECTORY",
                Path(directory),
            ):
                response = self.client.delete(
                    "/api/v1/evaluations/datasets/missing-evaluation"
                )

        self.assertEqual(response.status_code, 404)

    def test_rejects_an_invalid_dataset_name_during_deletion(self) -> None:
        response = self.client.delete(
            "/api/v1/evaluations/datasets/invalid.name"
        )

        self.assertEqual(response.status_code, 422)

    @patch(
        "app.api.controllers.evaluation.dataset_sha256",
        return_value="a" * 64,
    )
    @patch("app.api.controllers.evaluation.write_dataset")
    @patch("app.api.controllers.evaluation.generate_ground_truth_records")
    def test_generates_jsonl_only_when_endpoint_is_called(
        self,
        generate_records,
        write_dataset,
        _dataset_hash,
    ) -> None:
        generate_records.return_value = [ground_truth_record()]
        write_dataset.return_value = (
            config.EVALUATION_DATA_DIRECTORY / "employee-policies.jsonl"
        )

        response = self.client.post(
            "/api/v1/evaluations/datasets/generate",
            json={
                "dataset_name": "employee-policies",
                "questions": ["How many vacation days?"],
                "reference_provider": "openai",
            },
        )

        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.json()["review_required"])
        self.assertEqual(response.json()["record_count"], 1)
        self.assertEqual(response.json()["usage"]["llm_calls"], 1)
        self.assertEqual(response.json()["usage"]["estimated_cost_usd"], 0.004)
        generate_records.assert_called_once()

    @patch(
        "app.api.controllers.evaluation.generate_evaluation_questions"
    )
    @patch("app.api.controllers.evaluation.dataset_sha256", return_value="a" * 64)
    @patch("app.api.controllers.evaluation.write_dataset")
    @patch("app.api.controllers.evaluation.generate_ground_truth_records")
    def test_generates_questions_from_ingested_data_on_demand(
        self,
        generate_records,
        write_dataset,
        _dataset_hash,
        generate_questions,
    ) -> None:
        generated = [
            GeneratedEvaluationQuestion(
                question=f"Question for {category.value}?",
                category=category,
                relevant_evidence_numbers=(
                    []
                    if category is EvaluationCategory.UNANSWERABLE
                    else [1]
                ),
            )
            for category in EvaluationCategory
        ]
        generate_questions.return_value = (
            generated,
            [
                LLMUsage(
                    input_tokens=200,
                    output_tokens=100,
                    total_tokens=300,
                    token_count_source="provider",
                    estimated_cost_usd=0.01,
                    pricing_source="built_in",
                )
            ],
            {},
        )
        generate_records.return_value = [ground_truth_record()]
        write_dataset.return_value = (
            config.EVALUATION_DATA_DIRECTORY / "auto-evaluation.jsonl"
        )

        response = self.client.post(
            "/api/v1/evaluations/datasets/generate",
            json={
                "dataset_name": "auto-evaluation",
                "questions_per_category": 1,
                "reference_provider": "openai",
            },
        )

        self.assertEqual(response.status_code, 201)
        generate_questions.assert_called_once_with(LLMProvider.OPENAI, 1)
        call = generate_records.call_args
        self.assertEqual(len(call.args[0]), len(EvaluationCategory))
        self.assertEqual(
            set(call.kwargs["category_hints"].values()),
            set(EvaluationCategory),
        )
        self.assertEqual(call.kwargs["evidence_hints"], {})
        self.assertEqual(response.json()["usage"]["llm_calls"], 2)

    @patch("app.api.controllers.evaluation.run_evaluation")
    @patch("app.api.controllers.evaluation.read_dataset")
    def test_runs_selected_providers_on_demand(
        self,
        read_dataset,
        run_evaluation,
    ) -> None:
        read_dataset.return_value = [ground_truth_record()]
        run_evaluation.return_value = EvaluationRun(
            run_id="run-1",
            dataset_name="employee-policies",
            dataset_sha256="b" * 64,
            generated_at=datetime.now(timezone.utc),
            judge_provider=LLMProvider.OPENAI,
            summaries=[],
            results=[],
        )

        response = self.client.post(
            "/api/v1/evaluations/run",
            json={
                "dataset_name": "employee-policies",
                "answer_providers": ["anthropic", "grok"],
                "judge_provider": "openai",
                "reranking": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_id"], "run-1")
        self.assertEqual(response.json()["usage"]["llm_calls"], 0)
        providers = run_evaluation.call_args.args[2]
        self.assertEqual(
            providers,
            [LLMProvider.ANTHROPIC, LLMProvider.GROK],
        )

    def test_rejects_duplicate_answer_providers(self) -> None:
        response = self.client.post(
            "/api/v1/evaluations/run",
            json={
                "dataset_name": "employee-policies",
                "answer_providers": ["openai", "openai"],
                "judge_provider": "anthropic",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_uploads_a_valid_jsonl_dataset_under_the_requested_name(self) -> None:
        with TemporaryDirectory() as directory:
            project_root = Path(directory)
            evaluation_directory = project_root / "evaluation_data"
            content = ground_truth_record().model_dump_json().encode("utf-8")
            with (
                patch("app.api.config.PROJECT_ROOT", project_root),
                patch(
                    "app.api.config.EVALUATION_DATA_DIRECTORY",
                    evaluation_directory,
                ),
            ):
                response = self.client.post(
                    "/api/v1/evaluations/datasets/upload",
                    data={"dataset_name": "team-baseline"},
                    files={
                        "dataset_file": (
                            "original.jsonl",
                            content,
                            "application/x-ndjson",
                        )
                    },
                )

            self.assertEqual(response.status_code, 201)
            self.assertEqual(response.json()["dataset_name"], "team-baseline")
            self.assertEqual(response.json()["record_count"], 1)
            self.assertTrue(response.json()["review_required"])
            self.assertTrue(
                (evaluation_directory / "team-baseline.jsonl").is_file()
            )

    def test_upload_uses_the_jsonl_filename_when_name_is_omitted(self) -> None:
        with TemporaryDirectory() as directory:
            project_root = Path(directory)
            evaluation_directory = project_root / "evaluation_data"
            content = ground_truth_record().model_dump_json().encode("utf-8")
            with (
                patch("app.api.config.PROJECT_ROOT", project_root),
                patch(
                    "app.api.config.EVALUATION_DATA_DIRECTORY",
                    evaluation_directory,
                ),
            ):
                response = self.client.post(
                    "/api/v1/evaluations/datasets/upload",
                    files={
                        "dataset_file": (
                            "benefits-tests.jsonl",
                            content,
                            "application/x-ndjson",
                        )
                    },
                )

            self.assertEqual(response.status_code, 201)
            self.assertEqual(response.json()["dataset_name"], "benefits-tests")

    def test_rejects_an_invalid_uploaded_dataset_without_storing_it(self) -> None:
        with TemporaryDirectory() as directory:
            project_root = Path(directory)
            evaluation_directory = project_root / "evaluation_data"
            with (
                patch("app.api.config.PROJECT_ROOT", project_root),
                patch(
                    "app.api.config.EVALUATION_DATA_DIRECTORY",
                    evaluation_directory,
                ),
            ):
                response = self.client.post(
                    "/api/v1/evaluations/datasets/upload",
                    files={
                        "dataset_file": (
                            "broken.jsonl",
                            b'{"question": "Missing ground truth fields"}\n',
                            "application/x-ndjson",
                        )
                    },
                )

            self.assertEqual(response.status_code, 422)
            self.assertFalse(evaluation_directory.exists())

    def test_rejects_a_non_jsonl_dataset_upload(self) -> None:
        response = self.client.post(
            "/api/v1/evaluations/datasets/upload",
            files={
                "dataset_file": (
                    "dataset.json",
                    b"{}",
                    "application/json",
                )
            },
        )

        self.assertEqual(response.status_code, 415)


if __name__ == "__main__":
    unittest.main()
