from datetime import datetime, timezone
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import config
from app.api.llm import LLMProvider
from app.api.llm_usage import LLMUsage
from app.api.main import app
from app.evaluation.models import (
    EvaluationRun,
    GroundTruthRecord,
    RetrievalConfiguration,
)


def ground_truth_record() -> GroundTruthRecord:
    return GroundTruthRecord(
        id="case-1",
        question="How many vacation days?",
        answerable=False,
        reference_answer="The documents do not contain enough information.",
        required_facts=[],
        relevant_evidence=[],
        evidence_fingerprint="fingerprint",
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


if __name__ == "__main__":
    unittest.main()
