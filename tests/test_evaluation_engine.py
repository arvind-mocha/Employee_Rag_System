from datetime import datetime, timezone
import json
import unittest
from unittest.mock import Mock, patch

from app.api.llm import GeneratedAnswer, GeneratedText, LLMProvider
from app.evaluation.dataset import text_sha256
from app.evaluation.engine import (
    evaluate_record,
    generate_ground_truth_records,
    run_evaluation,
)
from app.evaluation.models import (
    EvidenceSnapshot,
    GroundTruthRecord,
    GroundTruthStatus,
    RetrievalConfiguration,
)
from app.retrieval.models import RetrievedChunk


def retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        content="Employees receive 15 vacation days annually.",
        relevance_score=0.95,
        chunk_id="chunk-1",
        source_file="handbook.md",
        source_type="markdown",
        section="Vacation",
        chunk_index=0,
    )


def record() -> GroundTruthRecord:
    evidence = EvidenceSnapshot(
        position=1,
        chunk_id="chunk-1",
        content=retrieved_chunk().content,
        content_sha256=text_sha256(retrieved_chunk().content),
        source_file="handbook.md",
        source_type="markdown",
        source_sha256=None,
        section="Vacation",
        chunk_index=0,
    )
    return GroundTruthRecord(
        id="case-1",
        question="How many vacation days?",
        answerable=True,
        reference_answer="Employees receive 15 vacation days annually.",
        required_facts=["15 vacation days", "annual entitlement"],
        relevant_evidence=[evidence],
        evidence_fingerprint="fingerprint",
        generated_at=datetime.now(timezone.utc),
        reference_provider=LLMProvider.OPENAI,
        reference_model="reference-model",
        retrieval=RetrievalConfiguration(),
    )


class GroundTruthGenerationTests(unittest.TestCase):
    @patch("app.evaluation.engine.invoke_chat_model")
    def test_generates_atomic_ground_truth_with_evidence(self, invoke) -> None:
        router = Mock()
        router.search.return_value = [retrieved_chunk()]
        invoke.return_value = GeneratedText(
            json.dumps(
                {
                    "answerable": True,
                    "reference_answer": "Employees receive 15 days annually.",
                    "required_facts": ["15 days", "annual entitlement"],
                    "relevant_evidence_numbers": [1],
                }
            ),
            LLMProvider.OPENAI,
            "reference-model",
        )

        records = generate_ground_truth_records(
            ["How many vacation days?"],
            LLMProvider.OPENAI,
            RetrievalConfiguration(),
            router=router,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].required_facts[0], "15 days")
        self.assertEqual(len(records[0].retrieved_evidence), 1)
        self.assertEqual(records[0].relevant_evidence[0].chunk_id, "chunk-1")
        self.assertEqual(records[0].review_status, "machine_generated")


class EvaluationEngineTests(unittest.TestCase):
    @patch("app.evaluation.engine.semantic_similarity", return_value=0.91)
    @patch("app.evaluation.engine.invoke_chat_model")
    @patch("app.evaluation.engine.generate_grounded_answer")
    @patch(
        "app.evaluation.engine.ground_truth_status",
        return_value=GroundTruthStatus.CURRENT,
    )
    def test_returns_retrieval_and_answer_scores(
        self,
        _status,
        generate_answer,
        invoke,
        _semantic,
    ) -> None:
        router = Mock()
        router.search.return_value = [retrieved_chunk()]
        generate_answer.return_value = GeneratedAnswer(
            "Employees receive 15 vacation days annually [1].",
            LLMProvider.ANTHROPIC,
            "answer-model",
        )
        invoke.return_value = GeneratedText(
            json.dumps(
                {
                    "correctness": 1.0,
                    "groundedness": 1.0,
                    "completeness": 1.0,
                    "relevance": 1.0,
                    "citation_accuracy": 1.0,
                    "citation_coverage": 1.0,
                    "abstention_correctness": 1.0,
                    "unsupported_claims": [],
                    "missing_facts": [],
                    "rationale": "The answer matches the reference evidence.",
                }
            ),
            LLMProvider.OPENAI,
            "judge-model",
        )

        result = evaluate_record(
            record(),
            LLMProvider.ANTHROPIC,
            LLMProvider.OPENAI,
            RetrievalConfiguration(),
            router=router,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.answer_scores.correctness, 1.0)
        self.assertEqual(result.answer_scores.semantic_similarity, 0.91)
        self.assertEqual(result.answer_scores.citation_validity, 1.0)
        self.assertEqual(result.retrieval_scores.recall_at_k, 1.0)
        self.assertEqual(result.retrieval_scores.source_recall_at_k, 1.0)

        with (
            patch(
                "app.evaluation.engine.evaluate_record",
                return_value=result,
            ),
            patch(
                "app.evaluation.engine.dataset_sha256",
                return_value="a" * 64,
            ),
        ):
            evaluation = run_evaluation(
                "employee-policies",
                [record()],
                [LLMProvider.ANTHROPIC],
                LLMProvider.OPENAI,
                RetrievalConfiguration(),
                router=router,
            )

        summary = evaluation.summaries[0]
        self.assertEqual(summary.evaluated, 1)
        self.assertEqual(summary.pass_rate, 1.0)
        self.assertEqual(summary.average_scores.overall, 1.0)


if __name__ == "__main__":
    unittest.main()
