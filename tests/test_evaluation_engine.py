from datetime import datetime, timezone
import json
import unittest
from unittest.mock import Mock, patch

from app.api.llm import GeneratedAnswer, GeneratedText, LLMProvider
from app.api.llm_usage import LLMUsage
from app.evaluation.dataset import text_sha256
from app.evaluation.engine import (
    _summary,
    evaluate_record,
    run_evaluation,
)
from app.evaluation.prompts import build_judge_messages
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


def llm_usage(cost: float) -> LLMUsage:
    return LLMUsage(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        token_count_source="provider",
        input_cost_per_million_usd=1.0,
        output_cost_per_million_usd=2.0,
        estimated_cost_usd=cost,
        pricing_source="built_in",
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


class EvaluationEngineTests(unittest.TestCase):
    def test_judge_receives_supplied_references_and_metadata(self):
        sample = record()
        sample.keywords = ["vacation"]
        messages = build_judge_messages(sample, "Candidate", [])
        self.assertIn(sample.reference_answer, messages[1].content)
        self.assertIn("required_facts", messages[1].content)
        self.assertIn("vacation", messages[1].content)
        self.assertIn("untrusted", messages[0].content)

    @patch("app.evaluation.engine.semantic_similarity", side_effect=RuntimeError("unavailable"))
    @patch("app.evaluation.engine.invoke_chat_model")
    @patch("app.evaluation.engine.generate_grounded_answer")
    def test_minimal_unanswerable_record_keeps_retrieval_metrics_unavailable(
        self, generate, judge, _similarity,
    ):
        router = Mock()
        router.search.return_value = []
        sample = GroundTruthRecord(question="Unknown?", reference_answer="Not known.", answerable=False)
        generate.return_value = GeneratedAnswer("Not known.", LLMProvider.OPENAI, "answer", llm_usage(0.001))
        judge.return_value = GeneratedText(json.dumps({
            "completeness": 1, "relevance": 1, "correctness": 1,
            "groundedness": 1, "abstention": 1, "rationale": "Correct abstention",
        }), LLMProvider.ANTHROPIC, "judge", llm_usage(0.002))
        result = evaluate_record(sample, LLMProvider.OPENAI, LLMProvider.ANTHROPIC,
                                 RetrievalConfiguration(), router=router)
        self.assertTrue(result.passed)
        self.assertIsNone(result.retrieval_scores)
        self.assertIsNone(result.answer_scores.semantic_similarity)
        self.assertEqual(result.answer_scores.abstention, 1)
        self.assertEqual(result.answer_scores.correctness, 1)
        self.assertTrue(any("Retrieval metrics unavailable" in value for value in result.warnings))
        summary = _summary(LLMProvider.OPENAI, [result])
        self.assertIsNone(summary.average_scores.recall_at_k)

    @patch("app.evaluation.engine.invoke_chat_model")
    @patch("app.evaluation.engine.generate_grounded_answer")
    def test_provider_errors_are_reported_as_errored_cases(self, generate, judge):
        router = Mock()
        router.search.return_value = []
        generate.side_effect = RuntimeError("private details")
        result = evaluate_record(record(), LLMProvider.OPENAI, LLMProvider.ANTHROPIC,
                                 RetrievalConfiguration(), router=router)
        self.assertIsNone(result.passed)
        self.assertIsNone(result.answer_scores)
        self.assertNotIn("private details", result.error)
        judge.assert_not_called()
        summary = _summary(LLMProvider.OPENAI, [result])
        self.assertEqual(summary.errors, 1)
        self.assertEqual(summary.failed, 0)
        self.assertEqual(summary.passed, 0)

    @patch("app.evaluation.engine.semantic_similarity", return_value=0.95)
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
        _similarity,
    ) -> None:
        router = Mock()
        router.search.return_value = [retrieved_chunk()]
        generate_answer.return_value = GeneratedAnswer(
            "Employees receive 15 vacation days annually [1].",
            LLMProvider.ANTHROPIC,
            "answer-model",
            llm_usage(0.002),
        )
        invoke.return_value = GeneratedText(
            json.dumps(
                {
                    "completeness": 1.0,
                    "relevance": 1.0,
                    "missing_facts": [],
                    "rationale": "The answer is relevant and complete.",
                }
            ),
            LLMProvider.OPENAI,
            "judge-model",
            llm_usage(0.003),
        )

        result = evaluate_record(
            record(),
            LLMProvider.ANTHROPIC,
            LLMProvider.OPENAI,
            RetrievalConfiguration(),
            router=router,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.category, "direct_fact")
        self.assertEqual(result.answer_scores.completeness, 1.0)
        self.assertEqual(result.answer_scores.relevance, 1.0)
        self.assertEqual(result.answer_scores.semantic_similarity, 0.95)
        self.assertEqual(result.answer_scores.citation_validity, 1.0)
        self.assertEqual(result.retrieval_scores.recall_at_k, 1.0)
        self.assertEqual(result.retrieval_scores.source_recall_at_k, 1.0)
        self.assertEqual(result.usage.llm_calls, 2)
        self.assertAlmostEqual(result.usage.estimated_cost_usd, 0.005)

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
        self.assertAlmostEqual(summary.usage.estimated_cost_usd, 0.005)
        self.assertAlmostEqual(evaluation.usage.estimated_cost_usd, 0.005)


if __name__ == "__main__":
    unittest.main()
