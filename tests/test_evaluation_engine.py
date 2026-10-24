from datetime import datetime, timezone
import json
import unittest
from unittest.mock import Mock, patch

from app.api.llm import GeneratedAnswer, GeneratedText, LLMProvider
from app.api.llm_usage import LLMUsage
from app.evaluation.dataset import text_sha256
from app.evaluation.engine import (
    QUESTION_GENERATION_BASE_OUTPUT_TOKENS,
    QUESTION_GENERATION_CATEGORY_BATCH_SIZE,
    QUESTION_GENERATION_OUTPUT_TOKENS_PER_QUESTION,
    evaluate_record,
    generate_evaluation_questions,
    generate_ground_truth_records,
    run_evaluation,
)
from app.evaluation.models import (
    EvidenceSnapshot,
    EvaluationCategory,
    GroundTruthRecord,
    GroundTruthStatus,
    QuestionGenerationDraft,
    RetrievalConfiguration,
)
from app.ingestion.chunker import DocumentChunk
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


class GroundTruthGenerationTests(unittest.TestCase):
    @patch("app.evaluation.engine.invoke_chat_model")
    def test_generates_balanced_questions_for_every_category(self, invoke) -> None:
        generated_questions = [
            {
                "question": f"Test question for {category.value}?",
                "category": category.value,
                "relevant_evidence_numbers": (
                    []
                    if category is EvaluationCategory.UNANSWERABLE
                    else [1]
                ),
            }
            for category in EvaluationCategory
        ]
        invoke.side_effect = [
            GeneratedText(
                json.dumps(
                    {
                        "questions": generated_questions[
                            start : start + QUESTION_GENERATION_CATEGORY_BATCH_SIZE
                        ]
                    }
                ),
                LLMProvider.OPENAI,
                "reference-model",
                llm_usage(0.001),
            )
            for start in range(
                0,
                len(generated_questions),
                QUESTION_GENERATION_CATEGORY_BATCH_SIZE,
            )
        ]
        chunks = [
            DocumentChunk(
                chunk_id="chunk-1",
                content="Employees receive 15 vacation days annually.",
                source_file="handbook.md",
                source_type="markdown",
                section="Vacation",
                chunk_index=0,
            )
        ]

        questions, usage, evidence_hints = generate_evaluation_questions(
            LLMProvider.OPENAI,
            1,
            chunks=chunks,
        )

        self.assertEqual(len(questions), len(EvaluationCategory))
        self.assertEqual(
            {question.category for question in questions},
            set(EvaluationCategory),
        )
        self.assertEqual(len(usage), 3)
        self.assertTrue(
            all(item and item.estimated_cost_usd == 0.001 for item in usage)
        )
        self.assertEqual(invoke.call_count, 3)
        self.assertEqual(
            evidence_hints[
                "test question for direct_fact?"
            ][0].chunk_id,
            "chunk-1",
        )
        self.assertEqual(
            evidence_hints["test question for unanswerable?"],
            [],
        )
        first_call = invoke.call_args_list[0]
        messages = first_call.args[0]
        self.assertIn("untrusted", messages[0].content)
        self.assertIn("handbook.md", messages[1].content)
        self.assertIn(
            "direct_fact, temporal, spanning",
            messages[1].content,
        )
        self.assertEqual(
            first_call.kwargs["max_output_tokens"],
            QUESTION_GENERATION_BASE_OUTPUT_TOKENS
            + QUESTION_GENERATION_CATEGORY_BATCH_SIZE
            * QUESTION_GENERATION_OUTPUT_TOKENS_PER_QUESTION,
        )
        self.assertIs(
            first_call.kwargs["response_schema"],
            QuestionGenerationDraft,
        )

    def test_rejects_generation_without_ingested_chunks(self) -> None:
        with self.assertRaisesRegex(ValueError, "No ingested content"):
            generate_evaluation_questions(
                LLMProvider.OPENAI,
                1,
                chunks=[],
            )

    @patch("app.evaluation.engine.invoke_chat_model")
    def test_rejects_a_blank_generated_question(self, invoke) -> None:
        generated_questions = [
            {
                "question": (
                    "   "
                    if category is EvaluationCategory.DIRECT_FACT
                    else f"Test question for {category.value}?"
                ),
                "category": category.value,
                "relevant_evidence_numbers": (
                    []
                    if category is EvaluationCategory.UNANSWERABLE
                    else [1]
                ),
            }
            for category in EvaluationCategory
        ]
        invoke.side_effect = [
            GeneratedText(
                json.dumps(
                    {
                        "questions": generated_questions[
                            start : start + QUESTION_GENERATION_CATEGORY_BATCH_SIZE
                        ]
                    }
                ),
                LLMProvider.OPENAI,
                "reference-model",
                llm_usage(0.001),
            )
            for start in range(
                0,
                len(generated_questions),
                QUESTION_GENERATION_CATEGORY_BATCH_SIZE,
            )
        ]

        with self.assertRaisesRegex(ValueError, "blank question"):
            generate_evaluation_questions(
                LLMProvider.OPENAI,
                1,
                chunks=[
                    DocumentChunk(
                        chunk_id="chunk-1",
                        content="Employees receive 15 vacation days annually.",
                        source_file="handbook.md",
                        source_type="markdown",
                        section="Vacation",
                        chunk_index=0,
                    )
                ],
            )

    @patch("app.evaluation.engine.invoke_chat_model")
    def test_generates_atomic_ground_truth_with_evidence(self, invoke) -> None:
        router = Mock()
        router.search.return_value = [retrieved_chunk()]
        invoke.return_value = GeneratedText(
            json.dumps(
                {
                    "answerable": True,
                    "reference_answer": "Employees receive 15 days annually.",
                    "category": "direct_fact",
                    "keywords": ["15 days", "vacation"],
                    "required_facts": ["15 days", "annual entitlement"],
                    "relevant_evidence_numbers": [1],
                }
            ),
            LLMProvider.OPENAI,
            "reference-model",
            llm_usage(0.001),
        )

        records = generate_ground_truth_records(
            ["How many vacation days?"],
            LLMProvider.OPENAI,
            RetrievalConfiguration(),
            router=router,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].required_facts[0], "15 days")
        self.assertEqual(records[0].category, "direct_fact")
        self.assertEqual(records[0].keywords, ["15 days", "vacation"])
        self.assertEqual(len(records[0].retrieved_evidence), 1)
        self.assertEqual(records[0].relevant_evidence[0].chunk_id, "chunk-1")
        self.assertEqual(records[0].review_status, "machine_generated")
        self.assertEqual(records[0].reference_usage.estimated_cost_usd, 0.001)

    @patch("app.evaluation.engine.invoke_chat_model")
    def test_uses_source_anchors_when_retrieval_misses(self, invoke) -> None:
        router = Mock()
        router.search.return_value = []
        invoke.return_value = GeneratedText(
            json.dumps(
                {
                    "answerable": True,
                    "reference_answer": "Employees receive 15 days annually.",
                    "category": "direct_fact",
                    "keywords": ["15 days", "vacation"],
                    "required_facts": ["15 days", "annual entitlement"],
                    "relevant_evidence_numbers": [1],
                }
            ),
            LLMProvider.OPENAI,
            "reference-model",
            llm_usage(0.001),
        )
        anchor = DocumentChunk(
            chunk_id="chunk-1",
            content="Employees receive 15 vacation days annually.",
            source_file="handbook.md",
            source_type="markdown",
            section="Vacation",
            chunk_index=0,
        )

        records = generate_ground_truth_records(
            ["How many vacation days?"],
            LLMProvider.OPENAI,
            RetrievalConfiguration(),
            router=router,
            evidence_hints={"how many vacation days?": [anchor]},
        )

        self.assertTrue(records[0].answerable)
        self.assertEqual(records[0].relevant_evidence[0].chunk_id, "chunk-1")
        messages = invoke.call_args.args[0]
        self.assertIn("Employees receive 15 vacation days", messages[1].content)

    @patch("app.evaluation.engine.invoke_chat_model")
    def test_assigns_unanswerable_category_when_evidence_is_insufficient(
        self,
        invoke,
    ) -> None:
        router = Mock()
        router.search.return_value = []
        invoke.return_value = GeneratedText(
            json.dumps(
                {
                    "answerable": False,
                    "reference_answer": "The documents do not contain this information.",
                    "category": "direct_fact",
                    "keywords": ["unsupported"],
                    "required_facts": [],
                    "relevant_evidence_numbers": [],
                }
            ),
            LLMProvider.OPENAI,
            "reference-model",
            llm_usage(0.001),
        )

        records = generate_ground_truth_records(
            ["What is the office Wi-Fi password?"],
            LLMProvider.OPENAI,
            RetrievalConfiguration(),
            router=router,
        )

        self.assertEqual(records[0].category, "unanswerable")
        self.assertEqual(records[0].keywords, [])


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
            llm_usage(0.002),
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
        self.assertEqual(result.answer_scores.correctness, 1.0)
        self.assertEqual(result.answer_scores.semantic_similarity, 0.91)
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
