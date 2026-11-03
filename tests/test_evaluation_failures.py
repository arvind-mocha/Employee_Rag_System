"""Regression coverage through real pipeline boundaries with mocked model I/O."""
import json
import logging
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from app.api.llm import ChatModelClient, LLMProvider, LLMProviderRequestError, _invoke_client
from app.core.safe_errors import log_safe_exception, provider_failure
from app.evaluation.dataset import EvaluationDatasetError, parse_dataset_jsonl
from app.evaluation.engine import evaluate_record, run_evaluation
from app.evaluation.models import GroundTruthRecord, JudgeDraft, RetrievalConfiguration
from app.evaluation.preflight import validate_runtime_preflight
from app.api.llm import GeneratedAnswer, GeneratedText
from app.retrieval.models import RetrievedChunk

OPENAI, GROK = LLMProvider.OPENAI, LLMProvider.GROK
GOOD_JUDGE = json.dumps(dict(completeness=1, relevance=1, rationale="Supported by the retrieved context."))


class UpstreamError(Exception):
    def __init__(self, status, code=None, message="private credential in upstream response"):
        super().__init__(message)
        self.status_code, self.code = status, code


@pytest.mark.parametrize("status,code,message,category,retryable", [
    (401, "invalid_api_key", "secret", "provider_authentication", False),
    (400, None, "Incorrect API key provided", "provider_authentication", False),
    (403, None, "secret", "provider_permission", False),
    (404, "model_not_found", "secret", "invalid_model", False),
    (400, None, "Unsupported response format", "invalid_model", False),
    (429, "insufficient_quota", "secret", "provider_quota", False),
    (429, None, "secret", "provider_rate_limit", True),
    (503, None, "secret", "provider_unavailable", True),
    (400, None, "request ID includes 401, 429, 404", "provider_request", False),
])
def test_provider_classification_uses_status_and_codes(status, code, message, category, retryable):
    result = provider_failure(UpstreamError(status, code, message))
    assert result[0] == category
    assert result[2] is retryable
    assert "secret" not in result[1]


@pytest.mark.parametrize("status,code,attempts", [(401, None, 1), (404, None, 1),
                                                   (429, "insufficient_quota", 1), (429, None, 3), (503, None, 3)])
def test_bounded_retries_only_for_transient_failures(status, code, attempts):
    model = Mock()
    model.invoke.side_effect = UpstreamError(status, code)
    with patch("app.api.llm.sleep") as sleep, pytest.raises(LLMProviderRequestError) as failure:
        _invoke_client(ChatModelClient(OPENAI, "configured-model", model), [HumanMessage(content="Hello")])
    assert model.invoke.call_count == attempts
    assert sleep.call_count == attempts - 1
    assert failure.value.model == "configured-model"
    assert "private credential" not in str(failure.value)


def test_transient_failure_can_recover():
    model = Mock()
    model.invoke.side_effect = [UpstreamError(503), AIMessage(content="Hello")]
    with patch("app.api.llm.sleep"):
        response, _ = _invoke_client(ChatModelClient(OPENAI, "configured-model", model), [])
    assert response.content == "Hello"
    assert model.invoke.call_count == 2


def test_native_judge_parsing_error_is_not_misclassified_as_authentication():
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(content="not valid JSON"), "parsed": None,
        "parsing_error": ValueError("private-secret"),
    }
    with pytest.raises(LLMProviderRequestError) as failure:
        _invoke_client(ChatModelClient(GROK, "judge-model", model), [], response_schema=JudgeDraft)
    assert failure.value.category == "judge_response_parsing"
    assert "private-secret" not in str(failure.value)


@pytest.mark.parametrize("payload", [
    {"question": "Q"}, {"question": None, "reference_answer": "A"},
    {"question": "Q", "reference_answer": "A", "schema_version": 999},
    {"question": "Q", "reference_answer": "A", "reference_evidence": [None]},
])
def test_invalid_jsonl_is_rejected_with_field_and_line(payload):
    with pytest.raises(EvaluationDatasetError, match="line 1"):
        parse_dataset_jsonl(json.dumps(payload).encode())


def test_optional_null_evidence_and_generation_metadata_are_not_required():
    sample = parse_dataset_jsonl(b'{"question":"Q", "reference_answer":"A", "reference_evidence":null, "required_facts":null, "metadata":null}')[0]
    assert sample.relevant_evidence == []
    assert sample.required_facts == []
    assert sample.reference_provider is None


def case(router=None):
    return evaluate_record(GroundTruthRecord(question="Q", reference_answer="A"), OPENAI, GROK,
                           RetrievalConfiguration(), router=router or Mock(search=Mock(return_value=[])))


@pytest.mark.parametrize("target,stage,category", [
    ("ground_truth_status", "reference_validation", "jsonl_validation"),
    ("get_query_router", "retrieval", "retrieval_failure"),
    ("generate_grounded_answer", "answer_generation", "internal_evaluation"),
    ("invoke_chat_model", "judge_invocation", "internal_evaluation"),
    ("lexical_similarity", "metrics", "metric_calculation"),
    ("_retrieval_scores", "metrics", "metric_calculation"),
])
def test_all_pipeline_failures_are_isolated(target, stage, category, caplog):
    with patch("app.evaluation.engine.generate_grounded_answer", return_value=GeneratedAnswer("A", OPENAI, "answer")), \
         patch("app.evaluation.engine.invoke_chat_model", return_value=GeneratedText(GOOD_JUDGE, GROK, "judge")), \
         patch("app.evaluation.engine.semantic_similarity", return_value=1), \
         patch(f"app.evaluation.engine.{target}", side_effect=RuntimeError("private-secret")):
        result = case() if target != "get_query_router" else evaluate_record(
            GroundTruthRecord(question="Q", reference_answer="A"), OPENAI, GROK, RetrievalConfiguration())
    assert result.error_category == category
    assert result.failed_stage == stage
    assert result.answer_scores is None
    assert result.passed is None
    assert "private-secret" not in result.error
    assert "private-secret" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_retrieval_exception_is_a_retrieval_failure():
    result = case(Mock(search=Mock(side_effect=RuntimeError("db-url-secret"))))
    assert result.failed_stage == "retrieval"
    assert result.error_category == "retrieval_failure"
    assert "db-url-secret" not in result.error


def test_bad_judge_json_preserves_generated_answer_without_fake_score():
    with patch("app.evaluation.engine.generate_grounded_answer", return_value=GeneratedAnswer("A", OPENAI, "answer")), \
         patch("app.evaluation.engine.invoke_chat_model", return_value=GeneratedText("not json", GROK, "judge")):
        result = case()
    assert result.answer == "A"
    assert result.failed_stage == "judge_parsing"
    assert result.error_category == "judge_response_parsing"
    assert result.answer_scores is None


def test_authentication_failure_records_provider_model_and_stage():
    with patch("app.evaluation.engine.generate_grounded_answer", side_effect=LLMProviderRequestError(
        OPENAI, "OpenAI rejected its configured API key.", category="provider_authentication", model="answer-model")):
        result = case()
    assert result.error_provider == OPENAI
    assert result.error_model == "answer-model"
    assert result.failed_stage == "answer_generation"
    assert result.error_category == "provider_authentication"


def test_success_and_low_scores_are_distinct_from_errors_and_missing_evidence():
    router = Mock(search=Mock(return_value=[RetrievedChunk(content="A", relevance_score=1, chunk_id="1", source_file="policy.md", source_type="markdown", section=None, chunk_index=0)]))
    with patch("app.evaluation.engine.generate_grounded_answer", return_value=GeneratedAnswer("A [1]", OPENAI, "answer")), \
         patch("app.evaluation.engine.invoke_chat_model", side_effect=[
             GeneratedText(GOOD_JUDGE, GROK, "judge"),
             GeneratedText('{"completeness":0,"relevance":0,"rationale":"Incomplete"}', GROK, "judge")]), \
         patch("app.evaluation.engine.semantic_similarity", return_value=1):
        good, low = case(router), case(router)
    assert good.passed is True and good.error is None
    assert good.retrieval_scores is None
    assert low.passed is False and low.error is None
    assert low.answer_scores.overall == 0


def test_partial_and_all_failed_runs_keep_case_results():
    sample = GroundTruthRecord(question="Q", reference_answer="A")
    failed = case(Mock(search=Mock(side_effect=RuntimeError("failure"))))
    with patch("app.evaluation.engine.generate_grounded_answer", return_value=GeneratedAnswer("A", OPENAI, "answer")), \
         patch("app.evaluation.engine.invoke_chat_model", return_value=GeneratedText(GOOD_JUDGE, GROK, "judge")), \
         patch("app.evaluation.engine.semantic_similarity", return_value=1):
        good = case()
    with patch("app.evaluation.engine.evaluate_record", side_effect=[failed, good]):
        partial = run_evaluation("dataset", [sample, sample], [OPENAI], GROK, RetrievalConfiguration(), content_sha256="hash")
    assert partial.status == "partial"
    assert partial.summaries[0].evaluated == 1 and partial.summaries[0].errors == 1
    with patch("app.evaluation.engine.evaluate_record", return_value=failed):
        failure = run_evaluation("dataset", [sample], [OPENAI], GROK, RetrievalConfiguration(), content_sha256="hash")
    assert failure.status == "failed" and failure.error
    assert failure.results[0].run_id == failure.run_id


def test_tracebacks_retain_frames_without_upstream_messages(caplog):
    try:
        try:
            raise ValueError("API key abc123; Authorization: Bearer secret")
        except ValueError as error:
            raise RuntimeError("postgresql://password") from error
    except RuntimeError as error:
        log_safe_exception(logging.getLogger(__name__), error, "case failed")
    assert "ValueError" in caplog.text and "RuntimeError" in caplog.text
    assert "test_tracebacks_retain_frames" in caplog.text
    assert "abc123" not in caplog.text and "Bearer" not in caplog.text
    assert "postgresql://password" not in caplog.text


@pytest.mark.parametrize("role", ["answer", "judge"])
def test_preflight_blocks_provider_failure_independently(role):
    with patch("app.evaluation.preflight.ingestion_runner") as ingestion, \
         patch("app.evaluation.preflight.invoke_chat_model") as invoke:
        ingestion.get_status.return_value.status = "completed"
        failure = LLMProviderRequestError(OPENAI if role == "answer" else GROK, "Provider rejected its configured API key.", category="provider_authentication")
        invoke.side_effect = [failure] if role == "answer" else [GeneratedText("Hello", OPENAI, "answer"), failure]
        with pytest.raises(HTTPException) as error:
            validate_runtime_preflight([OPENAI], GROK)
    assert error.value.status_code == 422
    assert role.capitalize() in error.value.detail
    assert "provider_authentication" in error.value.detail


def test_preflight_blocks_incomplete_ingestion_before_provider_calls():
    with patch("app.evaluation.preflight.ingestion_runner") as ingestion, \
         patch("app.evaluation.preflight.invoke_chat_model") as invoke:
        ingestion.get_status.return_value.status = "running"
        with pytest.raises(HTTPException, match="ingestion"):
            validate_runtime_preflight([OPENAI], GROK)
    invoke.assert_not_called()


def test_preflight_blocks_missing_worker():
    with patch("app.evaluation.preflight.ingestion_runner") as ingestion, \
         patch("app.evaluation.preflight.get_evaluation_queue"), \
         patch("app.evaluation.preflight.Worker.count", return_value=0):
        ingestion.get_status.return_value.status = "completed"
        with pytest.raises(HTTPException, match="worker"):
            validate_runtime_preflight([OPENAI], GROK, check_worker=True)


def test_preflight_validates_judge_schema():
    with patch("app.evaluation.preflight.ingestion_runner") as ingestion, \
         patch("app.evaluation.preflight.invoke_chat_model", side_effect=[GeneratedText("A", OPENAI, "answer"), GeneratedText("bad JSON", GROK, "judge")]):
        ingestion.get_status.return_value.status = "completed"
        with pytest.raises(HTTPException, match="judge_response_parsing"):
            validate_runtime_preflight([OPENAI], GROK)


def test_preflight_checks_both_roles_with_short_nonretrying_requests():
    with patch("app.evaluation.preflight.ingestion_runner") as ingestion, \
         patch("app.evaluation.preflight.invoke_chat_model", side_effect=[GeneratedText("A", OPENAI, "model"), GeneratedText(GOOD_JUDGE, OPENAI, "model")]) as invoke:
        ingestion.get_status.return_value.status = "completed"
        assert validate_runtime_preflight([OPENAI], OPENAI) == {OPENAI: "model"}
        assert invoke.call_count == 2
        for call in invoke.call_args_list:
            assert call.kwargs["timeout_seconds"] == 20
            assert call.kwargs["max_attempts"] == 1
        assert invoke.call_args_list[1].kwargs["response_schema"] is JudgeDraft


def test_preflight_classifies_transient_unavailability_as_503():
    with patch("app.evaluation.preflight.ingestion_runner") as ingestion, \
         patch("app.evaluation.preflight.invoke_chat_model", side_effect=LLMProviderRequestError(
             OPENAI, "Temporarily unavailable", category="provider_unavailable", retryable=True)):
        ingestion.get_status.return_value.status = "completed"
        with pytest.raises(HTTPException) as error:
            validate_runtime_preflight([OPENAI], GROK)
    assert error.value.status_code == 503


def test_preflight_rejects_invalid_retrieval_configuration():
    from dataclasses import replace
    from app.retrieval.config import RETRIEVAL_SETTINGS
    with patch("app.evaluation.preflight.ingestion_runner") as ingestion, \
         patch("app.evaluation.preflight.RETRIEVAL_SETTINGS", replace(RETRIEVAL_SETTINGS, number_of_chunks=0)):
        ingestion.get_status.return_value.status = "completed"
        with pytest.raises(HTTPException, match="Retrieval"):
            validate_runtime_preflight([OPENAI], GROK)


def test_retrieval_scores_survive_later_provider_failure():
    sample = parse_dataset_jsonl(b'{"question":"Q","reference_answer":"A","reference_evidence":[{"content":"A"}]}')[0]
    chunk = RetrievedChunk(content="A", relevance_score=1, chunk_id="1", source_file="policy.md",
                           source_type="markdown", section=None, chunk_index=0)
    with patch("app.evaluation.engine.generate_grounded_answer", side_effect=RuntimeError("private")):
        result = evaluate_record(sample, OPENAI, GROK, RetrievalConfiguration(),
                                 router=Mock(search=Mock(return_value=[chunk])))
    assert result.retrieval_scores.recall_at_k == 1
    assert result.answer_scores is None
    assert result.failed_stage == "answer_generation"


def test_case_serialization_failure_is_isolated():
    with patch("app.evaluation.engine.generate_grounded_answer", return_value=GeneratedAnswer("A", OPENAI, "answer")), \
         patch("app.evaluation.engine.invoke_chat_model", return_value=GeneratedText(GOOD_JUDGE, GROK, "judge")), \
         patch("app.evaluation.engine.semantic_similarity", return_value=1), \
         patch("app.evaluation.models.EvaluationCaseResult.model_dump", side_effect=RuntimeError("private-serialization-detail")):
        result = case()
    assert result.failed_stage == "serialization"
    assert result.error_category == "internal_evaluation"
    assert result.answer_scores is None
    assert "private-serialization-detail" not in result.error


def test_run_isolates_unexpected_case_boundary_exception():
    sample = GroundTruthRecord(question="Q", reference_answer="A")
    with patch("app.evaluation.engine.evaluate_record", side_effect=RuntimeError("private-error")) as evaluate:
        result = run_evaluation("dataset", [sample, sample], [OPENAI], GROK, RetrievalConfiguration(), content_sha256="hash")
    assert evaluate.call_count == 2
    assert len(result.results) == 2
    assert result.status == "failed"
    assert all(case.failed_stage == "internal_evaluation" and case.answer_scores is None for case in result.results)
