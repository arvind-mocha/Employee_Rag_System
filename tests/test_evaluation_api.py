"""Dataset library, explicit selections, and asynchronous reporting contracts."""

from datetime import datetime, timedelta, timezone
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from app.api.main import app
from app.api.llm import GeneratedAnswer, GeneratedText, LLMProvider
from app.api.llm_usage import LLMUsage, summarize_llm_usage
from app.api.schemas.evaluation import RunEvaluationRequest
from app.core.redis_client import RedisSettings
from app.core.credential_store import ActiveCredential, CredentialStoreUnavailable, StoredCredential
from app.evaluation.dataset import dataset_path, parse_dataset_jsonl
from app.evaluation.engine import _summary
from app.evaluation.jobs import run_evaluation_job
from app.evaluation.models import (
    AnswerScores, EvaluationCaseResult, EvaluationRun, EvaluationTiming,
    GroundTruthStatus, ReviewStatus,
)
from app.retrieval.models import RetrievedChunk


BASE = "/api/v1/evaluations"
RECORD = {"question": "How many vacation days?", "reference_answer": "15 days annually."}


def jsonl(*records):
    return "\n".join(json.dumps(record) for record in records).encode()


class EvaluationApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "evaluation_data"
        self.config = patch("app.api.config.EVALUATION_DATA_DIRECTORY", self.root)
        self.config.start()
        self.addCleanup(self.config.stop)
        self.client = TestClient(app)
        store_patch = patch("app.evaluation.jobs.EvaluationJobStore", autospec=True)
        store_mock = store_patch.start()
        store_mock.return_value.should_cancel.return_value = False
        self.addCleanup(store_patch.stop)
        from contextlib import asynccontextmanager
        class TestGovernor:
            @asynccontextmanager
            async def slot(self, *args):
                yield {"tokens": -1}
            async def cooldown(self, *args):
                pass
        governor_patch = patch("app.evaluation.engine.RedisProviderGovernor", return_value=TestGovernor())
        governor_patch.start()
        self.addCleanup(governor_patch.stop)
        for target in ("app.api.controllers.evaluation.validate_runtime_preflight",
                       "app.api.controllers.evaluation_jobs.validate_runtime_preflight"):
            preflight = patch(target, return_value={})
            preflight.start()
            self.addCleanup(preflight.stop)

        self.credential_store = Mock()
        now = datetime.now(timezone.utc)
        self.credential_store.list_active.return_value = [
            StoredCredential(provider.value, "test-model", "1234",
                             now + timedelta(hours=1), now, now)
            for provider in LLMProvider
        ]
        self.credential_store.get_active.side_effect = lambda provider: ActiveCredential(
            provider, "test-model", "private-test-key", now + timedelta(hours=1),
        )
        store_patch = patch("app.api.services.provider_configuration.get_credential_store",
                            return_value=self.credential_store)
        store_patch.start()
        self.addCleanup(store_patch.stop)

    def upload(self, *files):
        return self.client.post(f"{BASE}/datasets/upload", files=[
            ("files", (name, content, "application/x-ndjson")) for name, content in files
        ])

    def dataset(self, filename="policies.jsonl", records=None):
        response = self.upload((filename, jsonl(*(records or [RECORD]))))
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["datasets"][0]

    def request_run(self, ids, **kwargs):
        return {"dataset_ids": ids, "answer_providers": ["openai"],
                "judge_provider": "anthropic", **kwargs}

    @staticmethod
    def fake_run(dataset_id, records, providers, judge, retrieval, **kwargs):
        results = []
        for provider in providers:
            for record in records:
                score = record.metadata.get("score", 1.0)
                usage = LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15,
                                 token_count_source="provider", estimated_cost_usd=0.01,
                                 pricing_source="built_in")
                results.append(EvaluationCaseResult(
                    record_id=record.id, question=record.question, category=record.category,
                    reference_answer=record.reference_answer, required_facts=record.required_facts,
                    review_status=ReviewStatus.USER_PROVIDED,
                    ground_truth_status=GroundTruthStatus.UNVERIFIABLE,
                    answer_provider=provider, judge_provider=judge, answer="An answer",
                    answer_usage=usage, usage=summarize_llm_usage([usage]),
                    retrieved_chunk_ids=[], retrieved_sources=[],
                    answer_scores=AnswerScores(completeness=score, relevance=score, overall=score),
                    passed=score >= 0.8,
                    timing=EvaluationTiming(retrieval_ms=1, answer_generation_ms=2, grading_ms=3, total_ms=6),
                ))
                kwargs["on_case_complete"]()
        return EvaluationRun(
            run_id=f"run-{dataset_id}", dataset_name=dataset_id,
            dataset_sha256=kwargs["content_sha256"], generated_at=datetime.now(timezone.utc),
            judge_provider=judge, summaries=[_summary(provider, results) for provider in providers],
            results=results, usage=summarize_llm_usage(result.answer_usage for result in results),
        )

    def test_end_to_end_upload_worker_persistence_and_result_polling(self):
        from langchain_core.messages import AIMessage
        from rq.serializers import JSONSerializer
        from app.api.llm import ChatModelClient
        from app.evaluation.models import JudgeDraft

        evidence = "Employees receive 15 vacation days annually."
        item = self.dataset(records=[{**RECORD, "reference_evidence": [{"content": evidence}]}])
        chunk = RetrievedChunk(content=evidence, relevance_score=1, chunk_id="policy-1",
                               source_file="policy.md", source_type="markdown", section=None, chunk_index=0)
        answer_model, judge_model = Mock(), Mock()
        answer_model.invoke.return_value = AIMessage(content=evidence + " [1]", usage_metadata={
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        })
        answer_model.ainvoke = AsyncMock(return_value=answer_model.invoke.return_value)
        judge_model.with_structured_output.return_value.invoke.return_value = {
            "raw": AIMessage(content="", usage_metadata={"input_tokens": 15, "output_tokens": 10, "total_tokens": 25}),
            "parsed": JudgeDraft(completeness=1, relevance=1, correctness=1, groundedness=1,
                                 rationale="Supported by the retrieved policy."), "parsing_error": None,
        }
        judge_model.with_structured_output.return_value.ainvoke = AsyncMock(return_value=judge_model.with_structured_output.return_value.invoke.return_value)
        clients = {LLMProvider.OPENAI: ChatModelClient(LLMProvider.OPENAI, "answer-model", answer_model),
                   LLMProvider.ANTHROPIC: ChatModelClient(LLMProvider.ANTHROPIC, "judge-model", judge_model)}
        job = Mock(id="integration-job", meta={}, enqueued_at=None, started_at=None, ended_at=None)
        job.get_status.return_value = "finished"
        with patch("app.api.controllers.evaluation_jobs.get_evaluation_queue") as queue:
            queue.return_value.enqueue_call.return_value = job
            queued = self.client.post(f"{BASE}/run/async", json=self.request_run([item["dataset_id"]]))
            self.assertEqual(queued.status_code, 202, queued.text)
            payload = queue.return_value.enqueue_call.call_args.kwargs["args"][0]
        with patch("app.evaluation.jobs.get_current_job", return_value=job), \
             patch("app.evaluation.engine.get_query_router", return_value=Mock(search=Mock(return_value=[chunk]))), \
             patch("app.api.llm.get_chat_model_client", side_effect=lambda provider, **_: clients[provider]), \
             patch("app.evaluation.engine.semantic_similarity", return_value=1):
            output = run_evaluation_job(payload)
        self.assertTrue(output["ok"], output)
        # Same JSON serializer RQ uses for Redis result persistence; read it back through the HTTP handler.
        job.result = JSONSerializer.loads(JSONSerializer.dumps(output))
        job.get_meta.return_value = job.meta
        with patch("app.api.controllers.evaluation_jobs.Job.fetch", return_value=job), \
             patch("app.api.controllers.evaluation_jobs.get_redis_connection"), \
             patch("app.api.controllers.evaluation_jobs.get_evaluation_queue"):
            polled = self.client.get(f"{BASE}/jobs/{job.id}")
        self.assertEqual(polled.status_code, 200, polled.text)
        body = polled.json()
        self.assertEqual(body["status"], "completed")
        case = body["result"]["aggregate"]["results"][0]
        self.assertTrue(case["passed"])
        self.assertEqual(case["answer_scores"]["overall"], 1)
        self.assertEqual(case["retrieval_scores"]["recall_at_k"], 1)
        self.assertEqual(case["answer_scores"]["citation_validity"], 1)
        self.assertEqual(case["usage"]["total_tokens"], 40)
        self.assertEqual(case["dataset_id"], item["dataset_id"])
        self.assertEqual(case["record_id"], "line-1")
        self.assertEqual(case["run_id"], body["result"]["run_id"])
        self.assertEqual(case["job_id"], job.id)
        self.assertEqual(case["judge_model"], "judge-model")
        self.assertIsNone(case["error"])
        self.assertEqual(job.meta["completed_cases"], 1)
        self.assertNotIn("private-test-key", polled.text)

    def test_all_error_job_reports_failure_and_retains_case_details(self):
        item = self.dataset()
        job = Mock(id="failed-integration-job", meta={}, enqueued_at=None, started_at=None, ended_at=None)
        with patch("app.evaluation.jobs.get_current_job", return_value=job), \
             patch("app.evaluation.engine.get_query_router", side_effect=RuntimeError("private-database-url")):
            output = run_evaluation_job(self.request_run([item["dataset_id"]]))
        self.assertFalse(output["ok"])
        self.assertEqual(output["result"]["aggregate"]["status"], "failed")
        self.assertEqual(len(output["result"]["aggregate"]["results"]), 1)
        job.result = output
        job.get_status.return_value = "finished"
        job.get_meta.return_value = job.meta
        with patch("app.api.controllers.evaluation_jobs.Job.fetch", return_value=job), \
             patch("app.api.controllers.evaluation_jobs.get_redis_connection"), \
             patch("app.api.controllers.evaluation_jobs.get_evaluation_queue"):
            response = self.client.get(f"{BASE}/jobs/{job.id}")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "failed")
        self.assertIn("No cases", body["error"])
        case = body["result"]["aggregate"]["results"][0]
        self.assertEqual(case["error_category"], "retrieval_failure")
        self.assertEqual(case["failed_stage"], "retrieval")
        self.assertEqual(case["job_id"], job.id)
        self.assertIsNone(case["answer_scores"])
        self.assertNotIn("private-database-url", response.text)

    def test_mixed_successful_and_errored_cases_preserve_successful_results(self):
        item = self.dataset(records=[{**RECORD, "id": "good"}, {**RECORD, "id": "bad"}])
        chunk = RetrievedChunk(content="15 vacation days.", relevance_score=1, chunk_id="1",
                               source_file="policy.md", source_type="markdown", section=None, chunk_index=0)
        router = Mock()
        router.search.side_effect = [[chunk], RuntimeError("private-db-error")]
        job = Mock(id="partial-job", meta={})
        with patch("app.evaluation.jobs.get_current_job", return_value=job), \
             patch("app.evaluation.engine.get_query_router", return_value=router), \
             patch("app.evaluation.engine.generate_grounded_answer", return_value=GeneratedAnswer("15 days [1]", LLMProvider.OPENAI, "answer")), \
             patch("app.evaluation.engine.invoke_chat_model", return_value=GeneratedText(
                 '{"completeness":1,"relevance":1,"rationale":"Correct"}', LLMProvider.ANTHROPIC, "judge")), \
             patch("app.evaluation.engine.semantic_similarity", return_value=1):
            output = run_evaluation_job(self.request_run([item["dataset_id"]]))
        self.assertTrue(output["ok"])
        aggregate = output["result"]["aggregate"]
        self.assertEqual(aggregate["status"], "partial")
        self.assertTrue(aggregate["results"][0]["passed"])
        self.assertIsNone(aggregate["results"][1]["passed"])
        self.assertEqual(aggregate["summaries"][0]["evaluated"], 1)
        self.assertEqual(aggregate["summaries"][0]["errors"], 1)
        self.assertEqual(aggregate["summaries"][0]["average_scores"]["overall"], 1)
        self.assertEqual(job.meta["completed_cases"], 2)

    def test_single_upload_preserves_original_bytes_and_metadata(self):
        content = jsonl(RECORD)
        item = self.dataset()
        self.assertNotEqual(item["dataset_id"], "policies")
        self.assertEqual(item["filename"], "policies.jsonl")
        self.assertEqual(item["record_count"], 1)
        self.assertEqual(item["size_bytes"], len(content))
        self.assertEqual(item["validation_status"], "valid")
        self.assertTrue(item["uploaded_at"])
        self.assertEqual(dataset_path(item["dataset_id"]).read_bytes(), content)
        preview = self.client.get(f"{BASE}/datasets/{item['dataset_id']}").json()
        self.assertEqual(preview["records"][0]["reference_answer"], RECORD["reference_answer"])
        self.assertEqual(preview["records"][0]["id"], "line-1")

    def test_multiple_and_later_uploads_with_duplicate_names_persist(self):
        response = self.upload(("same.jsonl", jsonl(RECORD)), ("same.jsonl", jsonl(RECORD)))
        self.assertEqual(response.status_code, 201)
        first = response.json()["datasets"]
        self.assertNotEqual(first[0]["dataset_id"], first[1]["dataset_id"])
        later = self.dataset("same.jsonl")
        # A fresh client and disk-only lookup exercise restart persistence.
        listing = TestClient(app).get(f"{BASE}/datasets").json()["datasets"]
        self.assertEqual({item["dataset_id"] for item in listing},
                         {first[0]["dataset_id"], first[1]["dataset_id"], later["dataset_id"]})
        self.assertEqual(len(list(self.root.glob("*/dataset.jsonl"))), 3)

    def test_individual_deletion_leaves_other_files_intact(self):
        first, second = self.dataset(), self.dataset()
        response = self.client.delete(f"{BASE}/datasets/{first['dataset_id']}")
        self.assertEqual(response.status_code, 200)
        remaining = self.client.get(f"{BASE}/datasets").json()["datasets"]
        self.assertEqual([item["dataset_id"] for item in remaining], [second["dataset_id"]])
        self.assertEqual(self.client.delete(f"{BASE}/datasets/{first['dataset_id']}").status_code, 404)

    def test_invalid_files_report_line_numbers_without_partial_storage(self):
        for content in [jsonl(RECORD) + b'\n{broken}', jsonl(RECORD, {"question": "Missing reference"}),
                        jsonl(RECORD) + b'\n[]']:
            with self.subTest(content=content):
                response = self.upload(("invalid.jsonl", content))
                self.assertEqual(response.status_code, 422)
                self.assertIn("line 2", response.json()["detail"])
                self.assertFalse(self.root.exists())

    def test_mixed_batch_keeps_valid_file_and_reports_invalid_file(self):
        response = self.upload(("valid.jsonl", jsonl(RECORD)), ("bad.jsonl", b'{}'))
        self.assertEqual(response.status_code, 201)
        self.assertEqual(len(response.json()["datasets"]), 1)
        self.assertIn("line 1", response.json()["errors"][0]["error"])
        self.assertEqual(len(self.client.get(f"{BASE}/datasets").json()["datasets"]), 1)

    def test_rejects_bad_encodings_empty_files_and_unsafe_filenames(self):
        for filename, content in [("x.jsonl", b'\xff'), ("x.jsonl", b'\n'),
                                  ("../secret.jsonl", jsonl(RECORD)),
                                  ("..\\secret.jsonl", jsonl(RECORD)),
                                  ("x.json", jsonl(RECORD))]:
            with self.subTest(filename=filename, content=content):
                self.assertEqual(self.upload((filename, content)).status_code, 422)
        self.assertFalse(self.root.exists())

    def test_enforces_configured_size_record_and_batch_limits(self):
        with patch("app.api.config.MAX_EVALUATION_DATASET_SIZE", 5):
            self.assertEqual(self.upload(("large.jsonl", jsonl(RECORD))).status_code, 422)
        with patch("app.api.config.MAX_EVALUATION_RECORDS", 1):
            response = self.upload(("many.jsonl", jsonl(RECORD, RECORD)))
            self.assertEqual(response.status_code, 422)
            self.assertIn("line 2", response.json()["detail"])
        with patch("app.api.config.MAX_EVALUATION_UPLOAD_FILES", 1):
            self.assertEqual(self.upload(("a.jsonl", jsonl(RECORD)),
                                         ("b.jsonl", jsonl(RECORD))).status_code, 422)
        self.assertFalse(self.root.exists())

    def test_legacy_single_file_field_and_existing_disk_files_are_supported(self):
        response = self.client.post(f"{BASE}/datasets/upload", files={
            "dataset_file": ("legacy-field.jsonl", jsonl(RECORD), "application/x-ndjson"),
        })
        self.assertEqual(response.status_code, 201)
        (self.root / "old-dataset.jsonl").write_bytes(jsonl(RECORD))
        listing = self.client.get(f"{BASE}/datasets").json()["datasets"]
        self.assertIn("old-dataset", [item["dataset_id"] for item in listing])

    def test_deleting_legacy_file_named_dataset_preserves_library(self):
        item = self.dataset()
        (self.root / "dataset.jsonl").write_bytes(jsonl(RECORD))
        response = self.client.delete(f"{BASE}/datasets/dataset")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.root.is_dir())
        self.assertTrue(dataset_path(item["dataset_id"]).is_file())

    def test_preview_paginates_and_marks_tampered_files_invalid(self):
        item = self.dataset(records=[{**RECORD, "id": str(i)} for i in range(25)])
        preview = self.client.get(f"{BASE}/datasets/{item['dataset_id']}?offset=20&limit=20").json()
        self.assertEqual(len(preview["records"]), 5)
        dataset_path(item["dataset_id"]).write_bytes(b'bad json')
        listing = self.client.get(f"{BASE}/datasets").json()["datasets"]
        self.assertEqual(listing[0]["validation_status"], "invalid")
        with patch("app.api.controllers.evaluation.run_evaluation") as run:
            response = self.client.post(f"{BASE}/run", json=self.request_run([item["dataset_id"]]))
        self.assertEqual(response.status_code, 422)
        run.assert_not_called()

    def test_generation_routes_are_removed(self):
        for suffix in ["/datasets/generate", "/datasets/generate/async"]:
            self.assertIn(self.client.post(BASE + suffix, json={}).status_code, [404, 405])
        self.assertNotIn("/api/v1/evaluations/datasets/generate", app.openapi()["paths"])

    def test_requires_explicit_unique_selection_and_providers(self):
        for overrides in [{"dataset_ids": []}, {"dataset_ids": ["a", "a"]},
                          {"dataset_ids": ["../secret"]},
                          {"answer_providers": ["openai", "openai"]},
                          {"semantic_search": False, "keyword_search": False}]:
            payload = self.request_run(["a"], **{k: v for k, v in overrides.items() if k != "dataset_ids"})
            payload.update(overrides)
            self.assertEqual(self.client.post(f"{BASE}/run", json=payload).status_code, 422)

    @patch("app.api.controllers.evaluation.run_evaluation", side_effect=fake_run)
    def test_single_dataset_evaluates_only_selected_file(self, run):
        first, _unselected = self.dataset(), self.dataset("unselected.jsonl")
        response = self.client.post(f"{BASE}/run", json=self.request_run([first["dataset_id"]], reranking=True))
        self.assertEqual(response.status_code, 200, response.text)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], first["dataset_id"])
        self.assertTrue(run.call_args.args[4].reranking)
        self.assertEqual(response.json()["datasets"][0]["dataset_id"], first["dataset_id"])
        self.assertEqual(response.json()["aggregate"]["results"][0]["dataset_id"], first["dataset_id"])

    @patch("app.api.controllers.evaluation.run_evaluation", side_effect=fake_run)
    def test_multiple_datasets_have_independent_and_weighted_aggregate_results(self, run):
        first = self.dataset("same.jsonl")
        second = self.dataset("same.jsonl", records=[
            {**RECORD, "metadata": {"score": 0.0}}, {**RECORD, "metadata": {"score": 0.0}},
        ])
        unselected = self.dataset("unselected.jsonl")
        response = self.client.post(f"{BASE}/run", json=self.request_run([first["dataset_id"], second["dataset_id"]]))
        self.assertEqual(response.status_code, 200, response.text)
        batch = response.json()
        self.assertEqual(run.call_count, 2)
        self.assertNotIn(unselected["dataset_id"], [call.args[0] for call in run.call_args_list])
        self.assertEqual(len(batch["datasets"]), 2)
        self.assertEqual(batch["datasets"][0]["summaries"][0]["passed"], 1)
        self.assertEqual(batch["datasets"][1]["summaries"][0]["failed"], 2)
        aggregate = batch["aggregate"]
        self.assertAlmostEqual(aggregate["summaries"][0]["average_scores"]["overall"], 1 / 3)
        self.assertEqual(aggregate["usage"]["total_tokens"], 45)
        self.assertAlmostEqual(aggregate["usage"]["estimated_cost_usd"], 0.03)
        self.assertEqual({item["dataset_id"] for item in aggregate["results"]},
                         {first["dataset_id"], second["dataset_id"]})
        # A subsequent run uses its new selection, with a fresh batch ID.
        later = self.client.post(f"{BASE}/run", json=self.request_run([second["dataset_id"]])).json()
        self.assertNotEqual(later["run_id"], batch["run_id"])
        self.assertEqual(len(later["aggregate"]["results"]), 2)

    @patch("app.api.controllers.evaluation.run_evaluation")
    def test_checks_all_selected_files_before_running_any(self, run):
        item = self.dataset()
        response = self.client.post(f"{BASE}/run", json=self.request_run([item["dataset_id"], "missing"]))
        self.assertEqual(response.status_code, 404)
        run.assert_not_called()

    @patch("app.api.controllers.evaluation.run_evaluation", side_effect=fake_run)
    def test_record_id_filters_are_scoped_to_each_selected_file(self, run):
        item = self.dataset(records=[{**RECORD, "id": "one"}, {**RECORD, "id": "two"}])
        response = self.client.post(f"{BASE}/run", json=self.request_run(
            [item["dataset_id"]], record_ids=["two"],
        ))
        self.assertEqual(response.status_code, 200)
        self.assertEqual([record.id for record in run.call_args.args[1]], ["two"])
        run.reset_mock()
        response = self.client.post(f"{BASE}/run", json=self.request_run(
            [item["dataset_id"]], record_ids=["missing"],
        ))
        self.assertEqual(response.status_code, 404)
        run.assert_not_called()

    @patch("app.api.controllers.evaluation_jobs.get_redis_settings", return_value=RedisSettings())
    @patch("app.api.controllers.evaluation_jobs.get_evaluation_queue")
    @patch("app.api.controllers.evaluation.run_evaluation", side_effect=fake_run)
    def test_async_payload_selection_progress_and_changed_dataset_protection(self, run, queue, _settings):
        first = self.dataset()
        self.dataset("unselected.jsonl")
        queue.return_value.enqueue_call.return_value = Mock(id="job-1")
        response = self.client.post(f"{BASE}/run/async", json=self.request_run([first["dataset_id"]]))
        self.assertEqual(response.status_code, 202, response.text)
        call = queue.return_value.enqueue_call.call_args
        self.assertEqual(call.kwargs["func"], "app.evaluation.jobs.run_evaluation_job")
        payload = call.kwargs["args"][0]
        self.assertEqual(payload["dataset_ids"], [first["dataset_id"]])
        self.assertEqual(payload["expected_hashes"], {first["dataset_id"]: first["dataset_sha256"]})
        job = Mock(id="job-1", meta={})
        with patch("app.evaluation.jobs.get_current_job", return_value=job):
            result = run_evaluation_job(payload)
        self.assertTrue(result["ok"])
        self.assertEqual(job.meta["completed_cases"], 1)
        self.assertEqual(job.meta["total_cases"], 1)
        self.assertEqual(run.call_args.args[0], first["dataset_id"])
        run.reset_mock()
        dataset_path(first["dataset_id"]).write_bytes(jsonl({**RECORD, "reference_answer": "Changed"}))
        with patch("app.evaluation.jobs.get_current_job", return_value=None):
            changed = run_evaluation_job(payload)
        self.assertFalse(changed["ok"])
        self.assertIn("changed after selection", changed["error"])
        run.assert_not_called()
        dataset_path(first["dataset_id"]).unlink()
        with patch("app.evaluation.jobs.get_current_job", return_value=None):
            deleted = run_evaluation_job(payload)
        self.assertFalse(deleted["ok"])
        self.assertIn("not found", deleted["error"])
        run.assert_not_called()

    def test_rejects_symlink_escape(self):
        self.root.mkdir()
        outside = Path(self.temporary.name) / "outside.jsonl"
        outside.write_bytes(jsonl(RECORD))
        (self.root / "escape.jsonl").symlink_to(outside)
        self.assertEqual(self.client.get(f"{BASE}/datasets/escape").status_code, 422)
        self.assertEqual(self.client.delete(f"{BASE}/datasets/escape").status_code, 422)
        self.assertTrue(outside.exists())

    @patch("app.api.controllers.evaluation.run_evaluation")
    @patch("app.api.controllers.evaluation_jobs.get_evaluation_queue")
    def test_backend_blocks_unconfigured_answers_and_judges(self, queue, run):
        item = self.dataset()
        configured = self.credential_store.list_active.return_value
        for missing in ["openai", "anthropic"]:
            self.credential_store.list_active.return_value = [
                credential for credential in configured if credential.provider != missing
            ]
            for endpoint in ["run", "run/async"]:
                with self.subTest(provider=missing, endpoint=endpoint):
                    response = self.client.post(f"{BASE}/{endpoint}", json=self.request_run([item["dataset_id"]]))
                    self.assertEqual(response.status_code, 422)
                    self.assertIn("OpenAI" if missing == "openai" else "Anthropic", response.json()["detail"])
                    self.assertIn("Settings", response.json()["detail"])
        queue.assert_not_called()
        run.assert_not_called()

    @patch("app.api.controllers.evaluation.run_evaluation")
    def test_missing_model_and_credential_status_failure_block_runs(self, run):
        item = self.dataset()
        self.credential_store.list_active.return_value = [
            replace(credential, model_name=" ") if credential.provider == "openai" else credential
            for credential in self.credential_store.list_active.return_value
        ]
        response = self.client.post(f"{BASE}/run", json=self.request_run([item["dataset_id"]]))
        self.assertEqual(response.status_code, 422)
        self.assertIn("Model not configured", response.json()["detail"])
        self.credential_store.list_active.side_effect = CredentialStoreUnavailable("secret-error-value")
        response = self.client.post(f"{BASE}/run", json=self.request_run([item["dataset_id"]]))
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret-error-value", response.text)
        run.assert_not_called()

    @patch("app.api.controllers.evaluation_jobs.get_redis_settings", return_value=RedisSettings())
    @patch("app.api.controllers.evaluation_jobs.get_evaluation_queue")
    @patch("app.api.controllers.evaluation.run_evaluation", side_effect=fake_run)
    def test_worker_rechecks_expiry_deletion_and_replacement_after_queueing(self, run, queue, _settings):
        item = self.dataset()
        queue.return_value.enqueue_call.return_value = Mock(id="credential-job")
        response = self.client.post(f"{BASE}/run/async", json=self.request_run([item["dataset_id"]]))
        self.assertEqual(response.status_code, 202)
        payload = queue.return_value.enqueue_call.call_args.kwargs["args"][0]
        configured = self.credential_store.list_active.return_value
        self.assertNotIn("private-test-key", json.dumps(payload))
        for provider in ["openai", "anthropic"]:
            expired = [replace(credential, expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
                       if credential.provider == provider else credential for credential in configured]
            deleted = [credential for credential in configured if credential.provider != provider]
            for state, expected in [(expired, "expired"), (deleted, "not configured")]:
                self.credential_store.list_active.return_value = state
                with patch("app.evaluation.jobs.get_current_job", return_value=None):
                    result = run_evaluation_job(payload)
                self.assertFalse(result["ok"])
                self.assertIn(expected, result["error"])
                run.assert_not_called()
        self.credential_store.list_active.return_value = configured
        with patch("app.evaluation.jobs.get_current_job", return_value=None):
            result = run_evaluation_job(payload)
        self.assertTrue(result["ok"])
        self.assertNotIn("private-test-key", json.dumps(result))
        run.assert_called_once()

    def test_example_download_matches_tracked_file_and_upload_schema(self):
        source = Path(__file__).parents[1] / "examples" / "evaluation_dataset.example.jsonl"
        response = self.client.get(f"{BASE}/reference/example")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, source.read_bytes())
        self.assertIn('filename="evaluation_dataset.example.jsonl"', response.headers["content-disposition"])
        self.assertIn("application/x-ndjson", response.headers["content-type"])
        lines = response.content.decode("utf-8").splitlines()
        self.assertEqual(len(lines), 5)
        for line in lines:
            self.assertIsInstance(json.loads(line), dict)
        records = parse_dataset_jsonl(response.content)
        self.assertEqual({record.category.value for record in records},
                         {"direct_fact", "procedural", "spanning", "numerical", "unanswerable"})
        self.assertTrue(all(record.reference_provider is None for record in records))
        uploaded = self.upload((source.name, response.content))
        self.assertEqual(uploaded.status_code, 201)
        self.assertEqual(uploaded.json()["datasets"][0]["record_count"], 5)

    def test_schema_reference_uses_current_validation_limits(self):
        with patch("app.api.config.MAX_EVALUATION_RECORDS", 321):
            response = self.client.get(f"{BASE}/reference/schema")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["maximum_records"], 321)
        self.assertEqual(set(response.json()["record_schema"]["required"]), {"question", "reference_answer"})
        self.assertIn("unanswerable", response.json()["categories"])

    @patch("app.evaluation.engine.get_query_router")
    @patch("app.evaluation.engine.semantic_similarity", return_value=0.9)
    @patch("app.evaluation.engine.invoke_chat_model")
    @patch("app.evaluation.engine.generate_grounded_answer")
    def test_uploaded_example_runs_through_existing_evaluation_engine(self, generate, judge, _similarity, get_router):
        example = self.client.get(f"{BASE}/reference/example").content
        records = parse_dataset_jsonl(example)
        by_question = {record.question: record for record in records}
        router = get_router.return_value
        router.search.side_effect = lambda question, _options: [
            RetrievedChunk(content=evidence.content, relevance_score=1.0,
                           chunk_id=None, source_file=None, source_type=None,
                           section=evidence.section, chunk_index=None)
            for evidence in by_question[question].relevant_evidence
        ]
        generate.side_effect = lambda question, _chunks, provider: GeneratedAnswer(
            by_question[question].reference_answer, provider, "answer-model", None,
        )
        judge.return_value = GeneratedText(json.dumps({
            "completeness": 1, "relevance": 1, "correctness": 1,
            "groundedness": 1, "rationale": "Reference agreement",
        }), LLMProvider.ANTHROPIC, "judge-model", None)
        item = self.upload(("example.jsonl", example)).json()["datasets"][0]
        response = self.client.post(f"{BASE}/run", json=self.request_run([item["dataset_id"]]))
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["aggregate"]["summaries"][0]["passed"], 5)
        self.assertEqual(generate.call_count, 5)
        self.assertEqual(judge.call_count, 5)
        results = {result["record_id"]: result for result in body["datasets"][0]["results"]}
        self.assertEqual(results["fictional-leave"]["retrieval_scores"]["recall_at_k"], 1)
        self.assertIsNone(results["fictional-learning-budget"]["retrieval_scores"])
        self.assertIsNone(results["fictional-unknown-access-code"]["retrieval_scores"])


if __name__ == "__main__":
    unittest.main()
