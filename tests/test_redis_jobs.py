from datetime import datetime, timezone
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from app.api.main import app
from app.api.services.ingestion_runner import IngestionRunner
from app.core.redis_client import RedisSettings


class RedisEvaluationJobApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("app.api.controllers.evaluation_jobs.get_redis_settings")
    @patch("app.api.controllers.evaluation_jobs.get_evaluation_queue")
    def test_enqueues_dataset_generation(
        self,
        get_queue,
        get_settings,
    ) -> None:
        queued_job = Mock(id="job-123")
        get_queue.return_value.enqueue_call.return_value = queued_job
        get_settings.return_value = RedisSettings()

        response = self.client.post(
            "/api/v1/evaluations/datasets/generate/async",
            json={
                "dataset_name": "employee-policies",
                "questions_per_category": 1,
                "reference_provider": "openai",
            },
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["job_id"], "job-123")
        self.assertEqual(response.json()["status"], "queued")
        self.assertEqual(
            response.json()["status_url"],
            "/api/v1/evaluations/jobs/job-123",
        )
        enqueue_call = get_queue.return_value.enqueue_call.call_args
        self.assertEqual(
            enqueue_call.kwargs["func"],
            "app.evaluation.jobs.generate_dataset_job",
        )
        self.assertEqual(
            enqueue_call.kwargs["meta"]["job_type"],
            "dataset_generation",
        )
        self.assertIsNone(enqueue_call.kwargs["args"][0]["questions"])
        self.assertEqual(
            enqueue_call.kwargs["args"][0]["questions_per_category"],
            1,
        )

    @patch("app.api.controllers.evaluation_jobs.get_evaluation_queue")
    def test_reports_redis_enqueue_failure(self, get_queue) -> None:
        get_queue.return_value.enqueue_call.side_effect = RedisConnectionError(
            "connection refused"
        )

        response = self.client.post(
            "/api/v1/evaluations/run/async",
            json={
                "dataset_name": "employee-policies",
                "answer_providers": ["openai"],
                "judge_provider": "anthropic",
            },
        )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Redis is unavailable", response.json()["detail"])

    @patch("app.api.controllers.evaluation_jobs.get_evaluation_queue")
    @patch("app.api.controllers.evaluation_jobs.get_redis_connection")
    @patch("app.api.controllers.evaluation_jobs.Job.fetch")
    def test_returns_completed_job_result(
        self,
        fetch,
        _connection,
        get_queue,
    ) -> None:
        now = datetime.now(timezone.utc)
        job = Mock(
            id="job-123",
            result={"ok": True, "result": {"dataset_name": "employee-policies"}},
            enqueued_at=now,
            started_at=now,
            ended_at=now,
        )
        job.get_status.return_value = "finished"
        job.get_meta.return_value = {"job_type": "dataset_generation"}
        fetch.return_value = job
        get_queue.return_value.serializer = Mock()

        response = self.client.get("/api/v1/evaluations/jobs/job-123")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "completed")
        self.assertEqual(
            response.json()["result"]["dataset_name"],
            "employee-policies",
        )


class RedisIngestionStateTests(unittest.TestCase):
    @patch("app.api.services.ingestion_runner.run_ingestion_pipeline")
    def test_uses_redis_lock_and_shared_state(self, pipeline) -> None:
        redis_client = Mock()
        distributed_lock = Mock()
        distributed_lock.acquire.return_value = True
        redis_client.lock.return_value = distributed_lock
        runner = IngestionRunner(redis_client)

        self.assertTrue(runner.reserve())
        runner.run()

        pipeline.assert_called_once_with()
        self.assertGreaterEqual(redis_client.hset.call_count, 3)
        self.assertGreaterEqual(redis_client.persist.call_count, 3)
        distributed_lock.release.assert_called_once_with()

    def test_reads_ingestion_state_from_redis(self) -> None:
        redis_client = Mock()
        redis_client.hgetall.return_value = {
            b"status": b"completed",
            b"error": b"",
            b"started_at": b"2026-10-23T12:00:00+00:00",
            b"finished_at": b"2026-10-23T12:01:00+00:00",
        }

        snapshot = IngestionRunner(redis_client).get_status()

        self.assertEqual(snapshot.status, "completed")
        self.assertIsNotNone(snapshot.started_at)
        self.assertIsNotNone(snapshot.finished_at)
        redis_client.persist.assert_called_once_with(
            "employee-rag:ingestion:state"
        )

    def test_new_data_invalidates_a_completed_run(self) -> None:
        redis_client = Mock()
        runner = IngestionRunner(redis_client)

        runner.mark_data_changed()

        redis_client.hset.assert_called_once()
        state_mapping = redis_client.hset.call_args.kwargs["mapping"]
        self.assertEqual(state_mapping["status"], "idle")
        redis_client.persist.assert_called_once_with(
            "employee-rag:ingestion:state"
        )


if __name__ == "__main__":
    unittest.main()
