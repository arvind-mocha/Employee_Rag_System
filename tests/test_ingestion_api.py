import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.main import app
from app.api.services.ingestion_runner import IngestionRunner, IngestionSnapshot


class IngestionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("app.api.controllers.ingestion.ingestion_runner")
    def test_starts_ingestion_in_the_background(self, ingestion_runner) -> None:
        ingestion_runner.reserve.return_value = True

        response = self.client.post("/api/v1/ingestion/start")

        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            response.json(),
            {
                "message": "Ingestion process started.",
                "status": "accepted",
                "status_url": "/api/v1/ingestion/status",
            },
        )
        ingestion_runner.reserve.assert_called_once_with()
        ingestion_runner.run.assert_called_once_with()

    @patch("app.api.controllers.ingestion.ingestion_runner")
    def test_rejects_a_second_ingestion_run(self, ingestion_runner) -> None:
        ingestion_runner.reserve.return_value = False

        response = self.client.post("/api/v1/ingestion/start")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json(),
            {"detail": "An ingestion process is already running."},
        )
        ingestion_runner.run.assert_not_called()

    @patch("app.api.controllers.ingestion.ingestion_runner")
    def test_returns_failed_ingestion_status(self, ingestion_runner) -> None:
        started_at = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)
        finished_at = datetime(2026, 10, 22, 12, 1, tzinfo=timezone.utc)
        ingestion_runner.get_status.return_value = IngestionSnapshot(
            status="failed",
            error="OperationalError: connection refused",
            started_at=started_at,
            finished_at=finished_at,
        )

        response = self.client.get("/api/v1/ingestion/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "failed")
        self.assertEqual(
            response.json()["error"],
            "OperationalError: connection refused",
        )
        self.assertIsNotNone(response.json()["started_at"])
        self.assertIsNotNone(response.json()["finished_at"])


class IngestionRunnerTests(unittest.TestCase):
    def test_finishes_a_reserved_maintenance_operation(self) -> None:
        runner = IngestionRunner()

        self.assertTrue(runner.reserve())
        self.assertEqual(runner.get_status().status, "pending")
        runner.finish_maintenance()

        self.assertEqual(runner.get_status().status, "idle")
        self.assertTrue(runner.reserve())

    @patch("app.api.services.ingestion_runner.run_ingestion_pipeline")
    def test_records_completed_status(self, pipeline) -> None:
        runner = IngestionRunner()
        observed_statuses: list[str] = []
        pipeline.side_effect = lambda: observed_statuses.append(
            runner.get_status().status
        )

        self.assertTrue(runner.reserve())
        self.assertEqual(runner.get_status().status, "pending")
        runner.run()

        self.assertEqual(observed_statuses, ["running"])
        self.assertEqual(runner.get_status().status, "completed")
        self.assertIsNotNone(runner.get_status().finished_at)
        self.assertTrue(runner.reserve())

    @patch(
        "app.api.services.ingestion_runner.run_ingestion_pipeline",
        side_effect=ConnectionError("PostgreSQL connection refused"),
    )
    def test_records_failed_status_and_error(self, pipeline) -> None:
        runner = IngestionRunner()
        runner.reserve()

        with self.assertLogs(
            "app.api.services.ingestion_runner",
            level="ERROR",
        ):
            runner.run()

        snapshot = runner.get_status()
        self.assertEqual(snapshot.status, "failed")
        self.assertEqual(
            snapshot.error,
            "ConnectionError: PostgreSQL connection refused",
        )
        self.assertIsNotNone(snapshot.started_at)
        self.assertIsNotNone(snapshot.finished_at)


if __name__ == "__main__":
    unittest.main()
