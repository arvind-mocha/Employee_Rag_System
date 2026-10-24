"""API tests for vector-data management operations."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from app.api.main import app
class DataManagementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("app.api.controllers.data.get_semantic_search")
    @patch("app.api.controllers.data.get_query_router")
    @patch("app.api.controllers.data.drop_vector_store_table")
    @patch("app.api.controllers.data.ingestion_runner")
    def test_erases_vectors_and_preserves_sources(
        self,
        ingestion_runner,
        drop_vector_store_table,
        get_query_router,
        get_semantic_search,
    ) -> None:
        ingestion_runner.reserve.return_value = True

        response = self.client.delete("/api/v1/data/vector-database")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "erased")
        self.assertIn("source files were preserved", response.json()["message"])
        drop_vector_store_table.assert_called_once_with()
        ingestion_runner.reserve.assert_called_once_with()
        ingestion_runner.finish_maintenance.assert_called_once_with()
        get_query_router.cache_clear.assert_called_once_with()
        get_semantic_search.cache_clear.assert_called_once_with()

    @patch("app.api.controllers.data.drop_vector_store_table")
    @patch("app.api.controllers.data.ingestion_runner")
    def test_rejects_erase_during_ingestion(
        self,
        ingestion_runner,
        drop_vector_store_table,
    ) -> None:
        ingestion_runner.reserve.return_value = False

        response = self.client.delete("/api/v1/data/vector-database")

        self.assertEqual(response.status_code, 409)
        drop_vector_store_table.assert_not_called()
        ingestion_runner.finish_maintenance.assert_not_called()

    @patch("app.api.controllers.data.drop_vector_store_table")
    @patch("app.api.controllers.data.ingestion_runner")
    def test_requires_redis_before_erasing_vectors(
        self,
        ingestion_runner,
        drop_vector_store_table,
    ) -> None:
        ingestion_runner.reserve.side_effect = RedisConnectionError(
            "connection refused"
        )

        response = self.client.delete("/api/v1/data/vector-database")

        self.assertEqual(response.status_code, 503)
        drop_vector_store_table.assert_not_called()


if __name__ == "__main__":
    unittest.main()
