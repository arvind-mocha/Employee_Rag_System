from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.api.main import app
from app.core.credential_store import StoredCredential


def stored_credential(provider: str = "openai") -> StoredCredential:
    now = datetime.now(timezone.utc)
    return StoredCredential(
        provider=provider,
        model_name="test-model",
        key_hint="1234",
        expires_at=now + timedelta(hours=24),
        created_at=now,
        updated_at=now,
    )


class SettingsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("app.api.controllers.settings.get_credential_store")
    def test_saves_a_secret_without_returning_it(self, get_store) -> None:
        store = Mock()
        store.save.return_value = stored_credential()
        get_store.return_value = store

        response = self.client.post(
            "/api/v1/settings/providers",
            json={
                "provider": "openai",
                "model_name": "test-model",
                "api_key": "top-secret-1234",
                "expires_in_hours": 24,
            },
        )

        self.assertEqual(response.status_code, 201)
        self.assertNotIn("top-secret-1234", response.text)
        self.assertEqual(response.json()["key_hint"], "1234")
        self.assertEqual(
            store.save.call_args.kwargs["api_key"],
            "top-secret-1234",
        )

    @patch("app.api.controllers.settings.get_credential_store")
    def test_lists_only_masked_credential_metadata(self, get_store) -> None:
        get_store.return_value.list_active.return_value = [stored_credential()]

        response = self.client.get("/api/v1/settings/providers")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["credentials"]), 1)
        self.assertNotIn("api_key", response.text)

    @patch("app.api.controllers.settings.get_credential_store")
    def test_deletes_a_saved_credential(self, get_store) -> None:
        get_store.return_value.delete.return_value = True

        response = self.client.delete("/api/v1/settings/providers/openai")

        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.content, b"")
        get_store.return_value.delete.assert_called_once_with("openai")

    def test_rejects_short_api_keys(self) -> None:
        response = self.client.post(
            "/api/v1/settings/providers",
            json={
                "provider": "openai",
                "model_name": "test-model",
                "api_key": "short",
                "expires_in_hours": 24,
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertNotIn("short", response.text)


if __name__ == "__main__":
    unittest.main()
