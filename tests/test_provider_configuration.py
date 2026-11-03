from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import MagicMock, Mock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.llm import LLMProvider
from app.api.main import app
from app.api.services.provider_configuration import (
    configuration_statuses, validate_evaluation_providers,
)
from app.core.credential_store import (
    ActiveCredential, CredentialStore, CredentialStoreConfigurationError,
    CredentialStoreUnavailable, StoredCredential,
)


def credential(provider="openai", **changes):
    now = datetime.now(timezone.utc)
    return replace(StoredCredential(provider, "example-model", "1234",
                                    now + timedelta(hours=1), now, now), **changes)


class ProviderConfigurationTests(unittest.TestCase):
    def test_reports_all_four_providers_and_missing_keys(self):
        statuses = {item.provider: item for item in configuration_statuses([credential()])}
        self.assertEqual(len(statuses), 4)
        self.assertEqual(statuses[LLMProvider.OPENAI].status, "configured")
        self.assertEqual(statuses[LLMProvider.GEMINI].status, "not_configured")
        self.assertEqual(statuses[LLMProvider.GEMINI].message,
                         "API key not configured. Configure this provider in Settings before running evaluations.")
        self.assertNotIn("Verified", statuses[LLMProvider.OPENAI].message)

    def test_expiry_missing_model_and_missing_encrypted_key(self):
        records = [credential(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
                   credential("gemini", model_name=" "),
                   credential("grok", has_api_key=False)]
        statuses = {item.provider: item for item in configuration_statuses(records)}
        self.assertEqual(statuses[LLMProvider.OPENAI].status, "expired")
        self.assertIn("expired", statuses[LLMProvider.OPENAI].message)
        self.assertEqual(statuses[LLMProvider.GEMINI].status, "not_configured")
        self.assertIn("Model not configured", statuses[LLMProvider.GEMINI].message)
        self.assertEqual(statuses[LLMProvider.GROK].status, "not_configured")

    @patch("app.api.controllers.settings.get_credential_store")
    def test_existing_endpoint_reports_expired_metadata_without_secrets(self, get_store):
        get_store.return_value.list_active.return_value = [
            credential(), credential("grok", expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
        ]
        response = TestClient(app).get("/api/v1/settings/providers")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["credentials"]), 1)
        self.assertEqual(len(body["providers"]), 4)
        statuses = {item["provider"]: item for item in body["providers"]}
        self.assertEqual(statuses["grok"]["status"], "expired")
        self.assertNotIn("key_hint", statuses["openai"])
        self.assertNotIn("encrypted_api_key", response.text)
        self.assertNotIn("nonce", response.text)

    @patch("app.api.controllers.settings.get_credential_store")
    def test_status_endpoint_failure_is_safe_and_distinct(self, get_store):
        get_store.return_value.list_active.side_effect = CredentialStoreUnavailable("secret-in-error")
        response = TestClient(app).get("/api/v1/settings/providers")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret-in-error", response.text)
        self.assertIn("unavailable", response.json()["detail"])

    @patch("app.api.controllers.settings.get_credential_store")
    def test_deletion_and_replacement_are_reflected_by_status(self, get_store):
        state = [credential()]
        store = get_store.return_value
        store.list_active.side_effect = lambda **_: list(state)
        def delete(_provider):
            state.clear()
            return True
        def save(**kwargs):
            state[:] = [credential(model_name=kwargs["model_name"])]
            return state[0]
        store.delete.side_effect, store.save.side_effect = delete, save
        client = TestClient(app)
        def status():
            return next(item for item in client.get("/api/v1/settings/providers").json()["providers"]
                        if item["provider"] == "openai")
        self.assertEqual(status()["status"], "configured")
        self.assertEqual(client.delete("/api/v1/settings/providers/openai").status_code, 204)
        self.assertEqual(status()["status"], "not_configured")
        response = client.post("/api/v1/settings/providers", json={
            "provider": "openai", "model_name": "replacement-model",
            "api_key": "private-replacement-key", "expires_in_hours": 1,
        })
        self.assertEqual(response.status_code, 201)
        self.assertEqual(status()["model_name"], "replacement-model")
        self.assertEqual(status()["status"], "configured")
        self.assertNotIn("private-replacement-key", response.text)

    @patch("app.api.services.provider_configuration.get_credential_store")
    def test_preflight_checks_decryptability_and_sanitizes_store_errors(self, get_store):
        store = get_store.return_value
        store.list_active.return_value = [credential()]
        store.get_active.side_effect = CredentialStoreConfigurationError("private-key-in-error")
        with self.assertRaises(HTTPException) as failure:
            validate_evaluation_providers([LLMProvider.OPENAI], LLMProvider.OPENAI)
        self.assertEqual(failure.exception.status_code, 503)
        self.assertNotIn("private-key-in-error", failure.exception.detail)
        self.assertIn("OpenAI", failure.exception.detail)

    @patch("app.api.services.provider_configuration.get_credential_store")
    def test_rechecks_key_model_expiry_and_deletion_after_reading_metadata(self, get_store):
        store = get_store.return_value
        store.list_active.return_value = [credential()]
        now = datetime.now(timezone.utc)
        for active in [None,
                       ActiveCredential("openai", "model", " ", now + timedelta(hours=1)),
                       ActiveCredential("openai", " ", "private-key", now + timedelta(hours=1)),
                       ActiveCredential("openai", "model", "private-key", now - timedelta(seconds=1))]:
            with self.subTest(active=type(active).__name__):
                store.get_active.return_value = active
                with self.assertRaises(HTTPException) as failure:
                    validate_evaluation_providers([LLMProvider.OPENAI], LLMProvider.OPENAI)
                self.assertEqual(failure.exception.status_code, 422)
                self.assertNotIn("private-key", failure.exception.detail)

    def test_active_credential_repr_does_not_expose_key(self):
        active = ActiveCredential("openai", "model", "private-key", datetime.now(timezone.utc))
        self.assertNotIn("private-key", repr(active))


class CredentialExpiryStorageTests(unittest.TestCase):
    def setUp(self):
        self.store = CredentialStore(cipher=Mock())
        self.store._initialized = True
        self.connection = MagicMock()
        self.cursor = MagicMock()
        self.connection.__enter__.return_value.cursor.return_value.__enter__.return_value = self.cursor
        self.store._connect = Mock(return_value=self.connection)

    def test_cleanup_erases_secret_material_but_preserves_status_metadata(self):
        self.cursor.rowcount = 1
        self.assertEqual(self.store.delete_expired(), 1)
        query = self.cursor.execute.call_args.args[0]
        self.assertIn("encrypted_api_key = ''::bytea", query)
        self.assertIn("nonce = ''::bytea", query)
        self.assertIn("key_hint = ''", query)
        self.assertIn("expires_at <= CURRENT_TIMESTAMP", query)
        self.assertNotIn("DELETE FROM", query)
        self.assertNotIn("SET model_name", query)

    def test_active_reads_exclude_expired_keys_and_status_reads_can_include_them(self):
        self.cursor.fetchall.return_value = []
        self.store.list_active()
        self.assertFalse(self.cursor.execute.call_args.args[1][1])
        self.store.list_active(include_expired=True)
        self.assertTrue(self.cursor.execute.call_args.args[1][1])
        self.cursor.fetchone.return_value = None
        self.assertIsNone(self.store.get_active("openai"))
        self.assertIn("expires_at > CURRENT_TIMESTAMP", self.cursor.execute.call_args.args[0])
        self.store._cipher.decrypt.assert_not_called()
