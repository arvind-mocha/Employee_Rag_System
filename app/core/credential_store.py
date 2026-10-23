"""Encrypted PostgreSQL storage for customer-supplied LLM credentials."""

from base64 import b64decode, urlsafe_b64encode
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import os
from pathlib import Path
from threading import Lock

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import psycopg
from psycopg.rows import dict_row

from app.api.config import PROJECT_ROOT
from app.core.vector_store import PGVectorSettings


PROVIDERS = ("huggingface", "openai", "anthropic", "grok")
KEY_BYTES = 32
KEY_VERSION = 1
DEFAULT_KEY_PATH = PROJECT_ROOT / ".secrets" / "credential_encryption_key"


class CredentialStoreError(RuntimeError):
    """Base class for safe credential-store failures."""


class CredentialStoreConfigurationError(CredentialStoreError):
    """Raised when the encryption key or database URL is invalid."""


class CredentialStoreUnavailable(CredentialStoreError):
    """Raised when PostgreSQL cannot service a credential operation."""


@dataclass(frozen=True, slots=True)
class StoredCredential:
    provider: str
    model_name: str
    key_hint: str
    expires_at: datetime
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ActiveCredential:
    provider: str
    model_name: str
    api_key: str
    expires_at: datetime


class CredentialCipher:
    """Encrypt and decrypt provider keys with authenticated encryption."""

    def __init__(self, key_path: Path | None = None) -> None:
        configured_path = os.getenv(
            "CREDENTIAL_ENCRYPTION_KEY_FILE",
            "",
        ).strip()
        self._explicit_path = key_path is None and bool(configured_path)
        self.key_path = key_path or (
            Path(configured_path).expanduser()
            if configured_path
            else DEFAULT_KEY_PATH
        )
        self._key: bytes | None = None
        self._lock = Lock()

    def encrypt(self, provider: str, plaintext: str) -> tuple[bytes, bytes]:
        nonce = os.urandom(12)
        ciphertext = AESGCM(self._get_key()).encrypt(
            nonce,
            plaintext.encode("utf-8"),
            self._associated_data(provider),
        )
        return ciphertext, nonce

    def decrypt(self, provider: str, ciphertext: bytes, nonce: bytes) -> str:
        try:
            plaintext = AESGCM(self._get_key()).decrypt(
                nonce,
                ciphertext,
                self._associated_data(provider),
            )
        except InvalidTag as error:
            raise CredentialStoreConfigurationError(
                "Stored credentials cannot be decrypted with this installation's "
                "encryption key. Restore the original key or replace the credential."
            ) from error
        try:
            return plaintext.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CredentialStoreConfigurationError(
                "A stored credential contains invalid encrypted data."
            ) from error

    def _get_key(self) -> bytes:
        if self._key is not None:
            return self._key
        with self._lock:
            if self._key is None:
                self._key = self._load_or_create_key()
        return self._key

    def _load_or_create_key(self) -> bytes:
        if not self.key_path.exists():
            if self._explicit_path:
                raise CredentialStoreConfigurationError(
                    f"Credential encryption key file does not exist: {self.key_path}"
                )
            self._create_key_file()

        if not self._explicit_path and os.name == "posix":
            try:
                self.key_path.parent.chmod(0o700)
                self.key_path.chmod(0o600)
            except OSError as error:
                raise CredentialStoreConfigurationError(
                    "Credential encryption key file permissions could not be secured."
                ) from error

        try:
            encoded = self.key_path.read_bytes().strip()
            key = b64decode(encoded, altchars=b"-_", validate=True)
        except (OSError, ValueError) as error:
            raise CredentialStoreConfigurationError(
                "Credential encryption key file is unreadable or invalid."
            ) from error
        if len(key) != KEY_BYTES:
            raise CredentialStoreConfigurationError(
                "Credential encryption key must decode to exactly 32 bytes."
            )
        return key

    def _create_key_file(self) -> None:
        temporary_path = self.key_path.with_name(
            f".{self.key_path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
        )
        descriptor: int | None = None
        try:
            self.key_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(
                temporary_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            os.write(
                descriptor,
                urlsafe_b64encode(os.urandom(KEY_BYTES)) + b"\n",
            )
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            try:
                os.link(temporary_path, self.key_path)
            except FileExistsError:
                pass
        except OSError as error:
            raise CredentialStoreConfigurationError(
                "Could not create the credential encryption key file."
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _associated_data(provider: str) -> bytes:
        return f"employee-rag/provider-credential/v{KEY_VERSION}/{provider}".encode()


class CredentialStore:
    """Persist encrypted provider credentials and enforce their expiry."""

    def __init__(
        self,
        database_url: str | None = None,
        cipher: CredentialCipher | None = None,
    ) -> None:
        self._database_url = database_url
        self._cipher = cipher or CredentialCipher()
        self._initialized = False
        self._init_lock = Lock()

    def save(
        self,
        *,
        provider: str,
        model_name: str,
        api_key: str,
        expires_at: datetime,
    ) -> StoredCredential:
        self._validate_provider(provider)
        normalized_model = model_name.strip()
        normalized_key = api_key.strip()
        if not normalized_model:
            raise ValueError("Model name is required.")
        if not normalized_key:
            raise ValueError("API key is required.")
        if expires_at <= datetime.now(timezone.utc):
            raise ValueError("Credential expiry must be in the future.")

        ciphertext, nonce = self._cipher.encrypt(provider, normalized_key)
        self._ensure_schema()
        try:
            with self._connect() as connection:
                with connection.cursor(row_factory=dict_row) as cursor:
                    cursor.execute(
                        """
                        INSERT INTO provider_credentials (
                            provider,
                            model_name,
                            encrypted_api_key,
                            nonce,
                            key_hint,
                            key_version,
                            expires_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (provider) DO UPDATE SET
                            model_name = EXCLUDED.model_name,
                            encrypted_api_key = EXCLUDED.encrypted_api_key,
                            nonce = EXCLUDED.nonce,
                            key_hint = EXCLUDED.key_hint,
                            key_version = EXCLUDED.key_version,
                            expires_at = EXCLUDED.expires_at,
                            updated_at = CURRENT_TIMESTAMP
                        RETURNING provider, model_name, key_hint, expires_at,
                                  created_at, updated_at
                        """,
                        (
                            provider,
                            normalized_model,
                            ciphertext,
                            nonce,
                            normalized_key[-4:],
                            KEY_VERSION,
                            expires_at,
                        ),
                    )
                    row = cursor.fetchone()
        except psycopg.Error as error:
            raise CredentialStoreUnavailable(
                "PostgreSQL is unavailable; provider settings were not saved."
            ) from error
        if row is None:
            raise CredentialStoreUnavailable("Provider settings were not saved.")
        return self._stored_credential(row)

    def list_active(self) -> list[StoredCredential]:
        self._ensure_schema()
        try:
            with self._connect() as connection:
                with connection.cursor(row_factory=dict_row) as cursor:
                    cursor.execute(
                        "DELETE FROM provider_credentials "
                        "WHERE expires_at <= CURRENT_TIMESTAMP"
                    )
                    cursor.execute(
                        """
                        SELECT provider, model_name, key_hint, expires_at,
                               created_at, updated_at
                        FROM provider_credentials
                        ORDER BY provider
                        """
                    )
                    rows = cursor.fetchall()
        except psycopg.Error as error:
            raise CredentialStoreUnavailable(
                "PostgreSQL is unavailable; provider settings could not be read."
            ) from error
        return [self._stored_credential(row) for row in rows]

    def get_active(self, provider: str) -> ActiveCredential | None:
        self._validate_provider(provider)
        self._ensure_schema()
        try:
            with self._connect() as connection:
                with connection.cursor(row_factory=dict_row) as cursor:
                    cursor.execute(
                        "DELETE FROM provider_credentials "
                        "WHERE provider = %s AND expires_at <= CURRENT_TIMESTAMP",
                        (provider,),
                    )
                    cursor.execute(
                        """
                        SELECT provider, model_name, encrypted_api_key, nonce,
                               key_version, expires_at
                        FROM provider_credentials
                        WHERE provider = %s
                        """,
                        (provider,),
                    )
                    row = cursor.fetchone()
        except psycopg.Error as error:
            raise CredentialStoreUnavailable(
                "PostgreSQL is unavailable; provider settings could not be read."
            ) from error
        if row is None:
            return None
        if row["key_version"] != KEY_VERSION:
            raise CredentialStoreConfigurationError(
                "A stored credential uses an unsupported encryption key version."
            )
        api_key = self._cipher.decrypt(
            provider,
            bytes(row["encrypted_api_key"]),
            bytes(row["nonce"]),
        )
        return ActiveCredential(
            provider=row["provider"],
            model_name=row["model_name"],
            api_key=api_key,
            expires_at=row["expires_at"],
        )

    def delete(self, provider: str) -> bool:
        self._validate_provider(provider)
        self._ensure_schema()
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM provider_credentials WHERE provider = %s",
                        (provider,),
                    )
                    deleted = cursor.rowcount > 0
        except psycopg.Error as error:
            raise CredentialStoreUnavailable(
                "PostgreSQL is unavailable; provider settings were not deleted."
            ) from error
        return deleted

    def delete_expired(self) -> int:
        self._ensure_schema()
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM provider_credentials "
                        "WHERE expires_at <= CURRENT_TIMESTAMP"
                    )
                    return max(cursor.rowcount, 0)
        except psycopg.Error as error:
            raise CredentialStoreUnavailable(
                "PostgreSQL is unavailable; expired credentials were not deleted."
            ) from error

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            try:
                with self._connect() as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            """
                            CREATE TABLE IF NOT EXISTS provider_credentials (
                                provider VARCHAR(32) PRIMARY KEY,
                                model_name VARCHAR(255) NOT NULL,
                                encrypted_api_key BYTEA NOT NULL,
                                nonce BYTEA NOT NULL,
                                key_hint VARCHAR(4) NOT NULL,
                                key_version SMALLINT NOT NULL,
                                expires_at TIMESTAMPTZ NOT NULL,
                                created_at TIMESTAMPTZ NOT NULL
                                    DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMPTZ NOT NULL
                                    DEFAULT CURRENT_TIMESTAMP,
                                CONSTRAINT provider_credentials_provider_check
                                    CHECK (provider IN (
                                        'huggingface', 'openai',
                                        'anthropic', 'grok'
                                    ))
                            )
                            """
                        )
                        cursor.execute(
                            """
                            CREATE INDEX IF NOT EXISTS
                                provider_credentials_expires_at_idx
                            ON provider_credentials (expires_at)
                            """
                        )
            except psycopg.Error as error:
                raise CredentialStoreUnavailable(
                    "PostgreSQL is unavailable; provider settings cannot be initialized."
                ) from error
            self._initialized = True

    def _connect(self) -> psycopg.Connection:
        database_url = self._database_url
        if database_url is None:
            try:
                database_url = PGVectorSettings.from_env().database_url
            except ValueError as error:
                raise CredentialStoreConfigurationError(str(error)) from error
        if database_url.startswith("postgresql+psycopg://"):
            database_url = database_url.replace(
                "postgresql+psycopg://",
                "postgresql://",
                1,
            )
        if database_url.startswith("postgresql+asyncpg://"):
            raise CredentialStoreConfigurationError(
                "Provider settings require a PostgreSQL URL usable by psycopg."
            )
        return psycopg.connect(database_url, connect_timeout=5)

    @staticmethod
    def _validate_provider(provider: str) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def _stored_credential(row: dict[str, object]) -> StoredCredential:
        return StoredCredential(
            provider=str(row["provider"]),
            model_name=str(row["model_name"]),
            key_hint=str(row["key_hint"]),
            expires_at=row["expires_at"],  # type: ignore[arg-type]
            created_at=row["created_at"],  # type: ignore[arg-type]
            updated_at=row["updated_at"],  # type: ignore[arg-type]
        )


@lru_cache(maxsize=1)
def get_credential_store() -> CredentialStore:
    return CredentialStore()
