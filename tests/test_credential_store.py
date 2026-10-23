from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.core.credential_store import CredentialCipher


class CredentialCipherTests(unittest.TestCase):
    def test_encrypts_and_decrypts_without_writing_plaintext(self) -> None:
        with TemporaryDirectory() as directory:
            key_path = Path(directory) / "credential.key"
            cipher = CredentialCipher(key_path)

            ciphertext, nonce = cipher.encrypt("openai", "secret-api-key")

            self.assertNotIn(b"secret-api-key", ciphertext)
            self.assertEqual(
                cipher.decrypt("openai", ciphertext, nonce),
                "secret-api-key",
            )

    def test_ciphertext_is_bound_to_its_provider(self) -> None:
        with TemporaryDirectory() as directory:
            cipher = CredentialCipher(Path(directory) / "credential.key")
            ciphertext, nonce = cipher.encrypt("openai", "secret-api-key")

            with self.assertRaisesRegex(
                RuntimeError,
                "cannot be decrypted",
            ):
                cipher.decrypt("anthropic", ciphertext, nonce)


if __name__ == "__main__":
    unittest.main()
