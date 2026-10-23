import unittest
from unittest.mock import Mock, patch

from app.api import config
from app.core.embeddings import (
    EmbeddingSettings,
    create_embedding_model,
    get_embedding_dimension,
)


class EmbeddingSettingsTests(unittest.TestCase):
    def test_loads_central_embedding_configuration(self) -> None:
        settings = EmbeddingSettings.from_config()

        self.assertEqual(
            settings,
            EmbeddingSettings(
                model_name=config.EMBEDDING_MODEL,
                batch_size=config.EMBEDDING_BATCH_SIZE,
                device=config.EMBEDDING_DEVICE,
            ),
        )

    def test_rejects_non_positive_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be at least 1"):
            EmbeddingSettings(batch_size=0)


class EmbeddingModelTests(unittest.TestCase):
    @patch("app.core.embeddings.HuggingFaceEmbeddings")
    def test_creates_normalized_hugging_face_embeddings(
        self,
        embedding_class: Mock,
    ) -> None:
        settings = EmbeddingSettings(
            model_name="sentence-transformers/test-model",
            batch_size=8,
            device="cpu",
        )

        create_embedding_model(settings)

        embedding_class.assert_called_once_with(
            model="sentence-transformers/test-model",
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "batch_size": 8,
                "normalize_embeddings": True,
            },
            query_encode_kwargs={"normalize_embeddings": True},
            show_progress=False,
        )

    def test_reads_embedding_dimension_from_probe_vector(self) -> None:
        embedding_model = Mock()
        embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]

        self.assertEqual(get_embedding_dimension(embedding_model), 3)


if __name__ == "__main__":
    unittest.main()
