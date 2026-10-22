from dataclasses import dataclass
import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    model_name: str
    batch_size: int
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        load_dotenv()

        model_name = os.getenv("EMBEDDING_MODEL", "").strip()
        if not model_name:
            raise ValueError("EMBEDDING_MODEL must be set in the environment")

        raw_batch_size = os.getenv("EMBEDDING_BATCH_SIZE", "").strip()
        if not raw_batch_size:
            raise ValueError("EMBEDDING_BATCH_SIZE must be set in the environment")

        try:
            batch_size = int(raw_batch_size)
        except ValueError as error:
            raise ValueError("EMBEDDING_BATCH_SIZE must be an integer") from error

        if batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE must be at least 1")

        device = os.getenv("EMBEDDING_DEVICE", "cpu").strip() or "cpu"
        return cls(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
        )


def create_embedding_model(
    settings: EmbeddingSettings | None = None,
) -> HuggingFaceEmbeddings:
    """Create the configured local Hugging Face embedding model."""
    settings = settings or EmbeddingSettings.from_env()

    return HuggingFaceEmbeddings(
        model=settings.model_name,
        model_kwargs={"device": settings.device},
        encode_kwargs={
            "batch_size": settings.batch_size,
            "normalize_embeddings": True,
        },
        query_encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )


def get_embedding_dimension(embedding_model: Embeddings) -> int:
    """Determine the vector dimension produced by an embedding model."""
    probe_embedding = embedding_model.embed_query("embedding dimension probe")

    if not probe_embedding:
        raise ValueError("The embedding model returned an empty vector")

    return len(probe_embedding)
