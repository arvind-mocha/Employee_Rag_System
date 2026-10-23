from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.api import config


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    model_name: str = config.EMBEDDING_MODEL
    batch_size: int = config.EMBEDDING_BATCH_SIZE
    device: str = config.EMBEDDING_DEVICE

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("embedding model name cannot be blank")
        if self.batch_size < 1:
            raise ValueError("embedding batch size must be at least 1")
        if not self.device.strip():
            raise ValueError("embedding device cannot be blank")

    @classmethod
    def from_config(cls) -> "EmbeddingSettings":
        return cls()


def create_embedding_model(
    settings: EmbeddingSettings | None = None,
) -> HuggingFaceEmbeddings:
    """Create the configured local Hugging Face embedding model."""
    settings = settings or EmbeddingSettings.from_config()

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
