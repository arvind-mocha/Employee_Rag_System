"""HTTP handlers for retrieval operations."""

import logging

from fastapi import HTTPException, status

from app.api.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunkResponse,
)
from app.retrieval.config import RETRIEVAL_SETTINGS
from app.retrieval.semantic_search import get_semantic_search


logger = logging.getLogger(__name__)


def retrieve_chunks(request: RetrievalRequest) -> RetrievalResponse:
    """Return relevant chunks without generating an LLM answer."""
    try:
        chunks = get_semantic_search().search(request.question)
    except Exception as error:
        logger.exception("Semantic retrieval failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Retrieval is unavailable. Verify the embedding configuration "
                "and PostgreSQL connection."
            ),
        ) from error

    response_chunks = [
        RetrievedChunkResponse(
            content=chunk.content,
            relevance_score=chunk.relevance_score,
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
            source_type=chunk.source_type,
            section=chunk.section,
            chunk_index=chunk.chunk_index,
        )
        for chunk in chunks
    ]
    return RetrievalResponse(
        question=request.question,
        requested_chunks=RETRIEVAL_SETTINGS.number_of_chunks,
        returned_chunks=len(response_chunks),
        chunks=response_chunks,
    )
