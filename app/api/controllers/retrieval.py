"""HTTP handlers for retrieval operations."""

import logging
from typing import Annotated

from fastapi import HTTPException, Query, status

from app.api.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunkResponse,
)
from app.retrieval.config import RETRIEVAL_SETTINGS
from app.retrieval.query_router import (
    RetrievalOptions,
    get_query_router,
)


logger = logging.getLogger(__name__)


def retrieve_chunks(
    request: Annotated[RetrievalRequest, Query()],
) -> RetrievalResponse:
    """Return relevant chunks without generating an LLM answer."""
    options = RetrievalOptions(
        semantic_search=request.semantic_search,
        keyword_search=request.keyword_search,
        query_rewriting=request.query_rewriting,
        reranking=request.reranking,
        source_diversity=request.source_diversity,
    )
    try:
        chunks = get_query_router().search(
            request.question,
            options,
        )
    except Exception as error:
        logger.exception("Document retrieval failed")
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
        effective_query=options.effective_query(request.question),
        applied_strategies=options.applied_strategies(),
        requested_chunks=RETRIEVAL_SETTINGS.number_of_chunks,
        returned_chunks=len(response_chunks),
        chunks=response_chunks,
    )
