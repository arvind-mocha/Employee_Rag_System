"""HTTP handler for retrieval-grounded LLM answers."""

import logging

from fastapi import HTTPException, status

from app.api.llm import (
    LLMConfigurationError,
    LLMProviderRequestError,
    generate_grounded_answer,
    select_provider,
)
from app.api.schemas.answer import AnswerRequest, AnswerResponse
from app.api.schemas.retrieval import RetrievedChunkResponse
from app.retrieval.query_router import RetrievalOptions, get_query_router


logger = logging.getLogger(__name__)


def answer_question(request: AnswerRequest) -> AnswerResponse:
    """Retrieve context and generate one provider-selected answer."""
    provider = select_provider(
        hf=request.hf,
        openai=request.openai,
        anthropic=request.anthropic,
        grok=request.grok,
    )
    options = RetrievalOptions(
        semantic_search=request.semantic_search,
        keyword_search=request.keyword_search,
        query_rewriting=request.query_rewriting,
        reranking=request.reranking,
        source_diversity=request.source_diversity,
    )

    try:
        chunks = get_query_router().search(request.question, options)
    except Exception as error:
        logger.exception("Answer context retrieval failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Retrieval is unavailable. Verify the embedding configuration "
                "and PostgreSQL connection."
            ),
        ) from error

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant context was found for the question.",
        )

    try:
        generated = generate_grounded_answer(request.question, chunks, provider)
    except LLMConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except LLMProviderRequestError as error:
        logger.exception("LLM provider request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=error.detail,
        ) from error
    except Exception as error:
        logger.exception("LLM answer generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"The {provider.value} language model request failed.",
        ) from error

    sources = [
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
    return AnswerResponse(
        question=request.question,
        semantic_search=request.semantic_search,
        keyword_search=request.keyword_search,
        query_rewriting=request.query_rewriting,
        reranking=request.reranking,
        source_diversity=request.source_diversity,
        effective_query=options.effective_query(request.question),
        applied_strategies=options.applied_strategies(),
        provider=generated.provider.value,
        model=generated.model_name,
        answer=generated.answer,
        sources=sources,
    )
