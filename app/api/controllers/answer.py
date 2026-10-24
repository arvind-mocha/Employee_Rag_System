"""HTTP handler for tool-directed LLM answers."""

import logging

from fastapi import HTTPException, status

from app.api.llm import (
    LLMConfigurationError,
    LLMProviderRequestError,
    generate_tool_driven_answer,
    select_provider,
)
from app.api.schemas.answer import AnswerRequest, AnswerResponse
from app.api.schemas.retrieval import RetrievedChunkResponse
from app.retrieval.query_router import RetrievalOptions, get_query_router


logger = logging.getLogger(__name__)


def answer_question(request: AnswerRequest) -> AnswerResponse:
    """Let one provider-selected model decide whether retrieval is needed."""
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

    def search_employee_knowledge(query: str):
        try:
            return get_query_router().search(query, options)
        except Exception as error:
            logger.exception("Answer context retrieval failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Retrieval is unavailable. Verify the embedding "
                    "configuration and PostgreSQL connection."
                ),
            ) from error

    try:
        generated = generate_tool_driven_answer(
            request.question,
            provider,
            search_employee_knowledge,
        )
    except HTTPException:
        raise
    except LLMConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except LLMProviderRequestError as error:
        logger.warning("LLM provider request failed for %s", provider.value)
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
        for chunk in generated.chunks
    ]
    effective_query = (
        options.effective_query(generated.retrieval_query)
        if generated.retrieval_query is not None
        else request.question
    )
    return AnswerResponse(
        question=request.question,
        semantic_search=request.semantic_search,
        keyword_search=request.keyword_search,
        query_rewriting=request.query_rewriting,
        reranking=request.reranking,
        source_diversity=request.source_diversity,
        effective_query=effective_query,
        applied_strategies=(
            options.applied_strategies() if generated.retrieval_used else []
        ),
        retrieval_used=generated.retrieval_used,
        provider=generated.provider.value,
        model=generated.model_name,
        answer=generated.answer,
        usage=generated.usage,
        sources=sources,
    )
