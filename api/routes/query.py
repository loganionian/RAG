"""Query endpoint for RAG system."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

from api.schemas import QueryMetadata, QueryRequest, QueryResponse, SourceInfo

if TYPE_CHECKING:
    from generation.rag_chain import RAGChain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])

# Global RAG chain instance (initialized in main.py lifespan)
_rag_chain: "RAGChain | None" = None


def set_rag_chain(chain: "RAGChain") -> None:
    """Set the global RAG chain instance."""
    global _rag_chain
    _rag_chain = chain


def get_rag_chain() -> "RAGChain":
    """Get the global RAG chain instance."""
    if _rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized",
        )
    return _rag_chain


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Query the RAG system with a question.

    Retrieves relevant chunks from the vectorstore and generates an answer
    using the configured LLM provider.
    """
    rag_chain = get_rag_chain()

    try:
        # Measure retrieval time separately
        retrieval_start = time.perf_counter()
        retrieval_result = rag_chain.retrieve(request.question, k=request.k)
        retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

        # Build context and generate (LLM call only)
        generation_start = time.perf_counter()
        context = rag_chain._format_context(retrieval_result)
        system_prompt = rag_chain.config.system_prompt.format(context=context)
        answer = rag_chain.llm_client.generate(
            prompt=request.question,
            system_prompt=system_prompt,
            max_tokens=rag_chain.config.max_tokens,
            temperature=rag_chain.config.temperature,
        )
        generation_time_ms = (time.perf_counter() - generation_start) * 1000

        # Build response using retrieval result
        response_metadatas = retrieval_result.metadatas
        response_chunks = retrieval_result.chunks

        # Build source info from metadatas
        sources = []
        for i, metadata in enumerate(response_metadatas):
            # Extract relevant metadata
            doc_id = metadata.get("doc_id", "unknown")
            relative_path = metadata.get("relative_path", doc_id)
            chunk_id = metadata.get("chunk_id", f"{doc_id}::chunk-{i:04d}")
            page = metadata.get("page")

            # Get snippet from retrieved chunk
            snippet = ""
            if i < len(response_chunks):
                chunk_text = response_chunks[i]
                snippet = chunk_text[:200].replace("\n", " ").strip()
                if len(chunk_text) > 200:
                    snippet += "..."

            source = SourceInfo(
                doc_id=doc_id,
                filename=relative_path,
                chunk_id=chunk_id,
                snippet=snippet,
                page=page,
            )
            sources.append(source)

        query_metadata = QueryMetadata(
            retrieval_time_ms=round(retrieval_time_ms, 2),
            generation_time_ms=round(generation_time_ms, 2),
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            metadata=query_metadata,
        )

    except Exception as e:
        logger.exception("Query failed: %s", request.question[:50])
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {e}",
        ) from e
