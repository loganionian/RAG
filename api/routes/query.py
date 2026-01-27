"""Query endpoint for RAG system."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

from api.schemas import QueryMetadata, QueryRequest, QueryResponse, SourceInfo
from storage import AgentCatalog, SQLStore, SQLStoreConfig

if TYPE_CHECKING:
    from generation.rag_chain import RAGChain

logger = logging.getLogger(__name__)

# Default database path (can be overridden via environment variable)
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "data/vectorstore"))
DB_PATH = VECTORSTORE_DIR / "catalog.duckdb"

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


def _get_catalog() -> AgentCatalog:
    """Get an initialized AgentCatalog instance.

    Returns:
        AgentCatalog connected to the database.
    """
    config = SQLStoreConfig(db_path=DB_PATH)
    sql_store = SQLStore(config)
    catalog = AgentCatalog(sql_store)
    catalog.init_catalog()
    return catalog


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Query the RAG system with a question.

    Retrieves relevant chunks from the vectorstore and generates an answer
    using the configured LLM provider.

    If an agent_id is provided, the agent's system prompt, temperature, and
    max_tokens settings will be used instead of the defaults.
    """
    rag_chain = get_rag_chain()

    # Determine generation parameters (agent overrides or defaults)
    system_prompt_template = rag_chain.config.system_prompt
    temperature = rag_chain.config.temperature
    max_tokens = rag_chain.config.max_tokens
    agent_id_used = None

    if request.agent_id:
        try:
            catalog = _get_catalog()
            agent = catalog.get_by_id(request.agent_id)
            if agent is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Agent not found: {request.agent_id}",
                )
            # Apply agent overrides
            system_prompt_template = agent.system_prompt
            if agent.temperature is not None:
                temperature = agent.temperature
            if agent.max_tokens is not None:
                max_tokens = agent.max_tokens
            agent_id_used = request.agent_id
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Failed to fetch agent %s", request.agent_id)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch agent: {e}",
            ) from e

    try:
        # Measure retrieval time separately
        retrieval_start = time.perf_counter()
        retrieval_result = rag_chain.retrieve(request.question, k=request.k)
        retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

        # Build context and generate (LLM call only)
        generation_start = time.perf_counter()
        context = rag_chain._format_context(retrieval_result)
        system_prompt = system_prompt_template.format(context=context)
        answer = rag_chain.llm_client.generate(
            prompt=request.question,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
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
            agent_id=agent_id_used,
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            metadata=query_metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query failed: %s", request.question[:50])
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {e}",
        ) from e
