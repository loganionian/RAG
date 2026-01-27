"""Documents and health check endpoints."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import DocumentInfo, DocumentsResponse, HealthResponse
from indexing.bm25_store import BM25Config, BM25Store
from indexing.chroma_store import health_check
from ingestion.storage import StorageManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["documents"])

# Default paths (can be overridden via environment variables)
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "data/processed"))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "data/vectorstore"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pilot-docs")


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents() -> DocumentsResponse:
    """List all ingested documents.

    Returns a list of all documents in the manifest with their metadata.
    """
    try:
        storage = StorageManager(PROCESSED_DIR)
        manifest = storage._manifest

        documents = []
        for doc_id, entry in manifest.items():
            # Parse ingested timestamp
            ingested_str = entry.get("last_ingested", "")
            try:
                ingested_at = datetime.fromisoformat(ingested_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ingested_at = datetime.utcnow()

            doc_info = DocumentInfo(
                doc_id=doc_id,
                filename=entry.get("relative_path", doc_id),
                file_type=entry.get("file_extension", "unknown"),
                chunks=entry.get("chunk_count", 0),
                ingested_at=ingested_at,
            )
            documents.append(doc_info)

        # Sort by ingested_at descending (most recent first)
        documents.sort(key=lambda d: d.ingested_at, reverse=True)

        return DocumentsResponse(documents=documents, total=len(documents))

    except Exception as e:
        logger.exception("Failed to list documents")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {e}",
        ) from e


@router.get("/health", response_model=HealthResponse)
async def health_check_endpoint() -> HealthResponse:
    """Check health of the RAG system.

    Returns status of the vectorstore and LLM provider.
    """
    # Check vectorstore health
    vs_result = health_check(VECTORSTORE_DIR, collection_name=COLLECTION_NAME)
    vectorstore_status = {
        "healthy": vs_result.healthy,
        "message": vs_result.message,
        "collection_count": vs_result.collection_count,
        "document_count": vs_result.document_count,
    }

    # Check LLM provider availability
    llm_status = {"healthy": False, "message": "Not configured"}
    try:
        # Check if LLM environment variables are set
        provider = os.getenv("LLM_PROVIDER", "custom")
        if provider == "custom":
            api_key = os.getenv("API_KEY")
            base_url = os.getenv("BASE_URL")
            if api_key and base_url:
                llm_status = {"healthy": True, "message": f"Custom provider configured", "provider": provider}
            else:
                llm_status = {"healthy": False, "message": "Missing API_KEY or BASE_URL", "provider": provider}
        elif provider == "openai":
            if os.getenv("OPENAI_API_KEY"):
                llm_status = {"healthy": True, "message": "OpenAI configured", "provider": provider}
            else:
                llm_status = {"healthy": False, "message": "Missing OPENAI_API_KEY", "provider": provider}
        elif provider == "anthropic":
            if os.getenv("ANTHROPIC_API_KEY"):
                llm_status = {"healthy": True, "message": "Anthropic configured", "provider": provider}
            else:
                llm_status = {"healthy": False, "message": "Missing ANTHROPIC_API_KEY", "provider": provider}
        elif provider == "ollama":
            llm_status = {"healthy": True, "message": "Ollama (local)", "provider": provider}
        else:
            llm_status = {"healthy": False, "message": f"Unknown provider: {provider}", "provider": provider}
    except Exception as e:
        logger.warning("LLM health check failed: %s", e)
        llm_status = {"healthy": False, "message": str(e)}

    # Check BM25 index health
    bm25_status = None
    try:
        bm25_config = BM25Config(
            index_dir=VECTORSTORE_DIR,
            index_name=COLLECTION_NAME,
        )
        bm25_store = BM25Store(bm25_config)
        bm25_result = bm25_store.health_check()
        bm25_status = {
            "healthy": bm25_result.healthy,
            "message": bm25_result.message,
            "document_count": bm25_result.document_count,
            "index_name": bm25_result.index_name,
        }
    except Exception as e:
        logger.warning("BM25 health check failed: %s", e)
        bm25_status = {"healthy": False, "message": str(e)}

    # Check SQL Agent health
    sql_agent_status = None
    db_path = VECTORSTORE_DIR / "catalog.duckdb"
    if db_path.exists():
        try:
            from sql_agent import SQLAgentConfig, SQLChain
            from sql_agent.sql_chain import SQLHealthCheckResult

            # Create a temporary chain just for health check
            # (we don't need an LLM client for basic health check)
            from sql_agent.schema_extractor import SchemaExtractor

            sql_config = SQLAgentConfig(db_path=db_path)
            extractor = SchemaExtractor(sql_config)
            stats = extractor.get_statistics()
            extractor.close()

            if stats["table_count"] > 0:
                sql_agent_status = {
                    "healthy": True,
                    "message": f"SQL Agent healthy. {stats['table_count']} tables, {stats['total_rows']:,} total rows.",
                    "table_count": stats["table_count"],
                    "total_rows": stats["total_rows"],
                }
            else:
                sql_agent_status = {
                    "healthy": True,
                    "message": "SQL Agent healthy but no user tables found.",
                    "table_count": 0,
                    "total_rows": 0,
                }
        except Exception as e:
            logger.warning("SQL Agent health check failed: %s", e)
            sql_agent_status = {"healthy": False, "message": str(e)}

    # Overall status
    overall_status = "healthy" if vs_result.healthy else "degraded"

    return HealthResponse(
        status=overall_status,
        vectorstore=vectorstore_status,
        llm_provider=llm_status,
        bm25_index=bm25_status,
        sql_agent=sql_agent_status,
    )
