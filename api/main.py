"""FastAPI application for RAG system."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import agents_router, documents_router, ingest_router, query_router, sql_query_router
from api.routes.query import set_rag_chain
from api.routes.sql_query import set_sql_chain
from api.schemas import ErrorResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "data/vectorstore"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    logger.info("Starting RAG API server...")

    # Initialize RAG chain if vectorstore exists
    if VECTORSTORE_DIR.exists():
        try:
            from generation.factory import create_llm_client
            from generation.rag_chain import RAGChain, RAGConfig

            llm_client = create_llm_client()
            rag_config = RAGConfig(vectorstore_dir=VECTORSTORE_DIR)
            rag_chain = RAGChain(llm_client, rag_config)
            set_rag_chain(rag_chain)
            logger.info("RAG chain initialized successfully")

            # Initialize SQL chain if database exists
            db_path = VECTORSTORE_DIR / "catalog.duckdb"
            if db_path.exists():
                try:
                    from sql_agent import SQLAgentConfig, SQLChain

                    sql_config = SQLAgentConfig(db_path=db_path)
                    sql_chain = SQLChain(llm_client, sql_config)
                    set_sql_chain(sql_chain)
                    logger.info("SQL Agent initialized successfully")
                except Exception as e:
                    logger.warning("Failed to initialize SQL Agent: %s", e)
                    logger.info("SQL query endpoint will be unavailable")
            else:
                logger.info("DuckDB database not found at %s, SQL Agent not initialized", db_path)

        except Exception as e:
            logger.warning("Failed to initialize RAG chain: %s", e)
            logger.info("Query endpoint will be unavailable until RAG chain is configured")
    else:
        logger.warning("Vectorstore directory does not exist: %s", VECTORSTORE_DIR)
        logger.info("Run ingestion and indexing pipelines first")

    yield

    # Shutdown
    logger.info("Shutting down RAG API server...")


# Create FastAPI app
app = FastAPI(
    title="RAG API",
    description="REST API for Retrieval Augmented Generation system",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agents_router)
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(ingest_router)
app.include_router(sql_query_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for consistent error format."""
    logger.exception("Unhandled exception: %s", exc)
    error = ErrorResponse(
        detail=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content=error.model_dump(),
    )


@app.get("/")
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": "RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )
