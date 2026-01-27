"""SQL Query endpoint for SQL Agent."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

from api.schemas import SQLQueryMetadata, SQLQueryRequest, SQLQueryResponse

if TYPE_CHECKING:
    from sql_agent import SQLChain

logger = logging.getLogger(__name__)

# Default database path (can be overridden via environment variable)
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "data/vectorstore"))
DB_PATH = VECTORSTORE_DIR / "catalog.duckdb"

router = APIRouter(prefix="/api", tags=["sql"])

# Global SQL chain instance (initialized in main.py lifespan)
_sql_chain: "SQLChain | None" = None


def set_sql_chain(chain: "SQLChain") -> None:
    """Set the global SQL chain instance."""
    global _sql_chain
    _sql_chain = chain


def get_sql_chain() -> "SQLChain":
    """Get the global SQL chain instance."""
    if _sql_chain is None:
        raise HTTPException(
            status_code=503,
            detail="SQL Agent not initialized. Ensure the database exists and the server is configured.",
        )
    return _sql_chain


@router.post("/sql-query", response_model=SQLQueryResponse)
async def query_sql(request: SQLQueryRequest) -> SQLQueryResponse:
    """Query the database using natural language.

    Translates a natural language question into SQL, executes the query,
    and returns the results with a natural language summary.

    Only SELECT queries are allowed for safety.
    """
    sql_chain = get_sql_chain()

    try:
        result = sql_chain.query(
            question=request.question,
            max_rows=request.max_rows,
            summarize=True,
        )

        metadata = SQLQueryMetadata(
            generation_time_ms=result.generation_time_ms,
            execution_time_ms=result.execution_time_ms,
            summarization_time_ms=result.summarization_time_ms,
            tables_used=result.tables_used,
        )

        return SQLQueryResponse(
            answer=result.answer,
            data=result.data,
            columns=result.columns,
            row_count=result.row_count,
            generated_sql=result.generated_sql if request.show_sql else None,
            metadata=metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Import here to avoid circular imports
        from sql_agent.errors import (
            QueryExecutionError,
            QueryGenerationError,
            QueryValidationError,
        )

        # Map SQL agent errors to appropriate HTTP status codes
        if isinstance(e, QueryValidationError):
            logger.warning("Query validation failed: %s", e)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid query: {e}",
            ) from e
        elif isinstance(e, QueryGenerationError):
            logger.warning("Query generation failed: %s", e)
            raise HTTPException(
                status_code=400,
                detail=f"Could not generate SQL: {e}",
            ) from e
        elif isinstance(e, QueryExecutionError):
            logger.exception("Query execution failed")
            raise HTTPException(
                status_code=500,
                detail=f"Query execution failed: {e}",
            ) from e
        else:
            logger.exception("SQL query failed: %s", request.question[:50])
            raise HTTPException(
                status_code=500,
                detail=f"SQL query failed: {e}",
            ) from e


@router.get("/sql-tables")
async def list_sql_tables() -> dict:
    """List available tables for SQL queries.

    Returns the list of tables that can be queried (excludes system tables).
    """
    sql_chain = get_sql_chain()

    try:
        tables = sql_chain.get_available_tables()
        return {
            "tables": tables,
            "count": len(tables),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to list SQL tables")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tables: {e}",
        ) from e
