"""API route modules."""

from .agents import router as agents_router
from .documents import router as documents_router
from .ingest import router as ingest_router
from .query import router as query_router

__all__ = ["agents_router", "documents_router", "ingest_router", "query_router"]
