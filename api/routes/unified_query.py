"""Unified query endpoint with routing and ACL support.

This module provides a unified query endpoint that automatically routes
questions to document search (RAG), SQL queries, or both based on
question classification. Uses the LangGraph-based orchestrator for
query execution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, HTTPException

from api.schemas import (
    UnifiedQueryRequest,
    UnifiedQueryResponse,
)
from orchestrator import RAGOrchestrator, graph_state_to_response, has_fatal_errors
from router import QuestionClassifier, RouterConfig
from security import ACLConfig, ACLFilter, AuditLogger, SecurityContext, TableACL, TableACLConfig

if TYPE_CHECKING:
    from generation.rag_chain import RAGChain
    from sql_agent import SQLChain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["unified"])

# Global instances (initialized in main.py lifespan)
_orchestrator: Optional[RAGOrchestrator] = None
_rag_chain: Optional["RAGChain"] = None
_sql_chain: Optional["SQLChain"] = None
_classifier: Optional[QuestionClassifier] = None
_acl_filter: Optional[ACLFilter] = None
_table_acl: Optional[TableACL] = None
_audit_logger: Optional[AuditLogger] = None


def set_orchestrator(orchestrator: Optional[RAGOrchestrator]) -> None:
    """Set the global orchestrator instance."""
    global _orchestrator
    _orchestrator = orchestrator


def get_orchestrator() -> Optional[RAGOrchestrator]:
    """Get the orchestrator if available."""
    return _orchestrator


def set_unified_components(
    rag_chain: Optional["RAGChain"] = None,
    sql_chain: Optional["SQLChain"] = None,
    classifier: Optional[QuestionClassifier] = None,
    acl_filter: Optional[ACLFilter] = None,
    table_acl: Optional[TableACL] = None,
    audit_logger: Optional[AuditLogger] = None,
) -> None:
    """Set global unified query components (for backward compatibility)."""
    global _rag_chain, _sql_chain, _classifier, _acl_filter, _table_acl, _audit_logger
    if rag_chain is not None:
        _rag_chain = rag_chain
    if sql_chain is not None:
        _sql_chain = sql_chain
    if classifier is not None:
        _classifier = classifier
    if acl_filter is not None:
        _acl_filter = acl_filter
    if table_acl is not None:
        _table_acl = table_acl
    if audit_logger is not None:
        _audit_logger = audit_logger


def get_classifier() -> QuestionClassifier:
    """Get the question classifier."""
    if _classifier is None:
        return QuestionClassifier(RouterConfig())
    return _classifier


def get_rag_chain() -> Optional["RAGChain"]:
    """Get the RAG chain if available."""
    return _rag_chain


def get_sql_chain() -> Optional["SQLChain"]:
    """Get the SQL chain if available."""
    return _sql_chain


def get_acl_filter() -> ACLFilter:
    """Get the ACL filter."""
    if _acl_filter is None:
        return ACLFilter(ACLConfig())
    return _acl_filter


def get_table_acl() -> TableACL:
    """Get the table ACL."""
    if _table_acl is None:
        return TableACL(TableACLConfig())
    return _table_acl


def get_audit_logger() -> AuditLogger:
    """Get the audit logger."""
    if _audit_logger is None:
        return AuditLogger(enabled=False)
    return _audit_logger


@router.post("/unified-query", response_model=UnifiedQueryResponse)
async def unified_query(request: UnifiedQueryRequest) -> UnifiedQueryResponse:
    """Query the system with automatic routing.

    Automatically classifies the question and routes to:
    - Document search (RAG) for document-centric questions
    - SQL query for structured data questions
    - Both for hybrid questions

    Uses the LangGraph-based orchestrator for execution.
    Optionally applies ACL filtering based on security context.
    """
    orchestrator = get_orchestrator()
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Unified query not available. Orchestrator not initialized.",
        )

    # Validate force_route if provided
    if request.force_route:
        valid_routes = {"documents", "structured", "hybrid"}
        if request.force_route not in valid_routes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid force_route: {request.force_route}. Must be one of: {valid_routes}",
            )

    # Validate search_mode
    if request.search_mode not in ("vector", "lexical", "hybrid"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid search_mode: {request.search_mode}. Must be 'vector', 'lexical', or 'hybrid'.",
        )

    # Build security context
    security_context = None
    if request.security_context:
        security_context = SecurityContext(
            user_id=request.security_context.user_id,
            roles=request.security_context.roles,
        )
    else:
        security_context = SecurityContext(user_id="anonymous", roles=[])

    try:
        # Execute through the orchestrator
        state = await orchestrator.aquery(
            question=request.question,
            security_context=security_context,
            k=request.k,
            search_mode=request.search_mode,
            force_route=request.force_route,
            rerank=request.rerank,
        )

        # Check for fatal errors
        if has_fatal_errors(state):
            errors = state.get("errors", [])
            error_messages = [e.message for e in errors if not e.recoverable]
            raise HTTPException(
                status_code=500,
                detail=f"Query failed: {'; '.join(error_messages)}",
            )

        # Convert graph state to API response
        return graph_state_to_response(state)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unified query failed: %s", request.question[:50])
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {e}",
        ) from e
