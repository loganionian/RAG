"""Unified query endpoint with routing and ACL support.

This module provides a unified query endpoint that automatically routes
questions to document search (RAG), SQL queries, or both based on
question classification.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, HTTPException

from api.schemas import (
    RoutingMetadata,
    SourceInfo,
    SQLResultData,
    UnifiedQueryMetadata,
    UnifiedQueryRequest,
    UnifiedQueryResponse,
)
from router import QueryType, QuestionClassifier, RouterConfig
from security import ACLConfig, ACLFilter, AuditLogger, SecurityContext, TableACL, TableACLConfig

if TYPE_CHECKING:
    from generation.rag_chain import RAGChain
    from sql_agent import SQLChain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["unified"])

# Global instances (initialized in main.py lifespan)
_rag_chain: Optional["RAGChain"] = None
_sql_chain: Optional["SQLChain"] = None
_classifier: Optional[QuestionClassifier] = None
_acl_filter: Optional[ACLFilter] = None
_table_acl: Optional[TableACL] = None
_audit_logger: Optional[AuditLogger] = None


def set_unified_components(
    rag_chain: Optional["RAGChain"] = None,
    sql_chain: Optional["SQLChain"] = None,
    classifier: Optional[QuestionClassifier] = None,
    acl_filter: Optional[ACLFilter] = None,
    table_acl: Optional[TableACL] = None,
    audit_logger: Optional[AuditLogger] = None,
) -> None:
    """Set global unified query components."""
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
        # Create a default classifier if not initialized
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

    Optionally applies ACL filtering based on security context.
    """
    total_start = time.perf_counter()
    classifier = get_classifier()
    audit = get_audit_logger()

    # Build security context
    security_context = None
    if request.security_context:
        security_context = SecurityContext(
            user_id=request.security_context.user_id,
            roles=request.security_context.roles,
        )
    else:
        # Default anonymous context
        security_context = SecurityContext(user_id="anonymous", roles=[])

    # Classify the question (or use forced route)
    routing_start = time.perf_counter()

    if request.force_route:
        # Validate forced route
        valid_routes = {"documents", "structured", "hybrid"}
        if request.force_route not in valid_routes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid force_route: {request.force_route}. Must be one of: {valid_routes}",
            )
        query_type = QueryType(request.force_route)
        route_decision = None
        routing_metadata = RoutingMetadata(
            query_type=request.force_route,
            confidence=1.0,
            reasoning=f"Forced to {request.force_route}",
            routing_time_ms=0.0,
            forced=True,
        )
    else:
        route_decision = classifier.classify(request.question)
        query_type = route_decision.query_type
        routing_time_ms = (time.perf_counter() - routing_start) * 1000
        routing_metadata = RoutingMetadata(
            query_type=route_decision.query_type.value,
            confidence=route_decision.confidence,
            reasoning=route_decision.reasoning,
            routing_time_ms=round(routing_time_ms, 2),
            forced=False,
        )

    # Log routing decision
    audit.log_route_decision(
        security_context=security_context,
        query_type=routing_metadata.query_type,
        confidence=routing_metadata.confidence,
        reasoning=routing_metadata.reasoning,
        question=request.question,
        force_route=request.force_route,
    )

    # Execute based on query type
    try:
        if query_type == QueryType.DOCUMENTS:
            return await _execute_document_query(
                request, security_context, routing_metadata, audit, total_start
            )
        elif query_type == QueryType.STRUCTURED:
            return await _execute_sql_query(
                request, security_context, routing_metadata, audit, total_start
            )
        else:  # HYBRID
            return await _execute_hybrid_query(
                request, security_context, routing_metadata, audit, total_start
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unified query failed: %s", request.question[:50])
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {e}",
        ) from e


async def _execute_document_query(
    request: UnifiedQueryRequest,
    security_context: SecurityContext,
    routing_metadata: RoutingMetadata,
    audit: AuditLogger,
    total_start: float,
) -> UnifiedQueryResponse:
    """Execute a document-centric query using RAG."""
    rag_chain = get_rag_chain()
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Document search not available. RAG system not initialized.",
        )

    acl_filter = get_acl_filter()

    # Validate search_mode
    search_mode = request.search_mode
    if search_mode not in ("vector", "lexical", "hybrid"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid search_mode: {search_mode}. Must be 'vector', 'lexical', or 'hybrid'.",
        )

    # Determine reranking
    should_rerank = (
        request.rerank if request.rerank is not None else rag_chain.config.enable_reranking
    )

    # Retrieve with optional ACL filtering
    retrieval_start = time.perf_counter()
    retrieval_result = rag_chain.retrieve(
        request.question,
        k=request.k,
        mode=search_mode,
        rerank=should_rerank,
        security_context=security_context,
        acl_filter=acl_filter,
    )
    retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

    # Generate answer
    generation_start = time.perf_counter()
    if retrieval_result.chunks:
        context = rag_chain._format_context(retrieval_result)
        system_prompt = rag_chain.config.system_prompt.format(context=context)
        answer = rag_chain.llm_client.generate(
            prompt=request.question,
            system_prompt=system_prompt,
            max_tokens=rag_chain.config.max_tokens,
            temperature=rag_chain.config.temperature,
        )
    else:
        answer = "I couldn't find any relevant information in the knowledge base."
    generation_time_ms = (time.perf_counter() - generation_start) * 1000

    # Build sources
    sources = []
    for i, metadata in enumerate(retrieval_result.metadatas):
        doc_id = metadata.get("doc_id", "unknown")
        relative_path = metadata.get("relative_path", doc_id)
        chunk_id = metadata.get("chunk_id", f"{doc_id}::chunk-{i:04d}")
        page = metadata.get("page")

        snippet = ""
        if i < len(retrieval_result.chunks):
            chunk_text = retrieval_result.chunks[i]
            snippet = chunk_text[:200].replace("\n", " ").strip()
            if len(chunk_text) > 200:
                snippet += "..."

        sources.append(
            SourceInfo(
                doc_id=doc_id,
                filename=relative_path,
                chunk_id=chunk_id,
                snippet=snippet,
                page=page,
            )
        )

    # Log execution
    total_time_ms = (time.perf_counter() - total_start) * 1000
    audit.log_query_executed(
        security_context=security_context,
        query_type="documents",
        search_mode=search_mode,
        chunks_retrieved=len(retrieval_result.chunks),
        execution_time_ms=total_time_ms,
    )

    return UnifiedQueryResponse(
        answer=answer,
        sources=sources,
        sql_result=None,
        metadata=UnifiedQueryMetadata(
            routing=routing_metadata,
            retrieval_time_ms=round(retrieval_time_ms, 2),
            generation_time_ms=round(generation_time_ms, 2),
            search_mode=search_mode,
            reranking_applied=should_rerank,
        ),
    )


async def _execute_sql_query(
    request: UnifiedQueryRequest,
    security_context: SecurityContext,
    routing_metadata: RoutingMetadata,
    audit: AuditLogger,
    total_start: float,
) -> UnifiedQueryResponse:
    """Execute a structured data query using SQL."""
    sql_chain = get_sql_chain()
    if sql_chain is None:
        raise HTTPException(
            status_code=503,
            detail="SQL query not available. SQL Agent not initialized.",
        )

    table_acl = get_table_acl()

    # Execute SQL query with table ACL
    result = sql_chain.query(
        question=request.question,
        max_rows=100,
        summarize=True,
        security_context=security_context,
        table_acl=table_acl,
    )

    # Log execution
    total_time_ms = (time.perf_counter() - total_start) * 1000
    audit.log_query_executed(
        security_context=security_context,
        query_type="structured",
        tables_accessed=result.tables_used,
        execution_time_ms=total_time_ms,
    )

    return UnifiedQueryResponse(
        answer=result.answer,
        sources=[],
        sql_result=SQLResultData(
            data=result.data,
            columns=result.columns,
            row_count=result.row_count,
            generated_sql=result.generated_sql,
        ),
        metadata=UnifiedQueryMetadata(
            routing=routing_metadata,
            retrieval_time_ms=result.execution_time_ms,
            generation_time_ms=result.generation_time_ms + result.summarization_time_ms,
            tables_used=result.tables_used,
        ),
    )


async def _execute_hybrid_query(
    request: UnifiedQueryRequest,
    security_context: SecurityContext,
    routing_metadata: RoutingMetadata,
    audit: AuditLogger,
    total_start: float,
) -> UnifiedQueryResponse:
    """Execute a hybrid query combining documents and SQL."""
    rag_chain = get_rag_chain()
    sql_chain = get_sql_chain()

    if rag_chain is None and sql_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Neither document search nor SQL query is available.",
        )

    acl_filter = get_acl_filter()
    table_acl = get_table_acl()

    # Collect results from both sources
    doc_answer = None
    doc_sources = []
    doc_time = 0.0

    sql_answer = None
    sql_result_data = None
    sql_tables = []
    sql_time = 0.0

    # Execute document query if available
    if rag_chain is not None:
        search_mode = request.search_mode
        if search_mode not in ("vector", "lexical", "hybrid"):
            search_mode = "hybrid"

        should_rerank = (
            request.rerank if request.rerank is not None else rag_chain.config.enable_reranking
        )

        retrieval_start = time.perf_counter()
        retrieval_result = rag_chain.retrieve(
            request.question,
            k=request.k,
            mode=search_mode,
            rerank=should_rerank,
            security_context=security_context,
            acl_filter=acl_filter,
        )
        doc_time = (time.perf_counter() - retrieval_start) * 1000

        if retrieval_result.chunks:
            context = rag_chain._format_context(retrieval_result)
            system_prompt = rag_chain.config.system_prompt.format(context=context)
            doc_answer = rag_chain.llm_client.generate(
                prompt=request.question,
                system_prompt=system_prompt,
                max_tokens=rag_chain.config.max_tokens,
                temperature=rag_chain.config.temperature,
            )

            # Build sources
            for i, metadata in enumerate(retrieval_result.metadatas):
                doc_id = metadata.get("doc_id", "unknown")
                relative_path = metadata.get("relative_path", doc_id)
                chunk_id = metadata.get("chunk_id", f"{doc_id}::chunk-{i:04d}")
                page = metadata.get("page")

                snippet = ""
                if i < len(retrieval_result.chunks):
                    chunk_text = retrieval_result.chunks[i]
                    snippet = chunk_text[:200].replace("\n", " ").strip()
                    if len(chunk_text) > 200:
                        snippet += "..."

                doc_sources.append(
                    SourceInfo(
                        doc_id=doc_id,
                        filename=relative_path,
                        chunk_id=chunk_id,
                        snippet=snippet,
                        page=page,
                    )
                )

    # Execute SQL query if available
    if sql_chain is not None:
        try:
            sql_start = time.perf_counter()
            sql_result = sql_chain.query(
                question=request.question,
                max_rows=100,
                summarize=True,
                security_context=security_context,
                table_acl=table_acl,
            )
            sql_time = (time.perf_counter() - sql_start) * 1000

            sql_answer = sql_result.answer
            sql_tables = sql_result.tables_used
            sql_result_data = SQLResultData(
                data=sql_result.data,
                columns=sql_result.columns,
                row_count=sql_result.row_count,
                generated_sql=sql_result.generated_sql,
            )
        except Exception as e:
            logger.warning("SQL query failed in hybrid mode: %s", e)
            # Continue with document results only

    # Combine answers
    if doc_answer and sql_answer:
        combined_answer = f"**From Documents:**\n{doc_answer}\n\n**From Data:**\n{sql_answer}"
    elif doc_answer:
        combined_answer = doc_answer
    elif sql_answer:
        combined_answer = sql_answer
    else:
        combined_answer = "I couldn't find relevant information from documents or data."

    # Log execution
    total_time_ms = (time.perf_counter() - total_start) * 1000
    audit.log_query_executed(
        security_context=security_context,
        query_type="hybrid",
        search_mode=request.search_mode,
        tables_accessed=sql_tables,
        chunks_retrieved=len(doc_sources),
        execution_time_ms=total_time_ms,
    )

    return UnifiedQueryResponse(
        answer=combined_answer,
        sources=doc_sources,
        sql_result=sql_result_data,
        metadata=UnifiedQueryMetadata(
            routing=routing_metadata,
            retrieval_time_ms=round(doc_time + sql_time, 2),
            generation_time_ms=0.0,  # Included in retrieval for hybrid
            search_mode=request.search_mode,
            tables_used=sql_tables,
        ),
    )
