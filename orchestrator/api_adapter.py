"""API adapter for converting GraphState to API response schemas.

This module provides functions to convert the orchestrator's GraphState
into the Pydantic response models used by the API.
"""

from __future__ import annotations

from typing import Optional

from api.schemas import (
    RoutingMetadata,
    SourceInfo,
    SQLResultData,
    UnifiedQueryMetadata,
    UnifiedQueryResponse,
)

from .state import Evidence, GraphState


def graph_state_to_response(state: GraphState) -> UnifiedQueryResponse:
    """Convert a GraphState to UnifiedQueryResponse.

    Extracts the relevant information from the graph state and builds
    the API response format.

    Args:
        state: Final graph state from orchestrator execution.

    Returns:
        UnifiedQueryResponse suitable for API return.
    """
    # Build routing metadata
    route_decision = state.get("route_decision")
    force_route = state.get("force_route")

    if route_decision:
        routing_metadata = RoutingMetadata(
            query_type=route_decision.query_type.value,
            confidence=route_decision.confidence,
            reasoning=route_decision.reasoning,
            routing_time_ms=round(state.get("routing_time_ms", 0.0), 2),
            forced=force_route is not None,
        )
    else:
        # Routing failed or was forced
        routing_metadata = RoutingMetadata(
            query_type=force_route or "unknown",
            confidence=1.0 if force_route else 0.0,
            reasoning=f"Forced to {force_route}" if force_route else "Routing failed",
            routing_time_ms=round(state.get("routing_time_ms", 0.0), 2),
            forced=force_route is not None,
        )

    # Extract sources from docs evidence
    sources = []
    docs_evidence = _get_evidence_by_source(state, "docs")
    if docs_evidence:
        sources = _build_sources_from_evidence(docs_evidence)

    # Extract SQL result from sql evidence
    sql_result = None
    sql_evidence = _get_evidence_by_source(state, "sql")
    if sql_evidence:
        sql_result = _build_sql_result_from_evidence(sql_evidence)

    # Determine search mode and reranking
    search_mode = state.get("search_mode")
    rerank = state.get("rerank")

    # Get tables used from SQL evidence
    tables_used = sql_evidence.tables_used if sql_evidence else []

    # Build metadata
    metadata = UnifiedQueryMetadata(
        routing=routing_metadata,
        retrieval_time_ms=round(state.get("retrieval_time_ms", 0.0), 2),
        generation_time_ms=round(state.get("generation_time_ms", 0.0), 2),
        search_mode=search_mode,
        reranking_applied=bool(rerank),
        tables_used=tables_used,
    )

    return UnifiedQueryResponse(
        answer=state.get("final_answer", ""),
        sources=sources,
        sql_result=sql_result,
        metadata=metadata,
    )


def _get_evidence_by_source(state: GraphState, source: str) -> Optional[Evidence]:
    """Get evidence by source type.

    Args:
        state: Graph state.
        source: Source type ("docs" or "sql").

    Returns:
        Evidence object if found, None otherwise.
    """
    evidence_list = state.get("evidence", [])
    for evidence in evidence_list:
        if evidence.source == source:
            return evidence
    return None


def _build_sources_from_evidence(evidence: Evidence) -> list[SourceInfo]:
    """Build SourceInfo list from docs evidence.

    Args:
        evidence: Docs evidence object.

    Returns:
        List of SourceInfo objects.
    """
    sources = []

    for i, metadata in enumerate(evidence.metadatas):
        doc_id = metadata.get("doc_id", "unknown")
        relative_path = metadata.get("relative_path", doc_id)
        chunk_id = evidence.chunk_ids[i] if i < len(evidence.chunk_ids) else f"{doc_id}::chunk-{i:04d}"
        page = metadata.get("page")

        # Build snippet from chunk text
        snippet = ""
        if i < len(evidence.chunks):
            chunk_text = evidence.chunks[i]
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

    return sources


def _build_sql_result_from_evidence(evidence: Evidence) -> SQLResultData:
    """Build SQLResultData from sql evidence.

    Args:
        evidence: SQL evidence object.

    Returns:
        SQLResultData object.
    """
    return SQLResultData(
        data=evidence.sql_data,
        columns=evidence.sql_columns,
        row_count=len(evidence.sql_data),
        generated_sql=evidence.sql_query,
    )


def extract_errors_from_state(state: GraphState) -> list[dict]:
    """Extract error information from graph state.

    Useful for debugging and error reporting.

    Args:
        state: Graph state.

    Returns:
        List of error dictionaries.
    """
    errors = state.get("errors", [])
    return [error.to_dict() for error in errors]


def get_node_status_summary(state: GraphState) -> dict[str, str]:
    """Get a summary of node execution status.

    Args:
        state: Graph state.

    Returns:
        Dictionary mapping node names to status strings.
    """
    return state.get("node_status", {})


def has_fatal_errors(state: GraphState) -> bool:
    """Check if state contains non-recoverable errors.

    Args:
        state: Graph state.

    Returns:
        True if there are non-recoverable errors.
    """
    errors = state.get("errors", [])
    return any(not error.recoverable for error in errors)
