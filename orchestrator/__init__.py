"""LangGraph-based RAG orchestrator package.

This package provides a LangGraph-based state machine for orchestrating
RAG (Retrieval-Augmented Generation) queries across multiple backends:
document search (RAG) and structured data queries (SQL).

Example usage:
    from orchestrator import RAGOrchestrator, OrchestratorConfig
    from generation.rag_chain import RAGChain
    from sql_agent import SQLChain

    # Initialize components
    rag_chain = RAGChain(llm_client)
    sql_chain = SQLChain(llm_client)

    # Create orchestrator
    orchestrator = RAGOrchestrator(
        rag_chain=rag_chain,
        sql_chain=sql_chain,
    )

    # Execute query
    state = orchestrator.query("What skills are in demand?")

    # Convert to API response
    from orchestrator.api_adapter import graph_state_to_response
    response = graph_state_to_response(state)
"""

from .api_adapter import (
    extract_errors_from_state,
    get_node_status_summary,
    graph_state_to_response,
    has_fatal_errors,
)
from .config import OrchestratorConfig
from .errors import OrchestratorError, RetrievalError, RoutingError, SynthesisError
from .graph import RAGOrchestrator
from .state import (
    CritiqueResult,
    Evidence,
    ExecutionStatus,
    GraphState,
    NodeError,
    PartialAnswer,
    create_initial_state,
)

__all__ = [
    # Main orchestrator
    "RAGOrchestrator",
    # Config
    "OrchestratorConfig",
    # State
    "GraphState",
    "ExecutionStatus",
    "NodeError",
    "Evidence",
    "PartialAnswer",
    "CritiqueResult",
    "create_initial_state",
    # Errors
    "OrchestratorError",
    "RoutingError",
    "RetrievalError",
    "SynthesisError",
    # API adapter
    "graph_state_to_response",
    "extract_errors_from_state",
    "get_node_status_summary",
    "has_fatal_errors",
]
