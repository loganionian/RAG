"""State models for the orchestrator graph.

This module defines the GraphState TypedDict and supporting dataclasses
for tracking execution state through the LangGraph state machine.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from router.models import RouteDecision
from security.config import SecurityContext

from .config import OrchestratorConfig


class ExecutionStatus(Enum):
    """Status of a node's execution.

    Tracks the lifecycle of each node in the graph.

    Attributes:
        PENDING: Node has not started execution.
        IN_PROGRESS: Node is currently executing.
        SUCCESS: Node completed successfully.
        FAILED: Node encountered an error.
        SKIPPED: Node was skipped (e.g., wrong route).
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeError:
    """Captures error information from a node.

    Stores detailed error context for debugging and traceability.

    Attributes:
        node_name: Name of the node that failed.
        error_type: Exception class name.
        message: Error message.
        traceback_str: Full traceback as string for debugging.
        recoverable: Whether execution can continue despite this error.
    """

    node_name: str
    error_type: str
    message: str
    traceback_str: str
    recoverable: bool = True

    @classmethod
    def from_exception(
        cls,
        node_name: str,
        exc: Exception,
        recoverable: bool = True,
    ) -> "NodeError":
        """Create NodeError from an exception.

        Args:
            node_name: Name of the node that raised the exception.
            exc: The caught exception.
            recoverable: Whether the error allows continued execution.

        Returns:
            NodeError instance with extracted exception details.
        """
        return cls(
            node_name=node_name,
            error_type=type(exc).__name__,
            message=str(exc),
            traceback_str=traceback.format_exc(),
            recoverable=recoverable,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_name": self.node_name,
            "error_type": self.error_type,
            "message": self.message,
            "traceback": self.traceback_str,
            "recoverable": self.recoverable,
        }


@dataclass
class Evidence:
    """Evidence retrieved by an agent.

    Represents either document chunks or SQL query results.

    Attributes:
        source: Source of evidence ("docs" or "sql").
        chunks: Retrieved document chunks (for docs agent).
        metadatas: Chunk metadata (for docs agent).
        chunk_ids: Chunk identifiers (for docs agent).
        distances: Retrieval distances/scores (for docs agent).
        sql_data: Query result rows (for SQL agent).
        sql_columns: Column names (for SQL agent).
        sql_query: Generated SQL query (for SQL agent).
        tables_used: Tables referenced in SQL query (for SQL agent).
    """

    source: str
    chunks: List[str] = field(default_factory=list)
    metadatas: List[Dict[str, Any]] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    sql_data: List[Dict[str, Any]] = field(default_factory=list)
    sql_columns: List[str] = field(default_factory=list)
    sql_query: Optional[str] = None
    tables_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "chunks": self.chunks,
            "metadatas": self.metadatas,
            "chunk_ids": self.chunk_ids,
            "distances": self.distances,
            "sql_data": self.sql_data,
            "sql_columns": self.sql_columns,
            "sql_query": self.sql_query,
            "tables_used": self.tables_used,
        }


@dataclass
class PartialAnswer:
    """Partial answer from a single agent.

    Each agent produces a partial answer that the synthesizer combines.

    Attributes:
        source: Which agent produced this answer ("docs" or "sql").
        answer: The generated answer text.
        confidence: Confidence score for this answer (0.0 to 1.0).
        timing_ms: Time taken to generate this answer in milliseconds.
    """

    source: str
    answer: str
    confidence: float = 1.0
    timing_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "answer": self.answer,
            "confidence": self.confidence,
            "timing_ms": self.timing_ms,
        }


@dataclass
class CritiqueResult:
    """Result from the critic node.

    Placeholder for future critic evaluation.

    Attributes:
        passed: Whether the answer passed evaluation.
        feedback: Feedback for improvement (if not passed).
        revision_number: Current revision iteration.
    """

    passed: bool = True
    feedback: str = ""
    revision_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "feedback": self.feedback,
            "revision_number": self.revision_number,
        }


class GraphState(TypedDict, total=False):
    """Main state for the orchestrator graph.

    This TypedDict defines all state that flows through the LangGraph.
    Uses total=False to allow partial updates.

    Attributes:
        question: The user's original question.
        security_context: Security context for ACL filtering.
        config: Orchestrator configuration.
        route_decision: Result from the router node.
        evidence: List of evidence from retrieval agents.
        partial_answers: List of partial answers from agents.
        final_answer: Synthesized final answer.
        critique: Result from critic evaluation (if enabled).
        errors: List of errors encountered during execution.
        node_status: Mapping of node names to execution status.
        k: Number of chunks to retrieve.
        search_mode: Search mode for document queries.
        max_rows: Maximum rows for SQL queries.
        rerank: Whether to apply reranking.
        force_route: Forced routing override (if any).
        retrieval_time_ms: Total retrieval time in milliseconds.
        generation_time_ms: Total generation time in milliseconds.
        routing_time_ms: Time spent on routing in milliseconds.
    """

    # Input
    question: str
    security_context: Optional[SecurityContext]
    config: OrchestratorConfig

    # Routing
    route_decision: Optional[RouteDecision]
    force_route: Optional[str]

    # Retrieval settings
    k: int
    search_mode: str
    max_rows: int
    rerank: Optional[bool]

    # Evidence and answers
    evidence: List[Evidence]
    partial_answers: List[PartialAnswer]
    final_answer: str

    # Critic (placeholder)
    critique: Optional[CritiqueResult]

    # Error tracking
    errors: List[NodeError]
    node_status: Dict[str, str]

    # Timing
    retrieval_time_ms: float
    generation_time_ms: float
    routing_time_ms: float


def create_initial_state(
    question: str,
    security_context: Optional[SecurityContext] = None,
    config: Optional[OrchestratorConfig] = None,
    k: Optional[int] = None,
    search_mode: Optional[str] = None,
    max_rows: Optional[int] = None,
    rerank: Optional[bool] = None,
    force_route: Optional[str] = None,
) -> GraphState:
    """Create an initial graph state for a query.

    Args:
        question: The user's question.
        security_context: Optional security context for ACL.
        config: Orchestrator configuration.
        k: Number of chunks to retrieve.
        search_mode: Search mode for document queries.
        max_rows: Maximum rows for SQL queries.
        rerank: Whether to apply reranking.
        force_route: Force routing to specific type.

    Returns:
        Initialized GraphState ready for graph execution.
    """
    cfg = config or OrchestratorConfig()

    return GraphState(
        question=question,
        security_context=security_context,
        config=cfg,
        route_decision=None,
        force_route=force_route,
        k=k if k is not None else cfg.default_k,
        search_mode=search_mode if search_mode is not None else cfg.default_search_mode,
        max_rows=max_rows if max_rows is not None else cfg.default_max_rows,
        rerank=rerank,
        evidence=[],
        partial_answers=[],
        final_answer="",
        critique=None,
        errors=[],
        node_status={
            "router": ExecutionStatus.PENDING.value,
            "docs_agent": ExecutionStatus.PENDING.value,
            "sql_agent": ExecutionStatus.PENDING.value,
            "synthesizer": ExecutionStatus.PENDING.value,
            "critic": ExecutionStatus.PENDING.value,
        },
        retrieval_time_ms=0.0,
        generation_time_ms=0.0,
        routing_time_ms=0.0,
    )
