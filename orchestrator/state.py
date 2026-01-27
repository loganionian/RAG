"""State models for the orchestrator graph.

This module defines the GraphState TypedDict and supporting dataclasses
for tracking execution state through the LangGraph state machine.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypedDict

from router.models import RouteDecision
from security.config import SecurityContext

from .config import OrchestratorConfig

logger = logging.getLogger(__name__)


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


@dataclass
class ValidationResult:
    """Result of a state validation check.

    Attributes:
        passed: Whether the validation passed.
        message: Description of what was validated or why it failed.
        recoverable: Whether execution can continue despite failure.
    """

    passed: bool
    message: str
    recoverable: bool = True

    def to_node_error(self, node_name: str) -> NodeError:
        """Convert validation failure to NodeError.

        Args:
            node_name: Name of the node that failed validation.

        Returns:
            NodeError representing the validation failure.
        """
        return NodeError(
            node_name=node_name,
            error_type="ValidationError",
            message=self.message,
            traceback_str="",
            recoverable=self.recoverable,
        )


class StateValidator:
    """Validates graph state at node entry and exit points.

    Performs validation checks to ensure state integrity throughout
    the orchestration flow. Validation failures are recoverable by
    default - they log warnings and add errors but don't abort execution.

    Validation rules:
    - Router entry: question must be non-empty
    - Router exit: route_decision must be set
    - Docs agent exit: evidence must be added if execution was successful
    - SQL agent exit: evidence must be added if execution was successful
    - Synthesizer exit: final_answer must be non-empty
    """

    def __init__(self, strict: bool = False) -> None:
        """Initialize the validator.

        Args:
            strict: If True, validation failures are non-recoverable.
        """
        self.strict = strict

    def validate_router_entry(self, state: "GraphState") -> ValidationResult:
        """Validate state before router execution.

        Checks that the question is non-empty.

        Args:
            state: Current graph state.

        Returns:
            ValidationResult indicating pass/fail.
        """
        question = state.get("question", "")
        if not question or not question.strip():
            return ValidationResult(
                passed=False,
                message="Router entry: question is empty or missing",
                recoverable=not self.strict,
            )
        return ValidationResult(passed=True, message="Router entry: valid")

    def validate_router_exit(self, state: "GraphState") -> ValidationResult:
        """Validate state after router execution.

        Checks that route_decision is set.

        Args:
            state: Current graph state.

        Returns:
            ValidationResult indicating pass/fail.
        """
        route_decision = state.get("route_decision")
        if route_decision is None:
            # Check if there's a routing error
            errors = state.get("errors", [])
            has_routing_error = any(e.node_name == "router" for e in errors)
            if has_routing_error:
                return ValidationResult(
                    passed=True,
                    message="Router exit: route_decision not set but routing error recorded",
                )
            return ValidationResult(
                passed=False,
                message="Router exit: route_decision is not set",
                recoverable=not self.strict,
            )
        return ValidationResult(passed=True, message="Router exit: valid")

    def validate_docs_agent_exit(self, state: "GraphState") -> ValidationResult:
        """Validate state after docs agent execution.

        Checks that evidence was added if the node executed successfully.

        Args:
            state: Current graph state.

        Returns:
            ValidationResult indicating pass/fail.
        """
        node_status = state.get("node_status", {})
        docs_status = node_status.get("docs_agent")

        # Skip validation if node was skipped or failed
        if docs_status in (ExecutionStatus.SKIPPED.value, ExecutionStatus.FAILED.value):
            return ValidationResult(
                passed=True,
                message=f"Docs agent exit: skipped validation (status={docs_status})",
            )

        # If successful, evidence should have been added
        if docs_status == ExecutionStatus.SUCCESS.value:
            evidence_list = state.get("evidence", [])
            has_docs_evidence = any(e.source == "docs" for e in evidence_list)
            if not has_docs_evidence:
                return ValidationResult(
                    passed=False,
                    message="Docs agent exit: node succeeded but no docs evidence added",
                    recoverable=not self.strict,
                )

        return ValidationResult(passed=True, message="Docs agent exit: valid")

    def validate_sql_agent_exit(self, state: "GraphState") -> ValidationResult:
        """Validate state after SQL agent execution.

        Checks that evidence was added if the node executed successfully.

        Args:
            state: Current graph state.

        Returns:
            ValidationResult indicating pass/fail.
        """
        node_status = state.get("node_status", {})
        sql_status = node_status.get("sql_agent")

        # Skip validation if node was skipped or failed
        if sql_status in (ExecutionStatus.SKIPPED.value, ExecutionStatus.FAILED.value):
            return ValidationResult(
                passed=True,
                message=f"SQL agent exit: skipped validation (status={sql_status})",
            )

        # If successful, evidence should have been added
        if sql_status == ExecutionStatus.SUCCESS.value:
            evidence_list = state.get("evidence", [])
            has_sql_evidence = any(e.source == "sql" for e in evidence_list)
            if not has_sql_evidence:
                return ValidationResult(
                    passed=False,
                    message="SQL agent exit: node succeeded but no sql evidence added",
                    recoverable=not self.strict,
                )

        return ValidationResult(passed=True, message="SQL agent exit: valid")

    def validate_synthesizer_exit(self, state: "GraphState") -> ValidationResult:
        """Validate state after synthesizer execution.

        Checks that final_answer is non-empty.

        Args:
            state: Current graph state.

        Returns:
            ValidationResult indicating pass/fail.
        """
        final_answer = state.get("final_answer", "")
        if not final_answer or not final_answer.strip():
            return ValidationResult(
                passed=False,
                message="Synthesizer exit: final_answer is empty",
                recoverable=not self.strict,
            )
        return ValidationResult(passed=True, message="Synthesizer exit: valid")

    def validate_entry(self, node_name: str, state: "GraphState") -> ValidationResult:
        """Validate state at node entry.

        Dispatches to the appropriate entry validation method.

        Args:
            node_name: Name of the node being entered.
            state: Current graph state.

        Returns:
            ValidationResult indicating pass/fail.
        """
        validators: Dict[str, Callable[["GraphState"], ValidationResult]] = {
            "router": self.validate_router_entry,
        }
        validator = validators.get(node_name)
        if validator:
            return validator(state)
        return ValidationResult(passed=True, message=f"{node_name} entry: no validation")

    def validate_exit(self, node_name: str, state: "GraphState") -> ValidationResult:
        """Validate state at node exit.

        Dispatches to the appropriate exit validation method.

        Args:
            node_name: Name of the node that just executed.
            state: Current graph state.

        Returns:
            ValidationResult indicating pass/fail.
        """
        validators: Dict[str, Callable[["GraphState"], ValidationResult]] = {
            "router": self.validate_router_exit,
            "docs_agent": self.validate_docs_agent_exit,
            "sql_agent": self.validate_sql_agent_exit,
            "synthesizer": self.validate_synthesizer_exit,
        }
        validator = validators.get(node_name)
        if validator:
            return validator(state)
        return ValidationResult(passed=True, message=f"{node_name} exit: no validation")


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
