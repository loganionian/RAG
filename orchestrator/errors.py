"""Custom exceptions for the orchestrator.

This module defines exception classes for orchestrator-specific errors,
providing clear error boundaries between routing, retrieval, and synthesis.
"""

from __future__ import annotations


class OrchestratorError(Exception):
    """Base exception for orchestrator errors.

    All orchestrator-specific exceptions inherit from this base class,
    allowing for catch-all handling when needed.
    """

    pass


class RoutingError(OrchestratorError):
    """Error during question classification/routing.

    Raised when the router fails to classify a question or encounters
    an invalid routing configuration.
    """

    pass


class RetrievalError(OrchestratorError):
    """Error during document or SQL retrieval.

    Raised when a retrieval agent (docs or SQL) fails to fetch results.
    Contains information about which agent failed and the underlying cause.
    """

    def __init__(
        self,
        message: str,
        agent: str = "unknown",
        recoverable: bool = True,
    ) -> None:
        """Initialize retrieval error.

        Args:
            message: Error description.
            agent: Name of the agent that failed ("docs" or "sql").
            recoverable: Whether the error is recoverable (can continue with other agents).
        """
        super().__init__(message)
        self.agent = agent
        self.recoverable = recoverable


class SynthesisError(OrchestratorError):
    """Error during answer synthesis.

    Raised when the synthesizer fails to combine agent outputs
    into a coherent response.
    """

    pass
