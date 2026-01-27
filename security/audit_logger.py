"""Audit logger for security events.

This module provides audit logging for routing decisions and access control
events, writing to an append-only JSONL file.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import SecurityContext

logger = logging.getLogger(__name__)

# Default log directory
DEFAULT_LOG_DIR = Path("data/logs")


@dataclass
class AuditEvent:
    """Base class for audit events."""

    event_type: str
    timestamp: str
    user_id: str
    roles: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class RouteDecisionEvent(AuditEvent):
    """Audit event for query routing decisions."""

    query_type: str
    confidence: float
    reasoning: str
    question_preview: str  # First 100 chars of question
    force_route: Optional[str] = None


@dataclass
class AccessDeniedEvent(AuditEvent):
    """Audit event for access denied events."""

    resource_type: str  # "document" or "table"
    resource_id: str
    reason: str


@dataclass
class QueryExecutedEvent(AuditEvent):
    """Audit event for successful query execution."""

    query_type: str  # "documents", "structured", "hybrid"
    search_mode: Optional[str] = None
    tables_accessed: Optional[List[str]] = None
    chunks_retrieved: int = 0
    execution_time_ms: float = 0.0


class AuditLogger:
    """Logs security and routing events to JSONL file.

    Events are appended to a JSONL (JSON Lines) file for easy parsing
    and analysis. Each line is a complete JSON object.
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_filename: str = "audit.jsonl",
        enabled: bool = True,
    ) -> None:
        """Initialize the audit logger.

        Args:
            log_dir: Directory for log files. Defaults to data/logs.
            log_filename: Name of the log file.
            enabled: Whether audit logging is enabled.
        """
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        self.log_filename = log_filename
        self.enabled = enabled
        self._log_path: Optional[Path] = None

    def _get_log_path(self) -> Path:
        """Get or create the log file path."""
        if self._log_path is None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = self.log_dir / self.log_filename
        return self._log_path

    def _write_event(self, event: AuditEvent) -> None:
        """Write an event to the log file.

        Args:
            event: Audit event to write.
        """
        if not self.enabled:
            return

        try:
            log_path = self._get_log_path()
            event_dict = event.to_dict()

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_dict, default=str) + "\n")

        except Exception as e:
            logger.warning("Failed to write audit event: %s", e)

    def _now_iso(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def log_route_decision(
        self,
        security_context: SecurityContext,
        query_type: str,
        confidence: float,
        reasoning: str,
        question: str,
        force_route: Optional[str] = None,
    ) -> None:
        """Log a routing decision.

        Args:
            security_context: Security context for the request.
            query_type: Classified query type (documents, structured, hybrid).
            confidence: Classification confidence score (0.0-1.0).
            reasoning: Human-readable reasoning for the classification.
            question: The user's question (will be truncated).
            force_route: If routing was forced, the forced route.
        """
        event = RouteDecisionEvent(
            event_type="route_decision",
            timestamp=self._now_iso(),
            user_id=security_context.user_id,
            roles=security_context.roles,
            query_type=query_type,
            confidence=confidence,
            reasoning=reasoning,
            question_preview=question[:100] + ("..." if len(question) > 100 else ""),
            force_route=force_route,
        )
        self._write_event(event)

        logger.debug(
            "Route decision: user=%s, type=%s, confidence=%.2f",
            security_context.user_id,
            query_type,
            confidence,
        )

    def log_access_denied(
        self,
        security_context: SecurityContext,
        resource_type: str,
        resource_id: str,
        reason: str,
    ) -> None:
        """Log an access denied event.

        Args:
            security_context: Security context for the request.
            resource_type: Type of resource ("document" or "table").
            resource_id: Identifier of the resource.
            reason: Reason for denial.
        """
        event = AccessDeniedEvent(
            event_type="access_denied",
            timestamp=self._now_iso(),
            user_id=security_context.user_id,
            roles=security_context.roles,
            resource_type=resource_type,
            resource_id=resource_id,
            reason=reason,
        )
        self._write_event(event)

        logger.warning(
            "Access denied: user=%s, resource=%s/%s, reason=%s",
            security_context.user_id,
            resource_type,
            resource_id,
            reason,
        )

    def log_query_executed(
        self,
        security_context: SecurityContext,
        query_type: str,
        search_mode: Optional[str] = None,
        tables_accessed: Optional[List[str]] = None,
        chunks_retrieved: int = 0,
        execution_time_ms: float = 0.0,
    ) -> None:
        """Log a successful query execution.

        Args:
            security_context: Security context for the request.
            query_type: Type of query executed.
            search_mode: Search mode used (for document queries).
            tables_accessed: Tables accessed (for SQL queries).
            chunks_retrieved: Number of chunks retrieved.
            execution_time_ms: Total execution time in milliseconds.
        """
        event = QueryExecutedEvent(
            event_type="query_executed",
            timestamp=self._now_iso(),
            user_id=security_context.user_id,
            roles=security_context.roles,
            query_type=query_type,
            search_mode=search_mode,
            tables_accessed=tables_accessed,
            chunks_retrieved=chunks_retrieved,
            execution_time_ms=execution_time_ms,
        )
        self._write_event(event)

        logger.debug(
            "Query executed: user=%s, type=%s, time=%.2fms",
            security_context.user_id,
            query_type,
            execution_time_ms,
        )
