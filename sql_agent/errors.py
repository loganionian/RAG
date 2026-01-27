"""Custom exceptions for the SQL Agent.

This module defines exception classes for different failure modes in the
SQL agent pipeline: validation, generation, and execution.
"""

from __future__ import annotations


class SQLAgentError(Exception):
    """Base exception for SQL agent errors."""

    pass


class QueryValidationError(SQLAgentError):
    """Raised when a generated SQL query fails validation.

    This includes:
    - Non-SELECT queries (INSERT, UPDATE, DELETE, etc.)
    - Queries targeting excluded tables
    - Queries with forbidden operations
    """

    pass


class QueryGenerationError(SQLAgentError):
    """Raised when SQL generation fails.

    This includes:
    - LLM returned CANNOT_ANSWER
    - LLM response parsing failures
    - Invalid SQL syntax from LLM
    """

    pass


class QueryExecutionError(SQLAgentError):
    """Raised when SQL execution fails in DuckDB.

    This includes:
    - SQL syntax errors
    - Invalid table/column references
    - Execution timeouts
    """

    pass
