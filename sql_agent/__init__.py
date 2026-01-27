"""SQL Agent package for natural language to SQL queries.

This package provides an LLM-based SQL agent that translates natural language
questions into SQL queries against a DuckDB database with SELECT-only guardrails.
"""

from .config import SQLAgentConfig
from .errors import (
    QueryExecutionError,
    QueryGenerationError,
    QueryValidationError,
    SQLAgentError,
)
from .query_validator import QueryValidator
from .schema_extractor import SchemaExtractor
from .sql_chain import SQLChain, SQLQueryResult
from .sql_generator import SQLGenerator

__all__ = [
    # Config
    "SQLAgentConfig",
    # Core classes
    "SQLChain",
    "SQLQueryResult",
    "QueryValidator",
    "SchemaExtractor",
    "SQLGenerator",
    # Errors
    "SQLAgentError",
    "QueryValidationError",
    "QueryGenerationError",
    "QueryExecutionError",
]
