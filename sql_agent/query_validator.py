"""Query validation with SELECT-only guardrails.

This module provides validation for generated SQL queries to ensure
only safe, read-only operations are executed.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Set, Tuple

from .config import SQLAgentConfig
from .errors import QueryValidationError

logger = logging.getLogger(__name__)

# Forbidden SQL keywords (case-insensitive)
FORBIDDEN_KEYWORDS = {
    # Data modification
    "INSERT",
    "UPDATE",
    "DELETE",
    "UPSERT",
    "MERGE",
    # Schema modification
    "CREATE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    # Other dangerous operations
    "GRANT",
    "REVOKE",
    "VACUUM",
    "ATTACH",
    "DETACH",
    "COPY",
    "EXPORT",
    "IMPORT",
    "LOAD",
    "INSTALL",
    "CALL",
    "SET",
    "PRAGMA",
}

# Additional forbidden patterns (for more complex detection)
FORBIDDEN_PATTERNS = [
    # INTO clause (used with INSERT, SELECT INTO)
    r"\bINTO\s+\w+",
    # Multiple statements (semicolon followed by SQL keyword)
    r";\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)",
    # SQL comments that could hide malicious code
    r"--",
    r"/\*",
    # File operations
    r"read_csv\s*\(",
    r"read_parquet\s*\(",
    r"read_json\s*\(",
    r"write_csv\s*\(",
    r"COPY\s+",
]


class QueryValidator:
    """Validates SQL queries to ensure SELECT-only operations.

    This class implements multiple layers of validation:
    1. Keyword blocklist (INSERT, UPDATE, DELETE, etc.)
    2. Pattern matching for complex attacks
    3. Table name validation against excluded tables
    4. Automatic LIMIT injection for safety
    """

    def __init__(self, config: Optional[SQLAgentConfig] = None) -> None:
        """Initialize the query validator.

        Args:
            config: SQLAgentConfig with validation settings.
        """
        self.config = config or SQLAgentConfig()
        self._excluded_tables: Set[str] = set(
            t.lower() for t in self.config.excluded_tables
        )
        # Compile forbidden patterns
        self._forbidden_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in FORBIDDEN_PATTERNS
        ]

    def validate(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate a SQL query.

        Args:
            sql: SQL query string to validate.

        Returns:
            Tuple of (is_valid, error_message).
            If valid, error_message is None.
        """
        if not sql or not sql.strip():
            return False, "Empty query"

        # Normalize whitespace
        normalized = " ".join(sql.split())

        # Check for forbidden keywords
        error = self._check_forbidden_keywords(normalized)
        if error:
            return False, error

        # Check for forbidden patterns
        error = self._check_forbidden_patterns(normalized)
        if error:
            return False, error

        # Check query starts with SELECT or WITH (CTE)
        error = self._check_query_type(normalized)
        if error:
            return False, error

        # Check for excluded tables
        error = self._check_excluded_tables(normalized)
        if error:
            return False, error

        return True, None

    def validate_or_raise(self, sql: str) -> str:
        """Validate a SQL query and raise if invalid.

        Args:
            sql: SQL query string to validate.

        Returns:
            The validated (and possibly modified) SQL query.

        Raises:
            QueryValidationError: If the query is invalid.
        """
        is_valid, error = self.validate(sql)
        if not is_valid:
            raise QueryValidationError(f"Query validation failed: {error}")

        # Optionally add LIMIT if missing
        if self.config.auto_limit:
            sql = self._ensure_limit(sql)

        return sql

    def _check_forbidden_keywords(self, sql: str) -> Optional[str]:
        """Check for forbidden SQL keywords.

        Args:
            sql: Normalized SQL query.

        Returns:
            Error message if forbidden keyword found, None otherwise.
        """
        # Extract words from query
        words = set(re.findall(r"\b([A-Za-z_]+)\b", sql.upper()))

        forbidden_found = words & FORBIDDEN_KEYWORDS
        if forbidden_found:
            return f"Forbidden keyword(s): {', '.join(sorted(forbidden_found))}"

        return None

    def _check_forbidden_patterns(self, sql: str) -> Optional[str]:
        """Check for forbidden SQL patterns.

        Args:
            sql: Normalized SQL query.

        Returns:
            Error message if forbidden pattern found, None otherwise.
        """
        for pattern in self._forbidden_patterns:
            if pattern.search(sql):
                return f"Forbidden pattern detected: {pattern.pattern}"

        return None

    def _check_query_type(self, sql: str) -> Optional[str]:
        """Check that query starts with SELECT or WITH.

        Args:
            sql: Normalized SQL query.

        Returns:
            Error message if not a SELECT query, None otherwise.
        """
        stripped = sql.strip().upper()

        # Allow SELECT or WITH (for CTEs)
        if stripped.startswith("SELECT") or stripped.startswith("WITH"):
            return None

        # Find what it starts with
        first_word = stripped.split()[0] if stripped.split() else "empty"
        return f"Query must start with SELECT or WITH, found: {first_word}"

    def _check_excluded_tables(self, sql: str) -> Optional[str]:
        """Check for references to excluded tables.

        Args:
            sql: Normalized SQL query.

        Returns:
            Error message if excluded table found, None otherwise.
        """
        sql_lower = sql.lower()

        for table in self._excluded_tables:
            # Check for table in FROM, JOIN, or quoted identifiers
            patterns = [
                rf"\bFROM\s+{re.escape(table)}\b",
                rf"\bJOIN\s+{re.escape(table)}\b",
                rf'"{re.escape(table)}"',
                rf"'{re.escape(table)}'",
            ]
            for pattern in patterns:
                if re.search(pattern, sql_lower, re.IGNORECASE):
                    return f"Access to table '{table}' is not allowed"

        return None

    def _ensure_limit(self, sql: str) -> str:
        """Add LIMIT clause if missing.

        Args:
            sql: SQL query.

        Returns:
            SQL query with LIMIT clause added if it was missing.
        """
        # Check if LIMIT already exists
        if re.search(r"\bLIMIT\s+\d+", sql, re.IGNORECASE):
            return sql

        # Strip trailing semicolon if present
        sql = sql.rstrip().rstrip(";")

        # Add LIMIT
        sql = f"{sql} LIMIT {self.config.default_limit}"
        logger.debug("Added LIMIT %d to query", self.config.default_limit)

        return sql

    def extract_tables(self, sql: str) -> List[str]:
        """Extract table names from a SQL query.

        Args:
            sql: SQL query.

        Returns:
            List of table names referenced in the query.
        """
        tables = set()

        # Match FROM and JOIN clauses
        # This is a simplified extraction and may not catch all cases
        patterns = [
            r"\bFROM\s+([\"']?[\w.]+[\"']?)",
            r"\bJOIN\s+([\"']?[\w.]+[\"']?)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            for match in matches:
                # Remove quotes
                table = match.strip("\"'")
                tables.add(table)

        return sorted(tables)
