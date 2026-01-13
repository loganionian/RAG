"""DuckDB connection management for tabular data storage.

This module provides configuration and connection management for DuckDB,
enabling SQL-based storage and querying of tabular spreadsheet data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

logger = logging.getLogger(__name__)


@dataclass
class SQLStoreConfig:
    """Configuration for SQL tabular storage.

    Attributes:
        db_path: Path to the DuckDB database file.
        read_only: Open database in read-only mode.
        memory_limit: Memory limit for DuckDB operations (e.g., "4GB").
        threads: Number of threads for parallel execution.
    """

    db_path: Path
    read_only: bool = False
    memory_limit: str = "4GB"
    threads: int = 4

    def __post_init__(self) -> None:
        """Ensure db_path is a Path object."""
        if isinstance(self.db_path, str):
            self.db_path = Path(self.db_path)


class SQLStore:
    """DuckDB connection manager for tabular data storage.

    Provides context management, table operations, and query execution
    for the tabular ingestion pipeline.
    """

    def __init__(self, config: SQLStoreConfig) -> None:
        """Initialize the SQL store.

        Args:
            config: SQLStoreConfig with database path and settings.
        """
        self.config = config
        self._connection: Optional[duckdb.DuckDBPyConnection] = None

        # Ensure parent directory exists
        if not config.read_only:
            config.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Get or create a database connection.

        Returns:
            Active DuckDB connection.
        """
        if self._connection is None:
            self._connection = duckdb.connect(
                str(self.config.db_path),
                read_only=self.config.read_only,
            )
            # Configure DuckDB settings
            self._connection.execute(f"SET memory_limit='{self.config.memory_limit}'")
            self._connection.execute(f"SET threads={self.config.threads}")
            logger.debug(
                "Connected to DuckDB at %s (read_only=%s)",
                self.config.db_path,
                self.config.read_only,
            )
        return self._connection

    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.debug("Closed DuckDB connection")

    def execute(
        self, query: str, params: Optional[tuple] = None
    ) -> duckdb.DuckDBPyRelation:
        """Execute a SQL query.

        Args:
            query: SQL query string.
            params: Optional tuple of parameters for parameterized queries.

        Returns:
            DuckDB relation result.
        """
        conn = self.connect()
        if params:
            return conn.execute(query, params)
        return conn.execute(query)

    def fetchall(self, query: str, params: Optional[tuple] = None) -> List[tuple]:
        """Execute a query and fetch all results.

        Args:
            query: SQL query string.
            params: Optional tuple of parameters.

        Returns:
            List of result tuples.
        """
        result = self.execute(query, params)
        return result.fetchall()

    def fetchone(self, query: str, params: Optional[tuple] = None) -> Optional[tuple]:
        """Execute a query and fetch one result.

        Args:
            query: SQL query string.
            params: Optional tuple of parameters.

        Returns:
            Single result tuple or None.
        """
        result = self.execute(query, params)
        return result.fetchone()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: Name of the table to check.

        Returns:
            True if the table exists, False otherwise.
        """
        query = """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = ?
        """
        result = self.fetchone(query, (table_name,))
        return result is not None and result[0] > 0

    def drop_table(self, table_name: str) -> None:
        """Drop a table if it exists.

        Args:
            table_name: Name of the table to drop.
        """
        # Use identifier quoting to handle special characters
        self.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        logger.info("Dropped table: %s", table_name)

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get the schema of a table.

        Args:
            table_name: Name of the table.

        Returns:
            List of column dictionaries with name, type, and nullable info.
        """
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = ?
            ORDER BY ordinal_position
        """
        rows = self.fetchall(query, (table_name,))
        return [
            {"name": row[0], "type": row[1], "nullable": row[2] == "YES"}
            for row in rows
        ]

    def get_table_count(self, table_name: str) -> int:
        """Get the row count of a table.

        Args:
            table_name: Name of the table.

        Returns:
            Number of rows in the table.
        """
        result = self.fetchone(f'SELECT COUNT(*) FROM "{table_name}"')
        return result[0] if result else 0

    def list_tables(self) -> List[str]:
        """List all user tables in the database.

        Returns:
            List of table names (excluding system tables).
        """
        query = """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'main'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """
        rows = self.fetchall(query)
        return [row[0] for row in rows]

    def __enter__(self) -> "SQLStore":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
