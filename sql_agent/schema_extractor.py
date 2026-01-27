"""Schema extraction for SQL Agent prompts.

This module provides functionality to extract and format database schema
information for inclusion in LLM prompts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from storage.sql_store import SQLStore, SQLStoreConfig

from .config import SQLAgentConfig

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Information about a database column."""

    name: str
    data_type: str
    nullable: bool = True
    sample_values: List[Any] = None

    def __post_init__(self):
        if self.sample_values is None:
            self.sample_values = []


@dataclass
class TableInfo:
    """Information about a database table."""

    name: str
    columns: List[ColumnInfo]
    row_count: int = 0
    description: Optional[str] = None


@dataclass
class SchemaInfo:
    """Complete database schema information."""

    tables: List[TableInfo]
    total_tables: int = 0
    total_rows: int = 0


class SchemaExtractor:
    """Extracts database schema for LLM prompts.

    Connects to DuckDB and retrieves table/column information,
    optionally including sample values and descriptions from
    the MetadataCatalog.
    """

    def __init__(
        self,
        config: Optional[SQLAgentConfig] = None,
        sql_store: Optional[SQLStore] = None,
    ) -> None:
        """Initialize the schema extractor.

        Args:
            config: SQLAgentConfig with database settings.
            sql_store: Optional pre-configured SQLStore instance.
        """
        self.config = config or SQLAgentConfig()
        self._sql_store = sql_store
        self._schema_cache: Optional[SchemaInfo] = None

    def _get_sql_store(self) -> SQLStore:
        """Get or create SQLStore instance."""
        if self._sql_store is None:
            store_config = SQLStoreConfig(
                db_path=self.config.db_path,
                read_only=True,  # Always read-only for safety
            )
            self._sql_store = SQLStore(store_config)
        return self._sql_store

    def extract_schema(self, refresh: bool = False) -> SchemaInfo:
        """Extract complete database schema.

        Args:
            refresh: Force refresh of cached schema.

        Returns:
            SchemaInfo with all table and column information.
        """
        if self._schema_cache is not None and not refresh:
            return self._schema_cache

        sql_store = self._get_sql_store()

        # Get all user tables
        all_tables = sql_store.list_tables()

        # Filter out excluded tables
        excluded = set(t.lower() for t in self.config.excluded_tables)
        tables = [t for t in all_tables if t.lower() not in excluded]

        table_infos = []
        total_rows = 0

        for table_name in tables:
            try:
                table_info = self._extract_table_info(sql_store, table_name)
                table_infos.append(table_info)
                total_rows += table_info.row_count
            except Exception as e:
                logger.warning("Failed to extract schema for table '%s': %s", table_name, e)

        self._schema_cache = SchemaInfo(
            tables=table_infos,
            total_tables=len(table_infos),
            total_rows=total_rows,
        )

        logger.debug(
            "Extracted schema: %d tables, %d total rows",
            len(table_infos),
            total_rows,
        )

        return self._schema_cache

    def _extract_table_info(self, sql_store: SQLStore, table_name: str) -> TableInfo:
        """Extract information for a single table.

        Args:
            sql_store: SQLStore instance.
            table_name: Name of the table.

        Returns:
            TableInfo with columns and row count.
        """
        # Get column schema
        schema = sql_store.get_table_schema(table_name)
        row_count = sql_store.get_table_count(table_name)

        columns = []
        for col in schema:
            col_info = ColumnInfo(
                name=col["name"],
                data_type=col["type"],
                nullable=col.get("nullable", True),
            )

            # Get sample values if configured
            if self.config.max_rows_preview > 0:
                try:
                    samples = self._get_sample_values(
                        sql_store, table_name, col["name"]
                    )
                    col_info.sample_values = samples
                except Exception as e:
                    logger.debug("Could not get samples for %s.%s: %s", table_name, col["name"], e)

            columns.append(col_info)

        # Try to get description from catalog
        description = self._get_table_description(sql_store, table_name)

        return TableInfo(
            name=table_name,
            columns=columns,
            row_count=row_count,
            description=description,
        )

    def _get_sample_values(
        self, sql_store: SQLStore, table_name: str, column_name: str
    ) -> List[Any]:
        """Get sample values for a column.

        Args:
            sql_store: SQLStore instance.
            table_name: Name of the table.
            column_name: Name of the column.

        Returns:
            List of sample values (up to max_rows_preview distinct values).
        """
        try:
            # Use double quotes for identifiers
            query = f"""
                SELECT DISTINCT "{column_name}"
                FROM "{table_name}"
                WHERE "{column_name}" IS NOT NULL
                LIMIT {self.config.max_rows_preview}
            """
            rows = sql_store.fetchall(query)
            return [row[0] for row in rows]
        except Exception as e:
            logger.debug("Error getting samples: %s", e)
            return []

    def _get_table_description(self, sql_store: SQLStore, table_name: str) -> Optional[str]:
        """Get table description from metadata catalog.

        Args:
            sql_store: SQLStore instance.
            table_name: Name of the table.

        Returns:
            Description string if available, None otherwise.
        """
        try:
            # Check if catalog table exists
            if not sql_store.table_exists("_ingestion_catalog"):
                return None

            query = """
                SELECT summary FROM "_ingestion_catalog"
                WHERE table_name = ?
            """
            row = sql_store.fetchone(query, (table_name,))
            return row[0] if row and row[0] else None
        except Exception as e:
            logger.debug("Could not get description for %s: %s", table_name, e)
            return None

    def format_schema_for_prompt(self, schema: Optional[SchemaInfo] = None) -> str:
        """Format schema as markdown for LLM prompt.

        Args:
            schema: SchemaInfo to format (extracts if not provided).

        Returns:
            Formatted markdown string with schema information.
        """
        if schema is None:
            schema = self.extract_schema()

        if not schema.tables:
            return "No tables available in the database."

        lines = [
            f"Database contains {schema.total_tables} table(s) with {schema.total_rows:,} total rows.\n"
        ]

        for table in schema.tables:
            lines.append(f"### Table: `{table.name}` ({table.row_count:,} rows)")

            if table.description:
                lines.append(f"Description: {table.description}")

            lines.append("| Column | Type | Sample Values |")
            lines.append("|--------|------|---------------|")

            for col in table.columns:
                samples = ", ".join(str(v)[:30] for v in col.sample_values[:3])
                if samples:
                    samples = f"`{samples}`"
                else:
                    samples = "-"
                lines.append(f"| {col.name} | {col.data_type} | {samples} |")

            lines.append("")  # Blank line between tables

        return "\n".join(lines)

    def get_table_names(self) -> List[str]:
        """Get list of available table names.

        Returns:
            List of table names (excluding system tables).
        """
        schema = self.extract_schema()
        return [t.name for t in schema.tables]

    def get_statistics(self) -> Dict[str, Any]:
        """Get schema statistics.

        Returns:
            Dictionary with table_count and total_rows.
        """
        schema = self.extract_schema()
        return {
            "table_count": schema.total_tables,
            "total_rows": schema.total_rows,
        }

    def close(self) -> None:
        """Close database connection."""
        if self._sql_store is not None:
            self._sql_store.close()
            self._sql_store = None
        self._schema_cache = None
