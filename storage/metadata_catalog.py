"""Metadata catalog for tracking file-to-table mappings.

This module provides a catalog that tracks which source files have been
loaded into which SQL tables, enabling idempotent ingestion and discovery.
It also stores LLM-generated summaries for dataset discovery.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .sql_store import SQLStore

if TYPE_CHECKING:
    from generation.dataset_summarizer import DatasetSummary

logger = logging.getLogger(__name__)

# Catalog table name (prefixed with underscore to indicate system table)
CATALOG_TABLE_NAME = "_ingestion_catalog"

# Summary columns added in E3.5
SUMMARY_COLUMNS = [
    ("summary", "VARCHAR"),
    ("column_descriptions", "VARCHAR"),
    ("summary_generated_at", "TIMESTAMP"),
    ("summary_token_usage", "VARCHAR"),
]

# All columns for SELECT queries (in order)
ALL_COLUMNS = """table_name, source_file, sheet_name, ingestion_timestamp,
                 row_count, column_count, content_hash, domain_labels, column_schema,
                 summary, column_descriptions, summary_generated_at, summary_token_usage"""


@dataclass
class CatalogEntry:
    """Represents a table entry in the ingestion catalog.

    Attributes:
        table_name: Name of the SQL table.
        source_file: Path to the source file.
        sheet_name: Sheet name for Excel files (None for CSV).
        ingestion_timestamp: When the table was created/updated.
        row_count: Number of rows in the table.
        column_count: Number of columns in the table.
        content_hash: SHA256 hash of the source content.
        domain_labels: Optional list of domain labels for the data.
        column_schema: List of column definitions with name and type.
        summary: LLM-generated summary of the dataset.
        column_descriptions: LLM-generated descriptions for columns.
        summary_generated_at: When the summary was generated.
        summary_token_usage: Token usage for summary generation.
    """

    table_name: str
    source_file: str
    sheet_name: Optional[str]
    ingestion_timestamp: datetime
    row_count: int
    column_count: int
    content_hash: str
    domain_labels: List[str] = field(default_factory=list)
    column_schema: List[Dict[str, str]] = field(default_factory=list)
    summary: Optional[str] = None
    column_descriptions: Dict[str, str] = field(default_factory=dict)
    summary_generated_at: Optional[datetime] = None
    summary_token_usage: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary."""
        result = {
            "table_name": self.table_name,
            "source_file": self.source_file,
            "sheet_name": self.sheet_name,
            "ingestion_timestamp": self.ingestion_timestamp.isoformat(),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "content_hash": self.content_hash,
            "domain_labels": self.domain_labels,
            "column_schema": self.column_schema,
            "summary": self.summary,
            "column_descriptions": self.column_descriptions,
            "summary_generated_at": self.summary_generated_at.isoformat() if self.summary_generated_at else None,
            "summary_token_usage": self.summary_token_usage,
        }
        return result

    @property
    def has_summary(self) -> bool:
        """Check if this entry has an LLM-generated summary."""
        return self.summary is not None and len(self.summary) > 0

    @classmethod
    def from_row(cls, row: tuple) -> "CatalogEntry":
        """Create entry from database row.

        Args:
            row: Tuple of (table_name, source_file, sheet_name,
                 ingestion_timestamp, row_count, column_count,
                 content_hash, domain_labels_json, column_schema_json,
                 summary, column_descriptions_json, summary_generated_at,
                 summary_token_usage_json).

        Returns:
            CatalogEntry instance.
        """
        # Parse summary fields if present (columns 9-12)
        summary = row[9] if len(row) > 9 else None
        column_descriptions = json.loads(row[10]) if len(row) > 10 and row[10] else {}
        summary_generated_at = None
        if len(row) > 11 and row[11]:
            if isinstance(row[11], str):
                summary_generated_at = datetime.fromisoformat(row[11])
            else:
                summary_generated_at = row[11]
        summary_token_usage = json.loads(row[12]) if len(row) > 12 and row[12] else {}

        return cls(
            table_name=row[0],
            source_file=row[1],
            sheet_name=row[2],
            ingestion_timestamp=datetime.fromisoformat(row[3])
            if isinstance(row[3], str)
            else row[3],
            row_count=row[4],
            column_count=row[5],
            content_hash=row[6],
            domain_labels=json.loads(row[7]) if row[7] else [],
            column_schema=json.loads(row[8]) if row[8] else [],
            summary=summary,
            column_descriptions=column_descriptions,
            summary_generated_at=summary_generated_at,
            summary_token_usage=summary_token_usage,
        )


class MetadataCatalog:
    """Manages the _ingestion_catalog table for tracking file-to-table mappings."""

    def __init__(self, sql_store: SQLStore) -> None:
        """Initialize the metadata catalog.

        Args:
            sql_store: SQLStore instance for database operations.
        """
        self.sql_store = sql_store
        self._initialized = False

    def init_catalog(self) -> None:
        """Create the catalog table if it doesn't exist.

        In read-only mode, this only verifies the table exists without creating it.
        """
        if self._initialized:
            return

        # In read-only mode, just check if the table exists
        if self.sql_store.config.read_only:
            if self.sql_store.table_exists(CATALOG_TABLE_NAME):
                self._initialized = True
                logger.debug("Catalog table verified (read-only mode)")
            return

        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {CATALOG_TABLE_NAME} (
                table_name VARCHAR PRIMARY KEY,
                source_file VARCHAR NOT NULL,
                sheet_name VARCHAR,
                ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                content_hash VARCHAR NOT NULL,
                domain_labels VARCHAR,
                column_schema VARCHAR,
                summary VARCHAR,
                column_descriptions VARCHAR,
                summary_generated_at TIMESTAMP,
                summary_token_usage VARCHAR
            )
        """
        self.sql_store.execute(create_table_sql)

        # Migrate existing tables: add summary columns if they don't exist
        self._migrate_summary_columns()

        # Create indexes for common lookups
        self.sql_store.execute(
            f"CREATE INDEX IF NOT EXISTS idx_catalog_source "
            f"ON {CATALOG_TABLE_NAME}(source_file)"
        )
        self.sql_store.execute(
            f"CREATE INDEX IF NOT EXISTS idx_catalog_hash "
            f"ON {CATALOG_TABLE_NAME}(content_hash)"
        )

        self._initialized = True
        logger.debug("Initialized metadata catalog table")

    def _migrate_summary_columns(self) -> None:
        """Add summary columns to existing catalog tables (migration)."""
        try:
            # Check if summary column exists by trying to select it
            test_query = f"SELECT summary FROM {CATALOG_TABLE_NAME} LIMIT 1"
            self.sql_store.fetchone(test_query)
        except Exception:
            # Column doesn't exist, add it
            logger.info("Migrating catalog table: adding summary columns")
            for col_name, col_type in SUMMARY_COLUMNS:
                try:
                    alter_sql = f"ALTER TABLE {CATALOG_TABLE_NAME} ADD COLUMN {col_name} {col_type}"
                    self.sql_store.execute(alter_sql)
                    logger.debug("Added column '%s' to catalog", col_name)
                except Exception as e:
                    # Column might already exist
                    logger.debug("Column '%s' may already exist: %s", col_name, e)

    def get_entry(self, table_name: str) -> Optional[CatalogEntry]:
        """Get a catalog entry by table name.

        Args:
            table_name: Name of the table to look up.

        Returns:
            CatalogEntry if found, None otherwise.
        """
        self.init_catalog()
        query = f"""
            SELECT {ALL_COLUMNS}
            FROM {CATALOG_TABLE_NAME}
            WHERE table_name = ?
        """
        row = self.sql_store.fetchone(query, (table_name,))
        return CatalogEntry.from_row(row) if row else None

    def get_entry_by_hash(self, content_hash: str) -> Optional[CatalogEntry]:
        """Get a catalog entry by content hash.

        Args:
            content_hash: SHA256 hash of the content.

        Returns:
            CatalogEntry if found, None otherwise.
        """
        self.init_catalog()
        query = f"""
            SELECT {ALL_COLUMNS}
            FROM {CATALOG_TABLE_NAME}
            WHERE content_hash = ?
        """
        row = self.sql_store.fetchone(query, (content_hash,))
        return CatalogEntry.from_row(row) if row else None

    def is_up_to_date(self, table_name: str, content_hash: str) -> bool:
        """Check if a table is up-to-date with the given content hash.

        Args:
            table_name: Name of the table.
            content_hash: Expected content hash.

        Returns:
            True if the table exists and hash matches, False otherwise.
        """
        entry = self.get_entry(table_name)
        return entry is not None and entry.content_hash == content_hash

    def register_table(self, entry: CatalogEntry) -> None:
        """Register or update a table in the catalog.

        Args:
            entry: CatalogEntry to register.
        """
        self.init_catalog()

        # Delete existing entry first for upsert behavior (DuckDB doesn't support INSERT OR REPLACE)
        self.sql_store.execute(
            f"DELETE FROM {CATALOG_TABLE_NAME} WHERE table_name = ?",
            (entry.table_name,),
        )

        # Insert new entry
        insert_sql = f"""
            INSERT INTO {CATALOG_TABLE_NAME}
            (table_name, source_file, sheet_name, ingestion_timestamp,
             row_count, column_count, content_hash, domain_labels, column_schema)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.sql_store.execute(
            insert_sql,
            (
                entry.table_name,
                entry.source_file,
                entry.sheet_name,
                entry.ingestion_timestamp.isoformat(),
                entry.row_count,
                entry.column_count,
                entry.content_hash,
                json.dumps(entry.domain_labels),
                json.dumps(entry.column_schema),
            ),
        )
        logger.info(
            "Registered table '%s' from %s (rows=%d, cols=%d)",
            entry.table_name,
            entry.source_file,
            entry.row_count,
            entry.column_count,
        )

    def unregister_table(self, table_name: str) -> bool:
        """Remove a table from the catalog.

        Args:
            table_name: Name of the table to remove.

        Returns:
            True if the entry was removed, False if it didn't exist.
        """
        self.init_catalog()
        entry = self.get_entry(table_name)
        if entry is None:
            return False

        delete_sql = f"DELETE FROM {CATALOG_TABLE_NAME} WHERE table_name = ?"
        self.sql_store.execute(delete_sql, (table_name,))
        logger.info("Unregistered table '%s' from catalog", table_name)
        return True

    def list_tables(self) -> List[CatalogEntry]:
        """List all tables in the catalog.

        Returns:
            List of CatalogEntry objects.
        """
        self.init_catalog()
        query = f"""
            SELECT {ALL_COLUMNS}
            FROM {CATALOG_TABLE_NAME}
            ORDER BY table_name
        """
        rows = self.sql_store.fetchall(query)
        return [CatalogEntry.from_row(row) for row in rows]

    def find_by_source(self, source_file: str) -> List[CatalogEntry]:
        """Find all tables loaded from a source file.

        Args:
            source_file: Path to the source file.

        Returns:
            List of CatalogEntry objects from that file.
        """
        self.init_catalog()
        query = f"""
            SELECT {ALL_COLUMNS}
            FROM {CATALOG_TABLE_NAME}
            WHERE source_file = ?
            ORDER BY table_name
        """
        rows = self.sql_store.fetchall(query, (source_file,))
        return [CatalogEntry.from_row(row) for row in rows]

    def get_statistics(self) -> Dict[str, Any]:
        """Get catalog statistics.

        Returns:
            Dictionary with table_count, total_rows, total_columns, tables_with_summaries.
        """
        self.init_catalog()
        query = f"""
            SELECT COUNT(*), COALESCE(SUM(row_count), 0), COALESCE(SUM(column_count), 0),
                   SUM(CASE WHEN summary IS NOT NULL AND summary != '' THEN 1 ELSE 0 END)
            FROM {CATALOG_TABLE_NAME}
        """
        row = self.sql_store.fetchone(query)
        if row:
            return {
                "table_count": row[0],
                "total_rows": row[1],
                "total_columns": row[2],
                "tables_with_summaries": row[3] or 0,
            }
        return {"table_count": 0, "total_rows": 0, "total_columns": 0, "tables_with_summaries": 0}

    def update_summary(self, table_name: str, summary: "DatasetSummary") -> bool:
        """Update the summary for a table.

        Args:
            table_name: Name of the table.
            summary: DatasetSummary object with generated content.

        Returns:
            True if update was successful, False if table not found.
        """
        self.init_catalog()

        # Check table exists
        entry = self.get_entry(table_name)
        if entry is None:
            logger.warning("Cannot update summary: table '%s' not found", table_name)
            return False

        update_sql = f"""
            UPDATE {CATALOG_TABLE_NAME}
            SET summary = ?,
                column_descriptions = ?,
                summary_generated_at = ?,
                summary_token_usage = ?
            WHERE table_name = ?
        """
        self.sql_store.execute(
            update_sql,
            (
                summary.summary,
                json.dumps(summary.column_descriptions),
                summary.generated_at.isoformat(),
                json.dumps(summary.token_usage),
                table_name,
            ),
        )
        logger.info(
            "Updated summary for '%s' (tokens: %d)",
            table_name,
            summary.token_usage.get("input_tokens", 0) + summary.token_usage.get("output_tokens", 0),
        )
        return True

    def get_tables_without_summaries(self) -> List[CatalogEntry]:
        """Get all tables that don't have summaries yet.

        Returns:
            List of CatalogEntry objects without summaries.
        """
        self.init_catalog()
        query = f"""
            SELECT {ALL_COLUMNS}
            FROM {CATALOG_TABLE_NAME}
            WHERE summary IS NULL OR summary = ''
            ORDER BY table_name
        """
        rows = self.sql_store.fetchall(query)
        return [CatalogEntry.from_row(row) for row in rows]

    def get_tables_with_summaries(self) -> List[CatalogEntry]:
        """Get all tables that have summaries.

        Returns:
            List of CatalogEntry objects with summaries.
        """
        self.init_catalog()
        query = f"""
            SELECT {ALL_COLUMNS}
            FROM {CATALOG_TABLE_NAME}
            WHERE summary IS NOT NULL AND summary != ''
            ORDER BY table_name
        """
        rows = self.sql_store.fetchall(query)
        return [CatalogEntry.from_row(row) for row in rows]

    def clear_summary(self, table_name: str) -> bool:
        """Clear the summary for a table.

        Args:
            table_name: Name of the table.

        Returns:
            True if update was successful, False if table not found.
        """
        self.init_catalog()

        entry = self.get_entry(table_name)
        if entry is None:
            return False

        update_sql = f"""
            UPDATE {CATALOG_TABLE_NAME}
            SET summary = NULL,
                column_descriptions = NULL,
                summary_generated_at = NULL,
                summary_token_usage = NULL
            WHERE table_name = ?
        """
        self.sql_store.execute(update_sql, (table_name,))
        logger.info("Cleared summary for '%s'", table_name)
        return True
