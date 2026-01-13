"""Metadata catalog for tracking file-to-table mappings.

This module provides a catalog that tracks which source files have been
loaded into which SQL tables, enabling idempotent ingestion and discovery.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .sql_store import SQLStore

logger = logging.getLogger(__name__)

# Catalog table name (prefixed with underscore to indicate system table)
CATALOG_TABLE_NAME = "_ingestion_catalog"


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            "table_name": self.table_name,
            "source_file": self.source_file,
            "sheet_name": self.sheet_name,
            "ingestion_timestamp": self.ingestion_timestamp.isoformat(),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "content_hash": self.content_hash,
            "domain_labels": self.domain_labels,
            "column_schema": self.column_schema,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "CatalogEntry":
        """Create entry from database row.

        Args:
            row: Tuple of (table_name, source_file, sheet_name,
                 ingestion_timestamp, row_count, column_count,
                 content_hash, domain_labels_json, column_schema_json).

        Returns:
            CatalogEntry instance.
        """
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
                column_schema VARCHAR
            )
        """
        self.sql_store.execute(create_table_sql)

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

    def get_entry(self, table_name: str) -> Optional[CatalogEntry]:
        """Get a catalog entry by table name.

        Args:
            table_name: Name of the table to look up.

        Returns:
            CatalogEntry if found, None otherwise.
        """
        self.init_catalog()
        query = f"""
            SELECT table_name, source_file, sheet_name, ingestion_timestamp,
                   row_count, column_count, content_hash, domain_labels, column_schema
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
            SELECT table_name, source_file, sheet_name, ingestion_timestamp,
                   row_count, column_count, content_hash, domain_labels, column_schema
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
            SELECT table_name, source_file, sheet_name, ingestion_timestamp,
                   row_count, column_count, content_hash, domain_labels, column_schema
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
            SELECT table_name, source_file, sheet_name, ingestion_timestamp,
                   row_count, column_count, content_hash, domain_labels, column_schema
            FROM {CATALOG_TABLE_NAME}
            WHERE source_file = ?
            ORDER BY table_name
        """
        rows = self.sql_store.fetchall(query, (source_file,))
        return [CatalogEntry.from_row(row) for row in rows]

    def get_statistics(self) -> Dict[str, Any]:
        """Get catalog statistics.

        Returns:
            Dictionary with table_count, total_rows, total_columns.
        """
        self.init_catalog()
        query = f"""
            SELECT COUNT(*), COALESCE(SUM(row_count), 0), COALESCE(SUM(column_count), 0)
            FROM {CATALOG_TABLE_NAME}
        """
        row = self.sql_store.fetchone(query)
        if row:
            return {
                "table_count": row[0],
                "total_rows": row[1],
                "total_columns": row[2],
            }
        return {"table_count": 0, "total_rows": 0, "total_columns": 0}
