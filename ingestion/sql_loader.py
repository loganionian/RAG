"""SQL table loading for tabular spreadsheet data.

This module provides functionality to load pandas DataFrames into DuckDB tables
with automatic type inference, idempotency via content hashing, and metadata
catalog management.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from storage.metadata_catalog import CatalogEntry, MetadataCatalog
from storage.sql_store import SQLStore

from .type_inference import (
    ColumnTypeInfo,
    generate_create_table_sql,
    infer_dataframe_schema,
)

logger = logging.getLogger(__name__)


def slugify(value: str) -> str:
    """Convert a string to a URL/SQL-safe slug.

    Args:
        value: String to slugify.

    Returns:
        Lowercase string with special chars replaced by underscores.
    """
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return normalized or "table"


def hash_dataframe(df: pd.DataFrame) -> str:
    """Compute SHA256 hash of a DataFrame's content.

    Args:
        df: DataFrame to hash.

    Returns:
        Hex-encoded SHA256 hash.
    """
    # Convert to bytes in a deterministic way
    content = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


@dataclass
class SQLLoadResult:
    """Result of loading a table into SQL.

    Attributes:
        table_name: Name of the created/updated table.
        source_file: Path to the source file.
        sheet_name: Sheet name for Excel files (None for CSV).
        row_count: Number of rows loaded.
        column_count: Number of columns loaded.
        content_hash: SHA256 hash of the source content.
        columns: List of column definitions.
        load_duration_seconds: Time taken to load the table.
        was_replaced: True if an existing table was replaced.
        was_skipped: True if the table was skipped (already up-to-date).
    """

    table_name: str
    source_file: str
    sheet_name: Optional[str]
    row_count: int
    column_count: int
    content_hash: str
    columns: List[ColumnTypeInfo] = field(default_factory=list)
    load_duration_seconds: float = 0.0
    was_replaced: bool = False
    was_skipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "table_name": self.table_name,
            "source_file": self.source_file,
            "sheet_name": self.sheet_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "content_hash": self.content_hash,
            "columns": [c.to_dict() for c in self.columns],
            "load_duration_seconds": self.load_duration_seconds,
            "was_replaced": self.was_replaced,
            "was_skipped": self.was_skipped,
        }


class SQLTableLoader:
    """Loads DataFrames into DuckDB tables with idempotency."""

    def __init__(self, sql_store: SQLStore, catalog: MetadataCatalog) -> None:
        """Initialize the SQL table loader.

        Args:
            sql_store: SQLStore instance for database operations.
            catalog: MetadataCatalog for tracking file-to-table mappings.
        """
        self.sql_store = sql_store
        self.catalog = catalog

    def generate_table_name(
        self, filename: str, sheet_name: Optional[str] = None
    ) -> str:
        """Generate a SQL table name from file and sheet names.

        Format: {base_filename}__{sheet_name} (both slugified)
        Example: "Q4_Report.xlsx", "Sales Data" -> "q4_report__sales_data"

        Args:
            filename: Name of the source file.
            sheet_name: Optional sheet name for Excel files.

        Returns:
            Slugified table name.
        """
        # Extract base filename without extension
        base = Path(filename).stem
        base_slug = slugify(base)

        if sheet_name:
            sheet_slug = slugify(sheet_name)
            return f"{base_slug}__{sheet_slug}"
        return base_slug

    def load_dataframe(
        self,
        df: pd.DataFrame,
        source_file: Path,
        sheet_name: Optional[str] = None,
        content_hash: Optional[str] = None,
        domain_labels: Optional[List[str]] = None,
        table_name: Optional[str] = None,
    ) -> SQLLoadResult:
        """Load a DataFrame into a SQL table with idempotency.

        1. Generate table name
        2. Compute content hash
        3. Check catalog for existing entry with same hash (skip if up-to-date)
        4. If exists and hash differs, DROP TABLE
        5. Infer schema
        6. CREATE TABLE with inferred types
        7. INSERT data
        8. Register in catalog

        Args:
            df: DataFrame to load.
            source_file: Path to the source file.
            sheet_name: Optional sheet name for Excel files.
            content_hash: Pre-computed content hash (if available).
            domain_labels: Optional list of domain labels.
            table_name: Override for table name (if not auto-generated).

        Returns:
            SQLLoadResult with load statistics.
        """
        start_time = time.time()

        # Generate table name
        if table_name is None:
            table_name = self.generate_table_name(source_file.name, sheet_name)

        # Compute content hash
        if content_hash is None:
            content_hash = hash_dataframe(df)

        # Check idempotency
        if self.catalog.is_up_to_date(table_name, content_hash):
            logger.info("Table '%s' is up-to-date, skipping", table_name)
            entry = self.catalog.get_entry(table_name)
            return SQLLoadResult(
                table_name=table_name,
                source_file=str(source_file),
                sheet_name=sheet_name,
                row_count=entry.row_count if entry else len(df),
                column_count=entry.column_count if entry else len(df.columns),
                content_hash=content_hash,
                columns=[],
                load_duration_seconds=time.time() - start_time,
                was_replaced=False,
                was_skipped=True,
            )

        # Check if table exists with different hash
        was_replaced = False
        existing_entry = self.catalog.get_entry(table_name)
        if existing_entry:
            logger.info(
                "Table '%s' exists with different hash, replacing", table_name
            )
            self.sql_store.drop_table(table_name)
            self.catalog.unregister_table(table_name)
            was_replaced = True

        # Infer schema
        schema = infer_dataframe_schema(df)

        # Create table
        create_sql = generate_create_table_sql(table_name, schema)
        logger.debug("Creating table with SQL:\n%s", create_sql)
        self.sql_store.execute(create_sql)

        # Prepare data for insertion
        # Rename columns to match schema
        df_to_insert = df.copy()
        df_to_insert.columns = [col.name for col in schema]

        # Add metadata columns
        df_to_insert.insert(0, "_row_id", range(1, len(df) + 1))
        df_to_insert.insert(1, "_source_file", str(source_file))

        # Insert data using DuckDB's native DataFrame support
        conn = self.sql_store.connect()
        conn.execute(f'INSERT INTO "{table_name}" SELECT * FROM df_to_insert')

        # Register in catalog
        catalog_entry = CatalogEntry(
            table_name=table_name,
            source_file=str(source_file),
            sheet_name=sheet_name,
            ingestion_timestamp=datetime.now(timezone.utc),
            row_count=len(df),
            column_count=len(schema),
            content_hash=content_hash,
            domain_labels=domain_labels or [],
            column_schema=[col.to_dict() for col in schema],
        )
        self.catalog.register_table(catalog_entry)

        duration = time.time() - start_time
        logger.info(
            "Loaded table '%s' with %d rows, %d columns in %.2fs",
            table_name,
            len(df),
            len(schema),
            duration,
        )

        return SQLLoadResult(
            table_name=table_name,
            source_file=str(source_file),
            sheet_name=sheet_name,
            row_count=len(df),
            column_count=len(schema),
            content_hash=content_hash,
            columns=schema,
            load_duration_seconds=duration,
            was_replaced=was_replaced,
            was_skipped=False,
        )

    def load_csv_file(
        self,
        file_path: Path,
        domain_labels: Optional[List[str]] = None,
        delimiter: Optional[str] = None,
    ) -> SQLLoadResult:
        """Load a CSV file into a SQL table.

        Args:
            file_path: Path to the CSV file.
            domain_labels: Optional list of domain labels.
            delimiter: Optional delimiter override.

        Returns:
            SQLLoadResult with load statistics.
        """
        # Detect delimiter if not specified
        if delimiter is None:
            for delim in [",", ";", "\t", "|"]:
                try:
                    test_df = pd.read_csv(file_path, delimiter=delim, nrows=5)
                    if len(test_df.columns) > 1:
                        delimiter = delim
                        break
                except Exception:
                    continue
            if delimiter is None:
                delimiter = ","

        # Read the full file
        df = pd.read_csv(file_path, delimiter=delimiter)

        return self.load_dataframe(
            df=df,
            source_file=file_path,
            sheet_name=None,
            domain_labels=domain_labels,
        )

    def load_excel_file(
        self,
        file_path: Path,
        sheet_filter: Optional[List[str]] = None,
        domain_labels: Optional[List[str]] = None,
    ) -> List[SQLLoadResult]:
        """Load all sheets from an Excel file into SQL tables.

        Args:
            file_path: Path to the Excel file.
            sheet_filter: Optional list of sheet names to load (loads all if None).
            domain_labels: Optional list of domain labels.

        Returns:
            List of SQLLoadResult, one per loaded sheet.
        """
        results: List[SQLLoadResult] = []

        # Read Excel file to get sheet names
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names

        # Filter sheets if specified
        if sheet_filter:
            sheet_names = [s for s in sheet_names if s in sheet_filter]

        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                result = self.load_dataframe(
                    df=df,
                    source_file=file_path,
                    sheet_name=sheet_name,
                    domain_labels=domain_labels,
                )
                results.append(result)

            except Exception as exc:
                logger.error(
                    "Failed to load sheet '%s' from %s: %s",
                    sheet_name,
                    file_path,
                    exc,
                )

        return results

    def load_spreadsheet_file(
        self,
        file_path: Path,
        sheet_name: Optional[str] = None,
        domain_labels: Optional[List[str]] = None,
    ) -> List[SQLLoadResult]:
        """Load a spreadsheet file (CSV or Excel) into SQL tables.

        Args:
            file_path: Path to the spreadsheet file.
            sheet_name: For Excel files, specific sheet to load (loads all if None).
            domain_labels: Optional list of domain labels.

        Returns:
            List of SQLLoadResult (single item for CSV, multiple for Excel).
        """
        ext = file_path.suffix.lower()

        if ext in (".csv", ".tsv"):
            delimiter = "\t" if ext == ".tsv" else None
            result = self.load_csv_file(
                file_path, domain_labels=domain_labels, delimiter=delimiter
            )
            return [result]

        elif ext in (".xlsx", ".xls"):
            sheet_filter = [sheet_name] if sheet_name else None
            return self.load_excel_file(
                file_path, sheet_filter=sheet_filter, domain_labels=domain_labels
            )

        else:
            raise ValueError(f"Unsupported file format: {ext}")
