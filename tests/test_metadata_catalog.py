"""Unit tests for storage/metadata_catalog.py."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from storage.metadata_catalog import CatalogEntry, MetadataCatalog
from storage.sql_store import SQLStore, SQLStoreConfig


@pytest.fixture
def sql_store(tmp_path: Path) -> SQLStore:
    """Create a SQLStore instance for testing."""
    db_path = tmp_path / "test.duckdb"
    config = SQLStoreConfig(db_path=db_path)
    store = SQLStore(config)
    yield store
    store.close()


@pytest.fixture
def catalog(sql_store: SQLStore) -> MetadataCatalog:
    """Create a MetadataCatalog instance for testing."""
    return MetadataCatalog(sql_store)


class TestCatalogEntry:
    """Tests for CatalogEntry dataclass."""

    def test_entry_creation(self) -> None:
        """Test basic entry creation."""
        entry = CatalogEntry(
            table_name="test_table",
            source_file="/path/to/file.csv",
            sheet_name=None,
            ingestion_timestamp=datetime(2024, 1, 15, 10, 30, 0),
            row_count=100,
            column_count=5,
            content_hash="abc123",
            domain_labels=["sales"],
            column_schema=[{"name": "id", "type": "INTEGER"}],
        )

        assert entry.table_name == "test_table"
        assert entry.row_count == 100
        assert entry.domain_labels == ["sales"]

    def test_entry_to_dict(self) -> None:
        """Test entry serialization to dict."""
        entry = CatalogEntry(
            table_name="test_table",
            source_file="/path/to/file.csv",
            sheet_name="Sheet1",
            ingestion_timestamp=datetime(2024, 1, 15, 10, 30, 0),
            row_count=100,
            column_count=5,
            content_hash="abc123",
        )

        data = entry.to_dict()

        assert data["table_name"] == "test_table"
        assert data["sheet_name"] == "Sheet1"
        assert "2024-01-15" in data["ingestion_timestamp"]

    def test_entry_from_row(self) -> None:
        """Test entry creation from database row."""
        row = (
            "test_table",
            "/path/to/file.csv",
            "Sheet1",
            "2024-01-15T10:30:00",
            100,
            5,
            "abc123",
            '["sales", "finance"]',
            '[{"name": "id", "type": "INTEGER"}]',
        )

        entry = CatalogEntry.from_row(row)

        assert entry.table_name == "test_table"
        assert entry.sheet_name == "Sheet1"
        assert entry.domain_labels == ["sales", "finance"]
        assert len(entry.column_schema) == 1


class TestMetadataCatalog:
    """Tests for MetadataCatalog class."""

    def test_init_catalog(self, catalog: MetadataCatalog, sql_store: SQLStore) -> None:
        """Test catalog table initialization."""
        catalog.init_catalog()

        # Verify table exists
        assert sql_store.table_exists("_ingestion_catalog")

    def test_register_and_get_entry(self, catalog: MetadataCatalog) -> None:
        """Test registering and retrieving an entry."""
        entry = CatalogEntry(
            table_name="test_table",
            source_file="/path/to/file.csv",
            sheet_name=None,
            ingestion_timestamp=datetime.now(timezone.utc),
            row_count=100,
            column_count=5,
            content_hash="abc123",
        )

        catalog.register_table(entry)
        retrieved = catalog.get_entry("test_table")

        assert retrieved is not None
        assert retrieved.table_name == "test_table"
        assert retrieved.row_count == 100
        assert retrieved.content_hash == "abc123"

    def test_get_nonexistent_entry(self, catalog: MetadataCatalog) -> None:
        """Test getting an entry that doesn't exist."""
        catalog.init_catalog()
        entry = catalog.get_entry("nonexistent")
        assert entry is None

    def test_is_up_to_date(self, catalog: MetadataCatalog) -> None:
        """Test idempotency check."""
        entry = CatalogEntry(
            table_name="test_table",
            source_file="/path/to/file.csv",
            sheet_name=None,
            ingestion_timestamp=datetime.now(timezone.utc),
            row_count=100,
            column_count=5,
            content_hash="abc123",
        )

        catalog.register_table(entry)

        # Same hash should be up-to-date
        assert catalog.is_up_to_date("test_table", "abc123") is True

        # Different hash should not be up-to-date
        assert catalog.is_up_to_date("test_table", "different_hash") is False

        # Non-existent table should not be up-to-date
        assert catalog.is_up_to_date("nonexistent", "abc123") is False

    def test_unregister_table(self, catalog: MetadataCatalog) -> None:
        """Test unregistering a table."""
        entry = CatalogEntry(
            table_name="test_table",
            source_file="/path/to/file.csv",
            sheet_name=None,
            ingestion_timestamp=datetime.now(timezone.utc),
            row_count=100,
            column_count=5,
            content_hash="abc123",
        )

        catalog.register_table(entry)
        assert catalog.get_entry("test_table") is not None

        result = catalog.unregister_table("test_table")
        assert result is True
        assert catalog.get_entry("test_table") is None

        # Unregistering again should return False
        result = catalog.unregister_table("test_table")
        assert result is False

    def test_list_tables(self, catalog: MetadataCatalog) -> None:
        """Test listing all tables."""
        catalog.init_catalog()

        # Initially empty
        entries = catalog.list_tables()
        assert len(entries) == 0

        # Add some entries
        for i in range(3):
            entry = CatalogEntry(
                table_name=f"table_{i}",
                source_file=f"/path/to/file_{i}.csv",
                sheet_name=None,
                ingestion_timestamp=datetime.now(timezone.utc),
                row_count=100 + i,
                column_count=5,
                content_hash=f"hash_{i}",
            )
            catalog.register_table(entry)

        entries = catalog.list_tables()
        assert len(entries) == 3

        table_names = [e.table_name for e in entries]
        assert "table_0" in table_names
        assert "table_1" in table_names
        assert "table_2" in table_names

    def test_find_by_source(self, catalog: MetadataCatalog) -> None:
        """Test finding tables by source file."""
        # Register tables from same source
        for i in range(2):
            entry = CatalogEntry(
                table_name=f"table_{i}",
                source_file="/path/to/source.xlsx",
                sheet_name=f"Sheet{i}",
                ingestion_timestamp=datetime.now(timezone.utc),
                row_count=100,
                column_count=5,
                content_hash=f"hash_{i}",
            )
            catalog.register_table(entry)

        # Register table from different source
        entry = CatalogEntry(
            table_name="other_table",
            source_file="/path/to/other.csv",
            sheet_name=None,
            ingestion_timestamp=datetime.now(timezone.utc),
            row_count=50,
            column_count=3,
            content_hash="other_hash",
        )
        catalog.register_table(entry)

        # Find by source
        entries = catalog.find_by_source("/path/to/source.xlsx")
        assert len(entries) == 2

        entries = catalog.find_by_source("/path/to/other.csv")
        assert len(entries) == 1

        entries = catalog.find_by_source("/nonexistent.csv")
        assert len(entries) == 0

    def test_get_statistics(self, catalog: MetadataCatalog) -> None:
        """Test getting catalog statistics."""
        catalog.init_catalog()

        # Empty catalog
        stats = catalog.get_statistics()
        assert stats["table_count"] == 0
        assert stats["total_rows"] == 0
        assert stats["total_columns"] == 0

        # Add entries
        for i in range(3):
            entry = CatalogEntry(
                table_name=f"table_{i}",
                source_file=f"/path/to/file_{i}.csv",
                sheet_name=None,
                ingestion_timestamp=datetime.now(timezone.utc),
                row_count=100,
                column_count=5,
                content_hash=f"hash_{i}",
            )
            catalog.register_table(entry)

        stats = catalog.get_statistics()
        assert stats["table_count"] == 3
        assert stats["total_rows"] == 300
        assert stats["total_columns"] == 15

    def test_update_existing_entry(self, catalog: MetadataCatalog) -> None:
        """Test updating an existing entry (upsert behavior)."""
        entry1 = CatalogEntry(
            table_name="test_table",
            source_file="/path/to/file.csv",
            sheet_name=None,
            ingestion_timestamp=datetime.now(timezone.utc),
            row_count=100,
            column_count=5,
            content_hash="hash1",
        )
        catalog.register_table(entry1)

        # Update with new hash and row count
        entry2 = CatalogEntry(
            table_name="test_table",
            source_file="/path/to/file.csv",
            sheet_name=None,
            ingestion_timestamp=datetime.now(timezone.utc),
            row_count=200,
            column_count=5,
            content_hash="hash2",
        )
        catalog.register_table(entry2)

        # Should have updated values
        retrieved = catalog.get_entry("test_table")
        assert retrieved.row_count == 200
        assert retrieved.content_hash == "hash2"

        # Should still be only one entry
        entries = catalog.list_tables()
        assert len(entries) == 1
