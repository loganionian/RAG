"""Unit tests for ingestion/sql_loader.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ingestion.sql_loader import (
    SQLLoadResult,
    SQLTableLoader,
    hash_dataframe,
    slugify,
)
from storage.metadata_catalog import MetadataCatalog
from storage.sql_store import SQLStore, SQLStoreConfig


class TestSlugify:
    """Tests for slugify function."""

    def test_basic_slugify(self) -> None:
        """Test basic slugification."""
        assert slugify("Hello World") == "hello_world"
        assert slugify("Test-File") == "test_file"

    def test_special_characters(self) -> None:
        """Test special characters are replaced."""
        assert slugify("file (1).csv") == "file_1_csv"
        assert slugify("report@2024") == "report_2024"

    def test_empty_string(self) -> None:
        """Test empty string returns default."""
        assert slugify("") == "table"
        assert slugify("   ") == "table"


class TestHashDataframe:
    """Tests for hash_dataframe function."""

    def test_basic_hash(self) -> None:
        """Test basic DataFrame hashing."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        hash1 = hash_dataframe(df)

        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex

    def test_same_content_same_hash(self) -> None:
        """Test same content produces same hash."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 3]})

        assert hash_dataframe(df1) == hash_dataframe(df2)

    def test_different_content_different_hash(self) -> None:
        """Test different content produces different hash."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 4]})

        assert hash_dataframe(df1) != hash_dataframe(df2)


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


@pytest.fixture
def loader(sql_store: SQLStore, catalog: MetadataCatalog) -> SQLTableLoader:
    """Create a SQLTableLoader instance for testing."""
    return SQLTableLoader(sql_store, catalog)


class TestSQLTableLoader:
    """Tests for SQLTableLoader class."""

    def test_generate_table_name_csv(self, loader: SQLTableLoader) -> None:
        """Test table name generation for CSV files."""
        name = loader.generate_table_name("sales_data.csv")
        assert name == "sales_data"

    def test_generate_table_name_excel_with_sheet(self, loader: SQLTableLoader) -> None:
        """Test table name generation for Excel with sheet name."""
        name = loader.generate_table_name("Q4_Report.xlsx", "Sales Data")
        assert name == "q4_report__sales_data"

    def test_generate_table_name_special_chars(self, loader: SQLTableLoader) -> None:
        """Test table name generation with special characters."""
        name = loader.generate_table_name("Report (Final).xlsx", "Q4-2024")
        assert name == "report_final__q4_2024"

    def test_load_dataframe_basic(
        self, loader: SQLTableLoader, sql_store: SQLStore, tmp_path: Path
    ) -> None:
        """Test basic DataFrame loading."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1],
        })

        source_file = tmp_path / "test.csv"
        result = loader.load_dataframe(df, source_file)

        assert isinstance(result, SQLLoadResult)
        assert result.row_count == 3
        assert result.column_count == 3
        assert result.was_skipped is False
        assert result.was_replaced is False

        # Verify table exists and has data
        assert sql_store.table_exists(result.table_name)
        count = sql_store.get_table_count(result.table_name)
        assert count == 3

    def test_load_dataframe_idempotency(
        self, loader: SQLTableLoader, tmp_path: Path
    ) -> None:
        """Test that loading same data twice skips second load."""
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        source_file = tmp_path / "test.csv"

        # First load
        result1 = loader.load_dataframe(df, source_file)
        assert result1.was_skipped is False

        # Second load with same data
        result2 = loader.load_dataframe(df, source_file)
        assert result2.was_skipped is True

    def test_load_dataframe_replacement(
        self, loader: SQLTableLoader, tmp_path: Path
    ) -> None:
        """Test that loading modified data replaces the table."""
        source_file = tmp_path / "test.csv"

        # First load
        df1 = pd.DataFrame({"id": [1, 2, 3]})
        result1 = loader.load_dataframe(df1, source_file)
        assert result1.was_replaced is False

        # Second load with different data
        df2 = pd.DataFrame({"id": [1, 2, 3, 4]})
        result2 = loader.load_dataframe(df2, source_file)
        assert result2.was_replaced is True
        assert result2.row_count == 4

    def test_load_dataframe_with_domain_labels(
        self, loader: SQLTableLoader, catalog: MetadataCatalog, tmp_path: Path
    ) -> None:
        """Test loading with domain labels."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        source_file = tmp_path / "test.csv"

        result = loader.load_dataframe(
            df, source_file, domain_labels=["finance", "quarterly"]
        )

        entry = catalog.get_entry(result.table_name)
        assert entry is not None
        assert "finance" in entry.domain_labels
        assert "quarterly" in entry.domain_labels

    def test_load_dataframe_custom_table_name(
        self, loader: SQLTableLoader, sql_store: SQLStore, tmp_path: Path
    ) -> None:
        """Test loading with custom table name."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        source_file = tmp_path / "test.csv"

        result = loader.load_dataframe(df, source_file, table_name="custom_name")

        assert result.table_name == "custom_name"
        assert sql_store.table_exists("custom_name")

    def test_load_csv_file(
        self, loader: SQLTableLoader, sql_store: SQLStore, tmp_path: Path
    ) -> None:
        """Test loading a CSV file."""
        # Create a test CSV file
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        df.to_csv(csv_path, index=False)

        result = loader.load_csv_file(csv_path)

        assert result.row_count == 3
        assert result.column_count == 2
        assert sql_store.table_exists(result.table_name)

    def test_load_csv_file_with_semicolon_delimiter(
        self, loader: SQLTableLoader, tmp_path: Path
    ) -> None:
        """Test loading a CSV file with semicolon delimiter."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("col1;col2\n1;a\n2;b\n3;c")

        result = loader.load_csv_file(csv_path)

        assert result.row_count == 3
        assert result.column_count == 2

    def test_load_spreadsheet_file_csv(
        self, loader: SQLTableLoader, tmp_path: Path
    ) -> None:
        """Test load_spreadsheet_file with CSV."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"col1": [1, 2, 3]})
        df.to_csv(csv_path, index=False)

        results = loader.load_spreadsheet_file(csv_path)

        assert len(results) == 1
        assert results[0].row_count == 3

    def test_load_spreadsheet_file_tsv(
        self, loader: SQLTableLoader, tmp_path: Path
    ) -> None:
        """Test load_spreadsheet_file with TSV."""
        tsv_path = tmp_path / "test.tsv"
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        df.to_csv(tsv_path, sep="\t", index=False)

        results = loader.load_spreadsheet_file(tsv_path)

        assert len(results) == 1
        assert results[0].column_count == 2

    def test_load_spreadsheet_file_unsupported(
        self, loader: SQLTableLoader, tmp_path: Path
    ) -> None:
        """Test load_spreadsheet_file with unsupported format."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("some text")

        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load_spreadsheet_file(txt_path)


class TestSQLLoadResult:
    """Tests for SQLLoadResult dataclass."""

    def test_result_to_dict(self) -> None:
        """Test result serialization."""
        result = SQLLoadResult(
            table_name="test_table",
            source_file="/path/to/file.csv",
            sheet_name=None,
            row_count=100,
            column_count=5,
            content_hash="abc123",
            load_duration_seconds=1.5,
            was_replaced=False,
            was_skipped=False,
        )

        data = result.to_dict()

        assert data["table_name"] == "test_table"
        assert data["row_count"] == 100
        assert data["was_skipped"] is False
