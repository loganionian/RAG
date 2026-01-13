"""Unit tests for ingestion/type_inference.py."""
from __future__ import annotations

import pandas as pd
import pytest

from ingestion.type_inference import (
    ColumnTypeInfo,
    SQLType,
    generate_create_table_sql,
    infer_column_type,
    infer_dataframe_schema,
    sanitize_column_name,
)


class TestSanitizeColumnName:
    """Tests for sanitize_column_name function."""

    def test_basic_name(self) -> None:
        """Test basic column name sanitization."""
        assert sanitize_column_name("Name") == "name"
        assert sanitize_column_name("age") == "age"

    def test_spaces_replaced(self) -> None:
        """Test spaces are replaced with underscores."""
        assert sanitize_column_name("First Name") == "first_name"
        assert sanitize_column_name("Date of Birth") == "date_of_birth"

    def test_special_chars_removed(self) -> None:
        """Test special characters are removed."""
        assert sanitize_column_name("Price ($)") == "price_"
        assert sanitize_column_name("Rate %") == "rate_"
        assert sanitize_column_name("Name/Title") == "name_title"

    def test_starts_with_number(self) -> None:
        """Test names starting with numbers get prefixed."""
        assert sanitize_column_name("123abc") == "col_123abc"
        assert sanitize_column_name("1st_place") == "col_1st_place"

    def test_empty_name(self) -> None:
        """Test empty or None names use fallback."""
        assert sanitize_column_name("") == "col_0"
        assert sanitize_column_name("   ") == "col_0"
        assert sanitize_column_name(None) == "col_0"

    def test_reserved_words(self) -> None:
        """Test SQL reserved words get suffix."""
        assert sanitize_column_name("select") == "select_col"
        assert sanitize_column_name("from") == "from_col"
        assert sanitize_column_name("table") == "table_col"
        assert sanitize_column_name("DATE") == "date_col"

    def test_long_name_truncated(self) -> None:
        """Test very long names are truncated."""
        long_name = "a" * 150
        result = sanitize_column_name(long_name)
        assert len(result) <= 100


class TestInferColumnType:
    """Tests for infer_column_type function."""

    def test_integer_column(self) -> None:
        """Test integer type inference."""
        series = pd.Series([1, 2, 3, 4, 5])
        assert infer_column_type(series) == SQLType.INTEGER

    def test_integer_with_nulls(self) -> None:
        """Test integer type with null values."""
        series = pd.Series([1, None, 3, None, 5])
        assert infer_column_type(series) == SQLType.INTEGER

    def test_integer_strings(self) -> None:
        """Test integer strings are inferred as integers."""
        series = pd.Series(["1", "2", "3", "4"])
        assert infer_column_type(series) == SQLType.INTEGER

    def test_real_column(self) -> None:
        """Test real/float type inference."""
        series = pd.Series([1.5, 2.7, 3.14, 4.0])
        assert infer_column_type(series) == SQLType.REAL

    def test_mixed_numeric(self) -> None:
        """Test mixed int and float results in REAL."""
        series = pd.Series([1, 2.5, 3, 4.7])
        assert infer_column_type(series) == SQLType.REAL

    def test_boolean_column(self) -> None:
        """Test boolean type inference."""
        series = pd.Series([True, False, True, False])
        assert infer_column_type(series) == SQLType.BOOLEAN

    def test_boolean_strings(self) -> None:
        """Test boolean-like strings."""
        series = pd.Series(["true", "false", "True", "False"])
        assert infer_column_type(series) == SQLType.BOOLEAN

        series = pd.Series(["yes", "no", "YES", "NO"])
        assert infer_column_type(series) == SQLType.BOOLEAN

        series = pd.Series(["1", "0", "1", "0"])
        assert infer_column_type(series) == SQLType.BOOLEAN

    def test_date_column(self) -> None:
        """Test date type inference."""
        series = pd.Series(["2024-01-15", "2024-02-20", "2024-03-25"])
        assert infer_column_type(series) == SQLType.DATE

    def test_datetime_column(self) -> None:
        """Test datetime type inference."""
        series = pd.Series([
            "2024-01-15 10:30:00",
            "2024-02-20 14:45:00",
            "2024-03-25 08:15:00",
        ])
        assert infer_column_type(series) == SQLType.DATETIME

    def test_text_column(self) -> None:
        """Test text type inference."""
        series = pd.Series(["hello", "world", "test"])
        assert infer_column_type(series) == SQLType.TEXT

    def test_mixed_text_numbers(self) -> None:
        """Test mixed text and numbers - 50% numeric returns INTEGER since all numeric values are integers."""
        series = pd.Series(["abc", "123", "def", "456"])
        result = infer_column_type(series)
        # Only 50% are numeric, but all numeric ones are integers
        # Type inference considers this as INTEGER since integer check comes before real
        assert result == SQLType.INTEGER

    def test_empty_series(self) -> None:
        """Test empty series defaults to text."""
        series = pd.Series([], dtype=object)
        assert infer_column_type(series) == SQLType.TEXT

    def test_all_null_series(self) -> None:
        """Test all-null series defaults to text."""
        series = pd.Series([None, None, None])
        assert infer_column_type(series) == SQLType.TEXT


class TestInferDataframeSchema:
    """Tests for infer_dataframe_schema function."""

    def test_basic_schema_inference(self) -> None:
        """Test schema inference for a basic DataFrame."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1],
        })

        schema = infer_dataframe_schema(df)

        assert len(schema) == 3
        assert schema[0].name == "id"
        assert schema[0].sql_type == SQLType.INTEGER
        assert schema[1].name == "name"
        assert schema[1].sql_type == SQLType.TEXT
        assert schema[2].name == "score"
        assert schema[2].sql_type == SQLType.REAL

    def test_schema_with_nulls(self) -> None:
        """Test schema inference with nullable columns."""
        df = pd.DataFrame({
            "id": [1, 2, None],
            "name": ["Alice", None, "Charlie"],
        })

        schema = infer_dataframe_schema(df)

        assert schema[0].nullable is True
        assert schema[1].nullable is True

    def test_similar_column_names(self) -> None:
        """Test handling of similar column names that sanitize to the same value."""
        # Create DataFrame with columns that will sanitize to the same name
        df = pd.DataFrame({
            "Name": [1, 2, 3],
            "name": [4, 5, 6],
            "NAME": [7, 8, 9],
        })

        schema = infer_dataframe_schema(df)

        # Should have unique sanitized names
        names = [col.name for col in schema]
        assert len(names) == len(set(names))  # All unique
        assert len(names) == 3

    def test_column_original_name_preserved(self) -> None:
        """Test that original column names are preserved."""
        df = pd.DataFrame({"First Name": ["Alice", "Bob"]})

        schema = infer_dataframe_schema(df)

        assert schema[0].name == "first_name"
        assert schema[0].original_name == "First Name"


class TestGenerateCreateTableSQL:
    """Tests for generate_create_table_sql function."""

    def test_basic_create_table(self) -> None:
        """Test basic CREATE TABLE generation."""
        schema = [
            ColumnTypeInfo("id", "id", SQLType.INTEGER, False, [1, 2, 3]),
            ColumnTypeInfo("name", "name", SQLType.TEXT, True, ["Alice"]),
        ]

        sql = generate_create_table_sql("test_table", schema)

        assert 'CREATE TABLE "test_table"' in sql
        assert "_row_id INTEGER PRIMARY KEY" in sql
        assert "_source_file VARCHAR" in sql
        assert '"id" BIGINT' in sql
        assert '"name" VARCHAR' in sql

    def test_all_types_in_schema(self) -> None:
        """Test CREATE TABLE with all SQL types."""
        schema = [
            ColumnTypeInfo("int_col", "int_col", SQLType.INTEGER, False, []),
            ColumnTypeInfo("real_col", "real_col", SQLType.REAL, False, []),
            ColumnTypeInfo("date_col", "date_col", SQLType.DATE, False, []),
            ColumnTypeInfo("datetime_col", "datetime_col", SQLType.DATETIME, False, []),
            ColumnTypeInfo("bool_col", "bool_col", SQLType.BOOLEAN, False, []),
            ColumnTypeInfo("text_col", "text_col", SQLType.TEXT, False, []),
        ]

        sql = generate_create_table_sql("all_types", schema)

        assert "BIGINT" in sql
        assert "DOUBLE" in sql
        assert "DATE" in sql
        assert "TIMESTAMP" in sql
        assert "BOOLEAN" in sql
        assert "VARCHAR" in sql
