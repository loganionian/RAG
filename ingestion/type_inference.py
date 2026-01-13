"""Column type inference for SQL table creation.

This module provides heuristics to infer SQL column types from pandas DataFrame
columns, enabling automatic schema generation for tabular data ingestion.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Reserved SQL keywords that need escaping
SQL_RESERVED_WORDS = {
    "select", "from", "where", "order", "group", "by", "insert", "update",
    "delete", "create", "drop", "table", "index", "view", "join", "left",
    "right", "inner", "outer", "on", "and", "or", "not", "null", "true",
    "false", "as", "in", "is", "like", "between", "having", "limit",
    "offset", "union", "all", "distinct", "case", "when", "then", "else",
    "end", "primary", "key", "foreign", "references", "default", "check",
    "unique", "constraint", "alter", "add", "column", "set", "values",
    "into", "exists", "count", "sum", "avg", "min", "max", "date", "time",
    "timestamp", "integer", "real", "text", "varchar", "boolean", "double",
}


class SQLType(Enum):
    """SQL column types supported for type inference."""

    INTEGER = "BIGINT"
    REAL = "DOUBLE"
    DATE = "DATE"
    DATETIME = "TIMESTAMP"
    BOOLEAN = "BOOLEAN"
    TEXT = "VARCHAR"


@dataclass
class ColumnTypeInfo:
    """Type inference result for a single column.

    Attributes:
        name: Sanitized column name for SQL.
        original_name: Original column name from the source.
        sql_type: Inferred SQL type.
        nullable: Whether the column contains null values.
        sample_values: First few non-null values for debugging.
    """

    name: str
    original_name: str
    sql_type: SQLType
    nullable: bool
    sample_values: List[Any]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "original_name": self.original_name,
            "type": self.sql_type.value,
            "nullable": self.nullable,
        }


def sanitize_column_name(name: str, index: int = 0) -> str:
    """Convert a column name to a valid SQL identifier.

    Args:
        name: Original column name.
        index: Column index for fallback naming.

    Returns:
        Sanitized column name safe for SQL use.
    """
    if pd.isna(name) or name is None or str(name).strip() == "":
        return f"col_{index}"

    # Convert to string and strip
    name = str(name).strip()

    # Replace common separators with underscores
    name = re.sub(r"[\s\-./\\]+", "_", name)

    # Remove characters that aren't alphanumeric or underscore
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)

    # Ensure it starts with a letter or underscore
    if name and not name[0].isalpha() and name[0] != "_":
        name = f"col_{name}"

    # Handle empty result
    if not name:
        return f"col_{index}"

    # Convert to lowercase
    name = name.lower()

    # Handle reserved words
    if name in SQL_RESERVED_WORDS:
        name = f"{name}_col"

    # Truncate if too long (DuckDB supports up to 255 chars)
    if len(name) > 100:
        name = name[:100]

    return name


def _is_integer_like(series: pd.Series) -> bool:
    """Check if a series can be represented as integers.

    Args:
        series: Pandas Series to check.

    Returns:
        True if all non-null values are integer-like.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    # Already numeric integer type
    if pd.api.types.is_integer_dtype(series):
        return True

    # Try converting to numeric
    try:
        numeric = pd.to_numeric(non_null, errors="coerce")
        valid = numeric.dropna()
        if len(valid) == 0:
            return False
        # Check if all values are whole numbers
        return (valid == valid.astype(int)).all()
    except (ValueError, TypeError, OverflowError):
        return False


def _is_real_like(series: pd.Series) -> bool:
    """Check if a series can be represented as floating-point numbers.

    Args:
        series: Pandas Series to check.

    Returns:
        True if all non-null values are numeric.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    # Already numeric float type
    if pd.api.types.is_float_dtype(series):
        return True

    # Try converting to numeric
    try:
        numeric = pd.to_numeric(non_null, errors="coerce")
        valid_ratio = numeric.notna().sum() / len(non_null)
        # At least 90% should be convertible
        return valid_ratio >= 0.9
    except (ValueError, TypeError):
        return False


def _is_boolean_like(series: pd.Series) -> bool:
    """Check if a series can be represented as boolean.

    Args:
        series: Pandas Series to check.

    Returns:
        True if all non-null values are boolean-like.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    # Already boolean type
    if pd.api.types.is_bool_dtype(series):
        return True

    # Check for boolean-like string values
    bool_values = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}
    try:
        str_values = non_null.astype(str).str.lower().str.strip()
        unique_values = set(str_values.unique())
        return unique_values.issubset(bool_values) and len(unique_values) <= 2
    except (ValueError, TypeError):
        return False


def _is_date_like(series: pd.Series) -> bool:
    """Check if a series can be represented as dates.

    Args:
        series: Pandas Series to check.

    Returns:
        True if values can be parsed as dates.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    # Already datetime type
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Try parsing as dates (sample first 100 values for performance)
    sample = non_null.head(100)
    try:
        parsed = pd.to_datetime(sample, errors="coerce")
        valid_ratio = parsed.notna().sum() / len(sample)
        return valid_ratio >= 0.8
    except (ValueError, TypeError):
        return False


def _has_time_component(series: pd.Series) -> bool:
    """Check if datetime values have a time component.

    Args:
        series: Pandas Series with datetime values.

    Returns:
        True if any values have non-midnight times.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    try:
        if pd.api.types.is_datetime64_any_dtype(series):
            # Check if any times are not midnight
            times = non_null.dt.time
            return any(t.hour != 0 or t.minute != 0 or t.second != 0 for t in times)

        # Try parsing and check
        parsed = pd.to_datetime(non_null.head(100), errors="coerce")
        valid = parsed.dropna()
        if len(valid) == 0:
            return False
        return any(
            t.hour != 0 or t.minute != 0 or t.second != 0
            for t in valid.dt.time
        )
    except (ValueError, TypeError, AttributeError):
        return False


def infer_column_type(series: pd.Series) -> SQLType:
    """Infer the SQL type for a pandas Series.

    Type inference priority:
    1. Boolean (if matches boolean patterns)
    2. Integer (if all values are whole numbers)
    3. Real (if all values are numeric)
    4. DateTime (if parseable with time component)
    5. Date (if parseable as date only)
    6. Text (default fallback)

    Args:
        series: Pandas Series to analyze.

    Returns:
        Inferred SQLType.
    """
    # Handle empty series
    if len(series) == 0:
        return SQLType.TEXT

    # Check if all values are null
    is_all_na = series.isna().all()
    if isinstance(is_all_na, pd.Series):
        is_all_na = is_all_na.all()  # Handle duplicate column names
    if is_all_na:
        return SQLType.TEXT

    # Check in order of specificity
    if _is_boolean_like(series):
        return SQLType.BOOLEAN

    if _is_integer_like(series):
        return SQLType.INTEGER

    if _is_real_like(series):
        return SQLType.REAL

    if _is_date_like(series):
        if _has_time_component(series):
            return SQLType.DATETIME
        return SQLType.DATE

    # Default to text
    return SQLType.TEXT


def infer_dataframe_schema(df: pd.DataFrame) -> List[ColumnTypeInfo]:
    """Infer SQL schema for an entire DataFrame.

    Args:
        df: Pandas DataFrame to analyze.

    Returns:
        List of ColumnTypeInfo objects, one per column.
    """
    schema: List[ColumnTypeInfo] = []
    seen_names: set = set()

    for idx, col in enumerate(df.columns):
        # Sanitize column name
        sanitized = sanitize_column_name(col, idx)

        # Handle duplicate names
        original_sanitized = sanitized
        suffix = 1
        while sanitized in seen_names:
            sanitized = f"{original_sanitized}_{suffix}"
            suffix += 1
        seen_names.add(sanitized)

        # Get column data
        series = df[col]

        # Infer type
        sql_type = infer_column_type(series)

        # Check for nulls (convert to Python bool for JSON serialization)
        nullable = bool(series.isna().any())

        # Get sample values
        non_null = series.dropna()
        sample_values = non_null.head(5).tolist() if len(non_null) > 0 else []

        schema.append(
            ColumnTypeInfo(
                name=sanitized,
                original_name=str(col),
                sql_type=sql_type,
                nullable=nullable,
                sample_values=sample_values,
            )
        )

        logger.debug(
            "Column '%s' -> '%s' (%s, nullable=%s)",
            col,
            sanitized,
            sql_type.value,
            nullable,
        )

    return schema


def generate_create_table_sql(table_name: str, schema: List[ColumnTypeInfo]) -> str:
    """Generate CREATE TABLE SQL statement from schema.

    Args:
        table_name: Name for the SQL table.
        schema: List of ColumnTypeInfo objects.

    Returns:
        CREATE TABLE SQL statement.
    """
    columns = []

    # Add row ID column for tracking
    columns.append("_row_id INTEGER PRIMARY KEY")

    # Add source file reference
    columns.append("_source_file VARCHAR")

    # Add data columns
    for col_info in schema:
        col_def = f'"{col_info.name}" {col_info.sql_type.value}'
        columns.append(col_def)

    columns_sql = ",\n    ".join(columns)
    return f'CREATE TABLE "{table_name}" (\n    {columns_sql}\n)'
