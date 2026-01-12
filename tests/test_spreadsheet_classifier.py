"""Unit tests for spreadsheet classifier (tabular vs report-like detection)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ingestion.spreadsheet_classifier import (
    ClassificationResult,
    SpreadsheetClassificationConfig,
    classify_dataframe,
    classify_spreadsheet_file,
    _compute_numeric_ratio,
    _compute_long_text_ratio,
    _compute_empty_cell_ratio,
    _check_override_patterns,
)


class TestComputeMetrics:
    """Tests for metric computation functions."""

    def test_numeric_ratio_all_numeric(self):
        """Test numeric ratio with all numeric data."""
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [10.5, 20.5, 30.5, 40.5, 50.5],
        })
        ratio = _compute_numeric_ratio(df)
        assert ratio == 1.0

    def test_numeric_ratio_mixed(self):
        """Test numeric ratio with mixed data."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        })
        ratio = _compute_numeric_ratio(df)
        # 3 numeric cells out of 6 total
        assert ratio == 0.5

    def test_numeric_ratio_no_numeric(self):
        """Test numeric ratio with no numeric data."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "city": ["NYC", "LA"],
        })
        ratio = _compute_numeric_ratio(df)
        assert ratio == 0.0

    def test_numeric_ratio_empty(self):
        """Test numeric ratio with empty DataFrame."""
        df = pd.DataFrame()
        ratio = _compute_numeric_ratio(df)
        assert ratio == 0.0

    def test_long_text_ratio_none_long(self):
        """Test long text ratio with no long text."""
        df = pd.DataFrame({
            "short1": ["abc", "def"],
            "short2": ["ghi", "jkl"],
        })
        ratio = _compute_long_text_ratio(df, threshold=200)
        assert ratio == 0.0

    def test_long_text_ratio_all_long(self):
        """Test long text ratio with all long text."""
        long_text = "x" * 250
        df = pd.DataFrame({
            "text1": [long_text, long_text],
            "text2": [long_text, long_text],
        })
        ratio = _compute_long_text_ratio(df, threshold=200)
        assert ratio == 1.0

    def test_long_text_ratio_mixed(self):
        """Test long text ratio with mixed lengths."""
        long_text = "x" * 250
        df = pd.DataFrame({
            "text": [long_text, "short"],
        })
        ratio = _compute_long_text_ratio(df, threshold=200)
        assert ratio == 0.5

    def test_empty_cell_ratio_no_empty(self):
        """Test empty cell ratio with no empty cells."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        })
        ratio = _compute_empty_cell_ratio(df)
        assert ratio == 0.0

    def test_empty_cell_ratio_with_nulls(self):
        """Test empty cell ratio with null values."""
        df = pd.DataFrame({
            "col1": [1, None, 3],
            "col2": ["a", "b", None],
        })
        ratio = _compute_empty_cell_ratio(df)
        # Should detect null values as empty
        assert ratio > 0.0

    def test_empty_cell_ratio_empty_dataframe(self):
        """Test empty cell ratio with empty DataFrame."""
        df = pd.DataFrame()
        ratio = _compute_empty_cell_ratio(df)
        assert ratio == 1.0


class TestOverridePatterns:
    """Tests for filename override pattern matching."""

    def test_override_pattern_match_tabular(self):
        """Test override pattern matching for tabular."""
        patterns = [
            {"pattern": "*_data.csv", "classification": "tabular"},
            {"pattern": "*_report.xlsx", "classification": "report_like"},
        ]
        result = _check_override_patterns("sales_data.csv", patterns)
        assert result == "tabular"

    def test_override_pattern_match_report(self):
        """Test override pattern matching for report_like."""
        patterns = [
            {"pattern": "*_data.csv", "classification": "tabular"},
            {"pattern": "*_report.xlsx", "classification": "report_like"},
        ]
        result = _check_override_patterns("quarterly_report.xlsx", patterns)
        assert result == "report_like"

    def test_override_pattern_no_match(self):
        """Test override pattern with no match."""
        patterns = [
            {"pattern": "*_data.csv", "classification": "tabular"},
        ]
        result = _check_override_patterns("random_file.xlsx", patterns)
        assert result is None

    def test_override_pattern_case_insensitive(self):
        """Test that pattern matching is case insensitive."""
        patterns = [
            {"pattern": "*_data.csv", "classification": "tabular"},
        ]
        result = _check_override_patterns("SALES_DATA.CSV", patterns)
        assert result == "tabular"

    def test_override_pattern_empty_list(self):
        """Test override pattern with empty list."""
        result = _check_override_patterns("file.csv", [])
        assert result is None


class TestClassifyDataframe:
    """Tests for classify_dataframe function."""

    def test_classify_tabular_data(self):
        """Test classification of typical tabular data."""
        # Create tabular-like data: many rows, mostly numeric
        df = pd.DataFrame({
            "id": range(1, 51),
            "value1": range(100, 150),
            "value2": [x * 1.5 for x in range(50)],
            "category": ["A", "B", "C", "D", "E"] * 10,
        })

        result = classify_dataframe(df)

        assert result.classification == "tabular"
        assert result.confidence > 0.5
        assert len(result.reasons) > 0
        assert result.metrics["row_count"] == 50
        assert result.metrics["column_count"] == 4

    def test_classify_report_like_data(self):
        """Test classification of report-like data."""
        # Create report-like data: few rows, long text cells
        long_paragraph = "This is a very long paragraph that contains detailed information. " * 10
        df = pd.DataFrame({
            "section": ["Introduction", "Analysis"],
            "content": [long_paragraph, long_paragraph],
        })

        result = classify_dataframe(df)

        assert result.classification == "report_like"
        assert result.confidence < 0.5

    def test_classify_sparse_data(self):
        """Test classification of sparse data."""
        # Create very sparse data with mostly empty cells and long text
        long_text = "This is a very long paragraph. " * 20  # >200 chars
        df = pd.DataFrame({
            "col1": [long_text, None, None],
            "col2": [None, long_text, None],
        })

        config = SpreadsheetClassificationConfig(
            min_rows_for_tabular=10,  # 3 rows < 10
            empty_cell_ratio_max=0.3,  # Sparse data has higher empty ratio
        )
        result = classify_dataframe(df, config=config)

        # Should be report-like due to low row count and long text
        assert result.classification == "report_like"

    def test_classify_with_override_pattern(self):
        """Test classification with override pattern."""
        df = pd.DataFrame({"col": [1, 2, 3]})

        config = SpreadsheetClassificationConfig(
            override_patterns=[
                {"pattern": "*_report.csv", "classification": "report_like"}
            ]
        )

        result = classify_dataframe(df, config=config, filename="quarterly_report.csv")

        assert result.classification == "report_like"
        assert result.confidence == 1.0
        assert "override pattern" in result.reasons[0].lower()

    def test_classify_with_custom_config(self):
        """Test classification with custom configuration."""
        df = pd.DataFrame({
            "id": range(1, 6),  # Only 5 rows
            "value": range(10, 15),
        })

        # Default config: min_rows_for_tabular=10, so 5 rows won't pass
        default_result = classify_dataframe(df)

        # Custom config: lower threshold
        custom_config = SpreadsheetClassificationConfig(min_rows_for_tabular=3)
        custom_result = classify_dataframe(df, config=custom_config)

        # Custom config should give higher tabular score
        assert custom_result.confidence > default_result.confidence

    def test_classify_empty_dataframe(self):
        """Test classification of empty DataFrame."""
        df = pd.DataFrame()

        result = classify_dataframe(df)

        assert result.classification == "report_like"
        assert result.metrics["row_count"] == 0

    def test_metrics_in_result(self):
        """Test that metrics are properly included in result."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        })

        result = classify_dataframe(df)

        assert "row_count" in result.metrics
        assert "column_count" in result.metrics
        assert "numeric_ratio" in result.metrics
        assert "long_text_ratio" in result.metrics
        assert "empty_cell_ratio" in result.metrics


class TestClassifySpreadsheetFile:
    """Tests for classify_spreadsheet_file function."""

    def test_classify_csv_file(self, tmp_path: Path):
        """Test classification of CSV file."""
        csv_content = "id,value,name\n1,100,Alice\n2,200,Bob\n3,300,Charlie\n"
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(csv_content)

        result = classify_spreadsheet_file(csv_path)

        assert result.classification in ("tabular", "report_like")
        assert result.metrics["row_count"] == 3

    def test_classify_tsv_file(self, tmp_path: Path):
        """Test classification of TSV file."""
        tsv_content = "id\tvalue\tname\n1\t100\tAlice\n2\t200\tBob\n"
        tsv_path = tmp_path / "data.tsv"
        tsv_path.write_text(tsv_content)

        result = classify_spreadsheet_file(tsv_path)

        assert result.classification in ("tabular", "report_like")

    def test_classify_nonexistent_file(self):
        """Test classification of non-existent file."""
        with pytest.raises(FileNotFoundError):
            classify_spreadsheet_file(Path("/nonexistent/file.csv"))

    def test_classify_unsupported_format(self, tmp_path: Path):
        """Test classification of unsupported format returns default."""
        txt_path = tmp_path / "data.txt"
        txt_path.write_text("some text")

        # Unsupported formats return default classification instead of raising
        result = classify_spreadsheet_file(txt_path)

        assert result.classification == "report_like"
        assert result.confidence == 0.0

    def test_classify_xlsx_file(self, tmp_path: Path):
        """Test classification of Excel file with mocked pandas."""
        xlsx_path = tmp_path / "data.xlsx"
        xlsx_path.touch()

        with patch("pandas.read_excel") as mock_read:
            mock_read.return_value = pd.DataFrame({
                "col1": range(20),
                "col2": range(20),
            })

            result = classify_spreadsheet_file(xlsx_path)

            assert result.classification in ("tabular", "report_like")
            mock_read.assert_called_once()

    def test_classify_with_read_error(self, tmp_path: Path):
        """Test graceful handling of file read errors."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_bytes(b"\xff\xfe invalid")  # Invalid UTF-8

        result = classify_spreadsheet_file(csv_path)

        # Should return default classification on error
        assert result.classification == "report_like"
        assert result.confidence == 0.0
        assert "read_error" in result.metrics or "Could not read" in result.reasons[0]


class TestClassificationConfig:
    """Tests for SpreadsheetClassificationConfig."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = SpreadsheetClassificationConfig()

        assert config.min_rows_for_tabular == 10
        assert config.max_columns_for_tabular == 50
        assert config.numeric_ratio_threshold == 0.3
        assert config.long_text_threshold == 200
        assert config.long_text_ratio_max == 0.1
        assert config.empty_cell_ratio_max == 0.5
        assert config.override_patterns == []

    def test_custom_config(self):
        """Test custom configuration."""
        config = SpreadsheetClassificationConfig(
            min_rows_for_tabular=5,
            numeric_ratio_threshold=0.5,
            override_patterns=[{"pattern": "*.csv", "classification": "tabular"}],
        )

        assert config.min_rows_for_tabular == 5
        assert config.numeric_ratio_threshold == 0.5
        assert len(config.override_patterns) == 1


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_result_attributes(self):
        """Test ClassificationResult attributes."""
        result = ClassificationResult(
            classification="tabular",
            confidence=0.85,
            reasons=["High row count", "High numeric ratio"],
            metrics={"row_count": 100},
        )

        assert result.classification == "tabular"
        assert result.confidence == 0.85
        assert len(result.reasons) == 2
        assert result.metrics["row_count"] == 100

    def test_result_default_metrics(self):
        """Test ClassificationResult default metrics."""
        result = ClassificationResult(
            classification="report_like",
            confidence=0.3,
            reasons=["Few rows"],
        )

        assert result.metrics == {}
