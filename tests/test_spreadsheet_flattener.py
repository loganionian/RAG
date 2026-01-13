"""Unit tests for spreadsheet flattener (report-like to text conversion)."""
from __future__ import annotations

import pandas as pd
import pytest

from ingestion.spreadsheet_flattener import (
    SpreadsheetFlattener,
    SpreadsheetFlatteningConfig,
    _compute_empty_ratio,
    _has_header_row,
    _extract_notes,
)


class TestComputeEmptyRatio:
    """Tests for empty ratio computation."""

    def test_empty_ratio_no_empty(self):
        """Test empty ratio with no empty cells."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        })
        ratio = _compute_empty_ratio(df)
        assert ratio == 0.0

    def test_empty_ratio_all_empty(self):
        """Test empty ratio with all empty cells."""
        df = pd.DataFrame({
            "col1": [None, None],
            "col2": ["", ""],
        })
        ratio = _compute_empty_ratio(df)
        assert ratio == 1.0

    def test_empty_ratio_mixed(self):
        """Test empty ratio with mixed content."""
        df = pd.DataFrame({
            "col1": [1, None],
            "col2": ["a", ""],
        })
        ratio = _compute_empty_ratio(df)
        assert ratio == 0.5

    def test_empty_ratio_empty_df(self):
        """Test empty ratio with empty DataFrame."""
        df = pd.DataFrame()
        ratio = _compute_empty_ratio(df)
        assert ratio == 1.0


class TestHasHeaderRow:
    """Tests for header row detection."""

    def test_header_row_with_string_header(self):
        """Test detection of string header row."""
        df = pd.DataFrame({
            0: ["Name", "Alice", "Bob"],
            1: ["Age", "25", "30"],
        })
        assert _has_header_row(df) is True

    def test_header_row_with_numeric_first_row(self):
        """Test that numeric first row is not detected as header."""
        df = pd.DataFrame({
            0: [1, 2, 3],
            1: [4, 5, 6],
        })
        assert _has_header_row(df) is False

    def test_header_row_empty_df(self):
        """Test header detection with empty DataFrame."""
        df = pd.DataFrame()
        assert _has_header_row(df) is False

    def test_header_row_single_row(self):
        """Test header detection with single row."""
        df = pd.DataFrame({0: ["Name"], 1: ["Age"]})
        assert _has_header_row(df) is False


class TestExtractNotes:
    """Tests for footer note extraction."""

    def test_extract_notes_with_note_pattern(self):
        """Test extraction of notes matching patterns."""
        df = pd.DataFrame({
            0: ["Data", "More data", "Note: See appendix"],
            1: ["Value", "Value 2", ""],
        })
        df_clean, notes = _extract_notes(df, [r"^note[s]?:"])
        assert len(notes) == 1
        assert "See appendix" in notes[0]
        assert len(df_clean) == 2

    def test_extract_notes_no_notes(self):
        """Test extraction when no notes present."""
        df = pd.DataFrame({
            0: ["Data", "More data"],
            1: ["Value", "Value 2"],
        })
        df_clean, notes = _extract_notes(df, [r"^note[s]?:"])
        assert len(notes) == 0
        assert len(df_clean) == 2

    def test_extract_notes_multiple_patterns(self):
        """Test extraction with multiple patterns."""
        df = pd.DataFrame({
            0: ["Data", "Source: Company report", "*Important disclaimer"],
        })
        df_clean, notes = _extract_notes(df, [r"^source:", r"^\*"])
        assert len(notes) == 2


class TestSpreadsheetFlattenerLayoutDetection:
    """Tests for layout type detection."""

    def test_detect_table_layout(self):
        """Test detection of table layout."""
        flattener = SpreadsheetFlattener()
        df = pd.DataFrame({
            "col1": range(10),
            "col2": range(10),
            "col3": range(10),
        })
        layout = flattener.detect_layout_type(df)
        assert layout == "table"

    def test_detect_form_layout(self):
        """Test detection of form layout (sparse, few columns)."""
        config = SpreadsheetFlatteningConfig(
            empty_ratio_threshold_form=0.5,
            max_columns_for_form=4,
        )
        flattener = SpreadsheetFlattener(config)
        df = pd.DataFrame({
            "key": ["Name", "Date", None],
            "value": ["Report", None, None],
        })
        layout = flattener.detect_layout_type(df)
        assert layout == "form"

    def test_detect_mixed_layout(self):
        """Test detection of mixed layout."""
        flattener = SpreadsheetFlattener()
        df = pd.DataFrame({
            "col1": [1, None, 3, None, 5],
            "col2": ["a", None, "c", None, "e"],
            "col3": [None, "x", None, "y", None],
        })
        layout = flattener.detect_layout_type(df)
        assert layout == "mixed"

    def test_detect_empty_layout(self):
        """Test layout detection with empty DataFrame."""
        flattener = SpreadsheetFlattener()
        df = pd.DataFrame()
        layout = flattener.detect_layout_type(df)
        assert layout == "form"


class TestFlattenToMarkdown:
    """Tests for Markdown table flattening."""

    def test_flatten_simple_table(self):
        """Test flattening of simple table to Markdown."""
        flattener = SpreadsheetFlattener()
        df = pd.DataFrame({
            0: ["Name", "Alice", "Bob"],
            1: ["Age", "25", "30"],
        })
        result = flattener.flatten_to_markdown(df, "Test Sheet")

        assert "## Sheet: Test Sheet" in result
        assert "| Name | Age |" in result
        assert "| Alice | 25 |" in result
        assert "| Bob | 30 |" in result
        assert "---" in result  # Separator line

    def test_flatten_with_notes(self):
        """Test flattening with appended notes."""
        flattener = SpreadsheetFlattener()
        df = pd.DataFrame({0: ["Header", "Data"]})
        notes = ["See appendix A", "Data as of 2024"]
        result = flattener.flatten_to_markdown(df, "Sheet", notes)

        assert "**Notes:**" in result
        assert "- See appendix A" in result
        assert "- Data as of 2024" in result

    def test_flatten_empty_sheet(self):
        """Test flattening empty sheet."""
        flattener = SpreadsheetFlattener()
        df = pd.DataFrame()
        result = flattener.flatten_to_markdown(df, "Empty")

        assert "## Sheet: Empty" in result
        assert "*Empty sheet*" in result

    def test_flatten_escapes_pipes(self):
        """Test that pipe characters in cells are escaped."""
        flattener = SpreadsheetFlattener()
        df = pd.DataFrame({
            0: ["Header", "Value|With|Pipes"],
        })
        result = flattener.flatten_to_markdown(df, "Sheet")

        assert r"Value\|With\|Pipes" in result

    def test_flatten_respects_max_rows(self):
        """Test that large tables are truncated."""
        config = SpreadsheetFlatteningConfig(max_table_rows_markdown=5)
        flattener = SpreadsheetFlattener(config)
        df = pd.DataFrame({0: ["Header"] + [f"Row{i}" for i in range(20)]})
        result = flattener.flatten_to_markdown(df, "Sheet")

        assert "... and 15 more rows" in result


class TestFlattenToProse:
    """Tests for prose flattening."""

    def test_flatten_key_value_pairs(self):
        """Test flattening of key-value pairs."""
        flattener = SpreadsheetFlattener()
        df = pd.DataFrame({
            0: ["Name", "Date", "Author"],
            1: ["Q4 Report", "2024-01-15", "Finance Team"],
        })
        result = flattener.flatten_to_prose(df, "Summary")

        assert "Sheet: Summary" in result
        assert "Name: Q4 Report" in result
        assert "Date: 2024-01-15" in result
        assert "Author: Finance Team" in result

    def test_flatten_empty_sheet_prose(self):
        """Test prose flattening of empty sheet."""
        flattener = SpreadsheetFlattener()
        df = pd.DataFrame()
        result = flattener.flatten_to_prose(df, "Empty")

        assert "Sheet: Empty" in result
        assert "(Empty sheet)" in result

    def test_flatten_with_notes_prose(self):
        """Test prose flattening with notes."""
        flattener = SpreadsheetFlattener()
        df = pd.DataFrame({0: ["Key"], 1: ["Value"]})
        notes = ["Important note"]
        result = flattener.flatten_to_prose(df, "Sheet", notes)

        assert "Notes:" in result
        assert "- Important note" in result


class TestFlattenMain:
    """Tests for main flatten() method."""

    def test_flatten_single_sheet(self):
        """Test flattening single sheet text."""
        flattener = SpreadsheetFlattener()
        text = "[SHEET:Sales]\nProduct | Price | Qty\nApple | 1.50 | 100\nBanana | 0.75 | 200"
        metadata = {}

        result_text, result_meta = flattener.flatten(text, metadata)

        assert result_meta["flattening_applied"] is True
        assert len(result_meta["sheets_flattened"]) == 1
        assert result_meta["sheets_flattened"][0]["sheet_name"] == "Sales"
        assert "original_row_count" in result_meta

    def test_flatten_multiple_sheets(self):
        """Test flattening multiple sheets."""
        flattener = SpreadsheetFlattener()
        text = "[SHEET:Sheet1]\na | b\n1 | 2\n\n[SHEET:Sheet2]\nx | y\n3 | 4"
        metadata = {}

        result_text, result_meta = flattener.flatten(text, metadata)

        assert len(result_meta["sheets_flattened"]) == 2
        assert "Sheet1" in result_text
        assert "Sheet2" in result_text

    def test_flatten_no_sheets_found(self):
        """Test handling of text without sheet markers."""
        flattener = SpreadsheetFlattener()
        text = "Just some plain text without any sheet markers"
        metadata = {}

        result_text, result_meta = flattener.flatten(text, metadata)

        assert result_meta["flattening_applied"] is False
        assert result_text == text

    def test_flatten_preserves_metadata(self):
        """Test that flattening adds to existing metadata."""
        flattener = SpreadsheetFlattener()
        text = "[SHEET:Test]\na | b\n1 | 2"
        metadata = {"existing_key": "existing_value"}

        _, result_meta = flattener.flatten(text, metadata)

        # Original metadata should be preserved (not modified in place)
        assert "existing_key" not in result_meta  # We return new metadata
        assert "flattening_applied" in result_meta


class TestSpreadsheetFlatteningConfig:
    """Tests for SpreadsheetFlatteningConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SpreadsheetFlatteningConfig()

        assert config.empty_ratio_threshold_form == 0.6
        assert config.max_columns_for_form == 4
        assert config.min_rows_for_table == 3
        assert config.include_sheet_headers is True
        assert config.max_table_rows_markdown == 100
        assert len(config.notes_patterns) > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = SpreadsheetFlatteningConfig(
            empty_ratio_threshold_form=0.8,
            max_columns_for_form=6,
            include_sheet_headers=False,
        )

        assert config.empty_ratio_threshold_form == 0.8
        assert config.max_columns_for_form == 6
        assert config.include_sheet_headers is False


class TestParseSheetText:
    """Tests for internal _parse_sheet_text method."""

    def test_parse_single_sheet(self):
        """Test parsing single sheet."""
        flattener = SpreadsheetFlattener()
        text = "[SHEET:Data]\na | b | c\n1 | 2 | 3"

        sheets = flattener._parse_sheet_text(text)

        assert len(sheets) == 1
        assert sheets[0][0] == "Data"
        assert len(sheets[0][1]) == 2  # 2 rows

    def test_parse_multiple_sheets(self):
        """Test parsing multiple sheets."""
        flattener = SpreadsheetFlattener()
        text = "[SHEET:First]\na | b\n[SHEET:Second]\nx | y | z"

        sheets = flattener._parse_sheet_text(text)

        assert len(sheets) == 2
        assert sheets[0][0] == "First"
        assert sheets[1][0] == "Second"

    def test_parse_empty_sheet(self):
        """Test parsing empty sheet."""
        flattener = SpreadsheetFlattener()
        text = "[SHEET:Empty]\n[SHEET:HasData]\na | b"

        sheets = flattener._parse_sheet_text(text)

        assert len(sheets) == 2
        assert sheets[0][0] == "Empty"
        assert sheets[0][1].empty
        assert not sheets[1][1].empty

    def test_parse_normalizes_row_lengths(self):
        """Test that rows are normalized to same length."""
        flattener = SpreadsheetFlattener()
        text = "[SHEET:Test]\na | b | c\nx | y"  # Second row has fewer columns

        sheets = flattener._parse_sheet_text(text)

        df = sheets[0][1]
        assert all(len(row) == 3 for _, row in df.iterrows())
