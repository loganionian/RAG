"""Unit tests for CSV/TSV document loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.loader import (
    DocumentParseError,
    load_csv,
    DocumentLoader,
    HANDLERS,
    _detect_csv_delimiter,
)


class TestCSVBasicLoading:
    """Tests for basic CSV file loading functionality."""

    def test_load_csv_simple(self, tmp_path: Path):
        """Test loading simple CSV file with comma delimiter."""
        csv_content = "name,age,city\nAlice,25,NYC\nBob,30,LA\n"
        csv_path = tmp_path / "simple.csv"
        csv_path.write_text(csv_content)

        text, metadata = load_csv(csv_path)

        assert "[SHEET:simple]" in text
        assert "name | age | city" in text
        assert "Alice | 25 | NYC" in text
        assert "Bob | 30 | LA" in text
        assert metadata["total_rows"] == 3
        assert metadata["total_columns"] == 3
        assert metadata["title"] == "simple"
        assert metadata["sheet_count"] == 1
        assert metadata["sheet_names"] == ["simple"]

    def test_load_csv_with_headers(self, tmp_path: Path):
        """Test that header row is included in output."""
        csv_content = "ID,Product,Price\n1,Widget,19.99\n2,Gadget,29.99\n"
        csv_path = tmp_path / "products.csv"
        csv_path.write_text(csv_content)

        text, metadata = load_csv(csv_path)

        # Header should be first row
        lines = text.split("\n")
        assert lines[1] == "ID | Product | Price"
        assert metadata["total_rows"] == 3

    def test_load_tsv(self, tmp_path: Path):
        """Test loading TSV file with tab delimiter."""
        tsv_content = "name\tage\tcity\nAlice\t25\tNYC\nBob\t30\tLA\n"
        tsv_path = tmp_path / "data.tsv"
        tsv_path.write_text(tsv_content)

        text, metadata = load_csv(tsv_path)

        assert "[SHEET:data]" in text
        assert "name | age | city" in text
        assert "Alice | 25 | NYC" in text
        assert metadata["detected_delimiter"] == "\t"

    def test_load_csv_empty_file(self, tmp_path: Path):
        """Test loading empty CSV file."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")

        text, metadata = load_csv(csv_path)

        assert "[SHEET:empty]" in text
        assert metadata["total_rows"] == 0
        assert metadata["total_columns"] == 0

    def test_csv_handler_registered(self):
        """Test that CSV and TSV handlers are registered."""
        assert ".csv" in HANDLERS
        assert ".tsv" in HANDLERS
        assert HANDLERS[".csv"] == load_csv
        assert HANDLERS[".tsv"] == load_csv


class TestCSVDelimiterDetection:
    """Tests for CSV delimiter detection."""

    def test_detect_comma_delimiter(self):
        """Test detection of comma delimiter."""
        content = "a,b,c\n1,2,3\n4,5,6\n"
        delimiter = _detect_csv_delimiter(content)
        assert delimiter == ","

    def test_detect_semicolon_delimiter(self):
        """Test detection of semicolon delimiter."""
        content = "a;b;c\n1;2;3\n4;5;6\n"
        delimiter = _detect_csv_delimiter(content)
        assert delimiter == ";"

    def test_detect_tab_delimiter(self):
        """Test detection of tab delimiter."""
        content = "a\tb\tc\n1\t2\t3\n4\t5\t6\n"
        delimiter = _detect_csv_delimiter(content)
        assert delimiter == "\t"

    def test_detect_pipe_delimiter(self):
        """Test detection of pipe delimiter."""
        content = "a|b|c\n1|2|3\n4|5|6\n"
        delimiter = _detect_csv_delimiter(content)
        assert delimiter == "|"

    def test_detect_delimiter_single_column(self):
        """Test delimiter detection with single column data."""
        content = "value\n1\n2\n3\n"
        delimiter = _detect_csv_delimiter(content)
        # Should default to comma when no multi-column delimiter found
        assert delimiter == ","

    def test_load_csv_semicolon_delimiter(self, tmp_path: Path):
        """Test loading CSV with semicolon delimiter."""
        csv_content = "name;age;city\nAlice;25;NYC\nBob;30;LA\n"
        csv_path = tmp_path / "semicolon.csv"
        csv_path.write_text(csv_content)

        text, metadata = load_csv(csv_path)

        assert "name | age | city" in text
        assert metadata["detected_delimiter"] == ";"


class TestCSVEncodingHandling:
    """Tests for CSV encoding detection and fallback."""

    def test_utf8_encoding(self, tmp_path: Path):
        """Test loading UTF-8 encoded CSV."""
        csv_content = "name,city\nAmélie,Montréal\nMüller,München\n"
        csv_path = tmp_path / "utf8.csv"
        csv_path.write_text(csv_content, encoding="utf-8")

        text, metadata = load_csv(csv_path)

        assert "Amélie | Montréal" in text
        assert "Müller | München" in text
        assert "encoding_fallback" not in metadata

    def test_latin1_fallback_encoding(self, tmp_path: Path):
        """Test fallback to latin-1 encoding."""
        csv_content = "name,city\nAmélie,Montréal\n"
        csv_path = tmp_path / "latin1.csv"
        csv_path.write_bytes(csv_content.encode("latin-1"))

        text, metadata = load_csv(csv_path)

        assert "Amélie" in text
        assert metadata.get("encoding_fallback") in ["cp1252", "iso-8859-1", "latin-1"]

    def test_cp1252_fallback_encoding(self, tmp_path: Path):
        """Test fallback to cp1252 encoding."""
        # CP1252 specific character (euro sign at 0x80)
        csv_content = "price\n€100\n"
        csv_path = tmp_path / "cp1252.csv"
        csv_path.write_bytes(csv_content.encode("cp1252"))

        text, metadata = load_csv(csv_path)

        # Should load without error
        assert metadata.get("encoding_fallback") in ["cp1252", "iso-8859-1", "latin-1"]


class TestCSVContentFormatting:
    """Tests for CSV content extraction and formatting."""

    def test_csv_sheet_marker(self, tmp_path: Path):
        """Test that [SHEET:filename] marker is present."""
        csv_path = tmp_path / "test_data.csv"
        csv_path.write_text("a,b\n1,2\n")

        text, metadata = load_csv(csv_path)

        assert text.startswith("[SHEET:test_data]")

    def test_csv_pipe_separated_values(self, tmp_path: Path):
        """Test that values are pipe-separated in output."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("col1,col2,col3\nval1,val2,val3\n")

        text, metadata = load_csv(csv_path)

        assert "col1 | col2 | col3" in text
        assert "val1 | val2 | val3" in text

    def test_csv_empty_rows_skipped(self, tmp_path: Path):
        """Test that empty rows are skipped."""
        csv_content = "a,b\n1,2\n,,\n3,4\n"
        csv_path = tmp_path / "sparse.csv"
        csv_path.write_text(csv_content)

        text, metadata = load_csv(csv_path)

        # Empty row should be skipped
        assert metadata["total_rows"] == 3  # header + 2 data rows

    def test_csv_whitespace_trimmed(self, tmp_path: Path):
        """Test that cell values have whitespace trimmed."""
        csv_content = "name,value\n  Alice  ,  100  \n"
        csv_path = tmp_path / "whitespace.csv"
        csv_path.write_text(csv_content)

        text, metadata = load_csv(csv_path)

        assert "Alice | 100" in text

    def test_csv_quoted_fields(self, tmp_path: Path):
        """Test handling of quoted fields with embedded delimiters."""
        csv_content = 'name,description\nProduct,"A product, with comma"\n'
        csv_path = tmp_path / "quoted.csv"
        csv_path.write_text(csv_content)

        text, metadata = load_csv(csv_path)

        assert "Product | A product, with comma" in text


class TestCSVErrorHandling:
    """Tests for CSV error handling scenarios."""

    def test_csv_malformed_data(self, tmp_path: Path):
        """Test handling of malformed CSV data."""
        # CSV with inconsistent column counts (valid CSV, just inconsistent)
        csv_content = "a,b,c\n1,2\n3,4,5,6\n"
        csv_path = tmp_path / "malformed.csv"
        csv_path.write_text(csv_content)

        # Should not raise, just parse what it can
        text, metadata = load_csv(csv_path)

        assert "[SHEET:malformed]" in text
        assert metadata["total_rows"] == 3

    def test_csv_single_column(self, tmp_path: Path):
        """Test handling of single-column CSV."""
        csv_content = "value\n1\n2\n3\n"
        csv_path = tmp_path / "single.csv"
        csv_path.write_text(csv_content)

        text, metadata = load_csv(csv_path)

        assert "[SHEET:single]" in text
        assert metadata["total_columns"] == 1


class TestCSVMetadata:
    """Tests for CSV metadata extraction."""

    def test_csv_metadata_title(self, tmp_path: Path):
        """Test that title is extracted from filename."""
        csv_path = tmp_path / "my_data_file.csv"
        csv_path.write_text("a,b\n1,2\n")

        text, metadata = load_csv(csv_path)

        assert metadata["title"] == "my_data_file"

    def test_csv_metadata_counts(self, tmp_path: Path):
        """Test row and column counts in metadata."""
        csv_content = "a,b,c,d\n1,2,3,4\n5,6,7,8\n9,10,11,12\n"
        csv_path = tmp_path / "counts.csv"
        csv_path.write_text(csv_content)

        text, metadata = load_csv(csv_path)

        assert metadata["total_rows"] == 4  # Including header
        assert metadata["total_columns"] == 4

    def test_csv_metadata_sheet_info(self, tmp_path: Path):
        """Test sheet-related metadata for consistency with Excel."""
        csv_path = tmp_path / "sheet_test.csv"
        csv_path.write_text("a,b\n1,2\n")

        text, metadata = load_csv(csv_path)

        assert metadata["sheet_count"] == 1
        assert metadata["sheet_names"] == ["sheet_test"]

    def test_csv_metadata_delimiter(self, tmp_path: Path):
        """Test detected delimiter in metadata."""
        csv_path = tmp_path / "semicolon.csv"
        csv_path.write_text("a;b;c\n1;2;3\n")

        text, metadata = load_csv(csv_path)

        assert metadata["detected_delimiter"] == ";"


class TestCSVIntegration:
    """Integration tests for CSV loading through the pipeline."""

    def test_csv_through_document_loader(self, tmp_path: Path):
        """Test CSV ingestion through DocumentLoader."""
        csv_content = "id,name,value\n1,Widget,100\n2,Gadget,200\n"
        csv_path = tmp_path / "products.csv"
        csv_path.write_text(csv_content)

        loader = DocumentLoader(input_root=tmp_path)
        document, content_hash = loader.load(csv_path)

        assert document.source_type == "csv"
        assert "[SHEET:products]" in document.text
        assert "Widget" in document.text
        assert document.metadata["total_rows"] == 3
        assert content_hash is not None

    def test_tsv_through_document_loader(self, tmp_path: Path):
        """Test TSV ingestion through DocumentLoader."""
        tsv_content = "id\tname\tvalue\n1\tWidget\t100\n"
        tsv_path = tmp_path / "data.tsv"
        tsv_path.write_text(tsv_content)

        loader = DocumentLoader(input_root=tmp_path)
        document, content_hash = loader.load(tsv_path)

        assert document.source_type == "tsv"
        assert "[SHEET:data]" in document.text

    def test_csv_chunking(self, tmp_path: Path):
        """Test that CSV content can be chunked correctly."""
        from ingestion.chunker import chunk_document
        from ingestion.models import Document

        # Create document with substantial CSV content
        csv_content = "[SHEET:Sheet1]\n"
        csv_content += "ID | Name | Description\n"
        for i in range(100):
            csv_content += f"{i} | Item{i} | Description of item {i} with additional text\n"

        mock_document = Document(
            doc_id="test-csv",
            path=tmp_path / "test.csv",
            source_type="csv",
            text=csv_content,
            metadata={
                "sheet_count": 1,
                "total_rows": 101,
            },
        )

        chunks = chunk_document(mock_document)

        assert len(chunks) > 0
        assert "[SHEET:Sheet1]" in chunks[0].text
        assert all(c.chunk_id.startswith("test-csv::chunk-") for c in chunks)


class TestCSVEdgeCases:
    """Tests for CSV edge cases and boundary conditions."""

    def test_csv_unicode_content(self, tmp_path: Path):
        """Test handling of Unicode content in CSV."""
        csv_content = "name,greeting\n日本語,こんにちは\n中文,你好\n"
        csv_path = tmp_path / "unicode.csv"
        csv_path.write_text(csv_content, encoding="utf-8")

        text, metadata = load_csv(csv_path)

        assert "日本語 | こんにちは" in text
        assert "中文 | 你好" in text

    def test_csv_numeric_values(self, tmp_path: Path):
        """Test that numeric values are preserved."""
        csv_content = "int,float,negative\n12345,99.99,-42\n"
        csv_path = tmp_path / "numbers.csv"
        csv_path.write_text(csv_content)

        text, metadata = load_csv(csv_path)

        assert "12345 | 99.99 | -42" in text

    def test_csv_large_file_performance(self, tmp_path: Path):
        """Test loading of larger CSV file."""
        # Create CSV with 1000 rows
        rows = ["id,value"]
        for i in range(1000):
            rows.append(f"{i},{i * 10}")

        csv_content = "\n".join(rows)
        csv_path = tmp_path / "large.csv"
        csv_path.write_text(csv_content)

        text, metadata = load_csv(csv_path)

        assert metadata["total_rows"] == 1001  # Including header
        assert metadata["total_columns"] == 2

    def test_csv_special_characters_in_filename(self, tmp_path: Path):
        """Test handling of special characters in filename."""
        csv_path = tmp_path / "data (2024) - final.csv"
        csv_path.write_text("a,b\n1,2\n")

        text, metadata = load_csv(csv_path)

        assert "[SHEET:data (2024) - final]" in text
        assert metadata["title"] == "data (2024) - final"
