"""Spreadsheet-to-text flattening for report-like sheets.

This module converts report-like spreadsheets (classified by spreadsheet_classifier)
into human-readable text formats (Markdown tables or prose) for better semantic
chunking and retrieval.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SpreadsheetFlatteningConfig:
    """Configuration for spreadsheet flattening behavior.

    Attributes:
        empty_ratio_threshold_form: Empty cell ratio above which layout is "form".
        max_columns_for_form: Maximum columns to consider a layout as "form".
        min_rows_for_table: Minimum rows to consider a layout as "table".
        include_sheet_headers: Whether to include sheet name as section header.
        max_table_rows_markdown: Maximum rows before switching to prose for tables.
        notes_patterns: Regex patterns to detect footer notes in cells.
    """

    empty_ratio_threshold_form: float = 0.6
    max_columns_for_form: int = 4
    min_rows_for_table: int = 3
    include_sheet_headers: bool = True
    max_table_rows_markdown: int = 100
    notes_patterns: List[str] = field(
        default_factory=lambda: [
            r"^note[s]?:",
            r"^see ",
            r"^\*",
            r"^source:",
        ]
    )


LayoutType = Literal["table", "form", "mixed"]


def _compute_empty_ratio(df: pd.DataFrame) -> float:
    """Compute the ratio of empty/null cells in a DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        Ratio of empty cells (0.0 to 1.0).
    """
    if df.empty:
        return 1.0

    total_cells = df.size
    if total_cells == 0:
        return 1.0

    na_count = df.isna().sum().sum()
    empty_string_count = 0

    for col in df.columns:
        non_na_mask = df[col].notna()
        if non_na_mask.any():
            str_values = df.loc[non_na_mask, col].astype(str)
            empty_string_count += (str_values.str.strip() == "").sum()

    return (na_count + empty_string_count) / total_cells


def _has_header_row(df: pd.DataFrame) -> bool:
    """Detect if the first row looks like a header.

    Heuristics:
    - First row is all strings
    - First row values are distinct
    - Data rows have different types than header

    Args:
        df: Input DataFrame.

    Returns:
        True if first row appears to be a header.
    """
    if df.empty or len(df) < 2:
        return False

    first_row = df.iloc[0]

    # Check if first row is all non-numeric strings
    all_strings = True
    for val in first_row:
        if pd.isna(val):
            continue
        try:
            float(val)
            all_strings = False
            break
        except (ValueError, TypeError):
            pass

    if not all_strings:
        return False

    # Check if values are unique (typical for headers)
    non_empty_values = [str(v).strip() for v in first_row if pd.notna(v) and str(v).strip()]
    return len(non_empty_values) == len(set(non_empty_values))


def _extract_notes(df: pd.DataFrame, patterns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Extract footer notes from the DataFrame.

    Scans the last rows for cells matching note patterns and removes them.

    Args:
        df: Input DataFrame.
        patterns: List of regex patterns to match notes.

    Returns:
        Tuple of (DataFrame without notes, list of extracted notes).
    """
    notes = []
    rows_to_drop = []

    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    # Scan last 10 rows for notes
    scan_rows = min(10, len(df))
    for idx in range(len(df) - scan_rows, len(df)):
        row = df.iloc[idx]
        row_is_note = False

        for val in row:
            if pd.isna(val):
                continue
            val_str = str(val).strip()
            if not val_str:
                continue

            for pattern in compiled_patterns:
                if pattern.match(val_str):
                    notes.append(val_str)
                    row_is_note = True
                    break
            if row_is_note:
                break

        if row_is_note:
            rows_to_drop.append(idx)

    if rows_to_drop:
        df = df.drop(df.index[rows_to_drop])

    return df, notes


class SpreadsheetFlattener:
    """Convert report-like spreadsheets to readable text."""

    def __init__(self, config: Optional[SpreadsheetFlatteningConfig] = None):
        """Initialize the flattener.

        Args:
            config: Flattening configuration. Uses defaults if None.
        """
        self.config = config or SpreadsheetFlatteningConfig()

    def detect_layout_type(self, df: pd.DataFrame) -> LayoutType:
        """Detect the layout type of a DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Layout type: "table", "form", or "mixed".
        """
        if df.empty:
            return "form"

        empty_ratio = _compute_empty_ratio(df)
        col_count = len(df.columns)
        row_count = len(df)

        # Form layout: sparse data, few columns, key-value pairs
        if (
            empty_ratio >= self.config.empty_ratio_threshold_form
            and col_count <= self.config.max_columns_for_form
        ):
            return "form"

        # Table layout: regular grid with consistent columns
        if row_count >= self.config.min_rows_for_table and empty_ratio < 0.3:
            return "table"

        # Mixed: moderate sparsity or doesn't fit either pattern
        return "mixed"

    def flatten_to_markdown(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        notes: Optional[List[str]] = None,
    ) -> str:
        """Convert a DataFrame to Markdown table format.

        Args:
            df: Input DataFrame.
            sheet_name: Name of the sheet for the header.
            notes: Optional list of notes to append.

        Returns:
            Markdown-formatted string with table.
        """
        parts = []

        if self.config.include_sheet_headers:
            parts.append(f"## Sheet: {sheet_name}")
            parts.append("")

        if df.empty:
            parts.append("*Empty sheet*")
            return "\n".join(parts)

        # Determine if first row is header
        has_header = _has_header_row(df)

        if has_header:
            headers = [str(v).strip() if pd.notna(v) else "" for v in df.iloc[0]]
            data_df = df.iloc[1:]
        else:
            headers = [f"Col{i+1}" for i in range(len(df.columns))]
            data_df = df

        # Build markdown table
        # Header row
        header_line = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        parts.append(header_line)
        parts.append(separator)

        # Data rows (limit to max_table_rows_markdown)
        rows_to_render = min(len(data_df), self.config.max_table_rows_markdown)
        for idx in range(rows_to_render):
            row = data_df.iloc[idx]
            cells = [str(v).strip() if pd.notna(v) else "" for v in row]
            # Escape pipe characters in cell content
            cells = [c.replace("|", "\\|") for c in cells]
            row_line = "| " + " | ".join(cells) + " |"
            parts.append(row_line)

        if len(data_df) > self.config.max_table_rows_markdown:
            parts.append("")
            parts.append(f"*... and {len(data_df) - self.config.max_table_rows_markdown} more rows*")

        # Append notes if present
        if notes:
            parts.append("")
            parts.append("**Notes:**")
            for note in notes:
                parts.append(f"- {note}")

        return "\n".join(parts)

    def flatten_to_prose(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        notes: Optional[List[str]] = None,
    ) -> str:
        """Convert a DataFrame to prose format for sparse/form layouts.

        Args:
            df: Input DataFrame.
            sheet_name: Name of the sheet for the header.
            notes: Optional list of notes to append.

        Returns:
            Prose-formatted string.
        """
        parts = []

        if self.config.include_sheet_headers:
            parts.append(f"Sheet: {sheet_name}")
            parts.append("")

        if df.empty:
            parts.append("(Empty sheet)")
            return "\n".join(parts)

        # For form layout, try to detect key-value pairs
        col_count = len(df.columns)

        if col_count == 2:
            # Likely key-value pairs
            for _, row in df.iterrows():
                key = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
                value = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
                if key or value:
                    if key and value:
                        parts.append(f"{key}: {value}")
                    elif key:
                        parts.append(f"{key}")
                    else:
                        parts.append(f"  {value}")
        else:
            # Mixed/irregular layout - list all non-empty cells
            current_section = None

            for _, row in df.iterrows():
                non_empty_cells = []
                for val in row:
                    if pd.notna(val):
                        val_str = str(val).strip()
                        if val_str:
                            non_empty_cells.append(val_str)

                if not non_empty_cells:
                    continue

                # If single cell, might be a section header
                if len(non_empty_cells) == 1:
                    cell = non_empty_cells[0]
                    # Check if it looks like a header (short, no punctuation at end)
                    if len(cell) < 50 and not cell.endswith((".", ":", ",")):
                        if current_section:
                            parts.append("")
                        current_section = cell
                        parts.append(f"{cell}:")
                        continue

                # Multiple values - format as list or comma-separated
                if len(non_empty_cells) <= 3:
                    parts.append("- " + ", ".join(non_empty_cells))
                else:
                    # First cell might be a label
                    label = non_empty_cells[0]
                    values = non_empty_cells[1:]
                    parts.append(f"- {label}: {', '.join(values)}")

        # Append notes if present
        if notes:
            parts.append("")
            parts.append("Notes:")
            for note in notes:
                parts.append(f"- {note}")

        return "\n".join(parts)

    def _parse_sheet_text(self, text: str) -> List[Tuple[str, pd.DataFrame]]:
        """Parse pipe-separated text with [SHEET:name] markers into DataFrames.

        Args:
            text: Text with sheet markers and pipe-separated rows.

        Returns:
            List of (sheet_name, DataFrame) tuples.
        """
        sheets = []
        sheet_pattern = re.compile(r"\[SHEET:([^\]]+)\]")

        # Split by sheet markers
        parts = sheet_pattern.split(text)

        # parts[0] is content before first marker (usually empty)
        # parts[1] is first sheet name, parts[2] is its content, etc.
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                sheet_name = parts[i].strip()
                sheet_content = parts[i + 1].strip()

                if not sheet_content:
                    sheets.append((sheet_name, pd.DataFrame()))
                    continue

                # Parse pipe-separated rows
                rows = []
                for line in sheet_content.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    cells = [cell.strip() for cell in line.split("|")]
                    rows.append(cells)

                if rows:
                    # Normalize row lengths
                    max_cols = max(len(row) for row in rows)
                    normalized_rows = [
                        row + [""] * (max_cols - len(row)) for row in rows
                    ]
                    df = pd.DataFrame(normalized_rows)
                else:
                    df = pd.DataFrame()

                sheets.append((sheet_name, df))

        return sheets

    def flatten(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Flatten spreadsheet text to human-readable format.

        Main entry point for flattening. Parses the pipe-separated text,
        detects layout type, and applies appropriate flattening strategy.

        Args:
            text: Document text with [SHEET:name] markers and pipe-separated rows.
            metadata: Document metadata dict.

        Returns:
            Tuple of (flattened_text, flattening_metadata).
        """
        flatten_meta: Dict[str, Any] = {
            "flattening_applied": True,
            "sheets_flattened": [],
        }

        sheets = self._parse_sheet_text(text)

        if not sheets:
            logger.warning("No sheets found in text, returning original")
            flatten_meta["flattening_applied"] = False
            return text, flatten_meta

        flattened_parts = []
        total_rows = 0
        total_cols = 0

        for sheet_name, df in sheets:
            # Extract notes before processing
            df_clean, notes = _extract_notes(df, self.config.notes_patterns)

            # Detect layout type
            layout_type = self.detect_layout_type(df_clean)

            # Track stats
            total_rows += len(df)
            total_cols = max(total_cols, len(df.columns) if not df.empty else 0)

            # Apply appropriate flattening strategy
            if layout_type == "table":
                flattened = self.flatten_to_markdown(df_clean, sheet_name, notes)
                strategy = "markdown"
            else:
                flattened = self.flatten_to_prose(df_clean, sheet_name, notes)
                strategy = "prose"

            flattened_parts.append(flattened)

            flatten_meta["sheets_flattened"].append({
                "sheet_name": sheet_name,
                "layout_type": layout_type,
                "flattening_strategy": strategy,
                "row_count": len(df),
                "col_count": len(df.columns) if not df.empty else 0,
                "notes_extracted": len(notes),
            })

            logger.debug(
                "Flattened sheet '%s' using %s strategy (layout: %s)",
                sheet_name,
                strategy,
                layout_type,
            )

        # Use the dominant strategy for the overall metadata
        strategies = [s["flattening_strategy"] for s in flatten_meta["sheets_flattened"]]
        flatten_meta["flattening_strategy"] = max(set(strategies), key=strategies.count) if strategies else "prose"

        layouts = [s["layout_type"] for s in flatten_meta["sheets_flattened"]]
        flatten_meta["layout_type"] = max(set(layouts), key=layouts.count) if layouts else "mixed"

        flatten_meta["original_row_count"] = total_rows
        flatten_meta["original_col_count"] = total_cols

        return "\n\n".join(flattened_parts), flatten_meta
