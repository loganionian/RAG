"""Spreadsheet classification logic for tabular vs report-like detection.

This module provides heuristics to classify CSV/XLSX files as either structured
tabular data or report-like unstructured documents, enabling appropriate
downstream processing strategies.
"""
from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SpreadsheetClassificationConfig:
    """Configuration for spreadsheet classification heuristics.

    Attributes:
        min_rows_for_tabular: Minimum row count to suggest tabular data.
        max_columns_for_tabular: Maximum column count for typical tabular data.
        numeric_ratio_threshold: Minimum ratio of numeric cells for tabular.
        long_text_threshold: Character count above which a cell is "long text".
        long_text_ratio_max: Maximum ratio of long-text cells for tabular.
        empty_cell_ratio_max: Maximum ratio of empty cells for tabular.
        override_patterns: List of filename patterns to force classification.
            Each entry: {"pattern": "*.csv", "classification": "tabular"}
    """

    min_rows_for_tabular: int = 10
    max_columns_for_tabular: int = 50
    numeric_ratio_threshold: float = 0.3
    long_text_threshold: int = 200
    long_text_ratio_max: float = 0.1
    empty_cell_ratio_max: float = 0.5
    override_patterns: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Result of spreadsheet classification.

    Attributes:
        classification: Either "tabular" or "report_like".
        confidence: Confidence score from 0.0 to 1.0.
        reasons: List of human-readable reasons for the classification.
        metrics: Dictionary of computed metrics used for classification.
    """

    classification: Literal["tabular", "report_like"]
    confidence: float
    reasons: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)


def _compute_numeric_ratio(df: pd.DataFrame) -> float:
    """Compute the ratio of numeric cells in the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        Ratio of numeric cells (0.0 to 1.0).
    """
    if df.empty:
        return 0.0

    total_cells = df.size
    if total_cells == 0:
        return 0.0

    numeric_count = 0
    for col in df.columns:
        # Try to convert column to numeric
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        numeric_count += numeric_col.notna().sum()

    return numeric_count / total_cells


def _compute_long_text_ratio(df: pd.DataFrame, threshold: int) -> float:
    """Compute the ratio of cells containing long text.

    Args:
        df: Input DataFrame.
        threshold: Character count threshold for "long text".

    Returns:
        Ratio of long-text cells (0.0 to 1.0).
    """
    if df.empty:
        return 0.0

    total_cells = df.size
    if total_cells == 0:
        return 0.0

    long_text_count = 0
    for col in df.columns:
        str_col = df[col].astype(str)
        long_text_count += (str_col.str.len() > threshold).sum()

    return long_text_count / total_cells


def _compute_empty_cell_ratio(df: pd.DataFrame) -> float:
    """Compute the ratio of empty/null cells in the DataFrame.

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

    # Count NaN/None values
    na_count = df.isna().sum().sum()

    # Count empty strings (only for non-NA values)
    empty_string_count = 0
    for col in df.columns:
        # Only check non-NA values for empty strings
        non_na_mask = df[col].notna()
        if non_na_mask.any():
            str_values = df.loc[non_na_mask, col].astype(str)
            empty_string_count += (str_values.str.strip() == "").sum()

    return (na_count + empty_string_count) / total_cells


def _check_override_patterns(
    filename: str, patterns: List[Dict[str, str]]
) -> Optional[Literal["tabular", "report_like"]]:
    """Check if filename matches any override pattern.

    Args:
        filename: Name of the file to check.
        patterns: List of pattern dictionaries with "pattern" and "classification" keys.

    Returns:
        Classification if a pattern matches, None otherwise.
    """
    for pattern_entry in patterns:
        pattern = pattern_entry.get("pattern", "")
        classification = pattern_entry.get("classification", "")

        if fnmatch.fnmatch(filename.lower(), pattern.lower()):
            if classification in ("tabular", "report_like"):
                return classification

    return None


def classify_dataframe(
    df: pd.DataFrame,
    config: Optional[SpreadsheetClassificationConfig] = None,
    filename: Optional[str] = None,
) -> ClassificationResult:
    """Classify a DataFrame as tabular or report-like.

    Evaluates multiple heuristics to determine if the data represents
    structured tabular data or a report-like document.

    Args:
        df: Pandas DataFrame to classify.
        config: Classification configuration. Uses defaults if None.
        filename: Optional filename for override pattern matching.

    Returns:
        ClassificationResult with classification, confidence, and reasoning.
    """
    if config is None:
        config = SpreadsheetClassificationConfig()

    reasons: List[str] = []
    tabular_score = 0.0
    max_score = 0.0

    # Check override patterns first
    if filename and config.override_patterns:
        override = _check_override_patterns(filename, config.override_patterns)
        if override:
            return ClassificationResult(
                classification=override,
                confidence=1.0,
                reasons=[f"Filename matched override pattern, forced to {override}"],
                metrics={"override_applied": True, "filename": filename},
            )

    # Compute metrics
    row_count = len(df)
    col_count = len(df.columns)
    numeric_ratio = _compute_numeric_ratio(df)
    long_text_ratio = _compute_long_text_ratio(df, config.long_text_threshold)
    empty_ratio = _compute_empty_cell_ratio(df)

    metrics = {
        "row_count": row_count,
        "column_count": col_count,
        "numeric_ratio": round(numeric_ratio, 3),
        "long_text_ratio": round(long_text_ratio, 3),
        "empty_cell_ratio": round(empty_ratio, 3),
    }

    # Heuristic 1: Row count (weight: 1.0)
    max_score += 1.0
    if row_count >= config.min_rows_for_tabular:
        tabular_score += 1.0
        reasons.append(f"Row count ({row_count}) >= threshold ({config.min_rows_for_tabular})")
    else:
        reasons.append(f"Row count ({row_count}) < threshold ({config.min_rows_for_tabular})")

    # Heuristic 2: Column count (weight: 0.5)
    max_score += 0.5
    if 2 <= col_count <= config.max_columns_for_tabular:
        tabular_score += 0.5
        reasons.append(f"Column count ({col_count}) in typical tabular range (2-{config.max_columns_for_tabular})")
    else:
        reasons.append(f"Column count ({col_count}) outside typical tabular range")

    # Heuristic 3: Numeric ratio (weight: 1.5)
    max_score += 1.5
    if numeric_ratio >= config.numeric_ratio_threshold:
        tabular_score += 1.5
        reasons.append(
            f"Numeric ratio ({numeric_ratio:.1%}) >= threshold ({config.numeric_ratio_threshold:.0%})"
        )
    else:
        reasons.append(
            f"Numeric ratio ({numeric_ratio:.1%}) < threshold ({config.numeric_ratio_threshold:.0%})"
        )

    # Heuristic 4: Long text presence (weight: 1.5)
    max_score += 1.5
    if long_text_ratio <= config.long_text_ratio_max:
        tabular_score += 1.5
        reasons.append(
            f"Long text ratio ({long_text_ratio:.1%}) <= threshold ({config.long_text_ratio_max:.0%})"
        )
    else:
        reasons.append(
            f"Long text ratio ({long_text_ratio:.1%}) > threshold ({config.long_text_ratio_max:.0%}) - suggests report"
        )

    # Heuristic 5: Empty cell ratio (weight: 1.0)
    max_score += 1.0
    if empty_ratio <= config.empty_cell_ratio_max:
        tabular_score += 1.0
        reasons.append(
            f"Empty cell ratio ({empty_ratio:.1%}) <= threshold ({config.empty_cell_ratio_max:.0%})"
        )
    else:
        reasons.append(
            f"Empty cell ratio ({empty_ratio:.1%}) > threshold ({config.empty_cell_ratio_max:.0%}) - sparse data"
        )

    # Calculate confidence and make decision
    confidence = tabular_score / max_score if max_score > 0 else 0.0

    # Decision threshold: 0.5 (50% of weighted score)
    if confidence >= 0.5:
        classification: Literal["tabular", "report_like"] = "tabular"
    else:
        classification = "report_like"

    return ClassificationResult(
        classification=classification,
        confidence=round(confidence, 3),
        reasons=reasons,
        metrics=metrics,
    )


def classify_spreadsheet_file(
    path: Path,
    config: Optional[SpreadsheetClassificationConfig] = None,
    sheet_name: Optional[str] = None,
) -> ClassificationResult:
    """Classify a spreadsheet file (CSV or Excel) as tabular or report-like.

    Args:
        path: Path to the spreadsheet file.
        config: Classification configuration. Uses defaults if None.
        sheet_name: For Excel files, specific sheet to classify. If None,
            classifies the first sheet.

    Returns:
        ClassificationResult with classification, confidence, and reasoning.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    try:
        if ext == ".csv":
            # Try different delimiters
            for delimiter in [",", ";", "\t", "|"]:
                try:
                    df = pd.read_csv(path, delimiter=delimiter, nrows=1000)
                    if len(df.columns) > 1:
                        break
                except Exception:
                    continue
            else:
                df = pd.read_csv(path, nrows=1000)

        elif ext == ".tsv":
            df = pd.read_csv(path, delimiter="\t", nrows=1000)

        elif ext in (".xlsx", ".xls"):
            if sheet_name:
                df = pd.read_excel(path, sheet_name=sheet_name, nrows=1000)
            else:
                df = pd.read_excel(path, nrows=1000)

        else:
            raise ValueError(f"Unsupported file format: {ext}")

    except Exception as e:
        logger.warning(f"Could not read file for classification: {path} - {e}")
        # Return default classification on read error
        return ClassificationResult(
            classification="report_like",
            confidence=0.0,
            reasons=[f"Could not read file: {e}"],
            metrics={"read_error": str(e)},
        )

    return classify_dataframe(df, config=config, filename=path.name)
