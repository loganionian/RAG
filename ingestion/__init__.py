"""Ingestion pipeline package for transforming documents into retrievable chunks."""

from .normalizer import NormalizationConfig, NormalizationResult, TextNormalizer
from .pipeline import IngestionPipeline, PipelineConfig
from .spreadsheet_classifier import (
    ClassificationResult,
    SpreadsheetClassificationConfig,
    classify_dataframe,
    classify_spreadsheet_file,
)
from .spreadsheet_flattener import SpreadsheetFlattener, SpreadsheetFlatteningConfig

__all__ = [
    "ClassificationResult",
    "IngestionPipeline",
    "NormalizationConfig",
    "NormalizationResult",
    "PipelineConfig",
    "SpreadsheetClassificationConfig",
    "SpreadsheetFlattener",
    "SpreadsheetFlatteningConfig",
    "TextNormalizer",
    "classify_dataframe",
    "classify_spreadsheet_file",
]
