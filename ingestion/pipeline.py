from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .chunker import chunk_document
from .loader import DocumentLoader, UnsupportedDocumentError, discover_documents
from .models import Document, FailureInfo
from .normalizer import NormalizationConfig, TextNormalizer
from .spreadsheet_classifier import (
    ClassificationResult,
    SpreadsheetClassificationConfig,
    classify_dataframe,
)
from .storage import StorageManager

logger = logging.getLogger(__name__)

# Lazy imports for SQL components (only loaded if SQL ingestion is enabled)
SQLStore = None
SQLStoreConfig = None
MetadataCatalog = None
SQLTableLoader = None


def _load_sql_components():
    """Lazy load SQL components to avoid import errors if not installed."""
    global SQLStore, SQLStoreConfig, MetadataCatalog, SQLTableLoader
    if SQLStore is None:
        from storage.sql_store import SQLStore, SQLStoreConfig
        from storage.metadata_catalog import MetadataCatalog
        from .sql_loader import SQLTableLoader
    return SQLStore, SQLStoreConfig, MetadataCatalog, SQLTableLoader


@dataclass
class PipelineConfig:
    """Configuration for the ingestion pipeline.

    Attributes:
        input_dir: Directory containing source documents.
        output_dir: Directory for processed chunks and manifest.
        chunk_size_tokens: Target token count per chunk.
        chunk_overlap_percent: Overlap as percentage of chunk size (0.10-0.20).
        fail_fast: Stop on first failure instead of continuing.
        normalization_config: Optional configuration for text normalization.
        cleanup_deleted: Remove orphaned docs whose source files were deleted.
        spreadsheet_classification_config: Optional config for classifying
            spreadsheets as tabular vs report-like.
        enable_sql_tabular: Enable SQL ingestion for tabular spreadsheets.
        sql_db_path: Path to DuckDB database file for tabular data.
        tabular_confidence_threshold: Minimum confidence for tabular classification
            to trigger SQL ingestion (0.0-1.0).
    """

    input_dir: Path
    output_dir: Path
    chunk_size_tokens: int = 400
    chunk_overlap_percent: float = 0.15
    fail_fast: bool = False
    normalization_config: Optional[NormalizationConfig] = field(default=None)
    cleanup_deleted: bool = False
    spreadsheet_classification_config: Optional[SpreadsheetClassificationConfig] = field(default=None)
    enable_sql_tabular: bool = False
    sql_db_path: Optional[Path] = None
    tabular_confidence_threshold: float = 0.7


@dataclass
class PipelineResult:
    processed: int
    skipped: int
    failed: int
    chunk_count: int
    failures: List[FailureInfo]
    start_time: str
    end_time: str
    duration_seconds: float
    cleaned_up: int = 0
    sql_tables_created: int = 0
    sql_tables_skipped: int = 0
    sql_rows_ingested: int = 0


SPREADSHEET_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}


class IngestionPipeline:
    """Pipeline for ingesting documents into normalized, chunked format.

    The pipeline discovers documents, loads them, optionally normalizes text,
    chunks the content, and persists the results.
    """

    def __init__(self, config: PipelineConfig):
        """Initialize the ingestion pipeline.

        Args:
            config: Pipeline configuration including paths and normalization settings.
        """
        self.config = config

        # Create normalizer if config provided
        normalizer: Optional[TextNormalizer] = None
        if config.normalization_config is not None:
            normalizer = TextNormalizer(config.normalization_config)

        self.loader = DocumentLoader(config.input_dir, normalizer=normalizer)
        self.storage = StorageManager(config.output_dir)

        # Initialize SQL components if enabled
        self.sql_store = None
        self.sql_catalog = None
        self.sql_loader = None
        if config.enable_sql_tabular:
            self._init_sql_components()

    def _init_sql_components(self) -> None:
        """Initialize SQL storage components for tabular data."""
        SQLStore, SQLStoreConfig, MetadataCatalog, SQLTableLoader = _load_sql_components()

        # Default database path if not specified
        db_path = self.config.sql_db_path
        if db_path is None:
            db_path = self.config.output_dir.parent / "tabular" / "tables.duckdb"

        sql_config = SQLStoreConfig(db_path=db_path)
        self.sql_store = SQLStore(sql_config)
        self.sql_catalog = MetadataCatalog(self.sql_store)
        self.sql_loader = SQLTableLoader(self.sql_store, self.sql_catalog)

        logger.info("SQL tabular ingestion enabled, database: %s", db_path)

    def _should_use_sql_path(self, classification: ClassificationResult) -> bool:
        """Determine if a spreadsheet should use SQL ingestion path.

        Args:
            classification: ClassificationResult from spreadsheet classifier.

        Returns:
            True if the spreadsheet should be loaded into SQL.
        """
        if not self.config.enable_sql_tabular:
            return False
        if self.sql_loader is None:
            return False
        return (
            classification.classification == "tabular"
            and classification.confidence >= self.config.tabular_confidence_threshold
        )

    def _is_spreadsheet(self, document: Document) -> bool:
        """Check if document is a spreadsheet type.

        Args:
            document: Document to check.

        Returns:
            True if document is CSV, TSV, or Excel format.
        """
        ext = document.path.suffix.lower()
        return ext in SPREADSHEET_EXTENSIONS

    def _classify_spreadsheet(self, document: Document) -> ClassificationResult:
        """Classify a spreadsheet document as tabular or report-like.

        Args:
            document: Spreadsheet document to classify.

        Returns:
            ClassificationResult with classification and reasoning.
        """
        path = document.path
        ext = path.suffix.lower()

        try:
            # Load data into DataFrame for classification
            if ext == ".csv":
                # Detect delimiter from the loaded text
                delimiter = document.metadata.get("detected_delimiter", ",")
                df = pd.read_csv(path, delimiter=delimiter, nrows=1000)
            elif ext == ".tsv":
                df = pd.read_csv(path, delimiter="\t", nrows=1000)
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(path, nrows=1000)
            else:
                # Fallback for unexpected extension
                return ClassificationResult(
                    classification="report_like",
                    confidence=0.0,
                    reasons=[f"Unknown spreadsheet type: {ext}"],
                    metrics={},
                )

            return classify_dataframe(
                df,
                config=self.config.spreadsheet_classification_config,
                filename=path.name,
            )

        except Exception as e:
            logger.warning(f"Could not classify spreadsheet {path}: {e}")
            return ClassificationResult(
                classification="report_like",
                confidence=0.0,
                reasons=[f"Classification error: {e}"],
                metrics={"error": str(e)},
            )

    def run(self, document_paths: Optional[List[Path]] = None) -> PipelineResult:
        """Run the ingestion pipeline.

        Args:
            document_paths: Optional list of specific document paths to process.
                           If None, discovers all documents in input_dir.

        Returns:
            PipelineResult with processing statistics and failure details.
        """
        start = datetime.utcnow()
        documents = document_paths or discover_documents(self.config.input_dir)
        processed = skipped = failed = chunk_total = 0
        sql_tables_created = sql_tables_skipped = sql_rows_ingested = 0
        failures: List[FailureInfo] = []

        if not documents:
            logger.warning("No documents found under %s", self.config.input_dir)
            end = datetime.utcnow()
            return PipelineResult(
                0, 0, 0, 0, [],
                start.isoformat() + "Z",
                end.isoformat() + "Z",
                (end - start).total_seconds(),
            )

        for path in documents:
            try:
                document, content_hash = self.loader.load(path)
                if self.storage.is_up_to_date(document.doc_id, content_hash):
                    skipped += 1
                    logger.info("Skipping %s (no changes detected)", path)
                    continue

                # Classify spreadsheets if enabled
                classification = None
                if self._is_spreadsheet(document) and self.config.spreadsheet_classification_config:
                    classification = self._classify_spreadsheet(document)
                    document.metadata["spreadsheet_classification"] = classification.classification
                    document.metadata["classification_confidence"] = classification.confidence
                    document.metadata["classification_reasons"] = classification.reasons
                    document.metadata["classification_metrics"] = classification.metrics
                    logger.info(
                        "Classified %s as %s (confidence: %.1f%%)",
                        document.doc_id,
                        classification.classification,
                        classification.confidence * 100,
                    )

                # Route to SQL path for tabular spreadsheets
                if classification and self._should_use_sql_path(classification):
                    sql_results = self.sql_loader.load_spreadsheet_file(
                        file_path=path,
                        domain_labels=document.metadata.get("domain_labels"),
                    )
                    for sql_result in sql_results:
                        if sql_result.was_skipped:
                            sql_tables_skipped += 1
                        else:
                            sql_tables_created += 1
                            sql_rows_ingested += sql_result.row_count
                    logger.info(
                        "SQL loaded %s -> %d table(s), %d rows",
                        document.metadata.get("relative_path", document.doc_id),
                        len(sql_results),
                        sum(r.row_count for r in sql_results if not r.was_skipped),
                    )
                    processed += 1
                    continue

                # Standard chunk path for non-tabular documents
                chunks = chunk_document(
                    document,
                    chunk_size_tokens=self.config.chunk_size_tokens,
                    chunk_overlap_percent=self.config.chunk_overlap_percent,
                )

                if not chunks:
                    skipped += 1
                    logger.warning("No chunks produced for %s", path)
                    continue

                self.storage.persist_document(document, chunks, content_hash)
                processed += 1
                chunk_total += len(chunks)
                logger.info(
                    "Processed %s -> %d chunks",
                    document.metadata.get("relative_path", document.doc_id),
                    len(chunks),
                )
            except UnsupportedDocumentError as exc:
                failed += 1
                failure = FailureInfo(
                    source_path=str(path),
                    doc_id=None,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    traceback=traceback.format_exc(),
                    timestamp=datetime.utcnow().isoformat() + "Z",
                )
                failures.append(failure)
                logger.error("Unsupported document: %s", exc)
                if self.config.fail_fast:
                    raise
            except Exception as exc:  # pylint: disable=broad-except
                failed += 1
                failure = FailureInfo(
                    source_path=str(path),
                    doc_id=None,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    traceback=traceback.format_exc(),
                    timestamp=datetime.utcnow().isoformat() + "Z",
                )
                failures.append(failure)
                logger.exception("Failed to process %s", path)
                if self.config.fail_fast:
                    raise

        # Cleanup orphaned documents if enabled
        cleaned_up = 0
        if self.config.cleanup_deleted:
            current_paths = [str(p.resolve()) for p in documents]
            orphaned = self.storage.find_orphaned_docs(current_paths)
            if orphaned:
                cleaned_up = self.storage.cleanup_orphaned_docs(orphaned)
                logger.info("Cleaned up %d orphaned document(s)", cleaned_up)

        end = datetime.utcnow()
        result = PipelineResult(
            processed,
            skipped,
            failed,
            chunk_total,
            failures,
            start.isoformat() + "Z",
            end.isoformat() + "Z",
            (end - start).total_seconds(),
            cleaned_up,
            sql_tables_created,
            sql_tables_skipped,
            sql_rows_ingested,
        )

        # Save failures and report
        self.storage.save_failures(failures)
        self.storage.save_report(result)

        # Close SQL connection if opened
        if self.sql_store is not None:
            self.sql_store.close()

        return result
