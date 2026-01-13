from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ingestion.normalizer import NormalizationConfig, TextNormalizer
from ingestion.pipeline import IngestionPipeline, PipelineConfig
from ingestion.spreadsheet_classifier import SpreadsheetClassificationConfig
from ingestion.storage import StorageManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-ingest documents (PDF/DOCX/MD) into normalized chunks.",
    )
    parser.add_argument("--input-dir", default="data/raw", help="Directory containing source documents.")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where chunks + manifest will be stored.",
    )
    parser.add_argument("--chunk-size", type=int, default=400, help="Approximate target token count per chunk.")
    parser.add_argument(
        "--chunk-overlap",
        type=float,
        default=0.15,
        help="Overlap as percentage of chunk size (0.10-0.20, default: 0.15 = 15%%).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort immediately on the first failure instead of logging and continuing.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry only documents that failed in a previous run (reads from failures.json).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove orphaned documents whose source files have been deleted.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging output.")

    # Normalization arguments
    normalize_group = parser.add_argument_group("normalization", "Text normalization options")
    normalize_group.add_argument(
        "--normalize",
        choices=["default", "minimal", "aggressive", "none"],
        default="default",
        help="Normalization preset to use (default: default).",
    )
    normalize_group.add_argument(
        "--normalize-config",
        type=str,
        default=None,
        help="Path to YAML configuration file for custom normalization settings.",
    )
    normalize_group.add_argument(
        "--no-remove-page-numbers",
        action="store_true",
        help="Disable removal of page number patterns.",
    )
    normalize_group.add_argument(
        "--no-remove-headers-footers",
        action="store_true",
        help="Disable removal of repeated header/footer lines.",
    )
    normalize_group.add_argument(
        "--no-remove-boilerplate",
        action="store_true",
        help="Disable removal of boilerplate text (confidential, copyright, etc.).",
    )

    # Spreadsheet classification arguments
    classify_group = parser.add_argument_group(
        "spreadsheet classification", "Spreadsheet classification options"
    )
    classify_group.add_argument(
        "--classify-spreadsheets",
        action="store_true",
        help="Enable tabular vs report-like classification for CSV/Excel files.",
    )
    classify_group.add_argument(
        "--min-rows-tabular",
        type=int,
        default=10,
        help="Minimum row count to classify as tabular (default: 10).",
    )
    classify_group.add_argument(
        "--numeric-ratio",
        type=float,
        default=0.3,
        help="Minimum numeric cell ratio for tabular (default: 0.3 = 30%%).",
    )
    classify_group.add_argument(
        "--long-text-threshold",
        type=int,
        default=200,
        help="Character count threshold for long text detection (default: 200).",
    )
    classify_group.add_argument(
        "--max-columns-tabular",
        type=int,
        default=50,
        help="Maximum column count for tabular classification (default: 50).",
    )

    # SQL tabular ingestion arguments
    sql_group = parser.add_argument_group(
        "sql tabular ingestion", "SQL database ingestion for tabular data"
    )
    sql_group.add_argument(
        "--enable-sql-tabular",
        action="store_true",
        help="Enable SQL ingestion for tabular spreadsheets (loads to DuckDB instead of chunking).",
    )
    sql_group.add_argument(
        "--sql-db-path",
        type=str,
        default=None,
        help="Path to DuckDB database file (default: data/tabular/tables.duckdb).",
    )
    sql_group.add_argument(
        "--tabular-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence for tabular classification to use SQL path (default: 0.7 = 70%%).",
    )
    sql_group.add_argument(
        "--list-sql-tables",
        action="store_true",
        help="List all tables in the SQL catalog and exit.",
    )
    sql_group.add_argument(
        "--query-sql",
        type=str,
        default=None,
        help="Execute a SQL query against the tabular database and print results.",
    )

    return parser.parse_args()


def build_normalization_config(args: argparse.Namespace) -> Optional[NormalizationConfig]:
    """Build a NormalizationConfig from command line arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        NormalizationConfig or None if normalization is disabled.
    """
    # If a YAML config file is provided, load it
    if args.normalize_config:
        normalizer = TextNormalizer.from_yaml(Path(args.normalize_config))
        config = normalizer.config
        # Apply CLI overrides
        if args.no_remove_page_numbers:
            config.remove_page_numbers = False
        if args.no_remove_headers_footers:
            config.remove_headers_footers = False
        if args.no_remove_boilerplate:
            config.remove_boilerplate = False
        return config

    # Handle preset selection
    if args.normalize == "none":
        return None

    if args.normalize == "minimal":
        config = NormalizationConfig(
            remove_page_numbers=False,
            remove_headers_footers=False,
            remove_boilerplate=False,
            normalize_whitespace=True,
            normalize_special_chars=True,
            normalize_bullets=False,
            remove_zero_width=True,
            preserve_code_blocks=True,
        )
    elif args.normalize == "aggressive":
        config = NormalizationConfig(
            remove_page_numbers=True,
            remove_headers_footers=True,
            remove_boilerplate=True,
            normalize_whitespace=True,
            normalize_special_chars=True,
            normalize_bullets=True,
            remove_zero_width=True,
            preserve_code_blocks=True,
            min_line_length=3,
            header_footer_threshold=2,
        )
    else:  # "default"
        config = NormalizationConfig()

    # Apply CLI overrides
    if args.no_remove_page_numbers:
        config.remove_page_numbers = False
    if args.no_remove_headers_footers:
        config.remove_headers_footers = False
    if args.no_remove_boilerplate:
        config.remove_boilerplate = False

    return config


def build_classification_config(args: argparse.Namespace) -> Optional[SpreadsheetClassificationConfig]:
    """Build a SpreadsheetClassificationConfig from command line arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        SpreadsheetClassificationConfig or None if classification is disabled.
    """
    if not args.classify_spreadsheets:
        return None

    return SpreadsheetClassificationConfig(
        min_rows_for_tabular=args.min_rows_tabular,
        numeric_ratio_threshold=args.numeric_ratio,
        long_text_threshold=args.long_text_threshold,
        max_columns_for_tabular=args.max_columns_tabular,
    )


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")


def handle_sql_commands(args: argparse.Namespace) -> Optional[int]:
    """Handle SQL-specific commands (--list-sql-tables, --query-sql).

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code if a command was handled, None otherwise.
    """
    if not args.list_sql_tables and not args.query_sql:
        return None

    from storage.sql_store import SQLStore, SQLStoreConfig
    from storage.metadata_catalog import MetadataCatalog

    # Determine database path
    output_dir = Path(args.output_dir)
    db_path = Path(args.sql_db_path) if args.sql_db_path else output_dir.parent / "tabular" / "tables.duckdb"

    if not db_path.exists():
        logging.error("Database not found: %s", db_path)
        logging.error("Run ingestion with --enable-sql-tabular first to create the database.")
        return 1

    sql_config = SQLStoreConfig(db_path=db_path, read_only=True)
    sql_store = SQLStore(sql_config)

    try:
        if args.list_sql_tables:
            catalog = MetadataCatalog(sql_store)
            entries = catalog.list_tables()
            if not entries:
                logging.info("No tables found in catalog.")
                return 0

            logging.info("=" * 80)
            logging.info("SQL TABLES IN CATALOG")
            logging.info("=" * 80)
            for entry in entries:
                logging.info(
                    "  %-30s | %6d rows | %3d cols | %s",
                    entry.table_name[:30],
                    entry.row_count,
                    entry.column_count,
                    entry.source_file,
                )
            logging.info("=" * 80)
            logging.info("Total: %d table(s)", len(entries))
            return 0

        if args.query_sql:
            result = sql_store.execute(args.query_sql)
            rows = result.fetchall()
            columns = [desc[0] for desc in result.description]

            # Print results as a table
            logging.info("Query: %s", args.query_sql)
            logging.info("-" * 80)

            # Print header
            header = " | ".join(f"{col:>15}" for col in columns)
            logging.info(header)
            logging.info("-" * len(header))

            # Print rows
            for row in rows[:100]:  # Limit to 100 rows
                row_str = " | ".join(f"{str(val)[:15]:>15}" for val in row)
                logging.info(row_str)

            if len(rows) > 100:
                logging.info("... (%d more rows)", len(rows) - 100)

            logging.info("-" * 80)
            logging.info("Returned %d row(s)", len(rows))
            return 0

    finally:
        sql_store.close()

    return None


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    # Handle SQL-specific commands first
    sql_result = handle_sql_commands(args)
    if sql_result is not None:
        return sql_result

    output_dir = Path(args.output_dir)

    # Handle retry-failed mode
    document_paths = None
    if args.retry_failed:
        storage = StorageManager(output_dir)
        failures = storage.load_failures()
        if not failures:
            logging.info("No failed documents to retry (failures.json is empty or missing)")
            return 0
        document_paths = [Path(f.source_path) for f in failures]
        logging.info("Retrying %d previously failed document(s)", len(document_paths))

    # Build normalization configuration
    normalization_config = build_normalization_config(args)

    if normalization_config is not None:
        logging.info("Text normalization enabled with preset: %s", args.normalize)
    else:
        logging.info("Text normalization disabled")

    # Build spreadsheet classification configuration
    classification_config = build_classification_config(args)

    # SQL tabular ingestion requires classification to be enabled
    if args.enable_sql_tabular and classification_config is None:
        logging.info("Enabling spreadsheet classification (required for SQL tabular ingestion)")
        classification_config = SpreadsheetClassificationConfig()

    if classification_config is not None:
        logging.info("Spreadsheet classification enabled")
    else:
        logging.info("Spreadsheet classification disabled")

    # Build SQL configuration
    sql_db_path = Path(args.sql_db_path) if args.sql_db_path else None

    if args.enable_sql_tabular:
        logging.info("SQL tabular ingestion enabled (threshold: %.0f%%)", args.tabular_threshold * 100)

    config = PipelineConfig(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        chunk_size_tokens=args.chunk_size,
        chunk_overlap_percent=args.chunk_overlap,
        fail_fast=args.fail_fast,
        normalization_config=normalization_config,
        cleanup_deleted=args.cleanup,
        spreadsheet_classification_config=classification_config,
        enable_sql_tabular=args.enable_sql_tabular,
        sql_db_path=sql_db_path,
        tabular_confidence_threshold=args.tabular_threshold,
    )

    pipeline = IngestionPipeline(config)
    result = pipeline.run(document_paths=document_paths)

    # Print summary
    logging.info("=" * 60)
    logging.info("INGESTION SUMMARY")
    logging.info("=" * 60)
    logging.info("  Processed:    %d document(s)", result.processed)
    logging.info("  Skipped:      %d document(s)", result.skipped)
    logging.info("  Failed:       %d document(s)", result.failed)
    if result.cleaned_up > 0:
        logging.info("  Cleaned up:   %d orphaned document(s)", result.cleaned_up)
    logging.info("  Total chunks: %d", result.chunk_count)
    if args.enable_sql_tabular:
        logging.info("  SQL tables:   %d created, %d skipped", result.sql_tables_created, result.sql_tables_skipped)
        logging.info("  SQL rows:     %d ingested", result.sql_rows_ingested)
    logging.info("  Duration:     %.2f seconds", result.duration_seconds)
    logging.info("=" * 60)

    if result.failures:
        logging.info("FAILURE DETAILS:")
        for failure in result.failures:
            logging.error("  [%s] %s", failure.error_type, failure.source_path)
            logging.error("    Message: %s", failure.error_message)
        logging.info("See %s for full details", output_dir / "failures.json")

    logging.info("Report saved to: %s", output_dir / "ingestion-report.json")

    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
