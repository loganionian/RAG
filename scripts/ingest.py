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


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

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

    if classification_config is not None:
        logging.info("Spreadsheet classification enabled")
    else:
        logging.info("Spreadsheet classification disabled")

    config = PipelineConfig(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        chunk_size_tokens=args.chunk_size,
        chunk_overlap_percent=args.chunk_overlap,
        fail_fast=args.fail_fast,
        normalization_config=normalization_config,
        cleanup_deleted=args.cleanup,
        spreadsheet_classification_config=classification_config,
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
