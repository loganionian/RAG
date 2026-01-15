"""CLI for generating LLM-based summaries for tabular datasets.

This script generates natural-language summaries for datasets stored
in the SQL store and optionally indexes them in Chroma for discovery.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from generation import create_llm_client
from generation.cost_tracker import CostConfig
from generation.dataset_summarizer import DatasetSummarizer, SummaryConfig
from indexing.summary_indexer import SummaryIndexConfig, SummaryIndexer
from storage import MetadataCatalog, SQLStore, SQLStoreConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM summaries for tabular datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate summaries for all tables without summaries
  python -m scripts.generate_summaries --chroma-dir data/vectorstore

  # Generate for specific tables only
  python -m scripts.generate_summaries --tables sales_2024 inventory

  # Dry run to estimate tokens/cost
  python -m scripts.generate_summaries --dry-run

  # Set token budget
  python -m scripts.generate_summaries --max-tokens 10000

  # Skip indexing (only generate and store in catalog)
  python -m scripts.generate_summaries --no-index
        """,
    )

    # Input/output paths
    parser.add_argument(
        "--db-path",
        default="data/tabular.db",
        help="Path to DuckDB database (default: data/tabular.db).",
    )
    parser.add_argument(
        "--chroma-dir",
        default="data/vectorstore",
        help="Directory where Chroma stores data (default: data/vectorstore).",
    )

    # Table selection
    parser.add_argument(
        "--tables",
        nargs="+",
        help="Specific tables to summarize (default: all without summaries).",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate summaries even for tables that already have them.",
    )

    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50000,
        help="Maximum total tokens for the run (default: 50000).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature (default: 0.3).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5,
        help="Number of sample rows to include in prompt (default: 5).",
    )

    # Indexing options
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip Chroma indexing (only update catalog).",
    )
    parser.add_argument(
        "--collection-name",
        default="dataset-summaries",
        help="Chroma collection name for summaries (default: dataset-summaries).",
    )

    # Operation modes
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without making LLM calls.",
    )
    parser.add_argument(
        "--list-tables",
        action="store_true",
        help="List all tables and their summary status.",
    )

    # Output options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def list_tables(catalog: MetadataCatalog, output_json: bool) -> int:
    """List all tables with their summary status."""
    entries = catalog.list_tables()

    if output_json:
        data = []
        for entry in entries:
            data.append({
                "table_name": entry.table_name,
                "source_file": entry.source_file,
                "row_count": entry.row_count,
                "column_count": entry.column_count,
                "has_summary": entry.has_summary,
                "summary_generated_at": entry.summary_generated_at.isoformat() if entry.summary_generated_at else None,
            })
        print(json.dumps(data, indent=2))
    else:
        print(f"\n{'Table Name':<40} {'Rows':>8} {'Cols':>6} {'Summary':>10}")
        print("-" * 70)
        for entry in entries:
            status = "Yes" if entry.has_summary else "No"
            print(f"{entry.table_name:<40} {entry.row_count:>8} {entry.column_count:>6} {status:>10}")
        print(f"\nTotal: {len(entries)} tables")

        stats = catalog.get_statistics()
        print(f"Tables with summaries: {stats.get('tables_with_summaries', 0)}")

    return 0


def dry_run(
    summarizer: DatasetSummarizer,
    table_names: list[str],
    output_json: bool,
) -> int:
    """Show what would be generated without making LLM calls."""
    total_estimated = 0
    results = []

    for table_name in table_names:
        estimated = summarizer.estimate_tokens(table_name)
        total_estimated += estimated
        results.append({
            "table_name": table_name,
            "estimated_tokens": estimated,
        })

    if output_json:
        print(json.dumps({
            "tables": results,
            "total_estimated_tokens": total_estimated,
            "table_count": len(table_names),
        }, indent=2))
    else:
        print(f"\nDry run - Estimated token usage:")
        print(f"{'Table Name':<40} {'Est. Tokens':>12}")
        print("-" * 55)
        for r in results:
            print(f"{r['table_name']:<40} {r['estimated_tokens']:>12}")
        print("-" * 55)
        print(f"{'Total':<40} {total_estimated:>12}")
        print(f"\nTables to process: {len(table_names)}")

    return 0


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    # Initialize storage
    db_path = Path(args.db_path)
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        logger.info("Run the ingestion pipeline first with --enable-sql-tabular")
        return 1

    sql_store = SQLStore(SQLStoreConfig(db_path=db_path, read_only=True))
    catalog = MetadataCatalog(sql_store)
    catalog.init_catalog()

    # Handle list-tables mode
    if args.list_tables:
        return list_tables(catalog, args.json)

    # Determine tables to process
    if args.tables:
        table_names = args.tables
        # Validate tables exist
        for name in table_names:
            if catalog.get_entry(name) is None:
                logger.error("Table not found: %s", name)
                return 1
    elif args.regenerate:
        # All tables
        entries = catalog.list_tables()
        table_names = [e.table_name for e in entries]
    else:
        # Only tables without summaries
        entries = catalog.get_tables_without_summaries()
        table_names = [e.table_name for e in entries]

    if not table_names:
        logger.info("No tables need summaries. Use --regenerate to regenerate existing.")
        return 0

    logger.info("Found %d tables to summarize", len(table_names))

    # Close read-only connection before opening write connection
    # DuckDB doesn't allow mixed configurations on the same database
    sql_store.close()

    # Create write-enabled connection for updating summaries
    sql_store_write = SQLStore(SQLStoreConfig(db_path=db_path, read_only=False))
    catalog_write = MetadataCatalog(sql_store_write)

    # Handle dry-run mode
    if args.dry_run:
        # Create summarizer just for estimation
        llm_client = create_llm_client()
        summarizer = DatasetSummarizer(
            llm_client=llm_client,
            sql_store=sql_store_write,
            catalog=catalog_write,
            config=SummaryConfig(
                sample_rows=args.sample_rows,
                temperature=args.temperature,
            ),
        )
        return dry_run(summarizer, table_names, args.json)

    # Create LLM client and summarizer
    try:
        llm_client = create_llm_client()
    except Exception as e:
        logger.error("Failed to create LLM client: %s", e)
        logger.info("Ensure LLM_PROVIDER and credentials are configured.")
        return 1

    summary_config = SummaryConfig(
        sample_rows=args.sample_rows,
        temperature=args.temperature,
    )
    cost_config = CostConfig(
        max_total_tokens_per_run=args.max_tokens,
    )

    summarizer = DatasetSummarizer(
        llm_client=llm_client,
        sql_store=sql_store_write,
        catalog=catalog_write,
        config=summary_config,
        cost_config=cost_config,
    )

    # Generate summaries
    logger.info("Generating summaries for %d tables...", len(table_names))
    summaries = summarizer.generate_summaries_batch(table_names)

    if not summaries:
        logger.warning("No summaries generated")
        return 1

    # Store summaries in catalog
    for summary in summaries:
        catalog_write.update_summary(summary.table_name, summary)

    logger.info("Stored %d summaries in catalog", len(summaries))

    # Index in Chroma (unless --no-index)
    indexed_count = 0
    if not args.no_index:
        index_config = SummaryIndexConfig(
            chroma_dir=Path(args.chroma_dir),
            collection_name=args.collection_name,
        )
        indexer = SummaryIndexer(index_config)
        indexed_count = indexer.index_summaries_batch(summaries)
        logger.info("Indexed %d summaries in Chroma", indexed_count)

    # Get cost report
    report = summarizer.get_cost_report()

    # Output results
    if args.json:
        output = {
            "summaries_generated": len(summaries),
            "summaries_indexed": indexed_count,
            "token_usage": {
                "total_input_tokens": report.total_input_tokens,
                "total_output_tokens": report.total_output_tokens,
                "total_tokens": report.total_tokens,
                "estimated_cost": report.estimated_cost,
            },
            "summaries": [s.to_dict() for s in summaries],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nSummary Generation Complete")
        print("=" * 40)
        print(f"Summaries generated: {len(summaries)}")
        print(f"Summaries indexed:   {indexed_count}")
        print(f"\nToken Usage:")
        print(f"  Input tokens:  {report.total_input_tokens:,}")
        print(f"  Output tokens: {report.total_output_tokens:,}")
        print(f"  Total tokens:  {report.total_tokens:,}")
        if report.estimated_cost > 0:
            print(f"  Est. cost:     ${report.estimated_cost:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
