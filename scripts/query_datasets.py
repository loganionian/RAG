"""CLI for discovering and querying tabular datasets.

This script enables semantic search over dataset summaries
to help users find relevant data sources.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from indexing.summary_indexer import SummaryIndexConfig, SummaryIndexer
from storage import MetadataCatalog, SQLStore, SQLStoreConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and query tabular datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Semantic search for datasets
  python -m scripts.query_datasets --question "What sales data do we have?"

  # List all indexed datasets
  python -m scripts.query_datasets --list-all

  # Get details for a specific dataset
  python -m scripts.query_datasets --describe sales_2024

  # Search with more results
  python -m scripts.query_datasets -q "financial data" -k 10

  # JSON output for programmatic use
  python -m scripts.query_datasets -q "customer data" --json
        """,
    )

    # Query modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-q", "--question",
        help="Natural language query to search datasets.",
    )
    mode_group.add_argument(
        "--list-all",
        action="store_true",
        help="List all indexed datasets.",
    )
    mode_group.add_argument(
        "--describe",
        metavar="TABLE",
        help="Get detailed information about a specific table.",
    )
    mode_group.add_argument(
        "--stats",
        action="store_true",
        help="Show catalog and index statistics.",
    )

    # Paths
    parser.add_argument(
        "--chroma-dir",
        default="data/vectorstore",
        help="Chroma persistence directory (default: data/vectorstore).",
    )
    parser.add_argument(
        "--db-path",
        default="data/tabular.db",
        help="Path to DuckDB database (default: data/tabular.db).",
    )
    parser.add_argument(
        "--collection-name",
        default="dataset-summaries",
        help="Chroma collection name (default: dataset-summaries).",
    )

    # Search options
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    parser.add_argument(
        "--domain",
        help="Filter results by domain label.",
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


def search_datasets(
    indexer: SummaryIndexer,
    query: str,
    k: int,
    domain: str | None,
    output_json: bool,
) -> int:
    """Search for datasets matching a query."""
    results = indexer.search_datasets(query, k=k, filter_domain=domain)

    if not results:
        if output_json:
            print(json.dumps({"results": [], "query": query}, indent=2))
        else:
            print(f"\nNo datasets found matching: {query}")
        return 0

    if output_json:
        output = {
            "query": query,
            "result_count": len(results),
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nSearch results for: \"{query}\"")
        print("=" * 60)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.table_name}")
            print(f"   Source: {result.source_file}")
            print(f"   Rows: {result.row_count:,} | Columns: {result.column_count}")
            print(f"   Score: {result.relevance_score:.4f}")
            print(f"   Summary: {result.summary}")
            if result.domain_labels:
                print(f"   Domains: {', '.join(result.domain_labels)}")

        print(f"\n{len(results)} result(s) found")

    return 0


def list_all_datasets(indexer: SummaryIndexer, output_json: bool) -> int:
    """List all indexed datasets."""
    results = indexer.list_all()

    if not results:
        if output_json:
            print(json.dumps({"datasets": []}, indent=2))
        else:
            print("\nNo datasets indexed yet.")
            print("Run: python -m scripts.generate_summaries --chroma-dir data/vectorstore")
        return 0

    if output_json:
        output = {
            "dataset_count": len(results),
            "datasets": [r.to_dict() for r in results],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nIndexed Datasets ({len(results)})")
        print("=" * 70)
        print(f"{'Table Name':<35} {'Rows':>10} {'Cols':>6} {'Source':<25}")
        print("-" * 70)
        for r in sorted(results, key=lambda x: x.table_name):
            source = r.source_file[-22:] if len(r.source_file) > 22 else r.source_file
            print(f"{r.table_name:<35} {r.row_count:>10,} {r.column_count:>6} {source:<25}")

    return 0


def describe_dataset(
    indexer: SummaryIndexer,
    catalog: MetadataCatalog | None,
    table_name: str,
    output_json: bool,
) -> int:
    """Get detailed information about a specific dataset."""
    # Try to get from index first
    result = indexer.get_summary(table_name)

    # Also get from catalog if available
    catalog_entry = None
    if catalog:
        catalog_entry = catalog.get_entry(table_name)

    if not result and not catalog_entry:
        if output_json:
            print(json.dumps({"error": f"Dataset '{table_name}' not found"}, indent=2))
        else:
            print(f"\nDataset '{table_name}' not found.")
        return 1

    if output_json:
        output = {}
        if result:
            output["indexed"] = result.to_dict()
        if catalog_entry:
            output["catalog"] = catalog_entry.to_dict()
        print(json.dumps(output, indent=2))
    else:
        print(f"\nDataset: {table_name}")
        print("=" * 60)

        if catalog_entry:
            print(f"\nCatalog Information:")
            print(f"  Source file:     {catalog_entry.source_file}")
            print(f"  Sheet name:      {catalog_entry.sheet_name or 'N/A'}")
            print(f"  Row count:       {catalog_entry.row_count:,}")
            print(f"  Column count:    {catalog_entry.column_count}")
            print(f"  Ingested:        {catalog_entry.ingestion_timestamp}")

            if catalog_entry.column_schema:
                print(f"\n  Columns:")
                for col in catalog_entry.column_schema[:10]:
                    print(f"    - {col.get('name', 'unknown')}: {col.get('type', 'unknown')}")
                if len(catalog_entry.column_schema) > 10:
                    print(f"    ... and {len(catalog_entry.column_schema) - 10} more")

        if result:
            print(f"\nSummary:")
            print(f"  {result.summary}")

            if result.column_descriptions:
                print(f"\n  Column Descriptions:")
                for col, desc in list(result.column_descriptions.items())[:10]:
                    print(f"    - {col}: {desc}")
                if len(result.column_descriptions) > 10:
                    print(f"    ... and {len(result.column_descriptions) - 10} more")

            if result.domain_labels:
                print(f"\n  Domain Labels: {', '.join(result.domain_labels)}")
        elif catalog_entry and catalog_entry.summary:
            print(f"\nSummary:")
            print(f"  {catalog_entry.summary}")

            if catalog_entry.column_descriptions:
                print(f"\n  Column Descriptions:")
                for col, desc in list(catalog_entry.column_descriptions.items())[:10]:
                    print(f"    - {col}: {desc}")

    return 0


def show_stats(
    indexer: SummaryIndexer,
    catalog: MetadataCatalog | None,
    output_json: bool,
) -> int:
    """Show catalog and index statistics."""
    index_count = indexer.count()

    catalog_stats = {}
    if catalog:
        catalog_stats = catalog.get_statistics()

    if output_json:
        output = {
            "index": {
                "document_count": index_count,
            },
            "catalog": catalog_stats,
        }
        print(json.dumps(output, indent=2))
    else:
        print("\nDataset Statistics")
        print("=" * 40)
        print(f"\nChroma Index:")
        print(f"  Indexed summaries: {index_count}")

        if catalog_stats:
            print(f"\nCatalog:")
            print(f"  Total tables:        {catalog_stats.get('table_count', 0)}")
            print(f"  Total rows:          {catalog_stats.get('total_rows', 0):,}")
            print(f"  Tables w/ summaries: {catalog_stats.get('tables_with_summaries', 0)}")

    return 0


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    chroma_dir = Path(args.chroma_dir)
    if not chroma_dir.exists():
        logger.warning("Chroma directory not found: %s", chroma_dir)
        if not args.stats:
            logger.info("Run ingestion and summary generation first.")
            return 1

    # Initialize indexer
    index_config = SummaryIndexConfig(
        chroma_dir=chroma_dir,
        collection_name=args.collection_name,
    )
    indexer = SummaryIndexer(index_config)

    # Initialize catalog if DB exists
    catalog = None
    db_path = Path(args.db_path)
    if db_path.exists():
        sql_store = SQLStore(SQLStoreConfig(db_path=db_path, read_only=True))
        catalog = MetadataCatalog(sql_store)

    # Handle different modes
    if args.question:
        return search_datasets(indexer, args.question, args.top_k, args.domain, args.json)
    elif args.list_all:
        return list_all_datasets(indexer, args.json)
    elif args.describe:
        return describe_dataset(indexer, catalog, args.describe, args.json)
    elif args.stats:
        return show_stats(indexer, catalog, args.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
