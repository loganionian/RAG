#!/usr/bin/env python3
"""CLI for SQL Agent queries.

Usage:
    # Single question
    python -m scripts.sql_chat -q "What tables are available?"

    # Interactive mode
    python -m scripts.sql_chat --interactive

    # Show generated SQL
    python -m scripts.sql_chat -q "Top 5 rows from any table" --show-sql

    # JSON output
    python -m scripts.sql_chat -q "Count rows per table" --json

    # Custom database path
    python -m scripts.sql_chat -q "List tables" --db-path data/vectorstore/catalog.duckdb
"""

from __future__ import annotations

import core  # noqa: F401  # Initialize tiktoken cache before other imports

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_logging(verbose: bool) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Quiet noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def create_sql_chain(db_path: Path):
    """Create and return an SQLChain instance."""
    from generation.factory import create_llm_client
    from sql_agent import SQLAgentConfig, SQLChain

    llm_client = create_llm_client()
    config = SQLAgentConfig(db_path=db_path)
    return SQLChain(llm_client, config)


def format_result_pretty(result, show_sql: bool = False) -> str:
    """Format query result for pretty printing."""
    lines = []

    # Answer
    lines.append("\n" + "=" * 60)
    lines.append("ANSWER")
    lines.append("=" * 60)
    lines.append(result.answer)

    # SQL (if requested)
    if show_sql and result.generated_sql:
        lines.append("\n" + "-" * 60)
        lines.append("GENERATED SQL")
        lines.append("-" * 60)
        lines.append(result.generated_sql)

    # Data preview (first 10 rows)
    if result.data:
        lines.append("\n" + "-" * 60)
        lines.append(f"DATA ({result.row_count} rows)")
        lines.append("-" * 60)

        # Column headers
        if result.columns:
            header = " | ".join(str(c)[:20].ljust(20) for c in result.columns[:5])
            lines.append(header)
            lines.append("-" * len(header))

        # Rows (preview)
        for row in result.data[:10]:
            values = []
            for col in result.columns[:5]:
                val = str(row.get(col, ""))[:20].ljust(20)
                values.append(val)
            lines.append(" | ".join(values))

        if result.row_count > 10:
            lines.append(f"... and {result.row_count - 10} more rows")

    # Timing
    lines.append("\n" + "-" * 60)
    lines.append("TIMING")
    lines.append("-" * 60)
    lines.append(f"Generation: {result.generation_time_ms:.1f}ms")
    lines.append(f"Execution:  {result.execution_time_ms:.1f}ms")
    if result.summarization_time_ms > 0:
        lines.append(f"Summary:    {result.summarization_time_ms:.1f}ms")
    total = result.generation_time_ms + result.execution_time_ms + result.summarization_time_ms
    lines.append(f"Total:      {total:.1f}ms")

    if result.tables_used:
        lines.append(f"\nTables used: {', '.join(result.tables_used)}")

    return "\n".join(lines)


def format_result_json(result, show_sql: bool = False) -> str:
    """Format query result as JSON."""
    output = {
        "answer": result.answer,
        "data": result.data,
        "columns": result.columns,
        "row_count": result.row_count,
        "tables_used": result.tables_used,
        "timing": {
            "generation_ms": result.generation_time_ms,
            "execution_ms": result.execution_time_ms,
            "summarization_ms": result.summarization_time_ms,
        },
    }
    if show_sql:
        output["generated_sql"] = result.generated_sql

    return json.dumps(output, indent=2, default=str)


def run_interactive(chain, show_sql: bool = False) -> None:
    """Run interactive SQL chat mode."""
    print("=" * 60)
    print("SQL Agent - Interactive Mode")
    print("=" * 60)
    print("Ask questions about your data in natural language.")
    print("Type 'exit', 'quit', or Ctrl+C to end.\n")

    # Show available tables
    tables = chain.get_available_tables()
    if tables:
        print(f"Available tables: {', '.join(tables)}\n")
    else:
        print("No tables available in the database.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        try:
            result = chain.query(question)
            print(format_result_pretty(result, show_sql=show_sql))
            print()
        except Exception as e:
            print(f"\nError: {e}\n")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SQL Agent CLI - Query databases with natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "-q", "--question",
        help="Question to ask (omit for interactive mode)",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--show-sql",
        action="store_true",
        help="Show the generated SQL query",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100,
        help="Maximum rows to return (default: 100)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/vectorstore/catalog.duckdb"),
        help="Path to DuckDB database",
    )
    parser.add_argument(
        "--list-tables",
        action="store_true",
        help="List available tables and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Check database exists
    if not args.db_path.exists():
        print(f"Error: Database not found at {args.db_path}", file=sys.stderr)
        print("Run the ingestion and indexing pipelines first.", file=sys.stderr)
        return 1

    try:
        chain = create_sql_chain(args.db_path)

        # List tables mode
        if args.list_tables:
            tables = chain.get_available_tables()
            if args.json:
                print(json.dumps({"tables": tables, "count": len(tables)}, indent=2))
            else:
                print("Available tables:")
                for table in tables:
                    print(f"  - {table}")
                print(f"\nTotal: {len(tables)} table(s)")
            return 0

        # Interactive mode
        if args.interactive or (not args.question and not args.list_tables):
            run_interactive(chain, show_sql=args.show_sql)
            return 0

        # Single question mode
        if args.question:
            result = chain.query(args.question, max_rows=args.max_rows)

            if args.json:
                print(format_result_json(result, show_sql=args.show_sql))
            else:
                print(format_result_pretty(result, show_sql=args.show_sql))

            return 0

        # No action specified
        parser.print_help()
        return 1

    except Exception as e:
        logging.exception("SQL chat failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1

    finally:
        if "chain" in locals():
            chain.close()


if __name__ == "__main__":
    sys.exit(main())
