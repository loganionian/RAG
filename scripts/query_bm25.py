"""CLI script for running BM25 lexical queries against the index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from indexing.bm25_store import BM25Config, BM25Store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ad-hoc BM25 lexical queries against the index.",
    )
    parser.add_argument(
        "--vectorstore-dir",
        default="data/vectorstore",
        help="Directory where BM25 index files live (in bm25/ subdirectory).",
    )
    parser.add_argument(
        "--index-name",
        default="pilot-docs",
        help="BM25 index name to query.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Natural language question/query text.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-K results to return.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=400,
        help="Characters of chunk text to display per hit.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON payload instead of plain text summary.",
    )
    return parser.parse_args()


def format_result(
    index: int,
    text: str,
    metadata: dict,
    score: float,
    max_chars: int,
) -> str:
    """Format a single search result for display."""
    snippet = text.strip().replace("\n", " ")
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3].rstrip() + "..."

    rel_path = metadata.get("relative_path") or metadata.get("source_path")
    chunk_id = metadata.get("chunk_id")
    doc_id = metadata.get("doc_id")

    header = f"[{index}] doc_id={doc_id} chunk_id={chunk_id} path={rel_path}"
    header += f" | score={score:.4f}"

    # Build location info line
    location_parts = []

    # Page info (for PDFs)
    if "page" in metadata:
        location_parts.append(f"page {metadata['page']}")

    # Section info
    if "section" in metadata:
        section = metadata["section"]
        # Truncate long section names
        if len(section) > 40:
            section = section[:37] + "..."
        location_parts.append(f'section "{section}"')

    # Paragraph span
    if "paragraph_start" in metadata and "paragraph_end" in metadata:
        location_parts.append(
            f"paragraphs {metadata['paragraph_start']}-{metadata['paragraph_end']}"
        )

    location_line = ""
    if location_parts:
        location_line = f"\n    [{', '.join(location_parts)}]"

    # Timestamp for freshness indication
    timestamp_line = ""
    if "timestamp" in metadata and metadata["timestamp"]:
        timestamp_line = f"\n    indexed: {metadata['timestamp']}"

    return f"{header}{location_line}{timestamp_line}\n    {snippet}"


def main() -> int:
    args = parse_args()
    vectorstore_path = Path(args.vectorstore_dir)
    bm25_path = vectorstore_path / "bm25"

    if not bm25_path.exists():
        print(f"[error] BM25 index path {bm25_path} not found.", file=sys.stderr)
        print("Run indexing with BM25 enabled first:", file=sys.stderr)
        print("  python -m scripts.index_chunks --processed-dir data/processed --chroma-dir data/vectorstore", file=sys.stderr)
        return 1

    # Initialize BM25 store
    config = BM25Config(
        index_dir=vectorstore_path,
        index_name=args.index_name,
    )
    store = BM25Store(config)

    if store.count() == 0:
        print(f"[error] BM25 index '{args.index_name}' is empty.", file=sys.stderr)
        return 1

    # Run search
    results = store.search(args.question, k=max(1, args.k))

    if args.pretty:
        output = []
        for idx, result in enumerate(results, start=1):
            payload = {
                "rank": idx,
                "chunk_id": result.chunk_id,
                "doc_id": result.doc_id,
                "score": result.score,
                "metadata": result.metadata,
                "text": result.text,
            }
            output.append(payload)
        json.dump(output, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    if not results:
        print("No results found.")
        return 0

    print(f"Query: {args.question!r}")
    print(f"Index: {args.index_name} ({store.count()} chunks)")
    print()

    for idx, result in enumerate(results, start=1):
        print(format_result(
            idx,
            result.text,
            result.metadata,
            result.score,
            args.max_chars,
        ))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
