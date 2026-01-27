from __future__ import annotations

import core  # noqa: F401  # Initialize tiktoken cache before other imports

import argparse
import logging
import sys
from pathlib import Path

from indexing.pipeline import ChromaIndexingPipeline, IndexingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index processed chunks into a persistent Chroma collection.",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing manifest.json and chunks/ outputs from ingestion.",
    )
    parser.add_argument(
        "--chroma-dir",
        default="data/vectorstore",
        help="Directory where Chroma will persist data.",
    )
    parser.add_argument(
        "--collection-name",
        default="pilot-docs",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model to embed chunks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks to upsert per batch.",
    )
    parser.add_argument(
        "--doc-ids",
        nargs="+",
        help="Optional list of specific doc_ids to reindex (defaults to all).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify metadata integrity after indexing.",
    )
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=100,
        help="Number of chunks to sample for verification (default: 100).",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    config = IndexingConfig(
        processed_dir=Path(args.processed_dir),
        chroma_dir=Path(args.chroma_dir),
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
        doc_filter=args.doc_ids,
    )

    pipeline = ChromaIndexingPipeline(config)
    result = pipeline.run()

    logging.info(
        "Chroma indexing complete: docs=%d chunks=%d skipped=%d failed=%d",
        result.indexed_docs,
        result.indexed_chunks,
        result.skipped_docs,
        result.failed_chunks,
    )

    # Run metadata verification if requested
    if args.verify:
        verification = pipeline.verify_metadata(sample_size=args.verify_sample)
        if verification.missing_fields > 0:
            logging.warning(
                "Metadata verification found %d chunks with missing required fields.",
                verification.missing_fields,
            )
        else:
            logging.info("Metadata verification passed: all required fields present.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
