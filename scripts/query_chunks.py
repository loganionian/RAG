from __future__ import annotations

import core  # noqa: F401  # Initialize tiktoken cache before other imports

import argparse
import json
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from indexing.embeddings import get_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ad-hoc semantic queries against the Chroma vector store.",
    )
    parser.add_argument(
        "--vectorstore-dir",
        default="data/vectorstore",
        help="Directory where Chroma persistence files live.",
    )
    parser.add_argument(
        "--collection-name",
        default="pilot-docs",
        help="Chroma collection to query.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used to embed query text.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Natural language question/query text.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
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
    document: str,
    metadata: dict,
    distance: float | None,
    max_chars: int,
) -> str:
    snippet = document.strip().replace("\n", " ")
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3].rstrip() + "..."

    rel_path = metadata.get("relative_path") or metadata.get("source_path")
    chunk_id = metadata.get("chunk_id")
    doc_id = metadata.get("doc_id")

    header = f"[{index}] doc_id={doc_id} chunk_id={chunk_id} path={rel_path}"
    if distance is not None:
        header += f" | distance={distance:.4f}"

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
    if not vectorstore_path.exists():
        print(f"[error] Vector store path {vectorstore_path} not found.", file=sys.stderr)
        return 1

    client = chromadb.PersistentClient(path=str(vectorstore_path))
    # Resolve model path (prefers local model if available)
    resolved_model = get_model_path(args.embedding_model)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=resolved_model,
    )
    collection = client.get_collection(
        args.collection_name,
        embedding_function=embedding_fn,
    )
    result = collection.query(
        query_texts=[args.question],
        n_results=max(1, args.k),
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0] if result.get("distances") else []

    if args.pretty:
        output = []
        for idx, (doc_text, metadata, chunk_id) in enumerate(
            zip(documents, metadatas, ids),
            start=1,
        ):
            payload = {
                "rank": idx,
                "chunk_id": chunk_id,
                "distance": distances[idx - 1] if idx - 1 < len(distances) else None,
                "metadata": metadata,
                "text": doc_text,
            }
            output.append(payload)
        json.dump(output, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    if not documents:
        print("No results found.")
        return 0

    print(f"Query: {args.question!r}")
    for idx, (doc_text, metadata, chunk_id) in enumerate(
        zip(documents, metadatas, ids),
        start=1,
    ):
        metadata = {**metadata, "chunk_id": chunk_id}
        distance = distances[idx - 1] if idx - 1 < len(distances) else None
        print(format_result(idx, doc_text, metadata, distance, args.max_chars))

    return 0


if __name__ == "__main__":
    sys.exit(main())
