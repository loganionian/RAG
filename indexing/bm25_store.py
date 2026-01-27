"""BM25 lexical search index for document chunks.

This module provides a persistent BM25 index that operates alongside
the Chroma vector store, using the same chunk IDs for result mapping.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class BM25Config:
    """Configuration for BM25 index."""

    index_dir: Path
    index_name: str = "pilot-docs"
    k1: float = 1.5  # Term frequency saturation
    b: float = 0.75  # Length normalization


@dataclass
class BM25IndexEntry:
    """Entry to be indexed in BM25."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class BM25SearchResult:
    """Single search result from BM25."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class BM25HealthCheckResult:
    """Result of BM25 health check."""

    healthy: bool
    message: str
    document_count: int = 0
    index_name: str = ""


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer with lowercasing.

    Args:
        text: Input text.

    Returns:
        List of lowercase tokens.
    """
    return text.lower().split()


class BM25Store:
    """Persistent BM25 index with same chunk IDs as Chroma.

    The index stores tokenized documents and their metadata, allowing
    for fast lexical search. Results can be mapped to Chroma chunks
    using the shared chunk_id field.
    """

    def __init__(self, config: BM25Config):
        """Initialize BM25 store.

        Args:
            config: BM25 configuration.
        """
        self.config = config
        self._index_dir = config.index_dir / "bm25"
        self._index_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        self._chunk_ids: List[str] = []
        self._doc_ids: List[str] = []
        self._texts: List[str] = []
        self._metadatas: List[Dict[str, object]] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._dirty = False

        # Try to load existing index
        self._load()

    def _get_index_path(self, suffix: str) -> Path:
        """Get path for index component file."""
        return self._index_dir / f"{self.config.index_name}.{suffix}.pkl"

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from tokenized corpus."""
        if not self._tokenized_corpus:
            self._bm25 = None
            return

        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self.config.k1,
            b=self.config.b,
        )

    def add_documents(self, entries: List[BM25IndexEntry]) -> int:
        """Add documents to the index.

        Args:
            entries: List of index entries to add.

        Returns:
            Number of documents added.
        """
        if not entries:
            return 0

        added = 0
        existing_ids = set(self._chunk_ids)

        for entry in entries:
            if entry.chunk_id in existing_ids:
                # Skip duplicates (upsert semantics - for update, remove first)
                continue

            tokens = _tokenize(entry.text)
            self._chunk_ids.append(entry.chunk_id)
            self._doc_ids.append(entry.doc_id)
            self._texts.append(entry.text)
            self._metadatas.append(entry.metadata)
            self._tokenized_corpus.append(tokens)
            existing_ids.add(entry.chunk_id)
            added += 1

        if added > 0:
            self._rebuild_bm25()
            self._dirty = True

        return added

    def remove_by_doc_id(self, doc_id: str) -> int:
        """Remove all chunks belonging to a document.

        Args:
            doc_id: Document ID to remove.

        Returns:
            Number of chunks removed.
        """
        # Find indices to remove (in reverse order for safe deletion)
        indices_to_remove = [
            i for i, did in enumerate(self._doc_ids) if did == doc_id
        ]

        if not indices_to_remove:
            return 0

        # Remove in reverse order to maintain valid indices
        for i in reversed(indices_to_remove):
            del self._chunk_ids[i]
            del self._doc_ids[i]
            del self._texts[i]
            del self._metadatas[i]
            del self._tokenized_corpus[i]

        self._rebuild_bm25()
        self._dirty = True

        return len(indices_to_remove)

    def remove_by_chunk_id(self, chunk_id: str) -> bool:
        """Remove a specific chunk by its ID.

        Args:
            chunk_id: Chunk ID to remove.

        Returns:
            True if chunk was found and removed.
        """
        try:
            idx = self._chunk_ids.index(chunk_id)
        except ValueError:
            return False

        del self._chunk_ids[idx]
        del self._doc_ids[idx]
        del self._texts[idx]
        del self._metadatas[idx]
        del self._tokenized_corpus[idx]

        self._rebuild_bm25()
        self._dirty = True
        return True

    def search(self, query: str, k: int = 5) -> List[BM25SearchResult]:
        """Search the index.

        Args:
            query: Search query text.
            k: Number of results to return.

        Returns:
            List of search results sorted by relevance.
        """
        if self._bm25 is None or not self._chunk_ids:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Get top-k indices
        top_k = min(k, len(scores))
        # argsort returns ascending, we want descending by score
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in sorted_indices:
            score = scores[idx]
            if score <= 0:
                continue  # Skip zero scores

            results.append(
                BM25SearchResult(
                    chunk_id=self._chunk_ids[idx],
                    doc_id=self._doc_ids[idx],
                    text=self._texts[idx],
                    score=float(score),
                    metadata=self._metadatas[idx].copy(),
                )
            )

        return results

    def get_by_ids(self, chunk_ids: List[str]) -> List[BM25SearchResult]:
        """Get specific chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to retrieve.

        Returns:
            List of results (without scores).
        """
        results = []
        chunk_id_to_idx = {cid: i for i, cid in enumerate(self._chunk_ids)}

        for chunk_id in chunk_ids:
            idx = chunk_id_to_idx.get(chunk_id)
            if idx is not None:
                results.append(
                    BM25SearchResult(
                        chunk_id=self._chunk_ids[idx],
                        doc_id=self._doc_ids[idx],
                        text=self._texts[idx],
                        score=0.0,  # No score for direct retrieval
                        metadata=self._metadatas[idx].copy(),
                    )
                )

        return results

    def save(self) -> None:
        """Save index to disk."""
        if not self._dirty and self._get_index_path("chunks").exists():
            logger.debug("Index not dirty, skipping save.")
            return

        data = {
            "chunk_ids": self._chunk_ids,
            "doc_ids": self._doc_ids,
            "texts": self._texts,
            "metadatas": self._metadatas,
            "tokenized_corpus": self._tokenized_corpus,
            "config": {
                "k1": self.config.k1,
                "b": self.config.b,
                "index_name": self.config.index_name,
            },
        }

        index_path = self._get_index_path("chunks")
        with index_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._dirty = False
        logger.info(
            "Saved BM25 index: %d chunks to %s",
            len(self._chunk_ids),
            index_path,
        )

    def _load(self) -> bool:
        """Load index from disk if it exists.

        Returns:
            True if index was loaded successfully.
        """
        index_path = self._get_index_path("chunks")
        if not index_path.exists():
            logger.debug("No existing BM25 index found at %s", index_path)
            return False

        try:
            with index_path.open("rb") as f:
                data = pickle.load(f)

            self._chunk_ids = data["chunk_ids"]
            self._doc_ids = data["doc_ids"]
            self._texts = data["texts"]
            self._metadatas = data["metadatas"]
            self._tokenized_corpus = data["tokenized_corpus"]

            # Rebuild BM25 index from loaded data
            self._rebuild_bm25()
            self._dirty = False

            logger.info(
                "Loaded BM25 index: %d chunks from %s",
                len(self._chunk_ids),
                index_path,
            )
            return True

        except Exception as e:
            logger.warning("Failed to load BM25 index from %s: %s", index_path, e)
            return False

    def health_check(self) -> BM25HealthCheckResult:
        """Check health of the BM25 index.

        Returns:
            Health check result with status and document count.
        """
        try:
            count = len(self._chunk_ids)
            index_exists = self._get_index_path("chunks").exists()

            if count == 0 and not index_exists:
                return BM25HealthCheckResult(
                    healthy=True,
                    message="BM25 index empty (not yet created)",
                    document_count=0,
                    index_name=self.config.index_name,
                )

            return BM25HealthCheckResult(
                healthy=True,
                message=f"BM25 index healthy. {count} chunks indexed.",
                document_count=count,
                index_name=self.config.index_name,
            )

        except Exception as e:
            return BM25HealthCheckResult(
                healthy=False,
                message=f"BM25 health check failed: {e}",
                document_count=0,
                index_name=self.config.index_name,
            )

    def count(self) -> int:
        """Get number of indexed chunks.

        Returns:
            Total chunk count.
        """
        return len(self._chunk_ids)

    def clear(self) -> None:
        """Clear all data from the index."""
        self._chunk_ids = []
        self._doc_ids = []
        self._texts = []
        self._metadatas = []
        self._tokenized_corpus = []
        self._bm25 = None
        self._dirty = True
