"""Lexical search service using BM25.

This module provides a high-level wrapper around BM25Store that returns
results in a format compatible with RetrievalResult from the RAG chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .bm25_store import BM25Config, BM25SearchResult, BM25Store

logger = logging.getLogger(__name__)


@dataclass
class LexicalSearchResult:
    """Result from lexical search, compatible with RetrievalResult format."""

    chunks: List[str]
    metadatas: List[dict]
    scores: List[float]
    ids: List[str]


@dataclass
class LexicalSearchConfig:
    """Configuration for lexical search service."""

    index_dir: Path = field(default_factory=lambda: Path("data/vectorstore"))
    index_name: str = "pilot-docs"
    k1: float = 1.5
    b: float = 0.75


class LexicalSearchService:
    """High-level wrapper for BM25 search returning RetrievalResult-compatible format.

    This service provides the same interface as Chroma vector search,
    enabling easy switching or combination of search methods.
    """

    def __init__(self, config: Optional[LexicalSearchConfig] = None):
        """Initialize lexical search service.

        Args:
            config: Search configuration. Uses defaults if not provided.
        """
        self.config = config or LexicalSearchConfig()
        self._bm25_config = BM25Config(
            index_dir=self.config.index_dir,
            index_name=self.config.index_name,
            k1=self.config.k1,
            b=self.config.b,
        )
        self._store = BM25Store(self._bm25_config)

    @property
    def store(self) -> BM25Store:
        """Get underlying BM25 store for direct access."""
        return self._store

    def search(self, query: str, k: int = 5) -> LexicalSearchResult:
        """Search using BM25 lexical matching.

        Args:
            query: Search query text.
            k: Number of results to return.

        Returns:
            Search results in RetrievalResult-compatible format.
        """
        bm25_results = self._store.search(query, k=k)
        return self._convert_results(bm25_results)

    def get_by_ids(self, chunk_ids: List[str]) -> LexicalSearchResult:
        """Get specific chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to retrieve.

        Returns:
            Results in RetrievalResult-compatible format.
        """
        bm25_results = self._store.get_by_ids(chunk_ids)
        return self._convert_results(bm25_results)

    def _convert_results(self, bm25_results: List[BM25SearchResult]) -> LexicalSearchResult:
        """Convert BM25 results to LexicalSearchResult format.

        Args:
            bm25_results: Raw BM25 search results.

        Returns:
            Converted results.
        """
        chunks = [r.text for r in bm25_results]
        metadatas = [r.metadata for r in bm25_results]
        scores = [r.score for r in bm25_results]
        ids = [r.chunk_id for r in bm25_results]

        return LexicalSearchResult(
            chunks=chunks,
            metadatas=metadatas,
            scores=scores,
            ids=ids,
        )

    def health_check(self):
        """Check health of the lexical search service.

        Returns:
            BM25 health check result.
        """
        return self._store.health_check()

    def count(self) -> int:
        """Get number of indexed chunks.

        Returns:
            Total chunk count in BM25 index.
        """
        return self._store.count()
