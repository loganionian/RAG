"""Vector indexing package using Chroma for document chunk storage.

This package provides:
- ChromaIndexingPipeline: Index document chunks in Chroma for RAG retrieval
- SummaryIndexer: Index dataset summaries for semantic discovery
- EmbeddingService: Generate embeddings with retry logic
- BM25Store: Persistent BM25 index for lexical search
- LexicalSearchService: High-level wrapper for BM25 search
"""

from .bm25_store import (
    BM25Config,
    BM25HealthCheckResult,
    BM25IndexEntry,
    BM25SearchResult,
    BM25Store,
)
from .embeddings import (
    DEFAULT_DIMENSIONS,
    DEFAULT_MODEL,
    EmbeddingConfig,
    EmbeddingError,
    EmbeddingService,
    build_embedding_function,
)
from .lexical_search import LexicalSearchConfig, LexicalSearchResult, LexicalSearchService
from .pipeline import ChromaIndexingPipeline, IndexingConfig, IndexingResult
from .summary_indexer import DatasetSearchResult, SummaryIndexConfig, SummaryIndexer

__all__ = [
    # Document chunk indexing
    "ChromaIndexingPipeline",
    "IndexingConfig",
    "IndexingResult",
    # Embeddings
    "EmbeddingService",
    "EmbeddingConfig",
    "EmbeddingError",
    "build_embedding_function",
    "DEFAULT_MODEL",
    "DEFAULT_DIMENSIONS",
    # BM25 lexical search
    "BM25Store",
    "BM25Config",
    "BM25IndexEntry",
    "BM25SearchResult",
    "BM25HealthCheckResult",
    "LexicalSearchService",
    "LexicalSearchConfig",
    "LexicalSearchResult",
    # Dataset summary indexing
    "SummaryIndexer",
    "SummaryIndexConfig",
    "DatasetSearchResult",
]
