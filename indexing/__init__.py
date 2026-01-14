"""Vector indexing package using Chroma for document chunk storage.

This package provides:
- ChromaIndexingPipeline: Index document chunks in Chroma for RAG retrieval
- SummaryIndexer: Index dataset summaries for semantic discovery
- EmbeddingService: Generate embeddings with retry logic
"""

from .embeddings import (
    DEFAULT_DIMENSIONS,
    DEFAULT_MODEL,
    EmbeddingConfig,
    EmbeddingError,
    EmbeddingService,
    build_embedding_function,
)
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
    # Dataset summary indexing
    "SummaryIndexer",
    "SummaryIndexConfig",
    "DatasetSearchResult",
]
