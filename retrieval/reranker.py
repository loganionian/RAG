"""Cross-encoder reranker for improving retrieval relevance."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker_model_path(model_name: str) -> str:
    """Resolve reranker model path, preferring local models if available.

    Args:
        model_name: Model name or path.

    Returns:
        Resolved model path (local if exists, otherwise original name).
    """
    import os

    # If it's already a local path that exists, use it
    if Path(model_name).exists():
        logger.info("Using local reranker model: %s", model_name)
        return model_name

    # Check for local model in models/ directory
    local_model_name = model_name.replace("/", "_")
    local_path = Path("models") / local_model_name

    if local_path.exists():
        logger.info("Found local reranker model: %s", local_path)
        return str(local_path)

    # Check environment variable for model path
    env_model_path = os.getenv("RERANKER_MODEL_PATH")
    if env_model_path and Path(env_model_path).exists():
        logger.info("Using reranker model from RERANKER_MODEL_PATH: %s", env_model_path)
        return env_model_path

    # Fall back to downloading from HuggingFace
    logger.info("Using remote reranker model: %s", model_name)
    return model_name


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranker.

    Attributes:
        model_name: HuggingFace model name or local path for the cross-encoder.
        max_length: Maximum sequence length for the cross-encoder.
        batch_size: Batch size for reranking.
        device: Device to use ('cpu', 'cuda', 'cuda:0', etc.). None for auto-detect.
    """
    model_name: str = DEFAULT_RERANKER_MODEL
    max_length: int = 512
    batch_size: int = 32
    device: Optional[str] = None


@dataclass
class RerankResult:
    """Result from cross-encoder reranking.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        text: The chunk text content.
        metadata: Chunk metadata dictionary.
        original_score: Score before reranking (e.g., RRF score or distance).
        rerank_score: Cross-encoder relevance score.
    """
    chunk_id: str
    text: str
    metadata: Dict
    original_score: float
    rerank_score: float


class CrossEncoderReranker:
    """Cross-encoder reranker for improving retrieval relevance.

    Uses a cross-encoder model to score query-document pairs and rerank
    retrieval results based on relevance scores.

    The model is loaded lazily on first use to avoid startup overhead.
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        """Initialize the reranker.

        Args:
            config: Reranker configuration. Uses defaults if not provided.
        """
        self.config = config or RerankerConfig()
        self._model = None
        self._model_loaded = False

    def _get_model(self):
        """Lazy-load the cross-encoder model.

        Returns:
            CrossEncoder model instance.
        """
        if self._model is None:
            from sentence_transformers import CrossEncoder

            resolved_path = get_reranker_model_path(self.config.model_name)
            logger.info("Loading cross-encoder model: %s", resolved_path)

            self._model = CrossEncoder(
                resolved_path,
                max_length=self.config.max_length,
                device=self.config.device,
            )
            self._model_loaded = True
            logger.info("Cross-encoder model loaded successfully")

        return self._model

    @property
    def is_loaded(self) -> bool:
        """Check if the model has been loaded."""
        return self._model_loaded

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank candidate documents using the cross-encoder.

        Args:
            query: The search query.
            candidates: List of candidate documents with keys:
                - 'chunk_id': Unique identifier
                - 'text': Document text
                - 'metadata': Metadata dictionary
                - 'score': Original retrieval score
            top_k: Number of top results to return. None returns all.

        Returns:
            List of RerankResult objects sorted by rerank_score (descending).
        """
        if not candidates:
            return []

        model = self._get_model()

        # Prepare query-document pairs
        pairs = [(query, c["text"]) for c in candidates]

        # Score all pairs
        logger.debug("Reranking %d candidates", len(candidates))
        scores = model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        # Build results with rerank scores
        results = [
            RerankResult(
                chunk_id=candidates[i]["chunk_id"],
                text=candidates[i]["text"],
                metadata=candidates[i]["metadata"],
                original_score=candidates[i].get("score", 0.0),
                rerank_score=float(scores[i]),
            )
            for i in range(len(candidates))
        ]

        # Sort by rerank score (descending) and take top_k
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results
