"""Score normalization utilities for hybrid retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional

logger = logging.getLogger(__name__)

NormalizationMethod = Literal["min_max", "z_score", "rank"]


@dataclass
class NormalizationConfig:
    """Configuration for score normalization.

    Attributes:
        method: Normalization method - 'min_max', 'z_score', or 'rank'.
        min_max_epsilon: Small value to avoid division by zero in min-max normalization.
    """
    method: NormalizationMethod = "min_max"
    min_max_epsilon: float = 1e-10


@dataclass
class NormalizedScore:
    """A score with both original and normalized values preserved.

    Attributes:
        original: The original score value.
        normalized: The normalized score value (0-1 range for most methods).
        item_id: Optional identifier for the scored item.
    """
    original: float
    normalized: float
    item_id: Optional[str] = None


class ScoreNormalizer:
    """Normalizes scores from different retrieval methods to a common scale.

    Supports min-max normalization, z-score normalization, and rank-based
    normalization for combining scores from vector and lexical search.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """Initialize the score normalizer.

        Args:
            config: Normalization configuration. Uses defaults if not provided.
        """
        self.config = config or NormalizationConfig()

    def normalize_vector_distances(
        self,
        distances: List[float],
        ids: Optional[List[str]] = None,
    ) -> List[NormalizedScore]:
        """Normalize vector distances to similarity scores.

        Converts cosine distances (0-2, lower is better) to similarity scores
        (0-1, higher is better).

        Args:
            distances: List of cosine distances from vector search.
            ids: Optional list of chunk IDs corresponding to distances.

        Returns:
            List of NormalizedScore objects with original distances and
            normalized similarity scores.
        """
        if not distances:
            return []

        ids = ids or [None] * len(distances)

        # Convert distances to similarities: similarity = 1 - (distance / 2)
        # Cosine distance ranges from 0 (identical) to 2 (opposite)
        similarities = [1.0 - (d / 2.0) for d in distances]

        # Apply configured normalization method
        normalized = self._normalize(similarities)

        return [
            NormalizedScore(
                original=distances[i],
                normalized=normalized[i],
                item_id=ids[i],
            )
            for i in range(len(distances))
        ]

    def normalize_bm25_scores(
        self,
        scores: List[float],
        ids: Optional[List[str]] = None,
    ) -> List[NormalizedScore]:
        """Normalize BM25 scores to a 0-1 range.

        BM25 scores are unbounded positive values where higher is better.
        This method normalizes them to a 0-1 range for combination with
        vector similarity scores.

        Args:
            scores: List of BM25 scores (higher is better).
            ids: Optional list of chunk IDs corresponding to scores.

        Returns:
            List of NormalizedScore objects with original and normalized scores.
        """
        if not scores:
            return []

        ids = ids or [None] * len(scores)

        # Apply configured normalization method
        normalized = self._normalize(scores)

        return [
            NormalizedScore(
                original=scores[i],
                normalized=normalized[i],
                item_id=ids[i],
            )
            for i in range(len(scores))
        ]

    def _normalize(self, values: List[float]) -> List[float]:
        """Apply the configured normalization method.

        Args:
            values: List of values to normalize (higher is better).

        Returns:
            Normalized values in 0-1 range.
        """
        if not values:
            return []

        if self.config.method == "min_max":
            return self._min_max_normalize(values)
        elif self.config.method == "z_score":
            return self._z_score_normalize(values)
        elif self.config.method == "rank":
            return self._rank_normalize(values)
        else:
            logger.warning(
                "Unknown normalization method '%s', using min_max.",
                self.config.method,
            )
            return self._min_max_normalize(values)

    def _min_max_normalize(self, values: List[float]) -> List[float]:
        """Min-max normalization to 0-1 range.

        Args:
            values: Values to normalize.

        Returns:
            Normalized values where min becomes 0 and max becomes 1.
        """
        if len(values) == 1:
            return [1.0]  # Single value gets max score

        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        if range_val < self.config.min_max_epsilon:
            # All values are essentially equal
            return [1.0] * len(values)

        return [(v - min_val) / range_val for v in values]

    def _z_score_normalize(self, values: List[float]) -> List[float]:
        """Z-score normalization followed by sigmoid to get 0-1 range.

        Args:
            values: Values to normalize.

        Returns:
            Normalized values using z-score + sigmoid transformation.
        """
        if len(values) == 1:
            return [0.5]  # Single value at mean

        import math

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 1.0

        # Z-scores
        z_scores = [(v - mean) / std if std > 0 else 0.0 for v in values]

        # Sigmoid to map to 0-1
        return [1.0 / (1.0 + math.exp(-z)) for z in z_scores]

    def _rank_normalize(self, values: List[float]) -> List[float]:
        """Rank-based normalization to 0-1 range.

        Assigns scores based on rank position, with the highest value
        getting score 1.0 and the lowest getting score close to 0.

        Args:
            values: Values to normalize.

        Returns:
            Normalized values based on rank position.
        """
        if len(values) == 1:
            return [1.0]

        n = len(values)
        # Get ranks (higher value = higher rank)
        sorted_indices = sorted(range(n), key=lambda i: values[i], reverse=True)
        ranks = [0] * n
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank

        # Convert ranks to 0-1 scores (rank 0 = 1.0, rank n-1 = 1/n)
        return [(n - r) / n for r in ranks]
