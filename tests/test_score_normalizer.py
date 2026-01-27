"""Tests for score normalization functionality."""

from __future__ import annotations

import math

import pytest

from retrieval.score_normalizer import (
    NormalizationConfig,
    NormalizedScore,
    ScoreNormalizer,
)


class TestNormalizationConfig:
    """Tests for NormalizationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NormalizationConfig()
        assert config.method == "min_max"
        assert config.min_max_epsilon == 1e-10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = NormalizationConfig(method="z_score", min_max_epsilon=1e-6)
        assert config.method == "z_score"
        assert config.min_max_epsilon == 1e-6


class TestNormalizedScore:
    """Tests for NormalizedScore dataclass."""

    def test_score_creation(self):
        """Test creating a NormalizedScore."""
        score = NormalizedScore(original=0.5, normalized=0.75, item_id="chunk-1")
        assert score.original == 0.5
        assert score.normalized == 0.75
        assert score.item_id == "chunk-1"

    def test_score_without_item_id(self):
        """Test creating a NormalizedScore without item_id."""
        score = NormalizedScore(original=0.5, normalized=0.75)
        assert score.item_id is None


class TestScoreNormalizerMinMax:
    """Tests for min-max normalization method."""

    def test_normalize_empty_list(self):
        """Test normalizing empty list returns empty."""
        normalizer = ScoreNormalizer()
        result = normalizer.normalize_vector_distances([])
        assert result == []

    def test_normalize_single_value(self):
        """Test normalizing single value returns max score."""
        normalizer = ScoreNormalizer()
        result = normalizer.normalize_vector_distances([0.5])
        assert len(result) == 1
        assert result[0].normalized == 1.0

    def test_normalize_vector_distances(self):
        """Test normalizing cosine distances to similarities."""
        normalizer = ScoreNormalizer()
        # Cosine distance: 0 = identical, 2 = opposite
        distances = [0.0, 0.5, 1.0, 1.5, 2.0]
        result = normalizer.normalize_vector_distances(distances)

        assert len(result) == 5
        # Distance 0.0 -> similarity 1.0 (highest after normalization)
        # Distance 2.0 -> similarity 0.0 (lowest after normalization)
        assert result[0].normalized > result[4].normalized

    def test_normalize_bm25_scores(self):
        """Test normalizing BM25 scores to 0-1 range."""
        normalizer = ScoreNormalizer()
        # BM25: higher is better
        scores = [0.0, 5.0, 10.0, 15.0, 20.0]
        result = normalizer.normalize_bm25_scores(scores)

        assert len(result) == 5
        # Highest BM25 score should have highest normalized score
        assert result[4].normalized == 1.0
        # Lowest BM25 score should have lowest normalized score
        assert result[0].normalized == 0.0

    def test_normalize_with_ids(self):
        """Test that chunk IDs are preserved."""
        normalizer = ScoreNormalizer()
        distances = [0.1, 0.2]
        ids = ["chunk-1", "chunk-2"]
        result = normalizer.normalize_vector_distances(distances, ids)

        assert result[0].item_id == "chunk-1"
        assert result[1].item_id == "chunk-2"

    def test_normalize_identical_values(self):
        """Test normalizing identical values returns max scores."""
        normalizer = ScoreNormalizer()
        scores = [5.0, 5.0, 5.0]
        result = normalizer.normalize_bm25_scores(scores)

        # All identical values should normalize to 1.0
        for r in result:
            assert r.normalized == 1.0


class TestScoreNormalizerZScore:
    """Tests for z-score normalization method."""

    def test_z_score_normalization(self):
        """Test z-score normalization produces values in 0-1 range."""
        config = NormalizationConfig(method="z_score")
        normalizer = ScoreNormalizer(config)

        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = normalizer.normalize_bm25_scores(scores)

        # All values should be in 0-1 range (sigmoid output)
        for r in result:
            assert 0.0 <= r.normalized <= 1.0

        # Mean value should be around 0.5
        mean_score = result[2].normalized  # Middle value
        assert 0.4 < mean_score < 0.6

    def test_z_score_single_value(self):
        """Test z-score with single value returns 0.5."""
        config = NormalizationConfig(method="z_score")
        normalizer = ScoreNormalizer(config)

        result = normalizer.normalize_bm25_scores([10.0])
        assert result[0].normalized == 0.5

    def test_z_score_preserves_ordering(self):
        """Test z-score preserves relative ordering."""
        config = NormalizationConfig(method="z_score")
        normalizer = ScoreNormalizer(config)

        scores = [1.0, 5.0, 10.0]
        result = normalizer.normalize_bm25_scores(scores)

        assert result[0].normalized < result[1].normalized < result[2].normalized


class TestScoreNormalizerRank:
    """Tests for rank-based normalization method."""

    def test_rank_normalization(self):
        """Test rank-based normalization."""
        config = NormalizationConfig(method="rank")
        normalizer = ScoreNormalizer(config)

        scores = [10.0, 30.0, 20.0]  # Unordered
        result = normalizer.normalize_bm25_scores(scores)

        # Rank 0 (highest value 30.0) should get 1.0
        assert result[1].normalized == 1.0  # 30.0 is at index 1
        # Rank 1 (20.0) should get 2/3
        assert abs(result[2].normalized - 2/3) < 0.01
        # Rank 2 (10.0) should get 1/3
        assert abs(result[0].normalized - 1/3) < 0.01

    def test_rank_single_value(self):
        """Test rank normalization with single value returns 1.0."""
        config = NormalizationConfig(method="rank")
        normalizer = ScoreNormalizer(config)

        result = normalizer.normalize_bm25_scores([5.0])
        assert result[0].normalized == 1.0


class TestVectorDistanceConversion:
    """Tests for cosine distance to similarity conversion."""

    def test_distance_zero_is_max_similarity(self):
        """Test that distance 0 converts to similarity 1."""
        normalizer = ScoreNormalizer()
        result = normalizer.normalize_vector_distances([0.0])
        # After normalization, single value gets 1.0
        assert result[0].normalized == 1.0

    def test_distance_two_is_min_similarity(self):
        """Test that distance 2 converts to similarity 0."""
        normalizer = ScoreNormalizer()
        result = normalizer.normalize_vector_distances([0.0, 2.0])
        # Distance 2.0 should have lowest similarity
        assert result[1].normalized < result[0].normalized

    def test_distance_one_is_mid_similarity(self):
        """Test that distance 1 converts to similarity 0.5 (before normalization)."""
        normalizer = ScoreNormalizer()
        # Raw conversion: 1 - (1/2) = 0.5
        result = normalizer.normalize_vector_distances([0.0, 1.0, 2.0])
        # After min-max normalization:
        # 0.0 -> 1.0, 1.0 -> 0.5, 2.0 -> 0.0
        # Normalized: [1.0, 0.5, 0.0]
        assert result[0].normalized == 1.0
        assert result[1].normalized == 0.5
        assert result[2].normalized == 0.0
