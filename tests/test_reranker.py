"""Tests for cross-encoder reranker functionality."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from retrieval.reranker import (
    CrossEncoderReranker,
    RerankerConfig,
    RerankResult,
    get_reranker_model_path,
    DEFAULT_RERANKER_MODEL,
)


class TestRerankerConfig:
    """Tests for RerankerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RerankerConfig()
        assert config.model_name == DEFAULT_RERANKER_MODEL
        assert config.max_length == 512
        assert config.batch_size == 32
        assert config.device is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RerankerConfig(
            model_name="custom-model",
            max_length=256,
            batch_size=16,
            device="cuda:0",
        )
        assert config.model_name == "custom-model"
        assert config.max_length == 256
        assert config.batch_size == 16
        assert config.device == "cuda:0"


class TestRerankResult:
    """Tests for RerankResult dataclass."""

    def test_result_creation(self):
        """Test creating a RerankResult."""
        result = RerankResult(
            chunk_id="doc1::chunk-0001",
            text="Sample text",
            metadata={"page": 1, "doc_id": "doc1"},
            original_score=0.5,
            rerank_score=0.8,
        )
        assert result.chunk_id == "doc1::chunk-0001"
        assert result.text == "Sample text"
        assert result.metadata == {"page": 1, "doc_id": "doc1"}
        assert result.original_score == 0.5
        assert result.rerank_score == 0.8


class TestGetRerankerModelPath:
    """Tests for get_reranker_model_path function."""

    def test_existing_path_returned_as_is(self, tmp_path):
        """Test that existing path is returned unchanged."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        result = get_reranker_model_path(str(model_path))
        assert result == str(model_path)

    def test_local_model_detected(self, tmp_path, monkeypatch):
        """Test that local model in models/ directory is detected."""
        # Create mock models directory
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_dir = models_dir / "cross-encoder_test-model"
        model_dir.mkdir()

        # Change to tmp_path so models/ is found
        monkeypatch.chdir(tmp_path)

        result = get_reranker_model_path("cross-encoder/test-model")
        assert "cross-encoder_test-model" in result

    def test_env_var_model_path(self, tmp_path, monkeypatch):
        """Test that RERANKER_MODEL_PATH env var is respected."""
        model_path = tmp_path / "env_model"
        model_path.mkdir()
        monkeypatch.setenv("RERANKER_MODEL_PATH", str(model_path))

        result = get_reranker_model_path("some-remote-model")
        assert result == str(model_path)

    def test_remote_model_fallback(self):
        """Test that remote model name is returned when no local found."""
        result = get_reranker_model_path("cross-encoder/nonexistent-model")
        assert result == "cross-encoder/nonexistent-model"


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker class."""

    def test_reranker_not_loaded_initially(self):
        """Test that model is not loaded on initialization."""
        reranker = CrossEncoderReranker()
        assert reranker._model is None
        assert reranker.is_loaded is False

    def test_rerank_empty_candidates(self):
        """Test reranking empty candidates returns empty list."""
        reranker = CrossEncoderReranker()
        result = reranker.rerank("test query", [])
        assert result == []

    def test_rerank_with_mock_model(self):
        """Test reranking with mocked model."""
        reranker = CrossEncoderReranker()

        # Mock the CrossEncoder
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.3, 0.7]

        candidates = [
            {
                "chunk_id": "chunk-1",
                "text": "First document",
                "metadata": {"page": 1},
                "score": 0.5,
            },
            {
                "chunk_id": "chunk-2",
                "text": "Second document",
                "metadata": {"page": 2},
                "score": 0.4,
            },
            {
                "chunk_id": "chunk-3",
                "text": "Third document",
                "metadata": {"page": 3},
                "score": 0.3,
            },
        ]

        with patch.object(reranker, "_get_model", return_value=mock_model):
            result = reranker.rerank("test query", candidates)

        # Should be sorted by rerank_score descending
        assert len(result) == 3
        assert result[0].chunk_id == "chunk-1"  # 0.9
        assert result[1].chunk_id == "chunk-3"  # 0.7
        assert result[2].chunk_id == "chunk-2"  # 0.3

        # Check scores
        assert result[0].rerank_score == 0.9
        assert result[0].original_score == 0.5

    def test_rerank_with_top_k(self):
        """Test reranking with top_k limit."""
        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.3, 0.7, 0.5]

        candidates = [
            {"chunk_id": f"chunk-{i}", "text": f"Doc {i}", "metadata": {}, "score": 0.5}
            for i in range(4)
        ]

        with patch.object(reranker, "_get_model", return_value=mock_model):
            result = reranker.rerank("test query", candidates, top_k=2)

        assert len(result) == 2
        # Top 2 by score: 0.9 and 0.7
        assert result[0].rerank_score == 0.9
        assert result[1].rerank_score == 0.7

    def test_rerank_preserves_metadata(self):
        """Test that metadata is preserved through reranking."""
        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]

        candidates = [
            {
                "chunk_id": "chunk-1",
                "text": "Document text",
                "metadata": {"page": 5, "doc_id": "doc1", "section": "intro"},
                "score": 0.5,
            }
        ]

        with patch.object(reranker, "_get_model", return_value=mock_model):
            result = reranker.rerank("query", candidates)

        assert result[0].metadata == {"page": 5, "doc_id": "doc1", "section": "intro"}

    def test_rerank_batch_size_config(self):
        """Test that batch_size config is used in predict."""
        config = RerankerConfig(batch_size=8)
        reranker = CrossEncoderReranker(config)

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]

        candidates = [
            {"chunk_id": "chunk-1", "text": "Doc", "metadata": {}, "score": 0.5}
        ]

        with patch.object(reranker, "_get_model", return_value=mock_model):
            reranker.rerank("query", candidates)

        mock_model.predict.assert_called_once()
        call_kwargs = mock_model.predict.call_args[1]
        assert call_kwargs["batch_size"] == 8


class TestCrossEncoderLazyLoading:
    """Tests for lazy loading behavior of CrossEncoderReranker."""

    def test_model_loaded_on_first_rerank(self):
        """Test that model is loaded lazily on first rerank call."""
        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]

        with patch("sentence_transformers.CrossEncoder", return_value=mock_model) as mock_ce:
            candidates = [
                {"chunk_id": "c1", "text": "text", "metadata": {}, "score": 0.5}
            ]
            reranker.rerank("query", candidates)

            mock_ce.assert_called_once()
            assert reranker.is_loaded is True

    def test_model_not_reloaded_on_subsequent_calls(self):
        """Test that model is only loaded once."""
        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]

        with patch("sentence_transformers.CrossEncoder", return_value=mock_model) as mock_ce:
            candidates = [
                {"chunk_id": "c1", "text": "text", "metadata": {}, "score": 0.5}
            ]

            reranker.rerank("query1", candidates)
            reranker.rerank("query2", candidates)
            reranker.rerank("query3", candidates)

            # Should only be called once
            assert mock_ce.call_count == 1
