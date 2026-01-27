"""Tests for LexicalSearchService."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from indexing.bm25_store import BM25Config, BM25IndexEntry, BM25Store
from indexing.lexical_search import (
    LexicalSearchConfig,
    LexicalSearchResult,
    LexicalSearchService,
)


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for BM25 index files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def populated_index(temp_index_dir):
    """Create a BM25 store with test data."""
    config = BM25Config(
        index_dir=temp_index_dir,
        index_name="test-index",
    )
    store = BM25Store(config)

    entries = [
        BM25IndexEntry(
            chunk_id="doc1::chunk-0001",
            doc_id="doc1",
            text="The quick brown fox jumps over the lazy dog",
            metadata={"page": 1, "doc_id": "doc1"},
        ),
        BM25IndexEntry(
            chunk_id="doc1::chunk-0002",
            doc_id="doc1",
            text="Machine learning and artificial intelligence transform industries",
            metadata={"page": 2, "doc_id": "doc1"},
        ),
        BM25IndexEntry(
            chunk_id="doc2::chunk-0001",
            doc_id="doc2",
            text="Natural language processing enables text understanding",
            metadata={"page": 1, "doc_id": "doc2"},
        ),
    ]
    store.add_documents(entries)
    store.save()

    return temp_index_dir


class TestLexicalSearchConfig:
    """Tests for LexicalSearchConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LexicalSearchConfig()
        assert config.index_dir == Path("data/vectorstore")
        assert config.index_name == "pilot-docs"
        assert config.k1 == 1.5
        assert config.b == 0.75

    def test_custom_values(self, temp_index_dir):
        """Test custom configuration values."""
        config = LexicalSearchConfig(
            index_dir=temp_index_dir,
            index_name="custom-index",
            k1=1.2,
            b=0.6,
        )
        assert config.index_dir == temp_index_dir
        assert config.index_name == "custom-index"
        assert config.k1 == 1.2
        assert config.b == 0.6


class TestLexicalSearchService:
    """Tests for LexicalSearchService class."""

    def test_init(self, populated_index):
        """Test service initialization."""
        config = LexicalSearchConfig(
            index_dir=populated_index,
            index_name="test-index",
        )
        service = LexicalSearchService(config)
        assert service.count() == 3

    def test_search_returns_lexical_search_result(self, populated_index):
        """Test that search returns LexicalSearchResult."""
        config = LexicalSearchConfig(
            index_dir=populated_index,
            index_name="test-index",
        )
        service = LexicalSearchService(config)

        result = service.search("machine learning", k=2)
        assert isinstance(result, LexicalSearchResult)
        assert len(result.chunks) > 0
        assert len(result.chunks) == len(result.metadatas)
        assert len(result.chunks) == len(result.scores)
        assert len(result.chunks) == len(result.ids)

    def test_search_result_format_compatibility(self, populated_index):
        """Test that results are compatible with RetrievalResult format."""
        config = LexicalSearchConfig(
            index_dir=populated_index,
            index_name="test-index",
        )
        service = LexicalSearchService(config)

        result = service.search("fox dog", k=1)

        # Verify all required fields are present
        assert hasattr(result, "chunks")
        assert hasattr(result, "metadatas")
        assert hasattr(result, "scores")
        assert hasattr(result, "ids")

        # Verify types
        assert isinstance(result.chunks, list)
        assert isinstance(result.metadatas, list)
        assert isinstance(result.scores, list)
        assert isinstance(result.ids, list)

    def test_search_finds_relevant_content(self, populated_index):
        """Test that search returns relevant results."""
        config = LexicalSearchConfig(
            index_dir=populated_index,
            index_name="test-index",
        )
        service = LexicalSearchService(config)

        result = service.search("artificial intelligence", k=3)

        # Should find the AI-related chunk
        assert any("artificial" in chunk.lower() for chunk in result.chunks)

    def test_get_by_ids(self, populated_index):
        """Test retrieving specific chunks by ID."""
        config = LexicalSearchConfig(
            index_dir=populated_index,
            index_name="test-index",
        )
        service = LexicalSearchService(config)

        result = service.get_by_ids(["doc1::chunk-0001", "doc2::chunk-0001"])
        assert len(result.ids) == 2
        assert "doc1::chunk-0001" in result.ids
        assert "doc2::chunk-0001" in result.ids

    def test_health_check(self, populated_index):
        """Test health check returns expected result."""
        config = LexicalSearchConfig(
            index_dir=populated_index,
            index_name="test-index",
        )
        service = LexicalSearchService(config)

        health = service.health_check()
        assert health.healthy is True
        assert health.document_count == 3

    def test_store_property(self, populated_index):
        """Test access to underlying BM25 store."""
        config = LexicalSearchConfig(
            index_dir=populated_index,
            index_name="test-index",
        )
        service = LexicalSearchService(config)

        assert service.store is not None
        assert isinstance(service.store, BM25Store)


class TestLexicalSearchResult:
    """Tests for LexicalSearchResult dataclass."""

    def test_dataclass_fields(self):
        """Test that dataclass has all expected fields."""
        result = LexicalSearchResult(
            chunks=["text1", "text2"],
            metadatas=[{"a": 1}, {"b": 2}],
            scores=[0.9, 0.8],
            ids=["id1", "id2"],
        )

        assert result.chunks == ["text1", "text2"]
        assert result.metadatas == [{"a": 1}, {"b": 2}]
        assert result.scores == [0.9, 0.8]
        assert result.ids == ["id1", "id2"]

    def test_empty_result(self):
        """Test creating empty result."""
        result = LexicalSearchResult(
            chunks=[],
            metadatas=[],
            scores=[],
            ids=[],
        )

        assert len(result.chunks) == 0
        assert len(result.metadatas) == 0
        assert len(result.scores) == 0
        assert len(result.ids) == 0
