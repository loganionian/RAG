"""Tests for BM25 lexical search index."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from indexing.bm25_store import (
    BM25Config,
    BM25IndexEntry,
    BM25SearchResult,
    BM25Store,
)


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for BM25 index files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def bm25_config(temp_index_dir):
    """Create a BM25 config with temp directory."""
    return BM25Config(
        index_dir=temp_index_dir,
        index_name="test-index",
    )


@pytest.fixture
def sample_entries():
    """Create sample BM25 index entries."""
    return [
        BM25IndexEntry(
            chunk_id="doc1::chunk-0001",
            doc_id="doc1",
            text="The quick brown fox jumps over the lazy dog",
            metadata={"page": 1, "section": "intro"},
        ),
        BM25IndexEntry(
            chunk_id="doc1::chunk-0002",
            doc_id="doc1",
            text="Machine learning is a subset of artificial intelligence",
            metadata={"page": 2, "section": "background"},
        ),
        BM25IndexEntry(
            chunk_id="doc2::chunk-0001",
            doc_id="doc2",
            text="Natural language processing enables text analysis",
            metadata={"page": 1, "section": "methods"},
        ),
    ]


class TestBM25Config:
    """Tests for BM25Config dataclass."""

    def test_default_values(self, temp_index_dir):
        """Test default configuration values."""
        config = BM25Config(index_dir=temp_index_dir)
        assert config.index_name == "pilot-docs"
        assert config.k1 == 1.5
        assert config.b == 0.75

    def test_custom_values(self, temp_index_dir):
        """Test custom configuration values."""
        config = BM25Config(
            index_dir=temp_index_dir,
            index_name="custom-index",
            k1=1.2,
            b=0.6,
        )
        assert config.index_name == "custom-index"
        assert config.k1 == 1.2
        assert config.b == 0.6


class TestBM25Store:
    """Tests for BM25Store class."""

    def test_init_creates_directory(self, bm25_config):
        """Test that initialization creates the bm25 subdirectory."""
        store = BM25Store(bm25_config)
        assert (bm25_config.index_dir / "bm25").exists()

    def test_add_documents(self, bm25_config, sample_entries):
        """Test adding documents to the index."""
        store = BM25Store(bm25_config)
        added = store.add_documents(sample_entries)
        assert added == 3
        assert store.count() == 3

    def test_add_documents_skips_duplicates(self, bm25_config, sample_entries):
        """Test that duplicate chunk IDs are skipped."""
        store = BM25Store(bm25_config)
        store.add_documents(sample_entries)
        added = store.add_documents(sample_entries)
        assert added == 0
        assert store.count() == 3

    def test_add_documents_empty_list(self, bm25_config):
        """Test adding empty list returns 0."""
        store = BM25Store(bm25_config)
        added = store.add_documents([])
        assert added == 0

    def test_remove_by_doc_id(self, bm25_config, sample_entries):
        """Test removing all chunks by doc_id."""
        store = BM25Store(bm25_config)
        store.add_documents(sample_entries)

        removed = store.remove_by_doc_id("doc1")
        assert removed == 2
        assert store.count() == 1

    def test_remove_by_doc_id_nonexistent(self, bm25_config, sample_entries):
        """Test removing nonexistent doc_id returns 0."""
        store = BM25Store(bm25_config)
        store.add_documents(sample_entries)

        removed = store.remove_by_doc_id("nonexistent")
        assert removed == 0
        assert store.count() == 3

    def test_remove_by_chunk_id(self, bm25_config, sample_entries):
        """Test removing a specific chunk by ID."""
        store = BM25Store(bm25_config)
        store.add_documents(sample_entries)

        removed = store.remove_by_chunk_id("doc1::chunk-0001")
        assert removed is True
        assert store.count() == 2

    def test_remove_by_chunk_id_nonexistent(self, bm25_config, sample_entries):
        """Test removing nonexistent chunk_id returns False."""
        store = BM25Store(bm25_config)
        store.add_documents(sample_entries)

        removed = store.remove_by_chunk_id("nonexistent")
        assert removed is False
        assert store.count() == 3

    def test_search_basic(self, bm25_config, sample_entries):
        """Test basic search functionality."""
        store = BM25Store(bm25_config)
        store.add_documents(sample_entries)

        results = store.search("fox dog", k=2)
        assert len(results) >= 1
        assert results[0].chunk_id == "doc1::chunk-0001"
        assert results[0].score > 0

    def test_search_returns_metadata(self, bm25_config, sample_entries):
        """Test that search results include metadata."""
        store = BM25Store(bm25_config)
        store.add_documents(sample_entries)

        results = store.search("machine learning", k=1)
        assert len(results) == 1
        assert results[0].metadata["page"] == 2
        assert results[0].metadata["section"] == "background"

    def test_search_respects_k(self, bm25_config, sample_entries):
        """Test that search returns at most k results."""
        store = BM25Store(bm25_config)
        store.add_documents(sample_entries)

        results = store.search("the", k=2)
        assert len(results) <= 2

    def test_search_empty_query(self, bm25_config, sample_entries):
        """Test searching with empty query returns empty results."""
        store = BM25Store(bm25_config)
        store.add_documents(sample_entries)

        results = store.search("", k=5)
        assert results == []

    def test_search_empty_index(self, bm25_config):
        """Test searching empty index returns empty results."""
        store = BM25Store(bm25_config)
        results = store.search("test query", k=5)
        assert results == []

    def test_get_by_ids(self, bm25_config, sample_entries):
        """Test retrieving chunks by their IDs."""
        store = BM25Store(bm25_config)
        store.add_documents(sample_entries)

        results = store.get_by_ids(["doc1::chunk-0001", "doc2::chunk-0001"])
        assert len(results) == 2
        chunk_ids = {r.chunk_id for r in results}
        assert "doc1::chunk-0001" in chunk_ids
        assert "doc2::chunk-0001" in chunk_ids

    def test_get_by_ids_nonexistent(self, bm25_config, sample_entries):
        """Test get_by_ids ignores nonexistent IDs."""
        store = BM25Store(bm25_config)
        store.add_documents(sample_entries)

        results = store.get_by_ids(["doc1::chunk-0001", "nonexistent"])
        assert len(results) == 1

    def test_save_and_load(self, bm25_config, sample_entries):
        """Test saving and loading the index."""
        store = BM25Store(bm25_config)
        store.add_documents(sample_entries)
        store.save()

        # Create new store and verify it loads
        store2 = BM25Store(bm25_config)
        assert store2.count() == 3

        # Verify search still works
        results = store2.search("machine learning", k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "doc1::chunk-0002"

    def test_health_check_empty(self, bm25_config):
        """Test health check on empty index."""
        store = BM25Store(bm25_config)
        result = store.health_check()
        assert result.healthy is True
        assert result.document_count == 0

    def test_health_check_with_data(self, bm25_config, sample_entries):
        """Test health check with indexed data."""
        store = BM25Store(bm25_config)
        store.add_documents(sample_entries)

        result = store.health_check()
        assert result.healthy is True
        assert result.document_count == 3
        assert result.index_name == "test-index"

    def test_clear(self, bm25_config, sample_entries):
        """Test clearing all data from the index."""
        store = BM25Store(bm25_config)
        store.add_documents(sample_entries)
        assert store.count() == 3

        store.clear()
        assert store.count() == 0


class TestBM25SearchResultDataclass:
    """Tests for BM25SearchResult dataclass."""

    def test_result_fields(self):
        """Test that result has all expected fields."""
        result = BM25SearchResult(
            chunk_id="test::chunk-0001",
            doc_id="test",
            text="sample text",
            score=1.5,
            metadata={"key": "value"},
        )
        assert result.chunk_id == "test::chunk-0001"
        assert result.doc_id == "test"
        assert result.text == "sample text"
        assert result.score == 1.5
        assert result.metadata == {"key": "value"}
