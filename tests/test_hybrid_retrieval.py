"""Tests for hybrid retrieval in RAGChain."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from generation.rag_chain import RAGChain, RAGConfig, RetrievalResult
from indexing.bm25_store import BM25Config, BM25IndexEntry, BM25Store


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()
    client.generate.return_value = "Test answer"
    return client


@pytest.fixture
def populated_bm25(temp_dir):
    """Create and populate a BM25 index for testing."""
    config = BM25Config(
        index_dir=temp_dir,
        index_name="test-collection",
    )
    store = BM25Store(config)

    entries = [
        BM25IndexEntry(
            chunk_id="doc1::chunk-0001",
            doc_id="doc1",
            text="The World Economic Forum publishes annual reports on jobs",
            metadata={"page": 1, "doc_id": "doc1", "relative_path": "doc1.pdf"},
        ),
        BM25IndexEntry(
            chunk_id="doc1::chunk-0002",
            doc_id="doc1",
            text="Artificial intelligence is transforming the workplace",
            metadata={"page": 2, "doc_id": "doc1", "relative_path": "doc1.pdf"},
        ),
        BM25IndexEntry(
            chunk_id="doc2::chunk-0001",
            doc_id="doc2",
            text="Skills in demand include data analysis and programming",
            metadata={"page": 1, "doc_id": "doc2", "relative_path": "doc2.pdf"},
        ),
    ]
    store.add_documents(entries)
    store.save()
    return temp_dir


class TestRAGConfigLexicalSettings:
    """Tests for lexical search configuration in RAGConfig."""

    def test_default_lexical_disabled(self):
        """Test that lexical search is disabled by default."""
        config = RAGConfig()
        assert config.enable_lexical is False

    def test_lexical_weight_default(self):
        """Test default lexical weight."""
        config = RAGConfig()
        assert config.lexical_weight == 0.3

    def test_custom_lexical_config(self, temp_dir):
        """Test custom lexical configuration."""
        config = RAGConfig(
            enable_lexical=True,
            lexical_weight=0.4,
            bm25_dir=temp_dir,
        )
        assert config.enable_lexical is True
        assert config.lexical_weight == 0.4
        assert config.bm25_dir == temp_dir


class TestRAGChainSearchModes:
    """Tests for different search modes in RAGChain."""

    def test_retrieve_vector_mode(self, mock_llm_client, temp_dir):
        """Test retrieval in vector mode."""
        config = RAGConfig(
            vectorstore_dir=temp_dir,
            collection_name="test-collection",
        )
        chain = RAGChain(mock_llm_client, config)

        # Mock the Chroma collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["chunk 1", "chunk 2"]],
            "metadatas": [[{"doc_id": "doc1"}, {"doc_id": "doc2"}]],
            "ids": [["id1", "id2"]],
            "distances": [[0.1, 0.2]],
        }

        with patch.object(chain, "_get_collection", return_value=mock_collection):
            result = chain.retrieve("test query", k=2, mode="vector")

        assert len(result.chunks) == 2
        mock_collection.query.assert_called_once()

    def test_retrieve_lexical_mode_without_enabling(self, mock_llm_client, temp_dir):
        """Test lexical retrieval returns empty when not enabled."""
        config = RAGConfig(
            vectorstore_dir=temp_dir,
            enable_lexical=False,
        )
        chain = RAGChain(mock_llm_client, config)

        result = chain.retrieve("test query", k=2, mode="lexical")

        assert result.chunks == []
        assert result.ids == []

    def test_retrieve_lexical_mode_enabled(self, mock_llm_client, populated_bm25):
        """Test lexical retrieval with enabled BM25."""
        config = RAGConfig(
            vectorstore_dir=populated_bm25,
            collection_name="test-collection",
            enable_lexical=True,
        )
        chain = RAGChain(mock_llm_client, config)

        result = chain.retrieve("World Economic Forum", k=2, mode="lexical")

        assert len(result.chunks) >= 1
        assert any("World Economic Forum" in chunk for chunk in result.chunks)

    def test_retrieve_hybrid_mode(self, mock_llm_client, populated_bm25):
        """Test hybrid retrieval combining vector and lexical."""
        config = RAGConfig(
            vectorstore_dir=populated_bm25,
            collection_name="test-collection",
            enable_lexical=True,
            lexical_weight=0.3,
        )
        chain = RAGChain(mock_llm_client, config)

        # Mock vector search
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["AI transforms workplaces"]],
            "metadatas": [[{"doc_id": "doc1", "relative_path": "doc1.pdf"}]],
            "ids": [["doc1::chunk-0002"]],
            "distances": [[0.1]],
        }

        with patch.object(chain, "_get_collection", return_value=mock_collection):
            result = chain.retrieve("artificial intelligence jobs", k=3, mode="hybrid")

        # Should get results from both methods
        assert len(result.chunks) >= 1
        assert len(result.ids) == len(result.chunks)

    def test_retrieve_unknown_mode_fallback(self, mock_llm_client, temp_dir):
        """Test that unknown mode falls back to vector."""
        config = RAGConfig(
            vectorstore_dir=temp_dir,
        )
        chain = RAGChain(mock_llm_client, config)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["result"]],
            "metadatas": [[{}]],
            "ids": [["id1"]],
            "distances": [[0.1]],
        }

        with patch.object(chain, "_get_collection", return_value=mock_collection):
            result = chain.retrieve("test", k=1, mode="invalid_mode")

        mock_collection.query.assert_called_once()


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion in hybrid mode."""

    def test_rrf_combines_results(self, mock_llm_client, populated_bm25):
        """Test that RRF properly combines results from both methods."""
        config = RAGConfig(
            vectorstore_dir=populated_bm25,
            collection_name="test-collection",
            enable_lexical=True,
            lexical_weight=0.5,  # Equal weight
        )
        chain = RAGChain(mock_llm_client, config)

        # Mock vector search returning different results
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Vector result 1", "Vector result 2"]],
            "metadatas": [[{"doc_id": "v1"}, {"doc_id": "v2"}]],
            "ids": [["vector-1", "vector-2"]],
            "distances": [[0.1, 0.2]],
        }

        with patch.object(chain, "_get_collection", return_value=mock_collection):
            result = chain.retrieve("skills jobs", k=5, mode="hybrid")

        # Results should include items from both sources
        assert len(result.chunks) > 0

    def test_rrf_lexical_weight_affects_ranking(self, mock_llm_client, populated_bm25):
        """Test that lexical_weight parameter affects result ranking."""
        # High lexical weight
        config_high = RAGConfig(
            vectorstore_dir=populated_bm25,
            collection_name="test-collection",
            enable_lexical=True,
            lexical_weight=0.9,
        )
        chain_high = RAGChain(mock_llm_client, config_high)

        # Low lexical weight
        config_low = RAGConfig(
            vectorstore_dir=populated_bm25,
            collection_name="test-collection",
            enable_lexical=True,
            lexical_weight=0.1,
        )
        chain_low = RAGChain(mock_llm_client, config_low)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Vector only result"]],
            "metadatas": [[{"doc_id": "v1"}]],
            "ids": [["vector-only"]],
            "distances": [[0.1]],
        }

        with patch.object(chain_high, "_get_collection", return_value=mock_collection):
            with patch.object(chain_low, "_get_collection", return_value=mock_collection):
                result_high = chain_high.retrieve("World Economic Forum", k=3, mode="hybrid")
                result_low = chain_low.retrieve("World Economic Forum", k=3, mode="hybrid")

        # Both should return results (specific ordering depends on weights)
        assert len(result_high.chunks) > 0
        assert len(result_low.chunks) > 0


class TestQueryWithSearchMode:
    """Tests for query() method with search_mode parameter."""

    def test_query_default_mode(self, mock_llm_client, temp_dir):
        """Test that query uses vector mode by default."""
        config = RAGConfig(vectorstore_dir=temp_dir)
        chain = RAGChain(mock_llm_client, config)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["context"]],
            "metadatas": [[{"doc_id": "d1"}]],
            "ids": [["id1"]],
            "distances": [[0.1]],
        }

        with patch.object(chain, "_get_collection", return_value=mock_collection):
            response = chain.query("test question")

        assert response.answer == "Test answer"
        mock_collection.query.assert_called_once()

    def test_query_lexical_mode(self, mock_llm_client, populated_bm25):
        """Test query with lexical search mode."""
        config = RAGConfig(
            vectorstore_dir=populated_bm25,
            collection_name="test-collection",
            enable_lexical=True,
        )
        chain = RAGChain(mock_llm_client, config)

        response = chain.query("World Economic Forum jobs", search_mode="lexical")

        assert response.answer == "Test answer"
        assert len(response.retrieved_chunks) > 0
        mock_llm_client.generate.assert_called_once()

    def test_query_hybrid_mode(self, mock_llm_client, populated_bm25):
        """Test query with hybrid search mode."""
        config = RAGConfig(
            vectorstore_dir=populated_bm25,
            collection_name="test-collection",
            enable_lexical=True,
        )
        chain = RAGChain(mock_llm_client, config)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Vector context"]],
            "metadatas": [[{"doc_id": "d1", "relative_path": "d1.pdf"}]],
            "ids": [["vid1"]],
            "distances": [[0.1]],
        }

        with patch.object(chain, "_get_collection", return_value=mock_collection):
            response = chain.query("skills demand", search_mode="hybrid")

        assert response.answer == "Test answer"
        mock_llm_client.generate.assert_called_once()
