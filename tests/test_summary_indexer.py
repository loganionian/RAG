"""Unit tests for indexing/summary_indexer.py."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from generation.dataset_summarizer import DatasetSummary
from indexing.summary_indexer import DatasetSearchResult, SummaryIndexConfig, SummaryIndexer


@pytest.fixture
def chroma_dir(tmp_path: Path) -> Path:
    """Create a temporary Chroma directory."""
    chroma_path = tmp_path / "chroma"
    chroma_path.mkdir()
    return chroma_path


@pytest.fixture
def index_config(chroma_dir: Path) -> SummaryIndexConfig:
    """Create a SummaryIndexConfig for testing."""
    return SummaryIndexConfig(
        chroma_dir=chroma_dir,
        collection_name="test-summaries",
    )


@pytest.fixture
def indexer(index_config: SummaryIndexConfig) -> SummaryIndexer:
    """Create a SummaryIndexer for testing."""
    return SummaryIndexer(index_config)


@pytest.fixture
def sample_summary() -> DatasetSummary:
    """Create a sample DatasetSummary for testing."""
    return DatasetSummary(
        table_name="sales_2024",
        summary="Sales data for fiscal year 2024, including revenue by region and product category.",
        column_descriptions={
            "region": "Geographic sales region",
            "product_id": "Unique product identifier",
            "revenue": "Total revenue in USD",
            "quantity": "Number of units sold",
        },
        sample_values={
            "region": ["North", "South", "East"],
            "revenue": [1000.0, 2000.0, 1500.0],
        },
        generated_at=datetime.now(),
        token_usage={"input_tokens": 150, "output_tokens": 100},
        source_file="/data/sales_2024.csv",
        domain_labels=["sales", "finance"],
        row_count=10000,
        column_count=4,
    )


class TestSummaryIndexConfig:
    """Tests for SummaryIndexConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default configuration values."""
        config = SummaryIndexConfig(chroma_dir=tmp_path)

        assert config.collection_name == "dataset-summaries"
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_custom_values(self, tmp_path: Path) -> None:
        """Test custom configuration values."""
        config = SummaryIndexConfig(
            chroma_dir=tmp_path,
            collection_name="custom-collection",
            embedding_model="custom-model",
        )

        assert config.collection_name == "custom-collection"
        assert config.embedding_model == "custom-model"


class TestDatasetSearchResult:
    """Tests for DatasetSearchResult dataclass."""

    def test_creation(self) -> None:
        """Test basic result creation."""
        result = DatasetSearchResult(
            table_name="my_table",
            summary="A summary",
            source_file="/path/to/file.csv",
            relevance_score=0.5,
            row_count=100,
            column_count=5,
        )

        assert result.table_name == "my_table"
        assert result.relevance_score == 0.5

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        result = DatasetSearchResult(
            table_name="my_table",
            summary="A summary",
            source_file="/path/to/file.csv",
            relevance_score=0.5,
            row_count=100,
            column_count=5,
            domain_labels=["test"],
        )

        data = result.to_dict()

        assert data["table_name"] == "my_table"
        assert data["domain_labels"] == ["test"]


class TestSummaryIndexer:
    """Tests for SummaryIndexer class."""

    def test_index_summary(self, indexer: SummaryIndexer, sample_summary: DatasetSummary) -> None:
        """Test indexing a single summary."""
        doc_id = indexer.index_summary(sample_summary)

        assert doc_id == "dataset::sales_2024"
        assert indexer.count() == 1

    def test_index_summaries_batch(self, indexer: SummaryIndexer) -> None:
        """Test batch indexing multiple summaries."""
        summaries = [
            DatasetSummary(
                table_name=f"table_{i}",
                summary=f"Summary for table {i}",
                source_file=f"/data/table_{i}.csv",
                row_count=100 * (i + 1),
                column_count=5,
            )
            for i in range(3)
        ]

        count = indexer.index_summaries_batch(summaries)

        assert count == 3
        assert indexer.count() == 3

    def test_index_empty_batch(self, indexer: SummaryIndexer) -> None:
        """Test batch indexing with empty list."""
        count = indexer.index_summaries_batch([])

        assert count == 0
        assert indexer.count() == 0

    def test_search_datasets(self, indexer: SummaryIndexer, sample_summary: DatasetSummary) -> None:
        """Test semantic search for datasets."""
        indexer.index_summary(sample_summary)

        results = indexer.search_datasets("revenue by region", k=5)

        assert len(results) == 1
        assert results[0].table_name == "sales_2024"
        assert "revenue" in results[0].summary.lower() or "sales" in results[0].summary.lower()

    def test_search_empty_index(self, indexer: SummaryIndexer) -> None:
        """Test search on empty index."""
        results = indexer.search_datasets("any query")

        assert len(results) == 0

    def test_search_multiple_results(self, indexer: SummaryIndexer) -> None:
        """Test search returns multiple ranked results."""
        summaries = [
            DatasetSummary(
                table_name="sales_data",
                summary="Sales revenue data by region and quarter.",
                source_file="/data/sales.csv",
                row_count=1000,
                column_count=5,
            ),
            DatasetSummary(
                table_name="inventory",
                summary="Product inventory levels and stock counts.",
                source_file="/data/inventory.csv",
                row_count=500,
                column_count=4,
            ),
            DatasetSummary(
                table_name="customers",
                summary="Customer information including purchase history.",
                source_file="/data/customers.csv",
                row_count=2000,
                column_count=8,
            ),
        ]
        indexer.index_summaries_batch(summaries)

        results = indexer.search_datasets("sales revenue", k=3)

        assert len(results) >= 1
        # Sales data should be most relevant
        assert results[0].table_name == "sales_data"

    def test_search_with_k_limit(self, indexer: SummaryIndexer) -> None:
        """Test search respects k limit."""
        summaries = [
            DatasetSummary(
                table_name=f"table_{i}",
                summary=f"Data table number {i} with various information.",
                source_file=f"/data/table_{i}.csv",
                row_count=100,
                column_count=5,
            )
            for i in range(10)
        ]
        indexer.index_summaries_batch(summaries)

        results = indexer.search_datasets("data table", k=3)

        assert len(results) == 3

    def test_delete_summary(self, indexer: SummaryIndexer, sample_summary: DatasetSummary) -> None:
        """Test deleting a summary from the index."""
        indexer.index_summary(sample_summary)
        assert indexer.count() == 1

        result = indexer.delete_summary("sales_2024")

        assert result is True
        assert indexer.count() == 0

    def test_delete_nonexistent_summary(self, indexer: SummaryIndexer) -> None:
        """Test deleting a summary that doesn't exist."""
        result = indexer.delete_summary("nonexistent")

        assert result is False

    def test_get_summary(self, indexer: SummaryIndexer, sample_summary: DatasetSummary) -> None:
        """Test retrieving a specific summary by table name."""
        indexer.index_summary(sample_summary)

        result = indexer.get_summary("sales_2024")

        assert result is not None
        assert result.table_name == "sales_2024"
        assert result.source_file == "/data/sales_2024.csv"
        assert result.row_count == 10000

    def test_get_summary_nonexistent(self, indexer: SummaryIndexer) -> None:
        """Test retrieving a nonexistent summary."""
        result = indexer.get_summary("nonexistent")

        assert result is None

    def test_list_all(self, indexer: SummaryIndexer) -> None:
        """Test listing all indexed summaries."""
        summaries = [
            DatasetSummary(
                table_name=f"table_{i}",
                summary=f"Summary {i}",
                source_file=f"/data/table_{i}.csv",
                row_count=100,
                column_count=5,
            )
            for i in range(5)
        ]
        indexer.index_summaries_batch(summaries)

        all_results = indexer.list_all()

        assert len(all_results) == 5
        table_names = [r.table_name for r in all_results]
        for i in range(5):
            assert f"table_{i}" in table_names

    def test_list_all_empty(self, indexer: SummaryIndexer) -> None:
        """Test listing from empty index."""
        all_results = indexer.list_all()

        assert len(all_results) == 0

    def test_upsert_updates_existing(
        self, indexer: SummaryIndexer, sample_summary: DatasetSummary
    ) -> None:
        """Test that indexing same table updates rather than duplicates."""
        indexer.index_summary(sample_summary)
        assert indexer.count() == 1

        # Update with new summary
        updated_summary = DatasetSummary(
            table_name="sales_2024",
            summary="Updated summary with new information.",
            source_file="/data/sales_2024_v2.csv",
            row_count=15000,
            column_count=6,
        )
        indexer.index_summary(updated_summary)

        # Should still be 1, not 2
        assert indexer.count() == 1

        # Should have updated content
        result = indexer.get_summary("sales_2024")
        assert result.row_count == 15000
        assert "Updated" in result.summary

    def test_column_descriptions_stored_and_retrieved(
        self, indexer: SummaryIndexer, sample_summary: DatasetSummary
    ) -> None:
        """Test that column descriptions are stored and can be retrieved."""
        indexer.index_summary(sample_summary)

        result = indexer.get_summary("sales_2024")

        assert result is not None
        assert "region" in result.column_descriptions
        assert result.column_descriptions["region"] == "Geographic sales region"

    def test_domain_labels_stored_and_retrieved(
        self, indexer: SummaryIndexer, sample_summary: DatasetSummary
    ) -> None:
        """Test that domain labels are stored and can be retrieved."""
        indexer.index_summary(sample_summary)

        result = indexer.get_summary("sales_2024")

        assert result is not None
        assert "sales" in result.domain_labels
        assert "finance" in result.domain_labels
