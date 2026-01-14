"""Unit tests for generation/dataset_summarizer.py."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest

from generation.base import LLMResponse
from generation.cost_tracker import CostConfig
from generation.dataset_summarizer import (
    DatasetSummarizer,
    DatasetSummary,
    DatasetSummaryError,
    SummaryConfig,
)
from storage.metadata_catalog import CatalogEntry, MetadataCatalog
from storage.sql_store import SQLStore, SQLStoreConfig


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: List[str] | None = None) -> None:
        self.responses = responses or []
        self.call_count = 0
        self.last_messages = None

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.last_messages = messages
        self.call_count += 1

        if self.call_count <= len(self.responses):
            content = self.responses[self.call_count - 1]
        else:
            content = json.dumps({
                "summary": "Default mock summary.",
                "column_descriptions": {"col1": "Description for col1"},
            })

        return LLMResponse(
            content=content,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )


@pytest.fixture
def sql_store(tmp_path: Path) -> SQLStore:
    """Create a SQLStore instance for testing."""
    db_path = tmp_path / "test.duckdb"
    config = SQLStoreConfig(db_path=db_path)
    store = SQLStore(config)

    # Create a test table
    store.execute("""
        CREATE TABLE test_table (
            id INTEGER,
            name VARCHAR,
            value DECIMAL(10, 2)
        )
    """)
    store.execute("""
        INSERT INTO test_table VALUES
        (1, 'Alice', 100.00),
        (2, 'Bob', 200.00),
        (3, 'Charlie', 300.00)
    """)

    yield store
    store.close()


@pytest.fixture
def catalog_with_entry(sql_store: SQLStore) -> MetadataCatalog:
    """Create a MetadataCatalog with a test entry."""
    catalog = MetadataCatalog(sql_store)
    catalog.init_catalog()

    entry = CatalogEntry(
        table_name="test_table",
        source_file="/path/to/test.csv",
        sheet_name=None,
        ingestion_timestamp=datetime.now(),
        row_count=3,
        column_count=3,
        content_hash="abc123",
        domain_labels=["test"],
        column_schema=[
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "VARCHAR"},
            {"name": "value", "type": "DECIMAL"},
        ],
    )
    catalog.register_table(entry)

    return catalog


class TestSummaryConfig:
    """Tests for SummaryConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SummaryConfig()

        assert config.max_tokens == 400
        assert config.temperature == 0.3
        assert config.sample_rows == 5
        assert config.max_columns_in_prompt == 20
        assert config.retry_count == 3

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SummaryConfig(
            max_tokens=800,
            temperature=0.5,
            sample_rows=10,
            max_columns_in_prompt=30,
            retry_count=5,
        )

        assert config.max_tokens == 800
        assert config.temperature == 0.5
        assert config.sample_rows == 10


class TestDatasetSummary:
    """Tests for DatasetSummary dataclass."""

    def test_creation(self) -> None:
        """Test basic summary creation."""
        summary = DatasetSummary(
            table_name="sales_2024",
            summary="Sales data for 2024 by region.",
            column_descriptions={"region": "Geographic region"},
            row_count=1000,
            column_count=5,
            source_file="/path/to/sales.csv",
        )

        assert summary.table_name == "sales_2024"
        assert summary.summary == "Sales data for 2024 by region."
        assert "region" in summary.column_descriptions

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        summary = DatasetSummary(
            table_name="sales_2024",
            summary="Sales data for 2024.",
            column_descriptions={"col1": "desc1"},
            token_usage={"input_tokens": 100, "output_tokens": 50},
            row_count=100,
            column_count=5,
        )

        data = summary.to_dict()

        assert data["table_name"] == "sales_2024"
        assert data["summary"] == "Sales data for 2024."
        assert data["token_usage"]["input_tokens"] == 100
        assert "generated_at" in data


class TestDatasetSummarizer:
    """Tests for DatasetSummarizer class."""

    def test_generate_summary_success(
        self,
        sql_store: SQLStore,
        catalog_with_entry: MetadataCatalog,
    ) -> None:
        """Test successful summary generation."""
        mock_response = json.dumps({
            "summary": "Test table with IDs, names, and values.",
            "column_descriptions": {
                "id": "Unique identifier",
                "name": "Person name",
                "value": "Monetary value",
            },
        })
        llm_client = MockLLMClient(responses=[mock_response])

        summarizer = DatasetSummarizer(
            llm_client=llm_client,
            sql_store=sql_store,
            catalog=catalog_with_entry,
        )

        summary = summarizer.generate_summary("test_table")

        assert summary.table_name == "test_table"
        assert "Test table" in summary.summary
        assert "id" in summary.column_descriptions
        assert summary.row_count == 3
        assert summary.token_usage["input_tokens"] == 100

    def test_generate_summary_table_not_found(
        self,
        sql_store: SQLStore,
        catalog_with_entry: MetadataCatalog,
    ) -> None:
        """Test error when table is not in catalog."""
        llm_client = MockLLMClient()

        summarizer = DatasetSummarizer(
            llm_client=llm_client,
            sql_store=sql_store,
            catalog=catalog_with_entry,
        )

        with pytest.raises(DatasetSummaryError) as exc_info:
            summarizer.generate_summary("nonexistent_table")

        assert "not found in catalog" in str(exc_info.value)

    def test_generate_summary_retries_on_failure(
        self,
        sql_store: SQLStore,
        catalog_with_entry: MetadataCatalog,
    ) -> None:
        """Test that summarizer retries on LLM failure."""
        # Create a mock that fails twice then succeeds
        class FailingMockClient:
            def __init__(self) -> None:
                self.call_count = 0

            def chat(self, **kwargs) -> LLMResponse:
                self.call_count += 1
                if self.call_count < 3:
                    raise Exception("Simulated failure")
                return LLMResponse(
                    content='{"summary": "Success after retries", "column_descriptions": {}}',
                    usage={"prompt_tokens": 100, "completion_tokens": 50},
                )

        llm_client = FailingMockClient()
        config = SummaryConfig(retry_count=3)

        summarizer = DatasetSummarizer(
            llm_client=llm_client,
            sql_store=sql_store,
            catalog=catalog_with_entry,
            config=config,
        )

        summary = summarizer.generate_summary("test_table")

        assert summary.summary == "Success after retries"
        assert llm_client.call_count == 3

    def test_generate_summary_all_retries_fail(
        self,
        sql_store: SQLStore,
        catalog_with_entry: MetadataCatalog,
    ) -> None:
        """Test error when all retries fail."""
        class AlwaysFailingClient:
            def chat(self, **kwargs) -> LLMResponse:
                raise Exception("Always fails")

        llm_client = AlwaysFailingClient()
        config = SummaryConfig(retry_count=2)

        summarizer = DatasetSummarizer(
            llm_client=llm_client,
            sql_store=sql_store,
            catalog=catalog_with_entry,
            config=config,
        )

        with pytest.raises(DatasetSummaryError) as exc_info:
            summarizer.generate_summary("test_table")

        assert "after 2 attempts" in str(exc_info.value)

    def test_parse_json_response(
        self,
        sql_store: SQLStore,
        catalog_with_entry: MetadataCatalog,
    ) -> None:
        """Test parsing JSON from LLM response."""
        mock_response = '{"summary": "Valid JSON", "column_descriptions": {"a": "b"}}'
        llm_client = MockLLMClient(responses=[mock_response])

        summarizer = DatasetSummarizer(
            llm_client=llm_client,
            sql_store=sql_store,
            catalog=catalog_with_entry,
        )

        summary = summarizer.generate_summary("test_table")

        assert summary.summary == "Valid JSON"
        assert summary.column_descriptions == {"a": "b"}

    def test_parse_json_embedded_in_text(
        self,
        sql_store: SQLStore,
        catalog_with_entry: MetadataCatalog,
    ) -> None:
        """Test extracting JSON embedded in response text."""
        mock_response = """Here is the summary:
        {"summary": "Embedded JSON", "column_descriptions": {}}
        Hope this helps!"""
        llm_client = MockLLMClient(responses=[mock_response])

        summarizer = DatasetSummarizer(
            llm_client=llm_client,
            sql_store=sql_store,
            catalog=catalog_with_entry,
        )

        summary = summarizer.generate_summary("test_table")

        assert summary.summary == "Embedded JSON"

    def test_fallback_to_plain_text(
        self,
        sql_store: SQLStore,
        catalog_with_entry: MetadataCatalog,
    ) -> None:
        """Test fallback when response is not valid JSON."""
        mock_response = "This is just plain text without JSON."
        llm_client = MockLLMClient(responses=[mock_response])

        summarizer = DatasetSummarizer(
            llm_client=llm_client,
            sql_store=sql_store,
            catalog=catalog_with_entry,
        )

        summary = summarizer.generate_summary("test_table")

        assert "plain text" in summary.summary
        assert summary.column_descriptions == {}

    def test_cost_tracking(
        self,
        sql_store: SQLStore,
        catalog_with_entry: MetadataCatalog,
    ) -> None:
        """Test that token usage is tracked."""
        llm_client = MockLLMClient()

        summarizer = DatasetSummarizer(
            llm_client=llm_client,
            sql_store=sql_store,
            catalog=catalog_with_entry,
        )

        summarizer.generate_summary("test_table")

        report = summarizer.get_cost_report()

        assert report.total_requests == 1
        assert report.total_input_tokens == 100
        assert report.total_output_tokens == 50

    def test_budget_limit_stops_batch(
        self,
        sql_store: SQLStore,
        catalog_with_entry: MetadataCatalog,
    ) -> None:
        """Test that batch processing stops at budget limit."""
        llm_client = MockLLMClient()
        cost_config = CostConfig(max_total_tokens_per_run=100)  # Very low limit

        summarizer = DatasetSummarizer(
            llm_client=llm_client,
            sql_store=sql_store,
            catalog=catalog_with_entry,
            cost_config=cost_config,
        )

        # First call uses 150 tokens, exceeding 100 limit
        summaries = summarizer.generate_summaries_batch(["test_table", "test_table"])

        # Should only process one (budget exhausted after first)
        assert len(summaries) == 1

    def test_estimate_tokens(
        self,
        sql_store: SQLStore,
        catalog_with_entry: MetadataCatalog,
    ) -> None:
        """Test token estimation for a table."""
        llm_client = MockLLMClient()

        summarizer = DatasetSummarizer(
            llm_client=llm_client,
            sql_store=sql_store,
            catalog=catalog_with_entry,
        )

        estimate = summarizer.estimate_tokens("test_table")

        # Should return a reasonable positive estimate
        assert estimate > 0
        assert estimate < 10000  # Not unreasonably large

    def test_estimate_tokens_nonexistent_table(
        self,
        sql_store: SQLStore,
        catalog_with_entry: MetadataCatalog,
    ) -> None:
        """Test token estimation for nonexistent table returns 0."""
        llm_client = MockLLMClient()

        summarizer = DatasetSummarizer(
            llm_client=llm_client,
            sql_store=sql_store,
            catalog=catalog_with_entry,
        )

        estimate = summarizer.estimate_tokens("nonexistent")

        assert estimate == 0
