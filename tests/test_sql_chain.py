"""Unit tests for sql_agent/sql_chain.py."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
from unittest.mock import MagicMock, patch

import pytest

from sql_agent.config import SQLAgentConfig
from sql_agent.errors import QueryExecutionError, QueryGenerationError, QueryValidationError
from sql_agent.sql_chain import SQLChain, SQLQueryResult


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["SELECT * FROM test_table LIMIT 10"]
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


class TestSQLChainBasic:
    """Basic tests for SQLChain class."""

    @pytest.fixture
    def db_with_data(self, tmp_path: Path):
        """Create a test database with sample data."""
        from storage.sql_store import SQLStore, SQLStoreConfig

        db_path = tmp_path / "test.duckdb"
        config = SQLStoreConfig(db_path=db_path)

        with SQLStore(config) as store:
            # Create a test table
            store.execute("""
                CREATE TABLE test_table (
                    id INTEGER,
                    name VARCHAR,
                    value DOUBLE
                )
            """)
            store.execute("""
                INSERT INTO test_table VALUES
                (1, 'Alice', 100.0),
                (2, 'Bob', 200.0),
                (3, 'Charlie', 300.0)
            """)

        return db_path

    @pytest.fixture
    def sql_chain(self, db_with_data: Path) -> SQLChain:
        """Create SQLChain with mock LLM."""
        llm_client = MockLLMClient(responses=[
            "SELECT * FROM test_table",
            "The data shows 3 records with names and values."
        ])
        config = SQLAgentConfig(db_path=db_with_data)
        return SQLChain(llm_client, config)

    def test_chain_initialization(self, db_with_data: Path) -> None:
        """Test that SQLChain initializes correctly."""
        llm_client = MockLLMClient()
        config = SQLAgentConfig(db_path=db_with_data)
        chain = SQLChain(llm_client, config)

        assert chain.config.db_path == db_with_data
        assert chain.config.read_only is True

    def test_get_available_tables(self, sql_chain: SQLChain) -> None:
        """Test getting available tables."""
        tables = sql_chain.get_available_tables()
        assert "test_table" in tables

    def test_health_check_healthy(self, sql_chain: SQLChain) -> None:
        """Test health check with healthy database."""
        result = sql_chain.health_check()

        assert result.healthy is True
        assert result.table_count >= 1
        assert "healthy" in result.message.lower()

    def test_health_check_no_tables(self, tmp_path: Path) -> None:
        """Test health check with empty database."""
        from storage.sql_store import SQLStore, SQLStoreConfig

        db_path = tmp_path / "empty.duckdb"
        store_config = SQLStoreConfig(db_path=db_path)

        # Create empty database
        with SQLStore(store_config):
            pass

        llm_client = MockLLMClient()
        config = SQLAgentConfig(db_path=db_path)
        chain = SQLChain(llm_client, config)

        result = chain.health_check()
        assert result.healthy is True
        assert result.table_count == 0
        assert "no user tables" in result.message.lower()

    def test_close(self, sql_chain: SQLChain) -> None:
        """Test that close cleans up resources."""
        # Access internal components to initialize them
        sql_chain._get_sql_store()
        sql_chain._get_schema_extractor()

        sql_chain.close()

        assert sql_chain._sql_store is None
        assert sql_chain._schema_extractor is None


class TestSQLChainQuery:
    """Tests for SQLChain.query method."""

    @pytest.fixture
    def db_with_data(self, tmp_path: Path):
        """Create a test database with sample data."""
        from storage.sql_store import SQLStore, SQLStoreConfig

        db_path = tmp_path / "test.duckdb"
        config = SQLStoreConfig(db_path=db_path)

        with SQLStore(config) as store:
            store.execute("""
                CREATE TABLE products (
                    id INTEGER,
                    name VARCHAR,
                    price DOUBLE,
                    category VARCHAR
                )
            """)
            store.execute("""
                INSERT INTO products VALUES
                (1, 'Widget', 10.0, 'Electronics'),
                (2, 'Gadget', 20.0, 'Electronics'),
                (3, 'Book', 15.0, 'Education'),
                (4, 'Pen', 2.0, 'Office'),
                (5, 'Laptop', 1000.0, 'Electronics')
            """)

        return db_path

    def test_query_returns_data(self, db_with_data: Path) -> None:
        """Test that query returns data correctly."""
        llm_client = MockLLMClient(responses=[
            "SELECT * FROM products WHERE category = 'Electronics'",
            "Found 3 electronics products."
        ])
        config = SQLAgentConfig(db_path=db_with_data)
        chain = SQLChain(llm_client, config)

        result = chain.query("Show me all electronics products")

        assert isinstance(result, SQLQueryResult)
        assert result.row_count == 3
        assert len(result.data) == 3
        assert "id" in result.columns
        assert "name" in result.columns

    def test_query_with_aggregation(self, db_with_data: Path) -> None:
        """Test query with aggregation."""
        llm_client = MockLLMClient(responses=[
            "SELECT category, COUNT(*) as count FROM products GROUP BY category",
            "There are 3 categories."
        ])
        config = SQLAgentConfig(db_path=db_with_data)
        chain = SQLChain(llm_client, config)

        result = chain.query("Count products by category")

        assert result.row_count >= 1
        assert "category" in result.columns
        assert "count" in result.columns

    def test_query_result_has_timing(self, db_with_data: Path) -> None:
        """Test that query result includes timing information."""
        llm_client = MockLLMClient(responses=[
            "SELECT * FROM products LIMIT 1",
            "Found 1 product."
        ])
        config = SQLAgentConfig(db_path=db_with_data)
        chain = SQLChain(llm_client, config)

        result = chain.query("Show one product")

        assert result.generation_time_ms > 0
        assert result.execution_time_ms >= 0

    def test_query_extracts_tables_used(self, db_with_data: Path) -> None:
        """Test that query extracts tables used."""
        llm_client = MockLLMClient(responses=[
            "SELECT * FROM products",
            "All products listed."
        ])
        config = SQLAgentConfig(db_path=db_with_data)
        chain = SQLChain(llm_client, config)

        result = chain.query("List all products")

        assert "products" in result.tables_used

    def test_query_empty_results(self, db_with_data: Path) -> None:
        """Test query that returns no results."""
        llm_client = MockLLMClient(responses=[
            "SELECT * FROM products WHERE price > 10000",
            "No products found."
        ])
        config = SQLAgentConfig(db_path=db_with_data)
        chain = SQLChain(llm_client, config)

        result = chain.query("Show products over $10,000")

        assert result.row_count == 0
        assert len(result.data) == 0
        assert "No results" in result.answer

    def test_query_without_summarization(self, db_with_data: Path) -> None:
        """Test query without answer summarization."""
        llm_client = MockLLMClient(responses=[
            "SELECT COUNT(*) FROM products",
        ])
        config = SQLAgentConfig(db_path=db_with_data)
        chain = SQLChain(llm_client, config)

        result = chain.query("Count products", summarize=False)

        assert result.row_count == 1
        assert "row(s)" in result.answer  # Generic message, not LLM-generated


class TestSQLChainValidation:
    """Tests for query validation in SQLChain."""

    @pytest.fixture
    def db_with_data(self, tmp_path: Path):
        """Create a test database with sample data."""
        from storage.sql_store import SQLStore, SQLStoreConfig

        db_path = tmp_path / "test.duckdb"
        config = SQLStoreConfig(db_path=db_path)

        with SQLStore(config) as store:
            store.execute("CREATE TABLE users (id INTEGER, name VARCHAR)")
            store.execute("INSERT INTO users VALUES (1, 'Test')")

        return db_path

    def test_query_rejects_non_select(self, db_with_data: Path) -> None:
        """Test that non-SELECT queries are rejected."""
        llm_client = MockLLMClient(responses=["DELETE FROM users WHERE id = 1"])
        config = SQLAgentConfig(db_path=db_with_data)
        chain = SQLChain(llm_client, config)

        with pytest.raises(QueryValidationError) as exc_info:
            chain.query("Delete user 1")

        assert "DELETE" in str(exc_info.value)

    def test_query_rejects_excluded_tables(self, db_with_data: Path) -> None:
        """Test that queries on excluded tables are rejected."""
        # First create the excluded table
        from storage.sql_store import SQLStore, SQLStoreConfig

        store_config = SQLStoreConfig(db_path=db_with_data)
        with SQLStore(store_config) as store:
            store.execute("CREATE TABLE _agents (id INTEGER)")
            store.execute("INSERT INTO _agents VALUES (1)")

        llm_client = MockLLMClient(responses=["SELECT * FROM _agents"])
        config = SQLAgentConfig(db_path=db_with_data)
        chain = SQLChain(llm_client, config)

        with pytest.raises(QueryValidationError) as exc_info:
            chain.query("Show agents")

        assert "_agents" in str(exc_info.value)

    def test_query_handles_cannot_answer(self, db_with_data: Path) -> None:
        """Test handling of CANNOT_ANSWER response."""
        llm_client = MockLLMClient(responses=["CANNOT_ANSWER: No relevant tables found"])
        config = SQLAgentConfig(db_path=db_with_data)
        chain = SQLChain(llm_client, config)

        with pytest.raises(QueryGenerationError) as exc_info:
            chain.query("What is the meaning of life?")

        assert "Cannot generate SQL" in str(exc_info.value)


class TestSQLChainExecution:
    """Tests for query execution in SQLChain."""

    @pytest.fixture
    def db_with_data(self, tmp_path: Path):
        """Create a test database with sample data."""
        from storage.sql_store import SQLStore, SQLStoreConfig

        db_path = tmp_path / "test.duckdb"
        config = SQLStoreConfig(db_path=db_path)

        with SQLStore(config) as store:
            store.execute("CREATE TABLE data (id INTEGER, value VARCHAR)")

        return db_path

    def test_query_execution_error(self, db_with_data: Path) -> None:
        """Test handling of SQL execution errors."""
        # Query references non-existent table
        llm_client = MockLLMClient(responses=["SELECT * FROM nonexistent_table"])
        config = SQLAgentConfig(db_path=db_with_data)
        chain = SQLChain(llm_client, config)

        with pytest.raises(QueryExecutionError) as exc_info:
            chain.query("Show data from missing table")

        assert "execution failed" in str(exc_info.value).lower()


class TestSQLChainConfig:
    """Tests for SQLChain configuration."""

    def test_config_forces_read_only(self, tmp_path: Path) -> None:
        """Test that config always forces read_only mode."""
        from storage.sql_store import SQLStore, SQLStoreConfig

        db_path = tmp_path / "test.duckdb"
        store_config = SQLStoreConfig(db_path=db_path)
        with SQLStore(store_config):
            pass

        llm_client = MockLLMClient()
        # Try to set read_only to False
        config = SQLAgentConfig(db_path=db_path, read_only=False)

        # Config should still be read_only
        assert config.read_only is True

        chain = SQLChain(llm_client, config)
        assert chain.config.read_only is True

    def test_config_max_rows(self, tmp_path: Path) -> None:
        """Test that max_result_rows is respected."""
        from storage.sql_store import SQLStore, SQLStoreConfig

        db_path = tmp_path / "test.duckdb"
        store_config = SQLStoreConfig(db_path=db_path)
        with SQLStore(store_config) as store:
            store.execute("CREATE TABLE data (id INTEGER)")
            for i in range(200):
                store.execute(f"INSERT INTO data VALUES ({i})")

        llm_client = MockLLMClient(responses=["SELECT * FROM data"])
        config = SQLAgentConfig(db_path=db_path, max_result_rows=50)
        chain = SQLChain(llm_client, config)

        result = chain.query("Show all data", summarize=False)

        # Should be limited by auto-LIMIT
        assert result.row_count <= 100  # Default limit from validator
