"""Unit tests for api/routes/sql_query.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class MockSQLQueryResult:
    """Mock result for SQLChain.query."""

    def __init__(
        self,
        answer: str = "Test answer",
        data: list = None,
        columns: list = None,
        row_count: int = 0,
        generated_sql: str = "SELECT 1",
        tables_used: list = None,
        generation_time_ms: float = 10.0,
        execution_time_ms: float = 5.0,
        summarization_time_ms: float = 20.0,
    ):
        self.answer = answer
        self.data = data or []
        self.columns = columns or []
        self.row_count = row_count
        self.generated_sql = generated_sql
        self.tables_used = tables_used or []
        self.generation_time_ms = generation_time_ms
        self.execution_time_ms = execution_time_ms
        self.summarization_time_ms = summarization_time_ms


class TestSQLQueryEndpoint:
    """Tests for POST /api/sql-query endpoint."""

    @pytest.fixture
    def mock_sql_chain(self):
        """Create a mock SQL chain."""
        chain = MagicMock()
        chain.query.return_value = MockSQLQueryResult(
            answer="Found 3 products",
            data=[
                {"id": 1, "name": "Widget"},
                {"id": 2, "name": "Gadget"},
                {"id": 3, "name": "Gizmo"},
            ],
            columns=["id", "name"],
            row_count=3,
            generated_sql="SELECT id, name FROM products",
            tables_used=["products"],
        )
        chain.get_available_tables.return_value = ["products", "orders", "users"]
        return chain

    @pytest.fixture
    def client(self, mock_sql_chain):
        """Create a test client with mocked SQL chain."""
        from api.main import app
        from api.routes.sql_query import set_sql_chain

        set_sql_chain(mock_sql_chain)
        return TestClient(app)

    def test_sql_query_success(self, client: TestClient, mock_sql_chain) -> None:
        """Test successful SQL query."""
        response = client.post(
            "/api/sql-query",
            json={"question": "List all products"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "Found 3 products"
        assert data["row_count"] == 3
        assert len(data["data"]) == 3
        assert data["columns"] == ["id", "name"]
        assert "generated_sql" not in data or data["generated_sql"] is None

        # Check metadata
        assert "metadata" in data
        assert data["metadata"]["generation_time_ms"] > 0
        assert data["metadata"]["execution_time_ms"] >= 0
        assert data["metadata"]["tables_used"] == ["products"]

    def test_sql_query_with_show_sql(self, client: TestClient, mock_sql_chain) -> None:
        """Test SQL query with show_sql=true."""
        response = client.post(
            "/api/sql-query",
            json={"question": "List all products", "show_sql": True},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["generated_sql"] == "SELECT id, name FROM products"

    def test_sql_query_with_max_rows(self, client: TestClient, mock_sql_chain) -> None:
        """Test SQL query with custom max_rows."""
        response = client.post(
            "/api/sql-query",
            json={"question": "List all products", "max_rows": 50},
        )

        assert response.status_code == 200
        mock_sql_chain.query.assert_called_once()
        call_kwargs = mock_sql_chain.query.call_args
        assert call_kwargs.kwargs["max_rows"] == 50

    def test_sql_query_empty_question_rejected(self, client: TestClient, mock_sql_chain) -> None:
        """Test that empty questions are rejected."""
        from sql_agent.errors import QueryGenerationError

        mock_sql_chain.query.side_effect = QueryGenerationError("Empty question")

        response = client.post(
            "/api/sql-query",
            json={"question": ""},
        )

        # Should fail during generation
        assert response.status_code == 400

    def test_sql_query_validation_error(self, client: TestClient, mock_sql_chain) -> None:
        """Test handling of query validation errors."""
        from sql_agent.errors import QueryValidationError

        mock_sql_chain.query.side_effect = QueryValidationError("DELETE not allowed")

        response = client.post(
            "/api/sql-query",
            json={"question": "Delete all data"},
        )

        assert response.status_code == 400
        assert "DELETE" in response.json()["detail"]

    def test_sql_query_generation_error(self, client: TestClient, mock_sql_chain) -> None:
        """Test handling of query generation errors."""
        from sql_agent.errors import QueryGenerationError

        mock_sql_chain.query.side_effect = QueryGenerationError("Cannot answer this question")

        response = client.post(
            "/api/sql-query",
            json={"question": "What is the meaning of life?"},
        )

        assert response.status_code == 400
        # Check for either "generate sql" or "could not generate"
        detail = response.json()["detail"].lower()
        assert "generate" in detail and "sql" in detail

    def test_sql_query_execution_error(self, client: TestClient, mock_sql_chain) -> None:
        """Test handling of query execution errors."""
        from sql_agent.errors import QueryExecutionError

        mock_sql_chain.query.side_effect = QueryExecutionError("Table not found")

        response = client.post(
            "/api/sql-query",
            json={"question": "Show data from nonexistent table"},
        )

        assert response.status_code == 500
        assert "execution failed" in response.json()["detail"].lower()

    def test_sql_query_not_initialized(self, mock_sql_chain) -> None:
        """Test error when SQL chain not initialized."""
        from api.main import app
        from api.routes.sql_query import set_sql_chain

        # Set chain to None
        set_sql_chain(None)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/sql-query",
            json={"question": "List products"},
        )

        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()

        # Restore chain for other tests
        set_sql_chain(mock_sql_chain)


class TestSQLTablesEndpoint:
    """Tests for GET /api/sql-tables endpoint."""

    @pytest.fixture
    def mock_sql_chain(self):
        """Create a mock SQL chain."""
        chain = MagicMock()
        chain.get_available_tables.return_value = ["products", "orders", "users"]
        return chain

    @pytest.fixture
    def client(self, mock_sql_chain):
        """Create a test client with mocked SQL chain."""
        from api.main import app
        from api.routes.sql_query import set_sql_chain

        set_sql_chain(mock_sql_chain)
        return TestClient(app)

    def test_list_tables_success(self, client: TestClient) -> None:
        """Test successful table listing."""
        response = client.get("/api/sql-tables")

        assert response.status_code == 200
        data = response.json()

        assert data["count"] == 3
        assert "products" in data["tables"]
        assert "orders" in data["tables"]
        assert "users" in data["tables"]

    def test_list_tables_empty(self, client: TestClient, mock_sql_chain) -> None:
        """Test table listing with no tables."""
        mock_sql_chain.get_available_tables.return_value = []

        response = client.get("/api/sql-tables")

        assert response.status_code == 200
        data = response.json()

        assert data["count"] == 0
        assert data["tables"] == []

    def test_list_tables_not_initialized(self, mock_sql_chain) -> None:
        """Test error when SQL chain not initialized."""
        from api.main import app
        from api.routes.sql_query import set_sql_chain

        set_sql_chain(None)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/sql-tables")

        assert response.status_code == 503

        # Restore chain
        set_sql_chain(mock_sql_chain)


class TestHealthCheckWithSQLAgent:
    """Tests for health check with SQL Agent."""

    @pytest.fixture
    def db_with_data(self, tmp_path: Path):
        """Create a test database with sample data."""
        from storage.sql_store import SQLStore, SQLStoreConfig

        db_path = tmp_path / "catalog.duckdb"
        config = SQLStoreConfig(db_path=db_path)

        with SQLStore(config) as store:
            store.execute("CREATE TABLE test_data (id INTEGER, value VARCHAR)")
            store.execute("INSERT INTO test_data VALUES (1, 'a'), (2, 'b')")

        return db_path

    def test_health_check_includes_sql_agent(self, db_with_data: Path) -> None:
        """Test that health check includes SQL Agent status."""
        import os
        from api.main import app

        # Set environment variable for vectorstore dir
        os.environ["VECTORSTORE_DIR"] = str(db_with_data.parent)

        client = TestClient(app)
        response = client.get("/api/health")

        # Health check should include sql_agent field
        assert response.status_code == 200
        data = response.json()

        # sql_agent may be None if db doesn't exist, but field should be present
        assert "sql_agent" in data

        # If database exists, should be healthy
        if data["sql_agent"] is not None:
            assert "healthy" in data["sql_agent"]
