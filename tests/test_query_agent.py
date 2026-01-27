"""Tests for agent-aware query endpoint."""
from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from storage.agent_catalog import DEFAULT_AGENT_ID


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "vectorstore" / "catalog.duckdb"


@pytest.fixture
def mock_rag_chain() -> MagicMock:
    """Create a mock RAG chain with minimal behavior."""
    mock = MagicMock()

    # Mock config with default values
    mock.config.system_prompt = "Default system prompt with {context}"
    mock.config.temperature = 0.7
    mock.config.max_tokens = 1000

    # Mock retrieval result
    mock_retrieval_result = MagicMock()
    mock_retrieval_result.chunks = ["Chunk 1 content"]
    mock_retrieval_result.metadatas = [
        {"doc_id": "test-doc", "relative_path": "test.pdf", "chunk_id": "test-doc::chunk-0001"}
    ]
    mock.retrieve.return_value = mock_retrieval_result

    # Mock context formatting
    mock._format_context.return_value = "Formatted context"

    # Mock LLM generation
    mock.llm_client.generate.return_value = "Generated answer"

    return mock


@pytest.fixture
def client(
    test_db_path: Path, mock_rag_chain: MagicMock
) -> Generator[TestClient, None, None]:
    """Create a test client with isolated database and mocked RAG chain."""
    # Ensure parent directory exists
    test_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Import modules before patching
    from api.routes import query as query_module
    import api.main as main_module

    # Patch VECTORSTORE_DIR to a non-existent path so lifespan skips RAG init
    fake_vectorstore = test_db_path.parent / "nonexistent_vectorstore"

    with (
        patch("api.routes.agents.DB_PATH", test_db_path),
        patch.object(query_module, "DB_PATH", test_db_path),
        patch.object(main_module, "VECTORSTORE_DIR", fake_vectorstore),
    ):
        # Import app and set up mock after patches
        from api.main import app

        # Set the mock RAG chain using the proper interface
        query_module.set_rag_chain(mock_rag_chain)

        with TestClient(app) as test_client:
            yield test_client


class TestQueryWithoutAgent:
    """Tests for queries without agent_id (backward compatibility)."""

    def test_query_without_agent_uses_defaults(
        self, client: TestClient, mock_rag_chain: MagicMock
    ) -> None:
        """Test that query without agent_id uses default config."""
        response = client.post(
            "/api/query",
            json={"question": "What is the answer?", "k": 3},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["answer"] == "Generated answer"
        assert data["metadata"]["agent_id"] is None

        # Verify LLM was called with default parameters
        mock_rag_chain.llm_client.generate.assert_called_once()
        call_kwargs = mock_rag_chain.llm_client.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1000

    def test_query_response_has_sources(
        self, client: TestClient, mock_rag_chain: MagicMock
    ) -> None:
        """Test that query response includes sources."""
        response = client.post(
            "/api/query",
            json={"question": "What is the answer?", "k": 3},
        )
        assert response.status_code == 200

        data = response.json()
        assert "sources" in data
        assert len(data["sources"]) == 1
        assert data["sources"][0]["doc_id"] == "test-doc"


class TestQueryWithValidAgent:
    """Tests for queries with a valid agent_id."""

    def test_query_with_agent_uses_agent_system_prompt(
        self, client: TestClient, mock_rag_chain: MagicMock
    ) -> None:
        """Test that query with agent_id uses the agent's system prompt."""
        # Create a custom agent
        create_response = client.post(
            "/api/agents",
            json={
                "name": "Custom Agent",
                "system_prompt": "You are a custom assistant. Context: {context}",
            },
        )
        assert create_response.status_code == 201
        agent_id = create_response.json()["id"]

        # Query with the agent
        response = client.post(
            "/api/query",
            json={"question": "What is the answer?", "k": 3, "agent_id": agent_id},
        )
        assert response.status_code == 200

        # Verify the custom system prompt was used
        call_kwargs = mock_rag_chain.llm_client.generate.call_args.kwargs
        assert "You are a custom assistant" in call_kwargs["system_prompt"]

    def test_query_with_agent_uses_agent_temperature(
        self, client: TestClient, mock_rag_chain: MagicMock
    ) -> None:
        """Test that query with agent_id uses the agent's temperature override."""
        # Create an agent with custom temperature
        create_response = client.post(
            "/api/agents",
            json={
                "name": "Hot Agent",
                "system_prompt": "Prompt with {context}",
                "temperature": 1.5,
            },
        )
        assert create_response.status_code == 201
        agent_id = create_response.json()["id"]

        # Query with the agent
        response = client.post(
            "/api/query",
            json={"question": "What is the answer?", "k": 3, "agent_id": agent_id},
        )
        assert response.status_code == 200

        # Verify the custom temperature was used
        call_kwargs = mock_rag_chain.llm_client.generate.call_args.kwargs
        assert call_kwargs["temperature"] == pytest.approx(1.5)

    def test_query_with_agent_uses_agent_max_tokens(
        self, client: TestClient, mock_rag_chain: MagicMock
    ) -> None:
        """Test that query with agent_id uses the agent's max_tokens override."""
        # Create an agent with custom max_tokens
        create_response = client.post(
            "/api/agents",
            json={
                "name": "Verbose Agent",
                "system_prompt": "Prompt with {context}",
                "max_tokens": 5000,
            },
        )
        assert create_response.status_code == 201
        agent_id = create_response.json()["id"]

        # Query with the agent
        response = client.post(
            "/api/query",
            json={"question": "What is the answer?", "k": 3, "agent_id": agent_id},
        )
        assert response.status_code == 200

        # Verify the custom max_tokens was used
        call_kwargs = mock_rag_chain.llm_client.generate.call_args.kwargs
        assert call_kwargs["max_tokens"] == 5000

    def test_query_with_agent_falls_back_to_defaults_for_none(
        self, client: TestClient, mock_rag_chain: MagicMock
    ) -> None:
        """Test that unset agent params fall back to default config."""
        # Create an agent without temperature/max_tokens
        create_response = client.post(
            "/api/agents",
            json={
                "name": "Minimal Agent",
                "system_prompt": "Minimal prompt with {context}",
            },
        )
        assert create_response.status_code == 201
        agent_id = create_response.json()["id"]

        # Query with the agent
        response = client.post(
            "/api/query",
            json={"question": "What is the answer?", "k": 3, "agent_id": agent_id},
        )
        assert response.status_code == 200

        # Verify defaults were used for unset params
        call_kwargs = mock_rag_chain.llm_client.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7  # default
        assert call_kwargs["max_tokens"] == 1000  # default
        # But system prompt should be from agent
        assert "Minimal prompt" in call_kwargs["system_prompt"]

    def test_response_metadata_includes_agent_id(
        self, client: TestClient, mock_rag_chain: MagicMock
    ) -> None:
        """Test that response metadata includes the agent_id when used."""
        # Create an agent
        create_response = client.post(
            "/api/agents",
            json={
                "name": "Test Agent",
                "system_prompt": "Prompt with {context}",
            },
        )
        assert create_response.status_code == 201
        agent_id = create_response.json()["id"]

        # Query with the agent
        response = client.post(
            "/api/query",
            json={"question": "What is the answer?", "k": 3, "agent_id": agent_id},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["metadata"]["agent_id"] == agent_id

    def test_query_with_default_agent(
        self, client: TestClient, mock_rag_chain: MagicMock
    ) -> None:
        """Test that query with default agent_id works."""
        response = client.post(
            "/api/query",
            json={"question": "What is the answer?", "k": 3, "agent_id": DEFAULT_AGENT_ID},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["metadata"]["agent_id"] == DEFAULT_AGENT_ID


class TestQueryWithInvalidAgent:
    """Tests for queries with an invalid agent_id."""

    def test_query_with_nonexistent_agent_returns_404(
        self, client: TestClient, mock_rag_chain: MagicMock
    ) -> None:
        """Test that query with non-existent agent_id returns 404."""
        response = client.post(
            "/api/query",
            json={"question": "What is the answer?", "k": 3, "agent_id": "nonexistent-id"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_query_with_invalid_agent_does_not_call_llm(
        self, client: TestClient, mock_rag_chain: MagicMock
    ) -> None:
        """Test that query with invalid agent_id does not call LLM."""
        # Reset the mock call count
        mock_rag_chain.llm_client.generate.reset_mock()

        response = client.post(
            "/api/query",
            json={"question": "What is the answer?", "k": 3, "agent_id": "nonexistent-id"},
        )
        assert response.status_code == 404

        # LLM should not have been called
        mock_rag_chain.llm_client.generate.assert_not_called()


class TestQuerySchemaValidation:
    """Tests for query request schema validation."""

    def test_agent_id_is_optional(
        self, client: TestClient, mock_rag_chain: MagicMock
    ) -> None:
        """Test that agent_id is an optional field."""
        # Query without agent_id should work
        response = client.post(
            "/api/query",
            json={"question": "What is the answer?"},
        )
        assert response.status_code == 200

    def test_agent_id_can_be_null(
        self, client: TestClient, mock_rag_chain: MagicMock
    ) -> None:
        """Test that agent_id can be explicitly null."""
        response = client.post(
            "/api/query",
            json={"question": "What is the answer?", "agent_id": None},
        )
        assert response.status_code == 200
        assert response.json()["metadata"]["agent_id"] is None
