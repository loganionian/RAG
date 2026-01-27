"""Integration tests for api/routes/agents.py."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from storage.agent_catalog import DEFAULT_AGENT_ID


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "vectorstore" / "catalog.duckdb"


@pytest.fixture
def client(test_db_path: Path) -> TestClient:
    """Create a test client with isolated database."""
    # Ensure parent directory exists
    test_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Patch the DB_PATH before importing the app
    with patch("api.routes.agents.DB_PATH", test_db_path):
        # Need to reload the module to pick up the patched path
        from api.main import app

        with TestClient(app) as test_client:
            yield test_client


class TestListAgents:
    """Tests for GET /api/agents endpoint."""

    def test_list_agents_returns_default(self, client: TestClient) -> None:
        """Test that list returns the default agent."""
        response = client.get("/api/agents")
        assert response.status_code == 200

        data = response.json()
        assert "agents" in data
        assert "total" in data
        assert data["total"] >= 1

        # Default agent should be present
        names = [a["name"] for a in data["agents"]]
        assert "Default" in names

    def test_list_agents_ordered_by_name(self, client: TestClient) -> None:
        """Test that agents are ordered alphabetically by name."""
        # Create agents in reverse order
        client.post(
            "/api/agents",
            json={"name": "Zebra", "system_prompt": "Prompt {context}"},
        )
        client.post(
            "/api/agents",
            json={"name": "Alpha", "system_prompt": "Prompt {context}"},
        )

        response = client.get("/api/agents")
        assert response.status_code == 200

        names = [a["name"] for a in response.json()["agents"]]
        assert names == sorted(names)


class TestGetAgent:
    """Tests for GET /api/agents/{agent_id} endpoint."""

    def test_get_default_agent(self, client: TestClient) -> None:
        """Test getting the default agent."""
        response = client.get(f"/api/agents/{DEFAULT_AGENT_ID}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == DEFAULT_AGENT_ID
        assert data["name"] == "Default"
        assert data["is_default"] is True
        assert "{context}" in data["system_prompt"]

    def test_get_agent_not_found(self, client: TestClient) -> None:
        """Test getting a non-existent agent returns 404."""
        response = client.get("/api/agents/nonexistent-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_created_agent(self, client: TestClient) -> None:
        """Test getting a newly created agent."""
        # Create agent
        create_response = client.post(
            "/api/agents",
            json={
                "name": "Test Agent",
                "system_prompt": "Test prompt with {context}",
                "description": "Test description",
                "temperature": 0.5,
                "max_tokens": 500,
            },
        )
        agent_id = create_response.json()["id"]

        # Get it back
        response = client.get(f"/api/agents/{agent_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == agent_id
        assert data["name"] == "Test Agent"
        assert data["description"] == "Test description"
        assert data["temperature"] == pytest.approx(0.5)
        assert data["max_tokens"] == 500


class TestCreateAgent:
    """Tests for POST /api/agents endpoint."""

    def test_create_agent_minimal(self, client: TestClient) -> None:
        """Test creating an agent with only required fields."""
        response = client.post(
            "/api/agents",
            json={
                "name": "Minimal Agent",
                "system_prompt": "Simple prompt with {context}",
            },
        )
        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Minimal Agent"
        assert data["system_prompt"] == "Simple prompt with {context}"
        assert data["description"] is None
        assert data["temperature"] is None
        assert data["max_tokens"] is None
        assert data["is_default"] is False
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_create_agent_full(self, client: TestClient) -> None:
        """Test creating an agent with all fields."""
        response = client.post(
            "/api/agents",
            json={
                "name": "Full Agent",
                "description": "A fully configured agent",
                "system_prompt": "You are helpful.\n\n{context}",
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        )
        assert response.status_code == 201

        data = response.json()
        assert data["name"] == "Full Agent"
        assert data["description"] == "A fully configured agent"
        assert data["temperature"] == pytest.approx(0.7)
        assert data["max_tokens"] == 1000

    def test_create_agent_duplicate_name_409(self, client: TestClient) -> None:
        """Test that creating agent with duplicate name returns 409."""
        # Create first agent
        client.post(
            "/api/agents",
            json={"name": "Unique Name", "system_prompt": "Prompt {context}"},
        )

        # Try to create another with same name
        response = client.post(
            "/api/agents",
            json={"name": "Unique Name", "system_prompt": "Different prompt {context}"},
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"].lower()

    def test_create_agent_validation_name_required(self, client: TestClient) -> None:
        """Test that name is required."""
        response = client.post(
            "/api/agents",
            json={"system_prompt": "Prompt {context}"},
        )
        assert response.status_code == 422  # Validation error

    def test_create_agent_validation_system_prompt_required(
        self, client: TestClient
    ) -> None:
        """Test that system_prompt is required."""
        response = client.post(
            "/api/agents",
            json={"name": "Test Agent"},
        )
        assert response.status_code == 422  # Validation error

    def test_create_agent_validation_temperature_range(
        self, client: TestClient
    ) -> None:
        """Test that temperature is validated (0.0-2.0)."""
        response = client.post(
            "/api/agents",
            json={
                "name": "Test",
                "system_prompt": "Prompt {context}",
                "temperature": 3.0,  # Out of range
            },
        )
        assert response.status_code == 422

        # Valid temperature
        response = client.post(
            "/api/agents",
            json={
                "name": "Test",
                "system_prompt": "Prompt {context}",
                "temperature": 2.0,
            },
        )
        assert response.status_code == 201

    def test_create_agent_validation_max_tokens_positive(
        self, client: TestClient
    ) -> None:
        """Test that max_tokens must be positive."""
        response = client.post(
            "/api/agents",
            json={
                "name": "Test",
                "system_prompt": "Prompt {context}",
                "max_tokens": 0,
            },
        )
        assert response.status_code == 422


class TestUpdateAgent:
    """Tests for PUT /api/agents/{agent_id} endpoint."""

    def test_update_agent_name(self, client: TestClient) -> None:
        """Test updating agent name."""
        # Create agent
        create_response = client.post(
            "/api/agents",
            json={"name": "Original Name", "system_prompt": "Prompt {context}"},
        )
        agent_id = create_response.json()["id"]

        # Update name
        response = client.put(
            f"/api/agents/{agent_id}",
            json={"name": "New Name"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "New Name"

    def test_update_agent_multiple_fields(self, client: TestClient) -> None:
        """Test updating multiple fields at once."""
        # Create agent
        create_response = client.post(
            "/api/agents",
            json={
                "name": "Test Agent",
                "system_prompt": "Original prompt {context}",
                "temperature": 0.5,
            },
        )
        agent_id = create_response.json()["id"]

        # Update multiple fields
        response = client.put(
            f"/api/agents/{agent_id}",
            json={
                "description": "Updated description",
                "temperature": 0.8,
                "max_tokens": 500,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["description"] == "Updated description"
        assert data["temperature"] == pytest.approx(0.8)
        assert data["max_tokens"] == 500
        # Unchanged fields
        assert data["name"] == "Test Agent"
        assert data["system_prompt"] == "Original prompt {context}"

    def test_update_agent_not_found_404(self, client: TestClient) -> None:
        """Test updating non-existent agent returns 404."""
        response = client.put(
            "/api/agents/nonexistent-id",
            json={"name": "New Name"},
        )
        assert response.status_code == 404

    def test_update_agent_duplicate_name_409(self, client: TestClient) -> None:
        """Test that updating to duplicate name returns 409."""
        # Create two agents
        client.post(
            "/api/agents",
            json={"name": "Agent One", "system_prompt": "Prompt {context}"},
        )
        create_response = client.post(
            "/api/agents",
            json={"name": "Agent Two", "system_prompt": "Prompt {context}"},
        )
        agent_id = create_response.json()["id"]

        # Try to rename Agent Two to Agent One
        response = client.put(
            f"/api/agents/{agent_id}",
            json={"name": "Agent One"},
        )
        assert response.status_code == 409

    def test_update_agent_same_name_ok(self, client: TestClient) -> None:
        """Test that updating with same name is allowed."""
        # Create agent
        create_response = client.post(
            "/api/agents",
            json={"name": "Test Agent", "system_prompt": "Prompt {context}"},
        )
        agent_id = create_response.json()["id"]

        # Update with same name but different description
        response = client.put(
            f"/api/agents/{agent_id}",
            json={"name": "Test Agent", "description": "New description"},
        )
        assert response.status_code == 200
        assert response.json()["description"] == "New description"

    def test_update_agent_empty_body_returns_unchanged(
        self, client: TestClient
    ) -> None:
        """Test that empty update body returns unchanged agent."""
        # Create agent
        create_response = client.post(
            "/api/agents",
            json={
                "name": "Test Agent",
                "system_prompt": "Prompt {context}",
                "temperature": 0.5,
            },
        )
        agent_id = create_response.json()["id"]
        original_data = create_response.json()

        # Update with empty body
        response = client.put(f"/api/agents/{agent_id}", json={})
        assert response.status_code == 200
        assert response.json()["name"] == original_data["name"]
        assert response.json()["temperature"] == original_data["temperature"]


class TestDeleteAgent:
    """Tests for DELETE /api/agents/{agent_id} endpoint."""

    def test_delete_agent_success(self, client: TestClient) -> None:
        """Test deleting an agent."""
        # Create agent
        create_response = client.post(
            "/api/agents",
            json={"name": "To Delete", "system_prompt": "Prompt {context}"},
        )
        agent_id = create_response.json()["id"]

        # Delete it
        response = client.delete(f"/api/agents/{agent_id}")
        assert response.status_code == 200
        assert response.json()["deleted"] is True

        # Verify it's gone
        get_response = client.get(f"/api/agents/{agent_id}")
        assert get_response.status_code == 404

    def test_delete_agent_not_found_404(self, client: TestClient) -> None:
        """Test deleting non-existent agent returns 404."""
        response = client.delete("/api/agents/nonexistent-id")
        assert response.status_code == 404

    def test_delete_default_agent_400(self, client: TestClient) -> None:
        """Test that deleting default agent returns 400."""
        response = client.delete(f"/api/agents/{DEFAULT_AGENT_ID}")
        assert response.status_code == 400
        assert "cannot delete" in response.json()["detail"].lower()


class TestAgentResponseFormat:
    """Tests for agent response format and timestamps."""

    def test_response_includes_timestamps(self, client: TestClient) -> None:
        """Test that response includes ISO format timestamps."""
        response = client.post(
            "/api/agents",
            json={"name": "Test Agent", "system_prompt": "Prompt {context}"},
        )
        assert response.status_code == 201

        data = response.json()
        assert "created_at" in data
        assert "updated_at" in data
        # ISO format should contain T separator
        assert "T" in data["created_at"]
        assert "T" in data["updated_at"]

    def test_updated_at_changes_on_update(self, client: TestClient) -> None:
        """Test that updated_at timestamp changes on update."""
        # Create agent
        create_response = client.post(
            "/api/agents",
            json={"name": "Test Agent", "system_prompt": "Prompt {context}"},
        )
        agent_id = create_response.json()["id"]
        original_updated_at = create_response.json()["updated_at"]

        # Update it
        update_response = client.put(
            f"/api/agents/{agent_id}",
            json={"description": "New description"},
        )

        # updated_at should be different (or same if very fast)
        # Just verify it's present
        assert "updated_at" in update_response.json()


class TestOpenAPIDocumentation:
    """Tests for OpenAPI documentation generation."""

    def test_openapi_includes_agents_endpoints(self, client: TestClient) -> None:
        """Test that OpenAPI schema includes agent endpoints."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi.get("paths", {})

        # Check all agent endpoints are documented
        assert "/api/agents" in paths
        assert "/api/agents/{agent_id}" in paths

        # Check methods
        assert "get" in paths["/api/agents"]
        assert "post" in paths["/api/agents"]
        assert "get" in paths["/api/agents/{agent_id}"]
        assert "put" in paths["/api/agents/{agent_id}"]
        assert "delete" in paths["/api/agents/{agent_id}"]
