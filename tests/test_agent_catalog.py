"""Unit tests for storage/agent_catalog.py."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from storage.agent_catalog import (
    AgentCatalog,
    AgentEntry,
    AGENT_TABLE_NAME,
    DEFAULT_AGENT_ID,
    DEFAULT_AGENT_NAME,
    DEFAULT_SYSTEM_PROMPT,
)
from storage.sql_store import SQLStore, SQLStoreConfig


@pytest.fixture
def sql_store(tmp_path: Path) -> SQLStore:
    """Create a SQLStore instance for testing."""
    db_path = tmp_path / "test.duckdb"
    config = SQLStoreConfig(db_path=db_path)
    store = SQLStore(config)
    yield store
    store.close()


@pytest.fixture
def catalog(sql_store: SQLStore) -> AgentCatalog:
    """Create an AgentCatalog instance for testing."""
    return AgentCatalog(sql_store)


class TestAgentEntry:
    """Tests for AgentEntry dataclass."""

    def test_entry_creation(self) -> None:
        """Test basic entry creation."""
        entry = AgentEntry(
            id="test-id",
            name="Test Agent",
            description="A test agent",
            system_prompt="You are a test assistant.\n\n{context}",
            temperature=0.5,
            max_tokens=500,
            is_default=False,
        )

        assert entry.id == "test-id"
        assert entry.name == "Test Agent"
        assert entry.description == "A test agent"
        assert entry.system_prompt == "You are a test assistant.\n\n{context}"
        assert entry.temperature == 0.5
        assert entry.max_tokens == 500
        assert entry.is_default is False

    def test_entry_creation_minimal(self) -> None:
        """Test entry creation with only required fields."""
        entry = AgentEntry(
            id="test-id",
            name="Test Agent",
            system_prompt="Test prompt with {context}",
        )

        assert entry.id == "test-id"
        assert entry.name == "Test Agent"
        assert entry.description is None
        assert entry.temperature is None
        assert entry.max_tokens is None
        assert entry.is_default is False

    def test_entry_to_dict(self) -> None:
        """Test entry serialization to dict."""
        entry = AgentEntry(
            id="test-id",
            name="Test Agent",
            description="A test agent",
            system_prompt="Test prompt with {context}",
            temperature=0.7,
            max_tokens=1000,
            is_default=True,
            created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
        )

        data = entry.to_dict()

        assert data["id"] == "test-id"
        assert data["name"] == "Test Agent"
        assert data["description"] == "A test agent"
        assert data["temperature"] == 0.7
        assert data["max_tokens"] == 1000
        assert data["is_default"] is True
        assert "2024-01-15" in data["created_at"]
        assert "2024-01-15" in data["updated_at"]

    def test_entry_from_row(self) -> None:
        """Test entry creation from database row."""
        row = (
            "test-id",
            "Test Agent",
            "A test agent",
            "Test prompt with {context}",
            0.5,
            800,
            True,
            "2024-01-15T10:30:00",
            "2024-01-15T11:00:00",
        )

        entry = AgentEntry.from_row(row)

        assert entry.id == "test-id"
        assert entry.name == "Test Agent"
        assert entry.description == "A test agent"
        assert entry.temperature == 0.5
        assert entry.max_tokens == 800
        assert entry.is_default is True

    def test_entry_from_row_with_nulls(self) -> None:
        """Test entry creation from database row with null fields."""
        row = (
            "test-id",
            "Test Agent",
            None,  # description
            "Test prompt",
            None,  # temperature
            None,  # max_tokens
            False,
            "2024-01-15T10:30:00",
            "2024-01-15T10:30:00",
        )

        entry = AgentEntry.from_row(row)

        assert entry.id == "test-id"
        assert entry.description is None
        assert entry.temperature is None
        assert entry.max_tokens is None


class TestAgentCatalog:
    """Tests for AgentCatalog class."""

    def test_init_catalog(self, catalog: AgentCatalog, sql_store: SQLStore) -> None:
        """Test catalog table initialization."""
        catalog.init_catalog()

        # Verify table exists
        assert sql_store.table_exists(AGENT_TABLE_NAME)

    def test_default_agent_seeded(self, catalog: AgentCatalog) -> None:
        """Test that default agent is seeded on initialization."""
        catalog.init_catalog()

        default = catalog.get_default()
        assert default is not None
        assert default.id == DEFAULT_AGENT_ID
        assert default.name == DEFAULT_AGENT_NAME
        assert default.is_default is True
        assert "{context}" in default.system_prompt

    def test_get_by_id(self, catalog: AgentCatalog) -> None:
        """Test getting agent by ID."""
        catalog.init_catalog()

        # Get default agent
        agent = catalog.get_by_id(DEFAULT_AGENT_ID)
        assert agent is not None
        assert agent.name == DEFAULT_AGENT_NAME

        # Non-existent agent
        agent = catalog.get_by_id("nonexistent")
        assert agent is None

    def test_get_by_name(self, catalog: AgentCatalog) -> None:
        """Test getting agent by name."""
        catalog.init_catalog()

        # Get default agent by name
        agent = catalog.get_by_name(DEFAULT_AGENT_NAME)
        assert agent is not None
        assert agent.id == DEFAULT_AGENT_ID

        # Non-existent agent
        agent = catalog.get_by_name("Nonexistent Agent")
        assert agent is None

    def test_list_agents(self, catalog: AgentCatalog) -> None:
        """Test listing all agents."""
        catalog.init_catalog()

        # Should have default agent
        agents = catalog.list_agents()
        assert len(agents) >= 1
        assert any(a.name == DEFAULT_AGENT_NAME for a in agents)

    def test_create_agent(self, catalog: AgentCatalog) -> None:
        """Test creating a new agent."""
        catalog.init_catalog()

        agent = catalog.create_agent(
            name="HR Agent",
            system_prompt="You are an HR assistant.\n\n{context}",
            description="Handles HR queries",
            temperature=0.3,
            max_tokens=800,
        )

        assert agent.id is not None
        assert agent.name == "HR Agent"
        assert agent.description == "Handles HR queries"
        assert agent.temperature == 0.3
        assert agent.max_tokens == 800
        assert agent.is_default is False

        # Verify it's persisted
        retrieved = catalog.get_by_id(agent.id)
        assert retrieved is not None
        assert retrieved.name == "HR Agent"

    def test_create_agent_minimal(self, catalog: AgentCatalog) -> None:
        """Test creating an agent with only required fields."""
        catalog.init_catalog()

        agent = catalog.create_agent(
            name="Simple Agent",
            system_prompt="Simple prompt with {context}",
        )

        assert agent.name == "Simple Agent"
        assert agent.description is None
        assert agent.temperature is None
        assert agent.max_tokens is None

    def test_create_agent_duplicate_name(self, catalog: AgentCatalog) -> None:
        """Test that creating an agent with duplicate name raises error."""
        catalog.init_catalog()

        catalog.create_agent(
            name="Unique Agent",
            system_prompt="Prompt with {context}",
        )

        with pytest.raises(ValueError, match="already exists"):
            catalog.create_agent(
                name="Unique Agent",
                system_prompt="Different prompt with {context}",
            )

    def test_update_agent(self, catalog: AgentCatalog) -> None:
        """Test updating an agent."""
        catalog.init_catalog()

        # Create an agent
        agent = catalog.create_agent(
            name="Test Agent",
            system_prompt="Original prompt with {context}",
            temperature=0.5,
        )

        # Update it
        updated = catalog.update_agent(
            agent_id=agent.id,
            name="Updated Agent",
            description="New description",
            temperature=0.8,
        )

        assert updated is not None
        assert updated.name == "Updated Agent"
        assert updated.description == "New description"
        assert updated.temperature == pytest.approx(0.8)
        # System prompt unchanged
        assert updated.system_prompt == "Original prompt with {context}"

    def test_update_agent_not_found(self, catalog: AgentCatalog) -> None:
        """Test updating non-existent agent returns None."""
        catalog.init_catalog()

        result = catalog.update_agent(
            agent_id="nonexistent",
            name="New Name",
        )
        assert result is None

    def test_update_agent_duplicate_name(self, catalog: AgentCatalog) -> None:
        """Test that updating to duplicate name raises error."""
        catalog.init_catalog()

        agent1 = catalog.create_agent(
            name="Agent One",
            system_prompt="Prompt with {context}",
        )
        catalog.create_agent(
            name="Agent Two",
            system_prompt="Prompt with {context}",
        )

        with pytest.raises(ValueError, match="already exists"):
            catalog.update_agent(
                agent_id=agent1.id,
                name="Agent Two",
            )

    def test_update_agent_same_name(self, catalog: AgentCatalog) -> None:
        """Test that updating with same name doesn't raise error."""
        catalog.init_catalog()

        agent = catalog.create_agent(
            name="Test Agent",
            system_prompt="Prompt with {context}",
        )

        # Update with same name should work
        updated = catalog.update_agent(
            agent_id=agent.id,
            name="Test Agent",
            description="New description",
        )

        assert updated is not None
        assert updated.description == "New description"

    def test_delete_agent(self, catalog: AgentCatalog) -> None:
        """Test deleting an agent."""
        catalog.init_catalog()

        # Create an agent
        agent = catalog.create_agent(
            name="To Delete",
            system_prompt="Prompt with {context}",
        )
        assert catalog.get_by_id(agent.id) is not None

        # Delete it
        result = catalog.delete_agent(agent.id)
        assert result is True
        assert catalog.get_by_id(agent.id) is None

    def test_delete_agent_not_found(self, catalog: AgentCatalog) -> None:
        """Test deleting non-existent agent returns False."""
        catalog.init_catalog()

        result = catalog.delete_agent("nonexistent")
        assert result is False

    def test_delete_default_agent_raises(self, catalog: AgentCatalog) -> None:
        """Test that deleting the default agent raises error."""
        catalog.init_catalog()

        with pytest.raises(ValueError, match="Cannot delete the default agent"):
            catalog.delete_agent(DEFAULT_AGENT_ID)

    def test_get_statistics(self, catalog: AgentCatalog) -> None:
        """Test getting catalog statistics."""
        catalog.init_catalog()

        # Initial stats (just default agent)
        stats = catalog.get_statistics()
        assert stats["agent_count"] >= 1
        assert stats["default_agent_name"] == DEFAULT_AGENT_NAME

        # Add more agents
        catalog.create_agent(
            name="Agent 1",
            system_prompt="Prompt with {context}",
        )
        catalog.create_agent(
            name="Agent 2",
            system_prompt="Prompt with {context}",
        )

        stats = catalog.get_statistics()
        assert stats["agent_count"] >= 3

    def test_idempotent_init(self, catalog: AgentCatalog) -> None:
        """Test that multiple init calls are idempotent."""
        catalog.init_catalog()
        catalog.init_catalog()
        catalog.init_catalog()

        # Should still only have one default agent
        agents = catalog.list_agents()
        default_count = sum(1 for a in agents if a.is_default)
        assert default_count == 1

    def test_agents_ordered_by_name(self, catalog: AgentCatalog) -> None:
        """Test that list_agents returns agents ordered by name."""
        catalog.init_catalog()

        # Create agents in reverse order
        catalog.create_agent(name="Zebra Agent", system_prompt="{context}")
        catalog.create_agent(name="Alpha Agent", system_prompt="{context}")
        catalog.create_agent(name="Middle Agent", system_prompt="{context}")

        agents = catalog.list_agents()
        names = [a.name for a in agents]

        # Should be sorted
        assert names == sorted(names)
