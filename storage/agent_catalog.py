"""Agent catalog for managing LLM agent configurations.

This module provides a catalog that stores custom LLM agent configurations,
enabling users to select different personas for RAG queries. Each agent has
a custom system prompt, temperature, and max tokens settings.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .sql_store import SQLStore

logger = logging.getLogger(__name__)

# Agent table name (prefixed with underscore to indicate system table)
AGENT_TABLE_NAME = "_agents"

# Default system prompt for the default agent
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
- Answer the question using ONLY the information from the context below
- If the context doesn't contain enough information to answer, say so clearly
- Be concise and direct in your responses
- Cite specific details from the context when relevant

Context:
{context}"""

# Default agent configuration
DEFAULT_AGENT_ID = "default"
DEFAULT_AGENT_NAME = "Default"
DEFAULT_AGENT_DESCRIPTION = "Default RAG assistant"

# All columns for SELECT queries (in order)
ALL_COLUMNS = """id, name, description, system_prompt, temperature, max_tokens,
                 is_default, created_at, updated_at"""


@dataclass
class AgentEntry:
    """Represents an agent configuration in the catalog.

    Attributes:
        id: Unique agent identifier.
        name: Human-readable agent name (unique).
        description: Optional short description.
        system_prompt: System prompt template with {context} placeholder.
        temperature: Optional LLM temperature (0.0-1.0).
        max_tokens: Optional max response tokens.
        is_default: Whether this is the system default agent.
        created_at: When the agent was created.
        updated_at: When the agent was last updated.
    """

    id: str
    name: str
    system_prompt: str
    description: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    is_default: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_row(cls, row: tuple) -> "AgentEntry":
        """Create entry from database row.

        Args:
            row: Tuple of (id, name, description, system_prompt, temperature,
                 max_tokens, is_default, created_at, updated_at).

        Returns:
            AgentEntry instance.
        """
        created_at = row[7]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        updated_at = row[8]
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return cls(
            id=row[0],
            name=row[1],
            description=row[2],
            system_prompt=row[3],
            temperature=row[4],
            max_tokens=row[5],
            is_default=bool(row[6]),
            created_at=created_at,
            updated_at=updated_at,
        )


class AgentCatalog:
    """Manages the _agents table for storing LLM agent configurations."""

    def __init__(self, sql_store: SQLStore) -> None:
        """Initialize the agent catalog.

        Args:
            sql_store: SQLStore instance for database operations.
        """
        self.sql_store = sql_store
        self._initialized = False

    def init_catalog(self) -> None:
        """Create the agents table if it doesn't exist.

        In read-only mode, this only verifies the table exists without creating it.
        """
        if self._initialized:
            return

        # In read-only mode, just check if the table exists
        if self.sql_store.config.read_only:
            if self.sql_store.table_exists(AGENT_TABLE_NAME):
                self._initialized = True
                logger.debug("Agents table verified (read-only mode)")
            return

        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {AGENT_TABLE_NAME} (
                id VARCHAR PRIMARY KEY,
                name VARCHAR UNIQUE NOT NULL,
                description VARCHAR,
                system_prompt TEXT NOT NULL,
                temperature FLOAT,
                max_tokens INTEGER,
                is_default BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        self.sql_store.execute(create_table_sql)

        # Create index on name for fast lookups
        self.sql_store.execute(
            f"CREATE INDEX IF NOT EXISTS idx_agents_name ON {AGENT_TABLE_NAME}(name)"
        )

        # Seed default agent if it doesn't exist
        self._seed_default_agent()

        self._initialized = True
        logger.debug("Initialized agents catalog table")

    def _seed_default_agent(self) -> None:
        """Create the default agent if it doesn't exist.

        Note: This method queries the database directly to avoid recursion,
        since it's called during init_catalog() before _initialized is set.
        """
        # Direct query to avoid recursion through get_by_id -> init_catalog
        query = f"""
            SELECT {ALL_COLUMNS}
            FROM {AGENT_TABLE_NAME}
            WHERE id = ?
        """
        row = self.sql_store.fetchone(query, (DEFAULT_AGENT_ID,))
        if row is not None:
            logger.debug("Default agent already exists")
            return

        default_agent = AgentEntry(
            id=DEFAULT_AGENT_ID,
            name=DEFAULT_AGENT_NAME,
            description=DEFAULT_AGENT_DESCRIPTION,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            is_default=True,
        )
        self._insert_agent(default_agent)
        logger.info("Seeded default agent")

    def _insert_agent(self, agent: AgentEntry) -> None:
        """Insert an agent into the database.

        Args:
            agent: AgentEntry to insert.
        """
        insert_sql = f"""
            INSERT INTO {AGENT_TABLE_NAME}
            (id, name, description, system_prompt, temperature, max_tokens,
             is_default, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.sql_store.execute(
            insert_sql,
            (
                agent.id,
                agent.name,
                agent.description,
                agent.system_prompt,
                agent.temperature,
                agent.max_tokens,
                agent.is_default,
                agent.created_at.isoformat(),
                agent.updated_at.isoformat(),
            ),
        )

    def get_by_id(self, agent_id: str) -> Optional[AgentEntry]:
        """Get an agent by ID.

        Args:
            agent_id: Agent identifier.

        Returns:
            AgentEntry if found, None otherwise.
        """
        self.init_catalog()
        query = f"""
            SELECT {ALL_COLUMNS}
            FROM {AGENT_TABLE_NAME}
            WHERE id = ?
        """
        row = self.sql_store.fetchone(query, (agent_id,))
        return AgentEntry.from_row(row) if row else None

    def get_by_name(self, name: str) -> Optional[AgentEntry]:
        """Get an agent by name.

        Args:
            name: Agent name.

        Returns:
            AgentEntry if found, None otherwise.
        """
        self.init_catalog()
        query = f"""
            SELECT {ALL_COLUMNS}
            FROM {AGENT_TABLE_NAME}
            WHERE name = ?
        """
        row = self.sql_store.fetchone(query, (name,))
        return AgentEntry.from_row(row) if row else None

    def get_default(self) -> Optional[AgentEntry]:
        """Get the default agent.

        Returns:
            The default AgentEntry if found, None otherwise.
        """
        self.init_catalog()
        query = f"""
            SELECT {ALL_COLUMNS}
            FROM {AGENT_TABLE_NAME}
            WHERE is_default = TRUE
            LIMIT 1
        """
        row = self.sql_store.fetchone(query)
        return AgentEntry.from_row(row) if row else None

    def list_agents(self) -> List[AgentEntry]:
        """List all agents in the catalog.

        Returns:
            List of AgentEntry objects, ordered by name.
        """
        self.init_catalog()
        query = f"""
            SELECT {ALL_COLUMNS}
            FROM {AGENT_TABLE_NAME}
            ORDER BY name
        """
        rows = self.sql_store.fetchall(query)
        return [AgentEntry.from_row(row) for row in rows]

    def create_agent(
        self,
        name: str,
        system_prompt: str,
        description: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AgentEntry:
        """Create a new agent.

        Args:
            name: Unique agent name.
            system_prompt: System prompt template with {context} placeholder.
            description: Optional short description.
            temperature: Optional LLM temperature (0.0-1.0).
            max_tokens: Optional max response tokens.

        Returns:
            Created AgentEntry.

        Raises:
            ValueError: If an agent with the same name already exists.
        """
        self.init_catalog()

        # Check for existing agent with same name
        existing = self.get_by_name(name)
        if existing is not None:
            raise ValueError(f"Agent with name '{name}' already exists")

        agent = AgentEntry(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            is_default=False,
        )
        self._insert_agent(agent)
        logger.info("Created agent '%s' (id=%s)", name, agent.id)
        return agent

    def update_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[AgentEntry]:
        """Update an existing agent.

        Args:
            agent_id: Agent identifier.
            name: New name (must be unique if provided).
            description: New description.
            system_prompt: New system prompt.
            temperature: New temperature.
            max_tokens: New max tokens.

        Returns:
            Updated AgentEntry if found, None if agent doesn't exist.

        Raises:
            ValueError: If new name conflicts with existing agent.
        """
        self.init_catalog()

        existing = self.get_by_id(agent_id)
        if existing is None:
            return None

        # Check for name conflict if name is being changed
        if name is not None and name != existing.name:
            conflict = self.get_by_name(name)
            if conflict is not None:
                raise ValueError(f"Agent with name '{name}' already exists")

        # Build update fields
        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if system_prompt is not None:
            updates.append("system_prompt = ?")
            params.append(system_prompt)
        if temperature is not None:
            updates.append("temperature = ?")
            params.append(temperature)
        if max_tokens is not None:
            updates.append("max_tokens = ?")
            params.append(max_tokens)

        if not updates:
            return existing

        # Always update updated_at
        updates.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())

        # Add agent_id for WHERE clause
        params.append(agent_id)

        update_sql = f"""
            UPDATE {AGENT_TABLE_NAME}
            SET {', '.join(updates)}
            WHERE id = ?
        """
        self.sql_store.execute(update_sql, tuple(params))
        logger.info("Updated agent '%s'", agent_id)

        return self.get_by_id(agent_id)

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            True if deleted, False if not found.

        Raises:
            ValueError: If attempting to delete the default agent.
        """
        self.init_catalog()

        existing = self.get_by_id(agent_id)
        if existing is None:
            return False

        if existing.is_default:
            raise ValueError("Cannot delete the default agent")

        delete_sql = f"DELETE FROM {AGENT_TABLE_NAME} WHERE id = ?"
        self.sql_store.execute(delete_sql, (agent_id,))
        logger.info("Deleted agent '%s' (name=%s)", agent_id, existing.name)
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get catalog statistics.

        Returns:
            Dictionary with agent_count and default_agent_name.
        """
        self.init_catalog()
        query = f"SELECT COUNT(*) FROM {AGENT_TABLE_NAME}"
        row = self.sql_store.fetchone(query)
        count = row[0] if row else 0

        default_agent = self.get_default()
        default_name = default_agent.name if default_agent else None

        return {
            "agent_count": count,
            "default_agent_name": default_name,
        }
