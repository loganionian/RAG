"""Agent management API endpoints.

Provides CRUD operations for LLM agent configurations.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import (
    AgentCreate,
    AgentDeleteResponse,
    AgentResponse,
    AgentsListResponse,
    AgentUpdate,
)
from storage import AgentCatalog, AgentEntry, SQLStore, SQLStoreConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"])

# Default database path (can be overridden via environment variable)
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "data/vectorstore"))
DB_PATH = VECTORSTORE_DIR / "catalog.duckdb"


def _get_catalog() -> AgentCatalog:
    """Get an initialized AgentCatalog instance.

    Returns:
        AgentCatalog connected to the database.
    """
    config = SQLStoreConfig(db_path=DB_PATH)
    sql_store = SQLStore(config)
    catalog = AgentCatalog(sql_store)
    catalog.init_catalog()
    return catalog


def _entry_to_response(entry: AgentEntry) -> AgentResponse:
    """Convert an AgentEntry to an AgentResponse.

    Args:
        entry: AgentEntry from the catalog.

    Returns:
        AgentResponse for the API.
    """
    return AgentResponse(
        id=entry.id,
        name=entry.name,
        description=entry.description,
        system_prompt=entry.system_prompt,
        temperature=entry.temperature,
        max_tokens=entry.max_tokens,
        is_default=entry.is_default,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


@router.get("", response_model=AgentsListResponse)
async def list_agents() -> AgentsListResponse:
    """List all agents.

    Returns all configured LLM agents, ordered by name.
    """
    try:
        catalog = _get_catalog()
        entries = catalog.list_agents()
        agents = [_entry_to_response(entry) for entry in entries]
        return AgentsListResponse(agents=agents, total=len(agents))
    except Exception as e:
        logger.exception("Failed to list agents")
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {e}") from e


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str) -> AgentResponse:
    """Get a single agent by ID.

    Args:
        agent_id: The unique agent identifier.

    Returns:
        The agent details.

    Raises:
        HTTPException: 404 if agent not found.
    """
    try:
        catalog = _get_catalog()
        entry = catalog.get_by_id(agent_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        return _entry_to_response(entry)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get agent %s", agent_id)
        raise HTTPException(status_code=500, detail=f"Failed to get agent: {e}") from e


@router.post("", response_model=AgentResponse, status_code=201)
async def create_agent(agent: AgentCreate) -> AgentResponse:
    """Create a new agent.

    Args:
        agent: Agent configuration to create.

    Returns:
        The created agent.

    Raises:
        HTTPException: 409 if agent with same name already exists.
    """
    try:
        catalog = _get_catalog()
        entry = catalog.create_agent(
            name=agent.name,
            system_prompt=agent.system_prompt,
            description=agent.description,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
        )
        return _entry_to_response(entry)
    except ValueError as e:
        # Name conflict
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to create agent")
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {e}") from e


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: str, agent: AgentUpdate) -> AgentResponse:
    """Update an existing agent.

    Args:
        agent_id: The unique agent identifier.
        agent: Fields to update (only provided fields are updated).

    Returns:
        The updated agent.

    Raises:
        HTTPException: 404 if agent not found, 409 if name conflict.
    """
    try:
        catalog = _get_catalog()

        # Check if any fields are provided
        update_data = agent.model_dump(exclude_unset=True)
        if not update_data:
            # No fields provided - return existing agent
            entry = catalog.get_by_id(agent_id)
            if entry is None:
                raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
            return _entry_to_response(entry)

        entry = catalog.update_agent(
            agent_id=agent_id,
            name=agent.name,
            description=agent.description,
            system_prompt=agent.system_prompt,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
        )
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        return _entry_to_response(entry)
    except HTTPException:
        raise
    except ValueError as e:
        # Name conflict
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to update agent %s", agent_id)
        raise HTTPException(status_code=500, detail=f"Failed to update agent: {e}") from e


@router.delete("/{agent_id}", response_model=AgentDeleteResponse)
async def delete_agent(agent_id: str) -> AgentDeleteResponse:
    """Delete an agent.

    Args:
        agent_id: The unique agent identifier.

    Returns:
        Confirmation of deletion.

    Raises:
        HTTPException: 404 if agent not found, 400 if attempting to delete default agent.
    """
    try:
        catalog = _get_catalog()
        deleted = catalog.delete_agent(agent_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        return AgentDeleteResponse(deleted=True)
    except HTTPException:
        raise
    except ValueError as e:
        # Attempting to delete default agent
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to delete agent %s", agent_id)
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {e}") from e
