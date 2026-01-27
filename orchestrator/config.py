"""Configuration for the orchestrator.

This module defines the OrchestratorConfig dataclass with settings
for graph execution behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class OrchestratorConfig:
    """Configuration for the RAG orchestrator.

    Controls graph execution behavior including optional critic node,
    retry limits, and default query parameters.

    Attributes:
        enable_critic: Whether to enable the critic node for answer evaluation.
            Disabled by default as it's a placeholder for future enhancement.
        max_revisions: Maximum number of revision cycles when critic requests changes.
            Only relevant when enable_critic is True.
        default_k: Default number of chunks to retrieve in document search.
        default_search_mode: Default search mode for document queries.
        default_max_rows: Default maximum rows for SQL queries.
        fallback_on_error: Whether to continue with other agents if one fails.
            When True, errors in docs_agent won't abort sql_agent execution.
        parallel_hybrid: Whether to run docs and SQL agents in parallel for hybrid queries.
            Currently not implemented (reserved for future enhancement).
    """

    enable_critic: bool = False
    max_revisions: int = 1
    default_k: int = 5
    default_search_mode: Literal["vector", "lexical", "hybrid"] = "hybrid"
    default_max_rows: int = 100
    fallback_on_error: bool = True
    parallel_hybrid: bool = False  # Reserved for future parallel execution
