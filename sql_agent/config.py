"""Configuration for the SQL Agent.

This module defines the SQLAgentConfig dataclass with all settings for
the SQL agent, including safety guardrails and LLM parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

DEFAULT_SQL_SYSTEM_PROMPT = """You are a SQL expert. Generate a single, valid SQL SELECT query to answer the user's question.

## Database Schema:
{schema}

## Rules:
1. ONLY generate SELECT queries - no INSERT, UPDATE, DELETE, DROP, CREATE, or ALTER
2. Use proper table and column names exactly as shown in the schema
3. Add LIMIT {max_rows} unless the user explicitly asks for all rows
4. Use double quotes for identifiers with special characters or spaces
5. For text matching, prefer ILIKE for case-insensitive search
6. If you cannot answer the question with the available data, respond ONLY with: CANNOT_ANSWER: <reason>
7. Return ONLY the SQL query, no explanations or markdown formatting

## Question: {question}"""


@dataclass
class SQLAgentConfig:
    """Configuration for SQL Agent.

    Attributes:
        db_path: Path to the DuckDB database file.
        read_only: Always open database in read-only mode for safety.
        excluded_tables: Tables to exclude from queries (system tables).
        max_rows_preview: Number of sample rows to show in schema prompt.
        max_result_rows: Maximum rows to return from queries.
        query_timeout: Query execution timeout in seconds.
        max_tokens: Maximum tokens for LLM response.
        temperature: LLM temperature (0.0 for deterministic SQL).
        system_prompt: System prompt template for SQL generation.
        auto_limit: Automatically add LIMIT if missing.
        default_limit: Default LIMIT value when auto-adding.
    """

    db_path: Path = field(
        default_factory=lambda: Path("data/vectorstore/catalog.duckdb")
    )
    read_only: bool = True  # SAFETY: Always read-only
    excluded_tables: List[str] = field(
        default_factory=lambda: ["_agents", "_ingestion_catalog"]
    )
    max_rows_preview: int = 3  # Sample rows in schema prompt
    max_result_rows: int = 100
    query_timeout: int = 30
    max_tokens: int = 500
    temperature: float = 0.0  # Deterministic for SQL
    system_prompt: str = DEFAULT_SQL_SYSTEM_PROMPT
    auto_limit: bool = True
    default_limit: int = 100

    def __post_init__(self) -> None:
        """Ensure db_path is a Path object and read_only is True."""
        if isinstance(self.db_path, str):
            self.db_path = Path(self.db_path)
        # SAFETY: Force read-only mode
        self.read_only = True
