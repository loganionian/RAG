"""Storage package for SQL tabular data storage.

This package provides DuckDB-based storage for tabular spreadsheet data,
enabling SQL queries over structured data ingested from CSV/Excel files.
It also provides storage for LLM agent configurations.
"""
from .sql_store import SQLStore, SQLStoreConfig
from .metadata_catalog import MetadataCatalog, CatalogEntry
from .agent_catalog import AgentCatalog, AgentEntry, DEFAULT_SYSTEM_PROMPT

__all__ = [
    "SQLStore",
    "SQLStoreConfig",
    "MetadataCatalog",
    "CatalogEntry",
    "AgentCatalog",
    "AgentEntry",
    "DEFAULT_SYSTEM_PROMPT",
]
