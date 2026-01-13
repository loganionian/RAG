"""Storage package for SQL tabular data storage.

This package provides DuckDB-based storage for tabular spreadsheet data,
enabling SQL queries over structured data ingested from CSV/Excel files.
"""
from .sql_store import SQLStore, SQLStoreConfig
from .metadata_catalog import MetadataCatalog, CatalogEntry

__all__ = [
    "SQLStore",
    "SQLStoreConfig",
    "MetadataCatalog",
    "CatalogEntry",
]
