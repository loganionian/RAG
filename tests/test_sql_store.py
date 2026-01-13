"""Unit tests for storage/sql_store.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from storage.sql_store import SQLStore, SQLStoreConfig


class TestSQLStoreConfig:
    """Tests for SQLStoreConfig dataclass."""

    def test_config_creation(self, tmp_path: Path) -> None:
        """Test basic config creation."""
        db_path = tmp_path / "test.duckdb"
        config = SQLStoreConfig(db_path=db_path)

        assert config.db_path == db_path
        assert config.read_only is False
        assert config.memory_limit == "4GB"
        assert config.threads == 4

    def test_config_with_string_path(self, tmp_path: Path) -> None:
        """Test config creation with string path."""
        db_path = str(tmp_path / "test.duckdb")
        config = SQLStoreConfig(db_path=db_path)

        assert isinstance(config.db_path, Path)

    def test_config_custom_settings(self, tmp_path: Path) -> None:
        """Test config with custom settings."""
        db_path = tmp_path / "test.duckdb"
        config = SQLStoreConfig(
            db_path=db_path,
            read_only=True,
            memory_limit="2GB",
            threads=2,
        )

        assert config.read_only is True
        assert config.memory_limit == "2GB"
        assert config.threads == 2


class TestSQLStore:
    """Tests for SQLStore class."""

    def test_connect_creates_database(self, tmp_path: Path) -> None:
        """Test that connect creates database file."""
        db_path = tmp_path / "test.duckdb"
        config = SQLStoreConfig(db_path=db_path)
        store = SQLStore(config)

        conn = store.connect()
        assert conn is not None
        assert db_path.exists()

        store.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test context manager usage."""
        db_path = tmp_path / "test.duckdb"
        config = SQLStoreConfig(db_path=db_path)

        with SQLStore(config) as store:
            conn = store.connect()
            assert conn is not None

    def test_execute_query(self, tmp_path: Path) -> None:
        """Test executing a simple query."""
        db_path = tmp_path / "test.duckdb"
        config = SQLStoreConfig(db_path=db_path)

        with SQLStore(config) as store:
            result = store.execute("SELECT 1 + 1 as result")
            row = result.fetchone()
            assert row[0] == 2

    def test_table_operations(self, tmp_path: Path) -> None:
        """Test table creation, existence check, and drop."""
        db_path = tmp_path / "test.duckdb"
        config = SQLStoreConfig(db_path=db_path)

        with SQLStore(config) as store:
            # Create table
            store.execute("CREATE TABLE test_table (id INTEGER, name VARCHAR)")

            # Check existence
            assert store.table_exists("test_table") is True
            assert store.table_exists("nonexistent") is False

            # Get schema
            schema = store.get_table_schema("test_table")
            assert len(schema) == 2
            assert schema[0]["name"] == "id"
            assert schema[1]["name"] == "name"

            # Drop table
            store.drop_table("test_table")
            assert store.table_exists("test_table") is False

    def test_list_tables(self, tmp_path: Path) -> None:
        """Test listing all tables."""
        db_path = tmp_path / "test.duckdb"
        config = SQLStoreConfig(db_path=db_path)

        with SQLStore(config) as store:
            # Initially no tables
            assert store.list_tables() == []

            # Create some tables
            store.execute("CREATE TABLE table_a (id INTEGER)")
            store.execute("CREATE TABLE table_b (id INTEGER)")

            tables = store.list_tables()
            assert "table_a" in tables
            assert "table_b" in tables

    def test_get_table_count(self, tmp_path: Path) -> None:
        """Test getting row count of a table."""
        db_path = tmp_path / "test.duckdb"
        config = SQLStoreConfig(db_path=db_path)

        with SQLStore(config) as store:
            store.execute("CREATE TABLE test_table (id INTEGER)")
            store.execute("INSERT INTO test_table VALUES (1), (2), (3)")

            count = store.get_table_count("test_table")
            assert count == 3

    def test_fetchall_and_fetchone(self, tmp_path: Path) -> None:
        """Test fetchall and fetchone methods."""
        db_path = tmp_path / "test.duckdb"
        config = SQLStoreConfig(db_path=db_path)

        with SQLStore(config) as store:
            store.execute("CREATE TABLE test_table (id INTEGER, name VARCHAR)")
            store.execute("INSERT INTO test_table VALUES (1, 'a'), (2, 'b'), (3, 'c')")

            # Test fetchall
            rows = store.fetchall("SELECT * FROM test_table ORDER BY id")
            assert len(rows) == 3
            assert rows[0] == (1, "a")

            # Test fetchone
            row = store.fetchone("SELECT * FROM test_table WHERE id = 2")
            assert row == (2, "b")

            # Test fetchone with no results
            row = store.fetchone("SELECT * FROM test_table WHERE id = 999")
            assert row is None

    def test_parameterized_queries(self, tmp_path: Path) -> None:
        """Test parameterized query execution."""
        db_path = tmp_path / "test.duckdb"
        config = SQLStoreConfig(db_path=db_path)

        with SQLStore(config) as store:
            store.execute("CREATE TABLE test_table (id INTEGER, name VARCHAR)")
            store.execute("INSERT INTO test_table VALUES (1, 'a'), (2, 'b')")

            row = store.fetchone(
                "SELECT * FROM test_table WHERE name = ?",
                ("b",),
            )
            assert row == (2, "b")

    def test_parent_directory_created(self, tmp_path: Path) -> None:
        """Test that parent directory is created if it doesn't exist."""
        db_path = tmp_path / "subdir" / "nested" / "test.duckdb"
        config = SQLStoreConfig(db_path=db_path)

        store = SQLStore(config)
        store.connect()

        assert db_path.parent.exists()
        store.close()
