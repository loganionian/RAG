"""Unit tests for sql_agent/query_validator.py."""

from __future__ import annotations

import pytest

from sql_agent.config import SQLAgentConfig
from sql_agent.errors import QueryValidationError
from sql_agent.query_validator import QueryValidator


class TestQueryValidator:
    """Tests for QueryValidator class."""

    @pytest.fixture
    def validator(self) -> QueryValidator:
        """Create a validator with default config."""
        return QueryValidator()

    @pytest.fixture
    def custom_validator(self) -> QueryValidator:
        """Create a validator with custom excluded tables."""
        config = SQLAgentConfig(
            excluded_tables=["secret_table", "private_data"],
            default_limit=50,
        )
        return QueryValidator(config)

    # ============================================================
    # Valid SELECT queries
    # ============================================================

    def test_simple_select_valid(self, validator: QueryValidator) -> None:
        """Test that simple SELECT queries pass validation."""
        sql = "SELECT * FROM users"
        is_valid, error = validator.validate(sql)
        assert is_valid is True
        assert error is None

    def test_select_with_where_valid(self, validator: QueryValidator) -> None:
        """Test SELECT with WHERE clause passes."""
        sql = "SELECT name, email FROM users WHERE active = true"
        is_valid, error = validator.validate(sql)
        assert is_valid is True

    def test_select_with_join_valid(self, validator: QueryValidator) -> None:
        """Test SELECT with JOIN passes."""
        sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        is_valid, error = validator.validate(sql)
        assert is_valid is True

    def test_select_with_aggregation_valid(self, validator: QueryValidator) -> None:
        """Test SELECT with aggregation passes."""
        sql = "SELECT category, COUNT(*), SUM(amount) FROM sales GROUP BY category"
        is_valid, error = validator.validate(sql)
        assert is_valid is True

    def test_select_with_subquery_valid(self, validator: QueryValidator) -> None:
        """Test SELECT with subquery passes."""
        sql = "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)"
        is_valid, error = validator.validate(sql)
        assert is_valid is True

    def test_cte_query_valid(self, validator: QueryValidator) -> None:
        """Test CTE (WITH clause) queries pass."""
        sql = """
            WITH active_users AS (
                SELECT * FROM users WHERE active = true
            )
            SELECT * FROM active_users
        """
        is_valid, error = validator.validate(sql)
        assert is_valid is True

    def test_select_with_limit_valid(self, validator: QueryValidator) -> None:
        """Test SELECT with explicit LIMIT passes."""
        sql = "SELECT * FROM users LIMIT 10"
        is_valid, error = validator.validate(sql)
        assert is_valid is True

    # ============================================================
    # Forbidden keywords
    # ============================================================

    def test_insert_rejected(self, validator: QueryValidator) -> None:
        """Test that INSERT queries are rejected."""
        sql = "INSERT INTO users (name) VALUES ('test')"
        is_valid, error = validator.validate(sql)
        assert is_valid is False
        assert "INSERT" in error

    def test_update_rejected(self, validator: QueryValidator) -> None:
        """Test that UPDATE queries are rejected."""
        sql = "UPDATE users SET name = 'test' WHERE id = 1"
        is_valid, error = validator.validate(sql)
        assert is_valid is False
        assert "UPDATE" in error

    def test_delete_rejected(self, validator: QueryValidator) -> None:
        """Test that DELETE queries are rejected."""
        sql = "DELETE FROM users WHERE id = 1"
        is_valid, error = validator.validate(sql)
        assert is_valid is False
        assert "DELETE" in error

    def test_drop_rejected(self, validator: QueryValidator) -> None:
        """Test that DROP queries are rejected."""
        sql = "DROP TABLE users"
        is_valid, error = validator.validate(sql)
        assert is_valid is False
        assert "DROP" in error

    def test_create_rejected(self, validator: QueryValidator) -> None:
        """Test that CREATE queries are rejected."""
        sql = "CREATE TABLE new_table (id INT)"
        is_valid, error = validator.validate(sql)
        assert is_valid is False
        assert "CREATE" in error

    def test_alter_rejected(self, validator: QueryValidator) -> None:
        """Test that ALTER queries are rejected."""
        sql = "ALTER TABLE users ADD COLUMN email VARCHAR"
        is_valid, error = validator.validate(sql)
        assert is_valid is False
        assert "ALTER" in error

    def test_truncate_rejected(self, validator: QueryValidator) -> None:
        """Test that TRUNCATE queries are rejected."""
        sql = "TRUNCATE TABLE users"
        is_valid, error = validator.validate(sql)
        assert is_valid is False
        assert "TRUNCATE" in error

    def test_grant_rejected(self, validator: QueryValidator) -> None:
        """Test that GRANT queries are rejected."""
        sql = "GRANT SELECT ON users TO public"
        is_valid, error = validator.validate(sql)
        assert is_valid is False
        assert "GRANT" in error

    def test_set_rejected(self, validator: QueryValidator) -> None:
        """Test that SET queries are rejected."""
        sql = "SET memory_limit = '8GB'"
        is_valid, error = validator.validate(sql)
        assert is_valid is False
        assert "SET" in error

    # ============================================================
    # Forbidden patterns
    # ============================================================

    def test_select_into_rejected(self, validator: QueryValidator) -> None:
        """Test that SELECT INTO is rejected."""
        sql = "SELECT * INTO new_table FROM users"
        is_valid, error = validator.validate(sql)
        assert is_valid is False
        assert "INTO" in error.upper() or "pattern" in error.lower()

    def test_sql_comment_double_dash_rejected(self, validator: QueryValidator) -> None:
        """Test that -- comments are rejected."""
        sql = "SELECT * FROM users -- this is a comment"
        is_valid, error = validator.validate(sql)
        assert is_valid is False
        assert "pattern" in error.lower()

    def test_sql_comment_block_rejected(self, validator: QueryValidator) -> None:
        """Test that /* */ comments are rejected."""
        sql = "SELECT * FROM users /* comment */"
        is_valid, error = validator.validate(sql)
        assert is_valid is False
        assert "pattern" in error.lower()

    def test_multiple_statements_rejected(self, validator: QueryValidator) -> None:
        """Test that multiple statements are rejected."""
        sql = "SELECT * FROM users; DELETE FROM users"
        is_valid, error = validator.validate(sql)
        assert is_valid is False

    def test_file_read_rejected(self, validator: QueryValidator) -> None:
        """Test that file reading functions are rejected."""
        sql = "SELECT * FROM read_csv('data.csv')"
        is_valid, error = validator.validate(sql)
        assert is_valid is False

    # ============================================================
    # Excluded tables
    # ============================================================

    def test_excluded_table_in_from_rejected(self, custom_validator: QueryValidator) -> None:
        """Test that queries on excluded tables are rejected."""
        sql = "SELECT * FROM secret_table"
        is_valid, error = custom_validator.validate(sql)
        assert is_valid is False
        assert "secret_table" in error

    def test_excluded_table_in_join_rejected(self, custom_validator: QueryValidator) -> None:
        """Test that JOINs on excluded tables are rejected."""
        sql = "SELECT u.* FROM users u JOIN secret_table s ON u.id = s.user_id"
        is_valid, error = custom_validator.validate(sql)
        assert is_valid is False
        assert "secret_table" in error

    def test_excluded_table_quoted_rejected(self, custom_validator: QueryValidator) -> None:
        """Test that quoted excluded table names are rejected."""
        sql = 'SELECT * FROM "secret_table"'
        is_valid, error = custom_validator.validate(sql)
        assert is_valid is False
        assert "secret_table" in error

    def test_default_excluded_tables(self, validator: QueryValidator) -> None:
        """Test that default excluded tables (_agents, _ingestion_catalog) are blocked."""
        sql = "SELECT * FROM _agents"
        is_valid, error = validator.validate(sql)
        assert is_valid is False
        assert "_agents" in error

        sql = "SELECT * FROM _ingestion_catalog"
        is_valid, error = validator.validate(sql)
        assert is_valid is False
        assert "_ingestion_catalog" in error

    # ============================================================
    # Query type validation
    # ============================================================

    def test_non_select_query_rejected(self, validator: QueryValidator) -> None:
        """Test that queries not starting with SELECT or WITH are rejected."""
        sql = "SHOW TABLES"
        is_valid, error = validator.validate(sql)
        assert is_valid is False
        assert "must start with SELECT or WITH" in error

    def test_empty_query_rejected(self, validator: QueryValidator) -> None:
        """Test that empty queries are rejected."""
        is_valid, error = validator.validate("")
        assert is_valid is False
        assert "Empty" in error

        is_valid, error = validator.validate("   ")
        assert is_valid is False
        assert "Empty" in error

    # ============================================================
    # validate_or_raise method
    # ============================================================

    def test_validate_or_raise_valid(self, validator: QueryValidator) -> None:
        """Test validate_or_raise returns SQL for valid queries."""
        sql = "SELECT * FROM users"
        result = validator.validate_or_raise(sql)
        assert "SELECT" in result

    def test_validate_or_raise_invalid(self, validator: QueryValidator) -> None:
        """Test validate_or_raise raises for invalid queries."""
        sql = "DELETE FROM users"
        with pytest.raises(QueryValidationError) as exc_info:
            validator.validate_or_raise(sql)
        assert "DELETE" in str(exc_info.value)

    # ============================================================
    # Auto-LIMIT functionality
    # ============================================================

    def test_auto_limit_added(self, validator: QueryValidator) -> None:
        """Test that LIMIT is auto-added when missing."""
        sql = "SELECT * FROM users"
        result = validator.validate_or_raise(sql)
        assert "LIMIT" in result

    def test_auto_limit_not_added_when_present(self, validator: QueryValidator) -> None:
        """Test that LIMIT is not added when already present."""
        sql = "SELECT * FROM users LIMIT 10"
        result = validator.validate_or_raise(sql)
        # Should only have one LIMIT
        assert result.count("LIMIT") == 1
        assert "LIMIT 10" in result

    def test_auto_limit_respects_config(self, custom_validator: QueryValidator) -> None:
        """Test that auto-LIMIT uses config value."""
        sql = "SELECT * FROM users"
        result = custom_validator.validate_or_raise(sql)
        assert "LIMIT 50" in result

    def test_auto_limit_removes_trailing_semicolon(self, validator: QueryValidator) -> None:
        """Test that trailing semicolons are handled correctly."""
        sql = "SELECT * FROM users;"
        result = validator.validate_or_raise(sql)
        assert "LIMIT" in result
        assert not result.endswith(";")

    # ============================================================
    # Table extraction
    # ============================================================

    def test_extract_tables_simple(self, validator: QueryValidator) -> None:
        """Test table extraction from simple query."""
        sql = "SELECT * FROM users"
        tables = validator.extract_tables(sql)
        assert "users" in tables

    def test_extract_tables_with_join(self, validator: QueryValidator) -> None:
        """Test table extraction from JOIN query."""
        sql = "SELECT * FROM users u JOIN orders o ON u.id = o.user_id"
        tables = validator.extract_tables(sql)
        assert "users" in tables
        assert "orders" in tables

    def test_extract_tables_with_quotes(self, validator: QueryValidator) -> None:
        """Test table extraction with quoted identifiers."""
        sql = 'SELECT * FROM "my_table"'
        tables = validator.extract_tables(sql)
        assert "my_table" in tables

    def test_extract_tables_multiple_joins(self, validator: QueryValidator) -> None:
        """Test table extraction from multiple JOINs."""
        sql = """
            SELECT *
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            INNER JOIN products p ON o.product_id = p.id
        """
        tables = validator.extract_tables(sql)
        assert len(tables) >= 3
        assert "users" in tables
        assert "orders" in tables
        assert "products" in tables


class TestQueryValidatorEdgeCases:
    """Edge case tests for QueryValidator."""

    @pytest.fixture
    def validator(self) -> QueryValidator:
        return QueryValidator()

    def test_case_insensitive_keyword_detection(self, validator: QueryValidator) -> None:
        """Test that keyword detection is case-insensitive."""
        for keyword in ["INSERT", "insert", "Insert", "iNsErT"]:
            sql = f"{keyword} INTO users VALUES (1)"
            is_valid, error = validator.validate(sql)
            assert is_valid is False

    def test_whitespace_handling(self, validator: QueryValidator) -> None:
        """Test that queries with various whitespace are handled."""
        sql = """
            SELECT
                *
            FROM
                users
            WHERE
                active = true
        """
        is_valid, error = validator.validate(sql)
        assert is_valid is True

    def test_unicode_in_query(self, validator: QueryValidator) -> None:
        """Test that queries with unicode characters are handled."""
        sql = "SELECT * FROM users WHERE name = 'café'"
        is_valid, error = validator.validate(sql)
        assert is_valid is True

    def test_numeric_table_name(self, validator: QueryValidator) -> None:
        """Test handling of numeric-prefixed table names."""
        sql = 'SELECT * FROM "2024_sales"'
        is_valid, error = validator.validate(sql)
        assert is_valid is True
