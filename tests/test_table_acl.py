"""Tests for the table ACL."""

import pytest

from security import SecurityContext, TableACL, TableACLConfig


class TestTableACL:
    """Test cases for TableACL."""

    def test_no_config_allows_all(self):
        """Without ACL config, all tables should be allowed."""
        config = TableACLConfig()
        acl = TableACL(config)
        ctx = SecurityContext(user_id="user1", roles=["reader"])

        available = ["sales", "products", "orders"]
        allowed = acl.get_allowed_tables(available, ctx)
        assert allowed == available

    def test_role_mapping_filters_tables(self):
        """Tables should be filtered by role mapping."""
        config = TableACLConfig(
            role_table_map={
                "analyst": ["sales", "products"],
                "admin": ["*"],
            }
        )
        acl = TableACL(config)

        # Analyst can only see sales and products
        ctx = SecurityContext(user_id="user1", roles=["analyst"])
        available = ["sales", "products", "orders", "users"]
        allowed = acl.get_allowed_tables(available, ctx)
        assert set(allowed) == {"sales", "products"}

        # Admin can see all
        ctx_admin = SecurityContext(user_id="admin1", roles=["admin"])
        allowed_admin = acl.get_allowed_tables(available, ctx_admin)
        assert set(allowed_admin) == set(available)

    def test_multiple_roles_union(self):
        """Multiple roles should have union of allowed tables."""
        config = TableACLConfig(
            role_table_map={
                "sales_team": ["sales"],
                "product_team": ["products"],
            }
        )
        acl = TableACL(config)
        ctx = SecurityContext(user_id="user1", roles=["sales_team", "product_team"])

        available = ["sales", "products", "orders"]
        allowed = acl.get_allowed_tables(available, ctx)
        assert set(allowed) == {"sales", "products"}

    def test_wildcard_allows_all(self):
        """Wildcard '*' should allow all tables."""
        config = TableACLConfig(
            role_table_map={
                "admin": ["*"],
            }
        )
        acl = TableACL(config)
        ctx = SecurityContext(user_id="admin1", roles=["admin"])

        available = ["sales", "products", "orders", "users"]
        allowed = acl.get_allowed_tables(available, ctx)
        assert allowed == available

    def test_no_matching_role(self):
        """User with no matching role respects default policy."""
        config = TableACLConfig(
            role_table_map={
                "analyst": ["sales"],
            },
            default_allow_no_mapping=True,
        )
        acl = TableACL(config)
        ctx = SecurityContext(user_id="user1", roles=["unknown_role"])

        available = ["sales", "products"]
        allowed = acl.get_allowed_tables(available, ctx)
        assert allowed == available  # default_allow_no_mapping=True

    def test_no_matching_role_deny(self):
        """User with no matching role denied if default_allow_no_mapping=False."""
        config = TableACLConfig(
            role_table_map={
                "analyst": ["sales"],
            },
            default_allow_no_mapping=False,
        )
        acl = TableACL(config)
        ctx = SecurityContext(user_id="user1", roles=["unknown_role"])

        available = ["sales", "products"]
        allowed = acl.get_allowed_tables(available, ctx)
        assert allowed == []  # Denied all

    def test_get_excluded_tables(self):
        """Get list of excluded tables."""
        config = TableACLConfig(
            role_table_map={
                "analyst": ["sales", "products"],
            }
        )
        acl = TableACL(config)
        ctx = SecurityContext(user_id="user1", roles=["analyst"])

        available = ["sales", "products", "orders", "users"]
        excluded = acl.get_excluded_tables(available, ctx)
        assert set(excluded) == {"orders", "users"}

    def test_check_table_access(self):
        """Check access for individual tables."""
        config = TableACLConfig(
            role_table_map={
                "analyst": ["sales", "products"],
            }
        )
        acl = TableACL(config)
        ctx = SecurityContext(user_id="user1", roles=["analyst"])

        assert acl.check_table_access("sales", ctx) is True
        assert acl.check_table_access("orders", ctx) is False

    def test_validate_query_tables(self):
        """Validate that query only accesses allowed tables."""
        config = TableACLConfig(
            role_table_map={
                "analyst": ["sales", "products"],
            }
        )
        acl = TableACL(config)
        ctx = SecurityContext(user_id="user1", roles=["analyst"])

        # Valid query
        is_valid, unauthorized = acl.validate_query_tables(["sales", "products"], ctx)
        assert is_valid is True
        assert unauthorized == []

        # Invalid query
        is_valid, unauthorized = acl.validate_query_tables(["sales", "orders"], ctx)
        assert is_valid is False
        assert unauthorized == ["orders"]

    def test_case_insensitive_matching(self):
        """Table matching should be case-insensitive."""
        config = TableACLConfig(
            role_table_map={
                "analyst": ["Sales", "PRODUCTS"],
            }
        )
        acl = TableACL(config)
        ctx = SecurityContext(user_id="user1", roles=["analyst"])

        available = ["sales", "products", "orders"]
        allowed = acl.get_allowed_tables(available, ctx)
        assert set(allowed) == {"sales", "products"}


class TestTableACLConfig:
    """Test cases for TableACLConfig."""

    def test_get_allowed_tables_for_roles(self):
        """Get allowed tables for a set of roles."""
        config = TableACLConfig(
            role_table_map={
                "analyst": ["sales"],
                "admin": ["*"],
            }
        )

        # Single role
        allowed = config.get_allowed_tables_for_roles(["analyst"])
        assert allowed == ["sales"]

        # Admin role returns None (all allowed)
        allowed = config.get_allowed_tables_for_roles(["admin"])
        assert allowed is None

        # No matching role with default allow
        config2 = TableACLConfig(
            role_table_map={"analyst": ["sales"]},
            default_allow_no_mapping=True,
        )
        allowed = config2.get_allowed_tables_for_roles(["unknown"])
        assert allowed is None  # All allowed
