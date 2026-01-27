"""Table ACL for SQL access control.

This module provides role-based access control for SQL tables,
filtering which tables a user can query based on their roles.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set

from .config import SecurityContext, TableACLConfig

logger = logging.getLogger(__name__)


class TableACL:
    """Filters SQL table access based on user roles.

    Used to dynamically restrict which tables appear in schema prompts
    and which tables can be queried by users.
    """

    def __init__(self, config: Optional[TableACLConfig] = None) -> None:
        """Initialize the table ACL.

        Args:
            config: Table ACL configuration. Uses defaults if not provided.
        """
        self.config = config or TableACLConfig()

    def get_allowed_tables(
        self,
        available_tables: List[str],
        security_context: SecurityContext,
    ) -> List[str]:
        """Get the list of tables a user is allowed to access.

        Args:
            available_tables: List of all available table names.
            security_context: Security context with user roles.

        Returns:
            Filtered list of tables the user can access.
        """
        allowed = self.config.get_allowed_tables_for_roles(security_context.roles)

        if allowed is None:
            # All tables allowed
            logger.debug(
                "User %s has unrestricted table access",
                security_context.user_id,
            )
            return available_tables

        # Filter to only allowed tables (case-insensitive matching)
        allowed_lower = {t.lower() for t in allowed}
        filtered = [t for t in available_tables if t.lower() in allowed_lower]

        logger.debug(
            "User %s allowed tables: %s (from %d available)",
            security_context.user_id,
            filtered,
            len(available_tables),
        )
        return filtered

    def get_excluded_tables(
        self,
        available_tables: List[str],
        security_context: SecurityContext,
    ) -> List[str]:
        """Get the list of tables a user is NOT allowed to access.

        Useful for building the excluded_tables list for SchemaExtractor.

        Args:
            available_tables: List of all available table names.
            security_context: Security context with user roles.

        Returns:
            List of table names to exclude from access.
        """
        allowed = self.get_allowed_tables(available_tables, security_context)
        allowed_set = {t.lower() for t in allowed}

        excluded = [t for t in available_tables if t.lower() not in allowed_set]

        if excluded:
            logger.debug(
                "User %s excluded from tables: %s",
                security_context.user_id,
                excluded,
            )
        return excluded

    def check_table_access(
        self,
        table_name: str,
        security_context: SecurityContext,
    ) -> bool:
        """Check if a user can access a specific table.

        Args:
            table_name: Name of the table to check.
            security_context: Security context with user roles.

        Returns:
            True if the user can access the table.
        """
        allowed = self.config.get_allowed_tables_for_roles(security_context.roles)

        if allowed is None:
            # All tables allowed
            return True

        # Case-insensitive check
        return table_name.lower() in {t.lower() for t in allowed}

    def validate_query_tables(
        self,
        tables_used: List[str],
        security_context: SecurityContext,
    ) -> tuple[bool, List[str]]:
        """Validate that a query only accesses allowed tables.

        Args:
            tables_used: List of table names referenced in the query.
            security_context: Security context with user roles.

        Returns:
            Tuple of (is_valid, unauthorized_tables).
        """
        allowed = self.config.get_allowed_tables_for_roles(security_context.roles)

        if allowed is None:
            # All tables allowed
            return True, []

        allowed_lower = {t.lower() for t in allowed}
        unauthorized = [t for t in tables_used if t.lower() not in allowed_lower]

        return len(unauthorized) == 0, unauthorized
