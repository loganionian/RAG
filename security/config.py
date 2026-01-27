"""Configuration for security and access control.

This module defines security context and ACL configuration dataclasses
for document and table-level access control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SecurityContext:
    """Security context for a request.

    Contains user identity and role information for access control decisions.

    Attributes:
        user_id: Unique identifier for the user making the request.
        roles: List of role names assigned to the user.
    """

    user_id: str
    roles: List[str] = field(default_factory=list)

    def has_role(self, role: str) -> bool:
        """Check if the user has a specific role.

        Args:
            role: Role name to check.

        Returns:
            True if the user has the role, False otherwise.
        """
        return role in self.roles

    def has_any_role(self, roles: List[str]) -> bool:
        """Check if the user has any of the specified roles.

        Args:
            roles: List of role names to check.

        Returns:
            True if the user has at least one of the roles.
        """
        return bool(set(self.roles) & set(roles))


@dataclass
class ACLConfig:
    """Configuration for document ACL filtering.

    Controls whether and how ACL filtering is applied to document retrieval.

    Attributes:
        enforce_document_acl: Whether to enforce ACL filtering on documents.
            Disabled by default for backward compatibility.
        enforce_table_acl: Whether to enforce ACL filtering on SQL tables.
            Disabled by default for backward compatibility.
        default_allow_empty_acl: Whether documents without ACL metadata
            should be accessible to all users. True means public by default.
        acl_metadata_field: Name of the metadata field containing ACL info.
    """

    enforce_document_acl: bool = False
    enforce_table_acl: bool = False
    default_allow_empty_acl: bool = True
    acl_metadata_field: str = "acl_read"


@dataclass
class TableACLConfig:
    """Configuration for SQL table access control.

    Maps roles to allowed tables for fine-grained data access.

    Attributes:
        role_table_map: Mapping from role names to list of allowed tables.
            Use ["*"] to allow access to all tables.
            Example: {"analyst": ["sales", "products"], "admin": ["*"]}
        default_allow_no_mapping: Whether users without role mappings
            can access all tables. False means deny by default.
    """

    role_table_map: Dict[str, List[str]] = field(default_factory=dict)
    default_allow_no_mapping: bool = True

    def get_allowed_tables_for_roles(self, roles: List[str]) -> Optional[List[str]]:
        """Get the list of tables allowed for a set of roles.

        Args:
            roles: List of role names to check.

        Returns:
            List of allowed table names, or None if all tables are allowed
            (either via "*" wildcard or default_allow_no_mapping).
        """
        if not self.role_table_map:
            # No ACL config - allow all if default_allow_no_mapping is True
            return None if self.default_allow_no_mapping else []

        allowed_tables = set()
        has_mapping = False

        for role in roles:
            if role in self.role_table_map:
                has_mapping = True
                tables = self.role_table_map[role]
                if "*" in tables:
                    # Wildcard - allow all tables
                    return None
                allowed_tables.update(tables)

        if not has_mapping:
            # No matching roles - use default policy
            return None if self.default_allow_no_mapping else []

        return list(allowed_tables)
