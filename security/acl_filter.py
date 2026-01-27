"""ACL filter for document retrieval.

This module provides ACL-based filtering for both Chroma vector search
and BM25 lexical search results.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .config import ACLConfig, SecurityContext

logger = logging.getLogger(__name__)


class ACLFilter:
    """Filters document retrieval results based on ACL rules.

    Supports both pre-filtering via Chroma's where clause and post-filtering
    for BM25 results which don't support native metadata filtering.
    """

    def __init__(self, config: Optional[ACLConfig] = None) -> None:
        """Initialize the ACL filter.

        Args:
            config: ACL configuration. Uses defaults if not provided.
        """
        self.config = config or ACLConfig()

    def build_chroma_where(
        self,
        security_context: SecurityContext,
        existing_where: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a Chroma where clause for ACL filtering.

        Creates a filter that allows documents where:
        - The user's roles intersect with the document's ACL roles, OR
        - The document has no ACL metadata (if default_allow_empty_acl is True)

        Args:
            security_context: Security context with user roles.
            existing_where: Optional existing where clause to combine with.

        Returns:
            Combined where clause, or None if no filtering needed.
        """
        if not self.config.enforce_document_acl:
            return existing_where

        acl_field = self.config.acl_metadata_field
        user_roles = security_context.roles

        if not user_roles:
            # No roles - only allow documents with empty ACL (if configured)
            if self.config.default_allow_empty_acl:
                # Documents without ACL field are allowed
                # Chroma doesn't support "field doesn't exist" directly,
                # so we can't express this perfectly. Return None to skip filtering
                # and rely on post-filtering for stricter enforcement.
                logger.debug(
                    "User %s has no roles; relying on default_allow_empty_acl",
                    security_context.user_id,
                )
                return existing_where
            else:
                # Deny all - return impossible condition
                return {"$and": [{"_impossible_field": "deny_all"}]}

        # Build ACL filter: document's acl_read contains any of user's roles
        # Chroma uses $contains for array membership
        acl_conditions = []
        for role in user_roles:
            acl_conditions.append({acl_field: {"$contains": role}})

        if self.config.default_allow_empty_acl:
            # Also allow documents without ACL
            # Note: Chroma doesn't support "field is null" directly
            # This is a limitation - we'll handle it in post-filtering
            pass

        # Combine with OR
        if len(acl_conditions) == 1:
            acl_where = acl_conditions[0]
        else:
            acl_where = {"$or": acl_conditions}

        # Combine with existing where clause
        if existing_where:
            return {"$and": [existing_where, acl_where]}
        return acl_where

    def filter_results(
        self,
        results: List[Dict[str, Any]],
        security_context: SecurityContext,
    ) -> List[Dict[str, Any]]:
        """Post-filter retrieval results based on ACL.

        Used for BM25 results which don't support native metadata filtering.

        Args:
            results: List of result dictionaries with 'metadata' field.
            security_context: Security context with user roles.

        Returns:
            Filtered list of results the user is allowed to access.
        """
        if not self.config.enforce_document_acl:
            return results

        filtered = []
        user_roles = set(security_context.roles)
        acl_field = self.config.acl_metadata_field

        for result in results:
            metadata = result.get("metadata", {})
            doc_acl = metadata.get(acl_field)

            if self._check_access(doc_acl, user_roles):
                filtered.append(result)

        logger.debug(
            "ACL filtered %d -> %d results for user %s",
            len(results),
            len(filtered),
            security_context.user_id,
        )
        return filtered

    def filter_bm25_results(
        self,
        chunks: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        distances: List[float],
        security_context: SecurityContext,
    ) -> tuple[List[str], List[Dict[str, Any]], List[str], List[float]]:
        """Filter BM25 retrieval results based on ACL.

        Filters parallel lists from BM25 search results.

        Args:
            chunks: List of chunk texts.
            metadatas: List of chunk metadata dictionaries.
            ids: List of chunk IDs.
            distances: List of distance/score values.
            security_context: Security context with user roles.

        Returns:
            Tuple of filtered (chunks, metadatas, ids, distances).
        """
        if not self.config.enforce_document_acl:
            return chunks, metadatas, ids, distances

        user_roles = set(security_context.roles)
        acl_field = self.config.acl_metadata_field

        filtered_chunks = []
        filtered_metadatas = []
        filtered_ids = []
        filtered_distances = []

        for chunk, metadata, chunk_id, distance in zip(
            chunks, metadatas, ids, distances
        ):
            doc_acl = metadata.get(acl_field)

            if self._check_access(doc_acl, user_roles):
                filtered_chunks.append(chunk)
                filtered_metadatas.append(metadata)
                filtered_ids.append(chunk_id)
                filtered_distances.append(distance)

        logger.debug(
            "ACL filtered BM25 %d -> %d results for user %s",
            len(chunks),
            len(filtered_chunks),
            security_context.user_id,
        )
        return filtered_chunks, filtered_metadatas, filtered_ids, filtered_distances

    def _check_access(
        self,
        doc_acl: Optional[Any],
        user_roles: set,
    ) -> bool:
        """Check if user roles grant access to a document.

        Args:
            doc_acl: Document's ACL value (list of roles, string, or None).
            user_roles: Set of user's role names.

        Returns:
            True if access is allowed, False otherwise.
        """
        # No ACL on document
        if doc_acl is None or doc_acl == "":
            return self.config.default_allow_empty_acl

        # Normalize ACL to list
        if isinstance(doc_acl, str):
            doc_roles = [doc_acl]
        elif isinstance(doc_acl, list):
            doc_roles = doc_acl
        else:
            logger.warning("Unexpected ACL type: %s", type(doc_acl))
            return self.config.default_allow_empty_acl

        # Check for intersection
        return bool(user_roles & set(doc_roles))

    def check_document_access(
        self,
        metadata: Dict[str, Any],
        security_context: SecurityContext,
    ) -> bool:
        """Check if a user can access a specific document.

        Args:
            metadata: Document metadata dictionary.
            security_context: Security context with user roles.

        Returns:
            True if the user can access the document.
        """
        if not self.config.enforce_document_acl:
            return True

        doc_acl = metadata.get(self.config.acl_metadata_field)
        return self._check_access(doc_acl, set(security_context.roles))
