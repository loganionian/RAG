"""Tests for the ACL filter."""

import pytest

from security import ACLConfig, ACLFilter, SecurityContext


class TestACLFilter:
    """Test cases for ACLFilter."""

    def test_acl_disabled_returns_all(self):
        """When ACL is disabled, all results should pass."""
        config = ACLConfig(enforce_document_acl=False)
        filter = ACLFilter(config)
        ctx = SecurityContext(user_id="user1", roles=["reader"])

        # ACL disabled - should return input unchanged
        results = [{"metadata": {"acl_read": ["admin"]}}]
        filtered = filter.filter_results(results, ctx)
        assert len(filtered) == 1

    def test_acl_enabled_filters_by_role(self):
        """When ACL is enabled, results should be filtered by role."""
        config = ACLConfig(enforce_document_acl=True)
        filter = ACLFilter(config)
        ctx = SecurityContext(user_id="user1", roles=["reader"])

        results = [
            {"metadata": {"acl_read": ["reader"]}},  # Should pass
            {"metadata": {"acl_read": ["admin"]}},   # Should be filtered
            {"metadata": {"acl_read": ["reader", "admin"]}},  # Should pass
        ]
        filtered = filter.filter_results(results, ctx)
        assert len(filtered) == 2

    def test_default_allow_empty_acl_true(self):
        """Documents without ACL should be accessible by default."""
        config = ACLConfig(enforce_document_acl=True, default_allow_empty_acl=True)
        filter = ACLFilter(config)
        ctx = SecurityContext(user_id="user1", roles=["reader"])

        results = [
            {"metadata": {}},  # No ACL - should pass
            {"metadata": {"acl_read": None}},  # Null ACL - should pass
            {"metadata": {"acl_read": ""}},  # Empty ACL - should pass
        ]
        filtered = filter.filter_results(results, ctx)
        assert len(filtered) == 3

    def test_default_allow_empty_acl_false(self):
        """Documents without ACL should be blocked if configured."""
        config = ACLConfig(enforce_document_acl=True, default_allow_empty_acl=False)
        filter = ACLFilter(config)
        ctx = SecurityContext(user_id="user1", roles=["reader"])

        results = [
            {"metadata": {}},  # No ACL - should be blocked
            {"metadata": {"acl_read": ["reader"]}},  # Has matching ACL - should pass
        ]
        filtered = filter.filter_results(results, ctx)
        assert len(filtered) == 1

    def test_string_acl_value(self):
        """ACL can be a string instead of a list."""
        config = ACLConfig(enforce_document_acl=True)
        filter = ACLFilter(config)
        ctx = SecurityContext(user_id="user1", roles=["reader"])

        results = [
            {"metadata": {"acl_read": "reader"}},  # String ACL - should pass
        ]
        filtered = filter.filter_results(results, ctx)
        assert len(filtered) == 1

    def test_multiple_user_roles(self):
        """Users with multiple roles should access docs requiring any of them."""
        config = ACLConfig(enforce_document_acl=True)
        filter = ACLFilter(config)
        ctx = SecurityContext(user_id="user1", roles=["reader", "writer"])

        results = [
            {"metadata": {"acl_read": ["reader"]}},  # Matches reader
            {"metadata": {"acl_read": ["writer"]}},  # Matches writer
            {"metadata": {"acl_read": ["admin"]}},   # No match
        ]
        filtered = filter.filter_results(results, ctx)
        assert len(filtered) == 2

    def test_filter_bm25_results(self):
        """BM25 results should be filtered correctly."""
        config = ACLConfig(enforce_document_acl=True)
        filter = ACLFilter(config)
        ctx = SecurityContext(user_id="user1", roles=["reader"])

        chunks = ["chunk1", "chunk2", "chunk3"]
        metadatas = [
            {"acl_read": ["reader"]},
            {"acl_read": ["admin"]},
            {"acl_read": ["reader"]},
        ]
        ids = ["id1", "id2", "id3"]
        distances = [0.1, 0.2, 0.3]

        filtered_chunks, filtered_meta, filtered_ids, filtered_dist = filter.filter_bm25_results(
            chunks, metadatas, ids, distances, ctx
        )

        assert len(filtered_chunks) == 2
        assert filtered_chunks == ["chunk1", "chunk3"]
        assert filtered_ids == ["id1", "id3"]

    def test_check_document_access(self):
        """Check access for individual documents."""
        config = ACLConfig(enforce_document_acl=True)
        filter = ACLFilter(config)
        ctx = SecurityContext(user_id="user1", roles=["reader"])

        assert filter.check_document_access({"acl_read": ["reader"]}, ctx) is True
        assert filter.check_document_access({"acl_read": ["admin"]}, ctx) is False
        assert filter.check_document_access({}, ctx) is True  # Empty ACL allowed by default

    def test_build_chroma_where_disabled(self):
        """When ACL disabled, no where clause is generated."""
        config = ACLConfig(enforce_document_acl=False)
        filter = ACLFilter(config)
        ctx = SecurityContext(user_id="user1", roles=["reader"])

        where = filter.build_chroma_where(ctx)
        assert where is None

    def test_build_chroma_where_with_existing(self):
        """Existing where clauses should be combined with ACL."""
        config = ACLConfig(enforce_document_acl=True)
        filter = ACLFilter(config)
        ctx = SecurityContext(user_id="user1", roles=["reader"])

        existing = {"doc_type": "pdf"}
        where = filter.build_chroma_where(ctx, existing)

        # Should combine existing with ACL clause
        assert where is not None
        assert "$and" in where


class TestSecurityContext:
    """Test cases for SecurityContext."""

    def test_has_role(self):
        """Check if user has a specific role."""
        ctx = SecurityContext(user_id="user1", roles=["reader", "writer"])
        assert ctx.has_role("reader") is True
        assert ctx.has_role("admin") is False

    def test_has_any_role(self):
        """Check if user has any of the specified roles."""
        ctx = SecurityContext(user_id="user1", roles=["reader"])
        assert ctx.has_any_role(["reader", "writer"]) is True
        assert ctx.has_any_role(["admin", "superuser"]) is False

    def test_empty_roles(self):
        """User with no roles."""
        ctx = SecurityContext(user_id="anonymous")
        assert ctx.roles == []
        assert ctx.has_role("reader") is False
