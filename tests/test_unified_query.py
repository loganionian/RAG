"""Tests for the unified query endpoint."""

import pytest
from unittest.mock import MagicMock, patch

from api.schemas import (
    SecurityContextRequest,
    UnifiedQueryRequest,
    UnifiedQueryResponse,
    RoutingMetadata,
)
from router import QueryType, QuestionClassifier, RouteDecision
from security import ACLFilter, AuditLogger, SecurityContext, TableACL


class TestUnifiedQueryRequest:
    """Test cases for UnifiedQueryRequest schema."""

    def test_minimal_request(self):
        """Minimal request with just a question."""
        request = UnifiedQueryRequest(question="What is the policy?")
        assert request.question == "What is the policy?"
        assert request.k == 5  # default
        assert request.search_mode == "hybrid"  # default
        assert request.force_route is None
        assert request.security_context is None

    def test_full_request(self):
        """Full request with all options."""
        request = UnifiedQueryRequest(
            question="How many rows in sales?",
            security_context=SecurityContextRequest(
                user_id="user1",
                roles=["analyst"],
            ),
            k=10,
            search_mode="vector",
            force_route="structured",
            rerank=True,
        )
        assert request.question == "How many rows in sales?"
        assert request.security_context.user_id == "user1"
        assert request.security_context.roles == ["analyst"]
        assert request.k == 10
        assert request.search_mode == "vector"
        assert request.force_route == "structured"
        assert request.rerank is True

    def test_invalid_force_route(self):
        """Invalid force_route values should be caught by validation."""
        # Note: Validation happens at the API level, not schema level
        request = UnifiedQueryRequest(
            question="test",
            force_route="invalid",  # Will be validated by endpoint
        )
        assert request.force_route == "invalid"


class TestRoutingMetadata:
    """Test cases for RoutingMetadata schema."""

    def test_routing_metadata(self):
        """RoutingMetadata should serialize correctly."""
        metadata = RoutingMetadata(
            query_type="documents",
            confidence=0.85,
            reasoning="Document keywords detected",
            routing_time_ms=5.5,
            forced=False,
        )
        assert metadata.query_type == "documents"
        assert metadata.confidence == 0.85
        assert metadata.routing_time_ms == 5.5
        assert metadata.forced is False

    def test_forced_routing_metadata(self):
        """Forced routing should be marked."""
        metadata = RoutingMetadata(
            query_type="structured",
            confidence=1.0,
            reasoning="Forced to structured",
            routing_time_ms=0.0,
            forced=True,
        )
        assert metadata.forced is True


class TestClassifierIntegration:
    """Integration tests for classifier with unified query."""

    def test_classifier_routes_sql_question(self):
        """SQL questions should be routed correctly."""
        classifier = QuestionClassifier()
        decision = classifier.classify("How many products are in the database?")
        assert decision.query_type == QueryType.STRUCTURED

    def test_classifier_routes_doc_question(self):
        """Document questions should be routed correctly."""
        classifier = QuestionClassifier()
        decision = classifier.classify("What does the handbook say about PTO?")
        assert decision.query_type == QueryType.DOCUMENTS

    def test_classifier_with_known_tables(self):
        """Classifier should detect known table names."""
        from router import RouterConfig
        config = RouterConfig(known_tables=["employees", "departments"])
        classifier = QuestionClassifier(config)

        decision = classifier.classify("Show me data from employees")
        assert "employees" in decision.signals.get("matched_tables", [])


class TestSecurityIntegration:
    """Integration tests for security with unified query."""

    def test_security_context_conversion(self):
        """SecurityContextRequest should convert to SecurityContext."""
        request = SecurityContextRequest(user_id="user1", roles=["reader", "writer"])
        ctx = SecurityContext(
            user_id=request.user_id,
            roles=request.roles,
        )
        assert ctx.user_id == "user1"
        assert ctx.roles == ["reader", "writer"]

    def test_acl_filter_integration(self):
        """ACL filter should work with unified query flow."""
        from security import ACLConfig

        config = ACLConfig(enforce_document_acl=True)
        filter = ACLFilter(config)
        ctx = SecurityContext(user_id="user1", roles=["analyst"])

        # Simulate filtering results
        results = [
            {"metadata": {"acl_read": ["analyst"]}},
            {"metadata": {"acl_read": ["admin"]}},
        ]
        filtered = filter.filter_results(results, ctx)
        assert len(filtered) == 1

    def test_table_acl_integration(self):
        """Table ACL should work with unified query flow."""
        from security import TableACLConfig

        config = TableACLConfig(
            role_table_map={"analyst": ["sales", "products"]}
        )
        acl = TableACL(config)
        ctx = SecurityContext(user_id="user1", roles=["analyst"])

        available = ["sales", "products", "orders"]
        allowed = acl.get_allowed_tables(available, ctx)
        assert "orders" not in allowed


class TestAuditLoggerIntegration:
    """Integration tests for audit logger."""

    def test_audit_logger_logs_route_decision(self, tmp_path):
        """Audit logger should log routing decisions."""
        import json

        logger = AuditLogger(log_dir=tmp_path, enabled=True)
        ctx = SecurityContext(user_id="user1", roles=["reader"])

        logger.log_route_decision(
            security_context=ctx,
            query_type="documents",
            confidence=0.85,
            reasoning="Document keywords detected",
            question="What is the policy?",
        )

        # Check log file
        log_file = tmp_path / "audit.jsonl"
        assert log_file.exists()

        with open(log_file) as f:
            line = f.readline()
            event = json.loads(line)

        assert event["event_type"] == "route_decision"
        assert event["user_id"] == "user1"
        assert event["query_type"] == "documents"

    def test_audit_logger_logs_access_denied(self, tmp_path):
        """Audit logger should log access denied events."""
        import json

        logger = AuditLogger(log_dir=tmp_path, enabled=True)
        ctx = SecurityContext(user_id="user1", roles=["reader"])

        logger.log_access_denied(
            security_context=ctx,
            resource_type="table",
            resource_id="secret_data",
            reason="User lacks admin role",
        )

        log_file = tmp_path / "audit.jsonl"
        with open(log_file) as f:
            line = f.readline()
            event = json.loads(line)

        assert event["event_type"] == "access_denied"
        assert event["resource_type"] == "table"
        assert event["reason"] == "User lacks admin role"

    def test_audit_logger_disabled(self, tmp_path):
        """Disabled audit logger should not write files."""
        logger = AuditLogger(log_dir=tmp_path, enabled=False)
        ctx = SecurityContext(user_id="user1", roles=["reader"])

        logger.log_route_decision(
            security_context=ctx,
            query_type="documents",
            confidence=0.85,
            reasoning="Test",
            question="Test?",
        )

        log_file = tmp_path / "audit.jsonl"
        assert not log_file.exists()
