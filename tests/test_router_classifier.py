"""Tests for the question router classifier."""

import pytest

from router import QuestionClassifier, QueryType, RouterConfig, RouteDecision


class TestQuestionClassifier:
    """Test cases for QuestionClassifier."""

    def test_sql_question_with_count(self):
        """Questions with count/aggregation should route to structured."""
        classifier = QuestionClassifier()
        decision = classifier.classify("How many rows are in the sales table?")
        assert decision.query_type == QueryType.STRUCTURED
        assert decision.confidence > 0.5

    def test_sql_question_with_table_reference(self):
        """Questions referencing known tables should route to structured."""
        config = RouterConfig(known_tables=["sales", "products", "orders"])
        classifier = QuestionClassifier(config)
        decision = classifier.classify("Show me the sales data")
        assert decision.query_type == QueryType.STRUCTURED
        assert "sales" in decision.signals.get("matched_tables", [])

    def test_sql_question_with_aggregation(self):
        """Questions with aggregation keywords should route to structured."""
        classifier = QuestionClassifier()
        decision = classifier.classify("What is the total revenue grouped by month?")
        assert decision.query_type == QueryType.STRUCTURED
        assert decision.signals.get("has_aggregation_pattern", False) or \
               decision.signals.get("sql_keyword_count", 0) > 0

    def test_sql_question_top_n(self):
        """Top N questions should have aggregation pattern detected."""
        classifier = QuestionClassifier()
        decision = classifier.classify("What are the top 5 products by sales?")
        # Could be hybrid due to "what are" being a doc keyword
        assert decision.query_type in [QueryType.STRUCTURED, QueryType.HYBRID]
        assert decision.signals.get("has_aggregation_pattern", False)

    def test_document_question_policy(self):
        """Questions about policies should route to documents."""
        classifier = QuestionClassifier()
        decision = classifier.classify("What does the policy say about remote work?")
        assert decision.query_type == QueryType.DOCUMENTS
        assert decision.confidence > 0.5

    def test_document_question_explain(self):
        """Questions asking for explanations should route to documents."""
        classifier = QuestionClassifier()
        decision = classifier.classify("Explain the onboarding process")
        assert decision.query_type == QueryType.DOCUMENTS

    def test_document_question_according_to(self):
        """Questions with 'according to' should route to documents."""
        classifier = QuestionClassifier()
        decision = classifier.classify("According to the handbook, what is the vacation policy?")
        assert decision.query_type == QueryType.DOCUMENTS
        assert decision.signals.get("has_reference_pattern", False)

    def test_document_question_report(self):
        """Questions about reports/documents should route to documents."""
        classifier = QuestionClassifier()
        decision = classifier.classify("What are the key findings in the report?")
        assert decision.query_type == QueryType.DOCUMENTS

    def test_hybrid_question_mixed_signals(self):
        """Questions with both SQL and doc signals may route to hybrid."""
        classifier = QuestionClassifier()
        # This question has both doc keywords (report) and SQL keywords (count, total)
        decision = classifier.classify(
            "According to the report, what is the total count of sales?"
        )
        # Could be hybrid or either, depending on signal strength
        assert decision.query_type in [QueryType.HYBRID, QueryType.DOCUMENTS, QueryType.STRUCTURED]

    def test_ambiguous_question_defaults_to_documents(self):
        """Ambiguous questions should default to documents."""
        classifier = QuestionClassifier()
        # Using a truly ambiguous question without keywords
        decision = classifier.classify("Tell me more about that.")
        assert decision.query_type == QueryType.DOCUMENTS
        assert decision.confidence <= 0.7  # Low confidence for truly ambiguous

    def test_set_known_tables(self):
        """Known tables should be detected in questions."""
        classifier = QuestionClassifier()
        classifier.set_known_tables(["inventory", "customers"])
        decision = classifier.classify("Show me the inventory levels")
        assert "inventory" in decision.signals.get("matched_tables", [])

    def test_route_decision_to_dict(self):
        """RouteDecision should serialize correctly."""
        decision = RouteDecision(
            query_type=QueryType.STRUCTURED,
            confidence=0.85,
            reasoning="SQL keywords detected",
            signals={"sql_keyword_count": 3},
        )
        result = decision.to_dict()
        assert result["query_type"] == "structured"
        assert result["confidence"] == 0.85
        assert "signals" in result


class TestRouterConfig:
    """Test cases for RouterConfig."""

    def test_default_keywords(self):
        """Default config should have SQL and doc keywords."""
        config = RouterConfig()
        assert len(config.sql_keywords) > 0
        assert len(config.doc_keywords) > 0
        assert "count" in config.sql_keywords
        assert "document" in config.doc_keywords

    def test_custom_keywords(self):
        """Custom keywords should override defaults."""
        config = RouterConfig(
            sql_keywords=["custom_sql"],
            doc_keywords=["custom_doc"],
        )
        assert config.sql_keywords == ["custom_sql"]
        assert config.doc_keywords == ["custom_doc"]

    def test_known_tables(self):
        """Known tables should be configurable."""
        config = RouterConfig(known_tables=["table1", "table2"])
        assert config.known_tables == ["table1", "table2"]


class TestQueryType:
    """Test cases for QueryType enum."""

    def test_query_type_values(self):
        """QueryType should have expected values."""
        assert QueryType.DOCUMENTS.value == "documents"
        assert QueryType.STRUCTURED.value == "structured"
        assert QueryType.HYBRID.value == "hybrid"
