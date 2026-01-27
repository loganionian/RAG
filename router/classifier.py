"""Question classifier for routing decisions.

This module provides rule-based question classification to determine
whether a query should be routed to documents (RAG), structured data (SQL),
or both (hybrid).
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Set

from .config import RouterConfig
from .models import QueryType, RouteDecision

logger = logging.getLogger(__name__)


class QuestionClassifier:
    """Classifies questions to determine the best query route.

    Uses rule-based classification with keyword matching and pattern detection.
    Supports optional LLM fallback for uncertain cases (not yet implemented).
    """

    def __init__(self, config: Optional[RouterConfig] = None) -> None:
        """Initialize the classifier.

        Args:
            config: Router configuration. Uses defaults if not provided.
        """
        self.config = config or RouterConfig()
        self._sql_pattern: Optional[re.Pattern] = None
        self._doc_pattern: Optional[re.Pattern] = None

    def _build_keyword_pattern(self, keywords: List[str]) -> re.Pattern:
        """Build a regex pattern for keyword matching.

        Args:
            keywords: List of keywords to match.

        Returns:
            Compiled regex pattern for word-boundary matching.
        """
        # Escape special regex characters and join with OR
        escaped = [re.escape(kw) for kw in keywords]
        # Use word boundaries for whole-word matching
        pattern = r"\b(" + "|".join(escaped) + r")\b"
        return re.compile(pattern, re.IGNORECASE)

    def _get_sql_pattern(self) -> re.Pattern:
        """Get or build the SQL keyword pattern."""
        if self._sql_pattern is None:
            self._sql_pattern = self._build_keyword_pattern(self.config.sql_keywords)
        return self._sql_pattern

    def _get_doc_pattern(self) -> re.Pattern:
        """Get or build the document keyword pattern."""
        if self._doc_pattern is None:
            self._doc_pattern = self._build_keyword_pattern(self.config.doc_keywords)
        return self._doc_pattern

    def classify(self, question: str) -> RouteDecision:
        """Classify a question to determine routing.

        Args:
            question: The user's question.

        Returns:
            RouteDecision with query type, confidence, and reasoning.
        """
        # Compute signals
        signals = self._compute_signals(question)

        # Score each query type
        sql_score = signals["sql_keyword_count"] * 2.0 + signals["table_match_count"] * 3.0
        doc_score = signals["doc_keyword_count"] * 2.0

        # Adjust based on patterns
        if signals.get("has_aggregation_pattern"):
            sql_score += 2.0
        if signals.get("has_reference_pattern"):
            doc_score += 2.0

        # Determine query type and confidence
        total_score = sql_score + doc_score
        if total_score == 0:
            # No signals - default to documents (RAG)
            return RouteDecision(
                query_type=QueryType.DOCUMENTS,
                confidence=0.5,
                reasoning="No strong signals detected; defaulting to document search",
                signals=signals,
            )

        sql_ratio = sql_score / total_score if total_score > 0 else 0
        doc_ratio = doc_score / total_score if total_score > 0 else 0

        # Check for hybrid (both signals present significantly)
        if sql_ratio > 0.3 and doc_ratio > 0.3:
            confidence = min(sql_ratio, doc_ratio) * 2  # Higher when balanced
            return RouteDecision(
                query_type=QueryType.HYBRID,
                confidence=min(0.8, confidence),
                reasoning=f"Mixed signals: SQL keywords ({signals['sql_keyword_count']}), doc keywords ({signals['doc_keyword_count']})",
                signals=signals,
            )

        # Determine winner
        if sql_score > doc_score:
            confidence = sql_ratio
            reasoning_parts = []
            if signals["sql_keyword_count"] > 0:
                reasoning_parts.append(f"{signals['sql_keyword_count']} SQL keywords")
            if signals["table_match_count"] > 0:
                reasoning_parts.append(f"matched tables: {signals.get('matched_tables', [])}")
            if signals.get("has_aggregation_pattern"):
                reasoning_parts.append("aggregation pattern detected")

            return RouteDecision(
                query_type=QueryType.STRUCTURED,
                confidence=min(0.95, confidence),
                reasoning="; ".join(reasoning_parts) or "SQL patterns detected",
                signals=signals,
            )
        else:
            confidence = doc_ratio
            reasoning_parts = []
            if signals["doc_keyword_count"] > 0:
                reasoning_parts.append(f"{signals['doc_keyword_count']} document keywords")
            if signals.get("has_reference_pattern"):
                reasoning_parts.append("document reference pattern detected")

            return RouteDecision(
                query_type=QueryType.DOCUMENTS,
                confidence=min(0.95, confidence),
                reasoning="; ".join(reasoning_parts) or "Document patterns detected",
                signals=signals,
            )

    def _compute_signals(self, question: str) -> dict:
        """Compute classification signals from a question.

        Args:
            question: The user's question.

        Returns:
            Dictionary of computed signals.
        """
        question_lower = question.lower()

        # Count keyword matches
        sql_matches = self._get_sql_pattern().findall(question_lower)
        doc_matches = self._get_doc_pattern().findall(question_lower)

        # Check for table name matches
        matched_tables = []
        if self.config.known_tables:
            for table in self.config.known_tables:
                # Check if table name appears (case-insensitive, word boundary)
                table_pattern = re.compile(r"\b" + re.escape(table) + r"\b", re.IGNORECASE)
                if table_pattern.search(question):
                    matched_tables.append(table)

        # Check for aggregation patterns
        aggregation_patterns = [
            r"how many\b",
            r"how much\b",
            r"total\s+\w+",
            r"count\s+(of|the)?\s*\w+",
            r"sum\s+(of|the)?\s*\w+",
            r"average\s+(of|the)?\s*\w+",
            r"top\s+\d+",
            r"bottom\s+\d+",
            r"(highest|lowest|most|least)\s+\w+",
        ]
        has_aggregation = any(
            re.search(p, question_lower) for p in aggregation_patterns
        )

        # Check for document reference patterns
        reference_patterns = [
            r"according to\b",
            r"based on\b",
            r"mentioned in\b",
            r"stated in\b",
            r"says about\b",
            r"what does .+ say",
            r"in the (document|report|policy|manual)",
        ]
        has_reference = any(
            re.search(p, question_lower) for p in reference_patterns
        )

        return {
            "sql_keyword_count": len(sql_matches),
            "sql_keywords_matched": list(set(sql_matches)),
            "doc_keyword_count": len(doc_matches),
            "doc_keywords_matched": list(set(doc_matches)),
            "table_match_count": len(matched_tables),
            "matched_tables": matched_tables,
            "has_aggregation_pattern": has_aggregation,
            "has_reference_pattern": has_reference,
            "question_length": len(question),
        }

    def set_known_tables(self, tables: List[str]) -> None:
        """Update the list of known table names.

        Args:
            tables: List of table names to recognize.
        """
        self.config.known_tables = tables
        logger.debug("Updated known tables: %s", tables)
