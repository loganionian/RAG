"""Data models for the question router.

This module defines the QueryType enum and RouteDecision dataclass
for representing routing decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class QueryType(Enum):
    """Type of query for routing decisions.

    Attributes:
        DOCUMENTS: Query best answered by RAG over documents.
        STRUCTURED: Query best answered by SQL over tables.
        HYBRID: Query that may benefit from both document and SQL results.
    """

    DOCUMENTS = "documents"
    STRUCTURED = "structured"
    HYBRID = "hybrid"


@dataclass
class RouteDecision:
    """Result of question classification.

    Attributes:
        query_type: Classified query type.
        confidence: Confidence score for the classification (0.0 to 1.0).
        reasoning: Human-readable explanation for the classification.
        signals: Dictionary of signals/features used in classification.
    """

    query_type: QueryType
    confidence: float
    reasoning: str
    signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_type": self.query_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "signals": self.signals,
        }
