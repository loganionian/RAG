"""Configuration for the question router.

This module defines RouterConfig with settings for keyword patterns,
confidence thresholds, and optional LLM fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# Default SQL-related keywords (suggest structured query)
DEFAULT_SQL_KEYWORDS = [
    # Aggregation terms
    "count",
    "sum",
    "average",
    "avg",
    "total",
    "minimum",
    "maximum",
    "min",
    "max",
    # Data terms
    "table",
    "tables",
    "column",
    "columns",
    "row",
    "rows",
    "record",
    "records",
    "data",
    "database",
    # Query terms
    "query",
    "group by",
    "grouped",
    "filter",
    "sort",
    "sorted",
    "order by",
    "distinct",
    # Question patterns
    "how many",
    "how much",
    "list all",
    "show all",
    "top 5",
    "top 10",
    "top ten",
    "top five",
    "bottom",
    "highest",
    "lowest",
    "most",
    "least",
    "rank",
    "ranking",
    # Comparison
    "compare",
    "comparison",
    "versus",
    "vs",
    "between",
]

# Default document-related keywords (suggest document query)
DEFAULT_DOC_KEYWORDS = [
    # Document terms
    "document",
    "documents",
    "report",
    "reports",
    "policy",
    "policies",
    "procedure",
    "procedures",
    "guideline",
    "guidelines",
    "manual",
    "handbook",
    "whitepaper",
    "article",
    # Question patterns
    "explain",
    "describe",
    "what is",
    "what are",
    "what does",
    "according to",
    "based on",
    "mentioned in",
    "stated in",
    "says about",
    "definition of",
    "meaning of",
    # Content types
    "summary",
    "summarize",
    "overview",
    "background",
    "context",
    "history",
    "introduction",
    "conclusion",
]


@dataclass
class RouterConfig:
    """Configuration for question router.

    Attributes:
        sql_keywords: Keywords suggesting a SQL/structured query.
        doc_keywords: Keywords suggesting a document/RAG query.
        rule_confidence_threshold: Minimum confidence for rule-based classification.
            Below this threshold, classification is less certain.
        llm_fallback_enabled: Whether to use LLM for uncertain classifications.
            Currently not implemented (rule-based only).
        log_decisions: Whether to log routing decisions.
        known_tables: List of known table names for table detection.
    """

    sql_keywords: List[str] = field(default_factory=lambda: DEFAULT_SQL_KEYWORDS.copy())
    doc_keywords: List[str] = field(default_factory=lambda: DEFAULT_DOC_KEYWORDS.copy())
    rule_confidence_threshold: float = 0.7
    llm_fallback_enabled: bool = False  # Not implemented yet
    log_decisions: bool = True
    known_tables: List[str] = field(default_factory=list)
