"""Router package for query classification.

This package provides question routing to classify queries as document-centric,
data-centric (SQL), or hybrid based on question content and patterns.
"""

from .config import RouterConfig
from .models import QueryType, RouteDecision
from .classifier import QuestionClassifier

__all__ = [
    # Config
    "RouterConfig",
    # Models
    "QueryType",
    "RouteDecision",
    # Classes
    "QuestionClassifier",
]
