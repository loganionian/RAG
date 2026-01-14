"""Token usage tracking and cost estimation for LLM operations.

This module provides cost controls for LLM-based operations like
dataset summary generation, enabling budget limits and usage reporting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CostConfig:
    """Configuration for cost tracking.

    Attributes:
        max_tokens_per_request: Maximum tokens allowed per single request.
        max_total_tokens_per_run: Maximum total tokens for entire run.
        cost_per_1k_input_tokens: Cost per 1000 input tokens (provider-specific).
        cost_per_1k_output_tokens: Cost per 1000 output tokens (provider-specific).
    """

    max_tokens_per_request: int = 500
    max_total_tokens_per_run: int = 50000
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0


@dataclass
class UsageRecord:
    """Single usage record for tracking."""

    input_tokens: int
    output_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)
    operation: Optional[str] = None


@dataclass
class CostReport:
    """Summary report of token usage and costs.

    Attributes:
        total_requests: Number of LLM requests made.
        total_input_tokens: Sum of input tokens across all requests.
        total_output_tokens: Sum of output tokens across all requests.
        total_tokens: Combined input + output tokens.
        estimated_cost: Estimated monetary cost based on config rates.
        tokens_remaining: Tokens remaining before hitting budget limit.
        budget_exceeded: Whether the budget has been exceeded.
    """

    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    estimated_cost: float
    tokens_remaining: int
    budget_exceeded: bool


class CostTracker:
    """Tracks token usage and enforces cost limits.

    Usage:
        tracker = CostTracker(CostConfig(max_total_tokens_per_run=10000))

        if tracker.can_proceed():
            # Make LLM call
            response = llm.generate(prompt)
            tracker.track_usage(
                input_tokens=response.usage["prompt_tokens"],
                output_tokens=response.usage["completion_tokens"],
                operation="generate_summary"
            )

        report = tracker.get_report()
        print(f"Total tokens used: {report.total_tokens}")
    """

    def __init__(self, config: Optional[CostConfig] = None) -> None:
        """Initialize the cost tracker.

        Args:
            config: Cost configuration. Uses defaults if not provided.
        """
        self.config = config or CostConfig()
        self._records: List[UsageRecord] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def track_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        operation: Optional[str] = None,
    ) -> None:
        """Record token usage from an LLM operation.

        Args:
            input_tokens: Number of input/prompt tokens.
            output_tokens: Number of output/completion tokens.
            operation: Optional description of the operation.
        """
        record = UsageRecord(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            operation=operation,
        )
        self._records.append(record)
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

        logger.debug(
            "Tracked usage: input=%d, output=%d, total=%d/%d",
            input_tokens,
            output_tokens,
            self.total_tokens,
            self.config.max_total_tokens_per_run,
        )

    @property
    def total_tokens(self) -> int:
        """Get total tokens used so far."""
        return self._total_input_tokens + self._total_output_tokens

    @property
    def tokens_remaining(self) -> int:
        """Get remaining tokens before hitting budget limit."""
        return max(0, self.config.max_total_tokens_per_run - self.total_tokens)

    def can_proceed(self, estimated_tokens: int = 0) -> bool:
        """Check if another operation can proceed within budget.

        Args:
            estimated_tokens: Estimated tokens for the next operation.

        Returns:
            True if within budget, False otherwise.
        """
        if estimated_tokens > 0:
            return (self.total_tokens + estimated_tokens) <= self.config.max_total_tokens_per_run
        return self.total_tokens < self.config.max_total_tokens_per_run

    def would_exceed_budget(self, estimated_tokens: int) -> bool:
        """Check if an operation would exceed the budget.

        Args:
            estimated_tokens: Estimated tokens for the operation.

        Returns:
            True if operation would exceed budget.
        """
        return (self.total_tokens + estimated_tokens) > self.config.max_total_tokens_per_run

    def get_report(self) -> CostReport:
        """Generate a usage report.

        Returns:
            CostReport with usage statistics and cost estimates.
        """
        input_cost = (self._total_input_tokens / 1000) * self.config.cost_per_1k_input_tokens
        output_cost = (self._total_output_tokens / 1000) * self.config.cost_per_1k_output_tokens
        estimated_cost = input_cost + output_cost

        return CostReport(
            total_requests=len(self._records),
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            total_tokens=self.total_tokens,
            estimated_cost=round(estimated_cost, 6),
            tokens_remaining=self.tokens_remaining,
            budget_exceeded=self.total_tokens >= self.config.max_total_tokens_per_run,
        )

    def reset(self) -> None:
        """Reset the tracker, clearing all recorded usage."""
        self._records.clear()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        logger.debug("Cost tracker reset")

    def get_records(self) -> List[UsageRecord]:
        """Get all usage records.

        Returns:
            List of UsageRecord objects.
        """
        return self._records.copy()
