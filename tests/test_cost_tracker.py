"""Unit tests for generation/cost_tracker.py."""
from __future__ import annotations

import pytest

from generation.cost_tracker import CostConfig, CostReport, CostTracker


class TestCostConfig:
    """Tests for CostConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CostConfig()

        assert config.max_tokens_per_request == 500
        assert config.max_total_tokens_per_run == 50000
        assert config.cost_per_1k_input_tokens == 0.0
        assert config.cost_per_1k_output_tokens == 0.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CostConfig(
            max_tokens_per_request=1000,
            max_total_tokens_per_run=100000,
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.03,
        )

        assert config.max_tokens_per_request == 1000
        assert config.max_total_tokens_per_run == 100000
        assert config.cost_per_1k_input_tokens == 0.01
        assert config.cost_per_1k_output_tokens == 0.03


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_initial_state(self) -> None:
        """Test tracker starts with zero usage."""
        tracker = CostTracker()

        assert tracker.total_tokens == 0
        assert tracker.tokens_remaining == 50000  # default max
        assert tracker.can_proceed()

    def test_track_usage(self) -> None:
        """Test tracking token usage."""
        tracker = CostTracker()

        tracker.track_usage(input_tokens=100, output_tokens=50)

        assert tracker.total_tokens == 150
        assert tracker.tokens_remaining == 50000 - 150

    def test_track_multiple_usages(self) -> None:
        """Test tracking multiple operations."""
        tracker = CostTracker()

        tracker.track_usage(input_tokens=100, output_tokens=50)
        tracker.track_usage(input_tokens=200, output_tokens=100)
        tracker.track_usage(input_tokens=300, output_tokens=150)

        assert tracker.total_tokens == 900
        assert len(tracker.get_records()) == 3

    def test_track_usage_with_operation(self) -> None:
        """Test tracking usage with operation name."""
        tracker = CostTracker()

        tracker.track_usage(input_tokens=100, output_tokens=50, operation="summarize:sales_2024")

        records = tracker.get_records()
        assert len(records) == 1
        assert records[0].operation == "summarize:sales_2024"

    def test_can_proceed_within_budget(self) -> None:
        """Test can_proceed returns True within budget."""
        config = CostConfig(max_total_tokens_per_run=1000)
        tracker = CostTracker(config)

        tracker.track_usage(input_tokens=400, output_tokens=100)

        assert tracker.can_proceed()
        assert tracker.can_proceed(estimated_tokens=400)  # 500 + 400 = 900 < 1000

    def test_can_proceed_at_budget_limit(self) -> None:
        """Test can_proceed returns False at budget limit."""
        config = CostConfig(max_total_tokens_per_run=1000)
        tracker = CostTracker(config)

        tracker.track_usage(input_tokens=500, output_tokens=500)

        assert not tracker.can_proceed()  # at limit

    def test_can_proceed_with_estimate(self) -> None:
        """Test can_proceed with estimated tokens."""
        config = CostConfig(max_total_tokens_per_run=1000)
        tracker = CostTracker(config)

        tracker.track_usage(input_tokens=400, output_tokens=100)

        assert tracker.can_proceed(estimated_tokens=400)  # 500 + 400 = 900 <= 1000
        assert not tracker.can_proceed(estimated_tokens=600)  # 500 + 600 = 1100 > 1000

    def test_would_exceed_budget(self) -> None:
        """Test would_exceed_budget calculation."""
        config = CostConfig(max_total_tokens_per_run=1000)
        tracker = CostTracker(config)

        tracker.track_usage(input_tokens=400, output_tokens=100)

        assert not tracker.would_exceed_budget(500)  # 500 + 500 = 1000, not exceeding
        assert tracker.would_exceed_budget(501)  # 500 + 501 = 1001, exceeding

    def test_get_report(self) -> None:
        """Test generating cost report."""
        config = CostConfig(
            max_total_tokens_per_run=10000,
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.03,
        )
        tracker = CostTracker(config)

        tracker.track_usage(input_tokens=1000, output_tokens=500)
        tracker.track_usage(input_tokens=2000, output_tokens=1000)

        report = tracker.get_report()

        assert report.total_requests == 2
        assert report.total_input_tokens == 3000
        assert report.total_output_tokens == 1500
        assert report.total_tokens == 4500
        assert report.tokens_remaining == 5500
        assert not report.budget_exceeded

        # Cost calculation: (3000/1000)*0.01 + (1500/1000)*0.03 = 0.03 + 0.045 = 0.075
        assert report.estimated_cost == 0.075

    def test_get_report_budget_exceeded(self) -> None:
        """Test report shows budget exceeded."""
        config = CostConfig(max_total_tokens_per_run=500)
        tracker = CostTracker(config)

        tracker.track_usage(input_tokens=300, output_tokens=200)

        report = tracker.get_report()

        assert report.budget_exceeded
        assert report.tokens_remaining == 0

    def test_reset(self) -> None:
        """Test resetting tracker."""
        tracker = CostTracker()

        tracker.track_usage(input_tokens=100, output_tokens=50)
        tracker.track_usage(input_tokens=200, output_tokens=100)

        assert tracker.total_tokens == 450

        tracker.reset()

        assert tracker.total_tokens == 0
        assert len(tracker.get_records()) == 0
        assert tracker.can_proceed()

    def test_records_are_copies(self) -> None:
        """Test that get_records returns a copy."""
        tracker = CostTracker()

        tracker.track_usage(input_tokens=100, output_tokens=50)

        records = tracker.get_records()
        records.clear()

        # Original records should be unchanged
        assert len(tracker.get_records()) == 1

    def test_zero_cost_rates(self) -> None:
        """Test report with zero cost rates (default)."""
        tracker = CostTracker()

        tracker.track_usage(input_tokens=5000, output_tokens=2000)

        report = tracker.get_report()

        assert report.estimated_cost == 0.0
