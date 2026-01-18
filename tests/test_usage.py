"""Tests for utils/usage.py - LLM usage tracking."""

import pytest
from addm.utils.usage import (
    compute_cost,
    UsageTracker,
    accumulate_usage,
    MODEL_PRICING,
    LLMUsageRecord,
)


class TestComputeCost:
    """Test cost computation for different models."""

    def test_gpt5_nano_cost(self):
        """Test cost calculation for gpt-5-nano."""
        # $0.05 per 1M input, $0.40 per 1M output
        cost = compute_cost("gpt-5-nano", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.45)  # 0.05 + 0.40

    def test_gpt5_mini_cost(self):
        """Test cost for gpt-5-mini."""
        # $0.25 per 1M input, $2.00 per 1M output
        cost = compute_cost("gpt-5-mini", 1_000_000, 1_000_000)
        assert cost == pytest.approx(2.25)

    def test_small_token_counts(self):
        """Test cost with realistic token counts."""
        # 1000 input + 500 output tokens for gpt-5-nano
        cost = compute_cost("gpt-5-nano", 1000, 500)
        expected = (1000 / 1_000_000) * 0.05 + (500 / 1_000_000) * 0.40
        assert cost == pytest.approx(expected)

    def test_unknown_model_default_pricing(self):
        """Test that unknown models use default pricing."""
        cost = compute_cost("unknown-model", 1_000_000, 1_000_000)
        # Default: $1.0 input, $3.0 output
        assert cost == pytest.approx(4.0)

    def test_zero_tokens(self):
        """Test cost with zero tokens."""
        cost = compute_cost("gpt-5-nano", 0, 0)
        assert cost == 0.0

    def test_all_models_have_pricing(self):
        """Verify all models in pricing dict have both input and output."""
        for model, pricing in MODEL_PRICING.items():
            assert "input" in pricing, f"{model} missing input pricing"
            assert "output" in pricing, f"{model} missing output pricing"
            assert pricing["input"] >= 0, f"{model} has negative input pricing"
            assert pricing["output"] >= 0, f"{model} has negative output pricing"

    def test_reasoning_models_cost(self):
        """Test cost for reasoning models (o-series)."""
        # o1-pro should be expensive
        cost_o1_pro = compute_cost("o1-pro", 1_000_000, 1_000_000)
        assert cost_o1_pro == pytest.approx(750.0)  # 150 + 600

        # o1-mini should be cheaper
        cost_o1_mini = compute_cost("o1-mini", 1_000_000, 1_000_000)
        assert cost_o1_mini == pytest.approx(5.5)  # 1.10 + 4.40

    def test_claude_models_cost(self):
        """Test cost for Claude models."""
        cost_sonnet = compute_cost("claude-3-5-sonnet-20241022", 1_000_000, 1_000_000)
        assert cost_sonnet == pytest.approx(18.0)  # 3.00 + 15.00

        cost_opus = compute_cost("claude-opus-4-20250514", 1_000_000, 1_000_000)
        assert cost_opus == pytest.approx(90.0)  # 15.00 + 75.00


class TestUsageTracker:
    """Test UsageTracker singleton."""

    @pytest.fixture(autouse=True)
    def clear_tracker(self):
        """Clear tracker before each test."""
        tracker = UsageTracker()
        tracker.clear()
        yield
        tracker.clear()

    def test_singleton_pattern(self):
        """Test that UsageTracker is a singleton."""
        tracker1 = UsageTracker()
        tracker2 = UsageTracker()
        assert tracker1 is tracker2

    def test_record_basic(self):
        """Test recording a single LLM call."""
        tracker = UsageTracker()

        record = tracker.record(
            model="gpt-5-nano",
            provider="openai",
            prompt_tokens=1000,
            completion_tokens=500,
            latency_ms=250.5,
        )

        assert record.model == "gpt-5-nano"
        assert record.provider == "openai"
        assert record.prompt_tokens == 1000
        assert record.completion_tokens == 500
        assert record.latency_ms == 250.5
        assert record.cost_usd > 0

    def test_record_with_context(self):
        """Test recording with context metadata."""
        tracker = UsageTracker()

        context = {"sample_id": "sample_001", "method": "direct", "phase": "extraction"}

        record = tracker.record(
            model="gpt-5-nano",
            provider="openai",
            prompt_tokens=1000,
            completion_tokens=500,
            latency_ms=200.0,
            context=context,
        )

        assert record.context == context

    def test_record_with_previews(self):
        """Test recording with prompt/response previews."""
        tracker = UsageTracker()

        long_prompt = "a" * 300
        long_response = "b" * 300

        record = tracker.record(
            model="gpt-5-nano",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=100.0,
            prompt_preview=long_prompt,
            response_preview=long_response,
        )

        # Should truncate to 200 chars
        assert len(record.prompt_preview) == 200
        assert len(record.response_preview) == 200

    def test_get_records(self):
        """Test retrieving records."""
        tracker = UsageTracker()

        tracker.record("gpt-5-nano", "openai", 100, 50, 100.0)
        tracker.record("gpt-5-mini", "openai", 200, 100, 150.0)

        records = tracker.get_records()
        assert len(records) == 2

    def test_get_records_with_filter(self):
        """Test filtering records by context."""
        tracker = UsageTracker()

        tracker.record(
            "gpt-5-nano", "openai", 100, 50, 100.0, context={"sample_id": "s1"}
        )
        tracker.record(
            "gpt-5-nano", "openai", 200, 100, 150.0, context={"sample_id": "s2"}
        )
        tracker.record(
            "gpt-5-mini", "openai", 150, 75, 120.0, context={"sample_id": "s1"}
        )

        # Filter for sample_id=s1
        records = tracker.get_records(context_filter={"sample_id": "s1"})
        assert len(records) == 2
        assert all(r.context["sample_id"] == "s1" for r in records)

    def test_get_summary(self):
        """Test getting usage summary."""
        tracker = UsageTracker()

        tracker.record("gpt-5-nano", "openai", 1000, 500, 100.0)
        tracker.record("gpt-5-nano", "openai", 2000, 1000, 200.0)
        tracker.record("gpt-5-mini", "openai", 1500, 750, 150.0)

        summary = tracker.get_summary()

        assert summary["total_calls"] == 3
        assert summary["total_prompt_tokens"] == 4500  # 1000 + 2000 + 1500
        assert summary["total_completion_tokens"] == 2250  # 500 + 1000 + 750
        assert summary["total_tokens"] == 6750
        assert summary["total_latency_ms"] == 450.0  # 100 + 200 + 150
        assert summary["total_cost_usd"] > 0

    def test_get_summary_by_model(self):
        """Test summary breakdown by model."""
        tracker = UsageTracker()

        tracker.record("gpt-5-nano", "openai", 1000, 500, 100.0)
        tracker.record("gpt-5-nano", "openai", 2000, 1000, 200.0)
        tracker.record("gpt-5-mini", "openai", 1500, 750, 150.0)

        summary = tracker.get_summary()

        assert "by_model" in summary
        assert "gpt-5-nano" in summary["by_model"]
        assert "gpt-5-mini" in summary["by_model"]

        nano_usage = summary["by_model"]["gpt-5-nano"]
        assert nano_usage["calls"] == 2
        assert nano_usage["prompt_tokens"] == 3000
        assert nano_usage["completion_tokens"] == 1500

        mini_usage = summary["by_model"]["gpt-5-mini"]
        assert mini_usage["calls"] == 1
        assert mini_usage["prompt_tokens"] == 1500

    def test_clear_all_records(self):
        """Test clearing all records."""
        tracker = UsageTracker()

        tracker.record("gpt-5-nano", "openai", 100, 50, 100.0)
        tracker.record("gpt-5-mini", "openai", 200, 100, 150.0)

        assert len(tracker.get_records()) == 2

        tracker.clear()

        assert len(tracker.get_records()) == 0

    def test_clear_with_filter(self):
        """Test clearing records with context filter."""
        tracker = UsageTracker()

        tracker.record(
            "gpt-5-nano", "openai", 100, 50, 100.0, context={"sample_id": "s1"}
        )
        tracker.record(
            "gpt-5-nano", "openai", 200, 100, 150.0, context={"sample_id": "s2"}
        )

        # Clear only s1 records
        tracker.clear(context_filter={"sample_id": "s1"})

        records = tracker.get_records()
        assert len(records) == 1
        assert records[0].context["sample_id"] == "s2"


class TestAccumulateUsage:
    """Test usage accumulation helper."""

    def test_accumulate_basic(self, multiple_usages):
        """Test basic accumulation."""
        result = accumulate_usage(multiple_usages)

        assert result["prompt_tokens"] == 450  # 100 + 200 + 150
        assert result["completion_tokens"] == 225  # 50 + 75 + 100
        assert result["total_tokens"] == 675  # 150 + 275 + 250
        assert result["llm_calls"] == 3

    def test_accumulate_empty_list(self):
        """Test accumulation with empty list."""
        result = accumulate_usage([])

        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 0
        assert result["total_tokens"] == 0
        assert result["cost_usd"] == 0.0
        assert result["latency_ms"] == 0.0
        assert result["llm_calls"] == 0

    def test_accumulate_single_usage(self, sample_usage):
        """Test accumulation with single usage."""
        result = accumulate_usage([sample_usage])

        assert result["prompt_tokens"] == 150
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 200
        assert result["llm_calls"] == 1

    def test_accumulate_missing_fields(self):
        """Test accumulation handles missing fields gracefully."""
        usage_records = [
            {"prompt_tokens": 100},  # Missing completion_tokens
            {"completion_tokens": 50},  # Missing prompt_tokens
            {"cost_usd": 0.5},  # Missing token counts
        ]

        result = accumulate_usage(usage_records)

        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["cost_usd"] == 0.5
        assert result["llm_calls"] == 3
