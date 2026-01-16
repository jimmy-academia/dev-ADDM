"""Usage tracking for LLM calls - thread-safe for parallel execution."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from threading import RLock
from typing import Any


# Model pricing per 1M tokens (input, output) in USD
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 1.10, "output": 4.40},
    "o3-mini": {"input": 1.10, "output": 4.40},
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
}


@dataclass
class LLMUsageRecord:
    """Per-call usage record."""

    timestamp: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    context: dict[str, Any] = field(default_factory=dict)
    prompt_preview: str = ""  # First 200 chars
    response_preview: str = ""  # First 200 chars


@dataclass
class ModelUsage:
    """Usage aggregated by model."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute cost in USD from token counts.

    Args:
        model: Model name (e.g., "gpt-4o-mini")
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    # Default pricing for unknown models
    pricing = MODEL_PRICING.get(model, {"input": 1.0, "output": 3.0})
    return (prompt_tokens / 1_000_000) * pricing["input"] + (
        completion_tokens / 1_000_000
    ) * pricing["output"]


class UsageTracker:
    """Thread-safe singleton for tracking LLM usage."""

    _instance = None
    _lock = RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self._records: list[LLMUsageRecord] = []
        self._records_lock = RLock()

    def record(
        self,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        context: dict | None = None,
        prompt_preview: str = "",
        response_preview: str = "",
    ) -> LLMUsageRecord:
        """Record a single LLM call.

        Args:
            model: Model name
            provider: Provider name (openai, anthropic, mock)
            prompt_tokens: Input token count
            completion_tokens: Output token count
            latency_ms: API call duration in milliseconds
            context: Optional context dict (sample_id, method, phase, run_id)
            prompt_preview: First 200 chars of prompt for debugging
            response_preview: First 200 chars of response for debugging

        Returns:
            The recorded LLMUsageRecord
        """
        cost = compute_cost(model, prompt_tokens, completion_tokens)
        record = LLMUsageRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            context=context or {},
            prompt_preview=prompt_preview[:200] if prompt_preview else "",
            response_preview=response_preview[:200] if response_preview else "",
        )
        with self._records_lock:
            self._records.append(record)
        return record

    def get_records(
        self, context_filter: dict | None = None
    ) -> list[LLMUsageRecord]:
        """Get records, optionally filtered by context.

        Args:
            context_filter: Dict of context key-value pairs to match

        Returns:
            List of matching LLMUsageRecord objects
        """
        with self._records_lock:
            if context_filter is None:
                return list(self._records)
            return [
                r
                for r in self._records
                if all(r.context.get(k) == v for k, v in context_filter.items())
            ]

    def get_summary(self, context_filter: dict | None = None) -> dict:
        """Get aggregated usage summary.

        Args:
            context_filter: Optional filter for records

        Returns:
            Dict with total_calls, total_tokens, total_cost_usd, by_model breakdown
        """
        records = self.get_records(context_filter)
        by_model: dict[str, ModelUsage] = {}
        for r in records:
            if r.model not in by_model:
                by_model[r.model] = ModelUsage()
            m = by_model[r.model]
            m.calls += 1
            m.prompt_tokens += r.prompt_tokens
            m.completion_tokens += r.completion_tokens
            m.cost_usd += r.cost_usd

        return {
            "total_calls": len(records),
            "total_prompt_tokens": sum(r.prompt_tokens for r in records),
            "total_completion_tokens": sum(r.completion_tokens for r in records),
            "total_tokens": sum(
                r.prompt_tokens + r.completion_tokens for r in records
            ),
            "total_cost_usd": sum(r.cost_usd for r in records),
            "total_latency_ms": sum(r.latency_ms for r in records),
            "by_model": {k: asdict(v) for k, v in by_model.items()},
        }

    def clear(self, context_filter: dict | None = None):
        """Clear records, optionally filtered by context.

        Args:
            context_filter: If provided, only clear records matching this filter.
                           If None, clear all records.
        """
        with self._records_lock:
            if context_filter is None:
                self._records.clear()
            else:
                self._records = [
                    r
                    for r in self._records
                    if not all(
                        r.context.get(k) == v for k, v in context_filter.items()
                    )
                ]


# Global tracker instance
usage_tracker = UsageTracker()


def accumulate_usage(usage_records: list[dict]) -> dict:
    """Aggregate usage from multiple LLM calls.

    Args:
        usage_records: List of usage dicts from individual LLM calls

    Returns:
        Aggregated usage dict with totals
    """
    if not usage_records:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "llm_calls": 0,
        }
    return {
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage_records),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage_records),
        "total_tokens": sum(
            u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
            for u in usage_records
        ),
        "cost_usd": sum(u.get("cost_usd", 0) for u in usage_records),
        "latency_ms": sum(u.get("latency_ms", 0) for u in usage_records),
        "llm_calls": len(usage_records),
    }
