"""Method interfaces."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

from addm.data.types import Sample
from addm.llm import LLMService
from addm.utils.usage import accumulate_usage


class Method(ABC):
    """Base class for ADDM benchmark methods."""

    name: str = "base"

    @abstractmethod
    async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
        """Run method on a single sample.

        Args:
            sample: Input sample with query and context
            llm: LLM service for making API calls

        Returns:
            Dict containing at minimum 'sample_id' and 'output'.
            Should also include usage fields if tracking is desired:
            - prompt_tokens, completion_tokens, total_tokens
            - cost_usd, latency_ms, llm_calls
        """
        raise NotImplementedError

    def _accumulate_usage(self, usage_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate usage from multiple LLM calls.

        Args:
            usage_records: List of usage dicts from call_async_with_usage

        Returns:
            Aggregated usage dict with:
            - prompt_tokens, completion_tokens, total_tokens
            - cost_usd, latency_ms, llm_calls
        """
        return accumulate_usage(usage_records)

    def _make_result(
        self,
        sample_id: str,
        output: str,
        usage: Dict[str, Any],
        llm_calls: int = 1,
        usage_breakdown: Dict[str, Dict[str, Any]] | None = None,
        **extra_fields,
    ) -> Dict[str, Any]:
        """Build standard result dict with usage metrics.

        Args:
            sample_id: Sample identifier
            output: Method output (string)
            usage: Aggregated usage dict with prompt_tokens, completion_tokens, etc.
            llm_calls: Number of LLM calls made
            usage_breakdown: Optional per-phase/stage breakdown for detailed tracking
            **extra_fields: Method-specific fields (parsed, verdict, filter_stats, etc.)

        Returns:
            Standardized result dict
        """
        result = {
            "sample_id": sample_id,
            "output": output,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
            or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)),
            "cost_usd": usage.get("cost_usd", 0.0),
            "latency_ms": usage.get("latency_ms", 0.0),
            "llm_calls": llm_calls,
        }
        if usage_breakdown:
            result["usage_breakdown"] = usage_breakdown
        result.update(extra_fields)
        return result
