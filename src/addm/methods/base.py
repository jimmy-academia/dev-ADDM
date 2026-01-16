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
