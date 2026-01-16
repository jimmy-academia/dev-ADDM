"""Direct LLM baseline method."""

from typing import Dict, Any

from addm.data.types import Sample
from addm.llm import LLMService
from addm.methods.base import Method


def build_direct_prompt(sample: Sample, context: str) -> list[dict[str, str]]:
    """Build prompt messages for direct LLM evaluation."""
    system = "You are a precise evaluator. Answer strictly based on the provided context."
    user = f"Query: {sample.query}\n\nContext:\n{context}\n\nReturn answer required by the Query.".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class DirectMethod(Method):
    """Direct LLM baseline - single call with full context."""

    name = "direct"

    async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
        """Run direct evaluation on a sample.

        Args:
            sample: Input sample with query and context
            llm: LLM service

        Returns:
            Dict with sample_id, output, and usage metrics
        """
        context = sample.context or ""
        messages = build_direct_prompt(sample, context)

        # Call LLM with usage tracking
        response, usage = await llm.call_async_with_usage(
            messages,
            context={"sample_id": sample.sample_id, "method": self.name},
        )

        return {
            "sample_id": sample.sample_id,
            "output": response,
            # Usage metrics
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            "cost_usd": usage.get("cost_usd", 0.0),
            "latency_ms": usage.get("latency_ms", 0.0),
            "llm_calls": 1,
        }
