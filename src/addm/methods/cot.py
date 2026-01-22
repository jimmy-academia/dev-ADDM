"""Chain-of-Thought (CoT) baseline method.

Reference: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
Wei et al., NeurIPS 2022
https://arxiv.org/abs/2201.11903

Adapted from anot/methods/cot.py for ADDM benchmark.
"""

from typing import Any, Dict, Optional

from addm.data.types import Sample
from addm.llm import LLMService
from addm.methods.base import Method


# System prompt encouraging step-by-step reasoning
SYSTEM_PROMPT = """You are a precise evaluator. Think through this step-by-step before giving your final answer.

1. First, understand what the query is asking for
2. Analyze the relevant evidence in the context
3. Reason through how the evidence supports or contradicts the query
4. Provide your final answer

Answer strictly based on the provided context."""


def build_cot_prompt(sample: Sample, context: str) -> list[dict[str, str]]:
    """Build CoT prompt messages.

    Args:
        sample: Sample with query
        context: Context string

    Returns:
        List of message dicts for LLM
    """
    user_content = f"""Query: {sample.query}

Context:
{context}

Let's think through this step-by-step:"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class CoTMethod(Method):
    """Chain-of-Thought baseline - prompts LLM to reason step-by-step."""

    name = "cot"

    def __init__(self, system_prompt: Optional[str] = None):
        """Initialize CoT method.

        Args:
            system_prompt: Optional custom system prompt. If None, uses default.
        """
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
        """Run CoT evaluation on a sample.

        Args:
            sample: Input sample with query and context
            llm: LLM service

        Returns:
            Dict with sample_id, output, and usage metrics
        """
        context = sample.context or ""
        messages = build_cot_prompt(sample, context)

        # Override system prompt if custom one provided
        if self.system_prompt != SYSTEM_PROMPT:
            messages[0]["content"] = self.system_prompt

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
