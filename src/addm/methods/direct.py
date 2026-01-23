"""Direct LLM baseline method."""

from typing import Dict, Any, Optional

from addm.data.types import Sample
from addm.llm import LLMService
from addm.methods.base import Method


# Default system prompt (used if none provided)
DEFAULT_SYSTEM_PROMPT = "You are a precise evaluator. Answer strictly based on the provided context."


def build_direct_prompt(
    sample: Sample,
    context: str,
    system_prompt: Optional[str] = None,
) -> list[dict[str, str]]:
    """Build prompt messages for direct LLM evaluation.

    Args:
        sample: Sample with query
        context: Context string
        system_prompt: System prompt for output format. If None, uses default.

    Returns:
        List of message dicts for LLM
    """
    system = system_prompt or DEFAULT_SYSTEM_PROMPT
    user = f"Query: {sample.query}\n\nContext:\n{context}\n\nReturn answer required by the Query.".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class DirectMethod(Method):
    """Direct LLM baseline - single call with full context."""

    name = "direct"

    def __init__(self, system_prompt: Optional[str] = None):
        """Initialize Direct method.

        Args:
            system_prompt: System prompt for structured output format.
                          If None, uses default simple prompt.
        """
        self.system_prompt = system_prompt

    async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
        """Run direct evaluation on a sample.

        Args:
            sample: Input sample with query and context
            llm: LLM service

        Returns:
            Dict with sample_id, output, and usage metrics
        """
        context = sample.context or ""
        messages = build_direct_prompt(sample, context, self.system_prompt)

        # Call LLM with usage tracking
        response, usage = await llm.call_async_with_usage(
            messages,
            context={"sample_id": sample.sample_id, "method": self.name},
        )

        return self._make_result(sample.sample_id, response, usage)
