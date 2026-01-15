"""Direct LLM baseline method."""

from typing import Dict, Any

from addm.data.types import Sample
from addm.llm import LLMService
from addm.methods.base import Method
from addm.methods.prompts import build_direct_prompt


class DirectMethod(Method):
    name = "direct"

    async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
        context = sample.context or ""
        messages = build_direct_prompt(sample, context)
        response = await llm.call_async(messages)
        return {
            "sample_id": sample.sample_id,
            "output": response,
        }
