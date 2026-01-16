"""Direct LLM baseline method."""

from typing import Dict, Any

from addm.data.types import Sample
from addm.llm import LLMService
from addm.methods.base import Method
from addm.methods.prompts import build_direct_prompt


def build_direct_prompt(sample: Sample, context: str) -> list[dict[str, str]]:
    system = "You are a precise evaluator. Answer strictly based on the provided context."
    user = f"Query: {sample.query}\n\nContext:\n{context}\n\nReturn answer required by the Query.".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


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
