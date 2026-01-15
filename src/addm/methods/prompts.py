"""Prompt templates."""

from addm.data.types import Sample


def build_direct_prompt(sample: Sample, context: str) -> list[dict[str, str]]:
    system = "You are a precise evaluator. Answer strictly based on the provided context."
    user = f"Query: {sample.query}\n\nContext:\n{context}\n\nReturn a concise answer.".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
