"""Context curation helpers."""

from addm.data.types import Sample


def build_context(sample: Sample, max_chars: int | None = None) -> str:
    context = sample.context or ""
    if max_chars is not None and max_chars > 0:
        return context[:max_chars]
    return context
