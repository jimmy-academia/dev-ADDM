# Adding a New Method

This guide shows how to add a new baseline method to the ADDM benchmark.

## Overview

Methods live in `src/addm/methods/`. Each method is a Python class that:
1. Receives a sample with query and context (reviews)
2. Calls LLM(s) to produce a verdict
3. Returns structured result with usage tracking

## Step-by-Step

### 1. Create Method File

Copy `src/addm/methods/direct.py` as a template:

```bash
cp src/addm/methods/direct.py src/addm/methods/mymethod.py
```

### 2. Implement the Method Interface

Edit `mymethod.py` to implement the `Method` base class:

```python
"""My custom method."""

from typing import Dict, Any

from addm.data.types import Sample
from addm.llm import LLMService
from addm.methods.base import Method


class MyMethod(Method):
    """Description of what your method does."""

    name = "mymethod"  # CLI flag value: --method mymethod

    async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
        """Run method on a single sample.

        Args:
            sample: Input sample with query and context
            llm: LLM service for making API calls

        Returns:
            Dict with sample_id, output, and usage metrics
        """
        context = sample.context or ""

        # Build your prompt
        messages = [
            {"role": "system", "content": "Your system prompt here."},
            {"role": "user", "content": f"Query: {sample.query}\n\nContext:\n{context}"},
        ]

        # Call LLM with usage tracking
        response, usage = await llm.call_async_with_usage(
            messages,
            context={"sample_id": sample.sample_id, "method": self.name},
        )

        # Return result with usage metrics
        return {
            "sample_id": sample.sample_id,
            "output": response,
            # Usage metrics (required for tracking)
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            "cost_usd": usage.get("cost_usd", 0.0),
            "latency_ms": usage.get("latency_ms", 0.0),
            "llm_calls": 1,
        }
```

### 3. Register in __init__.py

Add your method to `src/addm/methods/__init__.py`:

```python
from addm.methods.mymethod import MyMethod  # Add import

def build_method_registry() -> MethodRegistry:
    registry = MethodRegistry()
    registry.register(DirectMethod)
    registry.register(RLMMethod)
    registry.register(RAGMethod)
    registry.register(AMOSMethod)
    registry.register(MyMethod)  # Add registration
    return registry
```

### 4. Test Locally

```bash
.venv/bin/python -m addm.tasks.cli.run_baseline \
    --policy G1_allergy_V2 -n 1 --method mymethod --dev
```

### 5. Document in BASELINES.md

Add a section to `docs/BASELINES.md` describing your method:

```markdown
### MyMethod

| Attribute | Value |
|-----------|-------|
| Method | `mymethod` |
| File | `src/addm/methods/mymethod.py` |
| Description | What it does |
| Token cost | ~X tokens per restaurant |
| Strengths | ... |
| Weaknesses | ... |
```

## Method Interface

The `Method` base class requires:

```python
class Method(ABC):
    name: str = "base"  # CLI flag value

    @abstractmethod
    async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
        """Run method on a single sample."""
        ...
```

**Required return fields:**
- `sample_id`: The input sample's ID
- `output`: The method's response/verdict

**Recommended usage fields:**
- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `cost_usd`, `latency_ms`, `llm_calls`

## Common Patterns

### Multiple LLM Calls

If your method makes multiple LLM calls, accumulate usage:

```python
async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
    usages = []

    # First call
    response1, usage1 = await llm.call_async_with_usage(messages1, ...)
    usages.append(usage1)

    # Second call
    response2, usage2 = await llm.call_async_with_usage(messages2, ...)
    usages.append(usage2)

    # Aggregate usage
    total_usage = self._accumulate_usage(usages)

    return {
        "sample_id": sample.sample_id,
        "output": response2,
        **total_usage,
    }
```

### Progress Output

Use the output manager for status updates:

```python
from addm.utils.output import output

output.status(f"Processing {sample.sample_id}...")
output.info("Found 5 relevant reviews")
```

### Error Handling

Raise exceptions with context:

```python
if not response:
    raise ValueError(f"Empty response for sample {sample.sample_id}")
```

## Example: Direct Method (Annotated)

```python
class DirectMethod(Method):
    """Direct LLM baseline - single call with full context."""

    name = "direct"  # --method direct

    async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
        context = sample.context or ""

        # Build prompt with all reviews
        messages = build_direct_prompt(sample, context)

        # Single LLM call with usage tracking
        response, usage = await llm.call_async_with_usage(
            messages,
            context={"sample_id": sample.sample_id, "method": self.name},
        )

        return {
            "sample_id": sample.sample_id,
            "output": response,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            "cost_usd": usage.get("cost_usd", 0.0),
            "latency_ms": usage.get("latency_ms", 0.0),
            "llm_calls": 1,
        }
```

## Checklist

- [ ] Created method file in `src/addm/methods/`
- [ ] Implemented `Method` base class with `name` and `run_sample`
- [ ] Registered in `src/addm/methods/__init__.py`
- [ ] Tested with `--method mymethod --dev`
- [ ] Added documentation to `docs/BASELINES.md`

## See Also

- [Method implementations](../../src/addm/methods/) - existing methods for reference
- [Baselines documentation](../BASELINES.md) - method comparison and results
- [Architecture overview](../architecture.md) - how methods fit in the system
