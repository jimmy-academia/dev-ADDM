# Usage Tracking Specification

## Overview

ADDM tracks LLM usage at three levels:
1. **Per-call**: Every API call recorded with tokens, cost, latency
2. **Per-sample**: Aggregated in results.json (in the `results` array)
3. **Per-run**: Summary metadata in results.json (top-level fields)

## Components

### UsageTracker (`src/addm/utils/usage.py`)

Thread-safe singleton for recording LLM calls.

```python
from addm.utils.usage import usage_tracker, compute_cost

# Record a call (done automatically by LLMService)
usage_tracker.record(
    model="gpt-4o-mini",
    provider="openai",
    prompt_tokens=1000,
    completion_tokens=500,
    latency_ms=850.0,
    context={"sample_id": "S001", "run_id": "..."},
)

# Get summary
summary = usage_tracker.get_summary()
# Returns: {total_calls, total_tokens, total_cost_usd, by_model: {...}}
```

### DebugLogger (`src/addm/utils/debug_logger.py`)

Captures full prompts and responses for debugging.

```python
from addm.utils.debug_logger import DebugLogger

logger = DebugLogger(output_dir=run_paths.root)
logger.log_llm_call(
    sample_id="S001",
    phase="main",
    prompt="full prompt text...",
    response="full response text...",
    model="gpt-4o-mini",
    latency_ms=850.0,
)
logger.flush()  # Writes to debug/{sample_id}.jsonl
```

### LLMService Integration

The `LLMService` automatically records usage:

```python
# Returns just response (usage recorded internally)
response = await llm.call_async(messages)

# Returns response + usage dict
response, usage = await llm.call_async_with_usage(messages, context={...})
# usage = {prompt_tokens, completion_tokens, latency_ms, cost_usd}

# Batch with usage
responses, usages = await llm.batch_call_with_usage(batch, context={...})
```

## Output Files

### results.json

Single JSON file per run containing both run-level metadata and per-sample results.

**Per-sample usage fields** (in `results` array):

```json
{
  "business_id": "S001",
  "response": "...",
  "verdict": "RISKY",
  "prompt_tokens": 1234,
  "completion_tokens": 567,
  "total_tokens": 1801,
  "cost_usd": 0.0045,
  "latency_ms": 892.5,
  "llm_calls": 1
}
```

**Run-level metadata** (top-level fields):

```json
{
  "run_id": "G1_allergy_V2",
  "method": "direct",
  "model": "gpt-5-nano",
  "k": 50,
  "n": 100,
  "timestamp": "20260116_143022",
  "accuracy": 0.85,
  "correct": 85,
  "total": 100,
  "results": [...]
}
```

**Note:** The separate `usage.json` file mentioned in older documentation is no longer created. All usage data is consolidated into `results.json`.

### Debug Logging (optional)

The `DebugLogger` can capture full prompts and responses when explicitly enabled. This is not used in standard baseline runs. See `src/addm/utils/debug_logger.py` for implementation details.

## Model Pricing

Pricing table in `MODEL_PRICING` (per 1M tokens). Source: [OpenAI Pricing](https://platform.openai.com/docs/pricing)

| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| gpt-5-nano | $0.05 | $0.40 | **Default for benchmarking** |
| gpt-5-mini | $0.25 | $2.00 | |
| gpt-5 | $1.25 | $10.00 | |
| gpt-5.1 | $1.25 | $10.00 | |
| gpt-5.2 | $1.75 | $14.00 | |
| o3-mini | $1.10 | $4.40 | Reasoning model |
| o4-mini | $1.10 | $4.40 | Reasoning model |
| o3 | $2.00 | $8.00 | Reasoning model |
| gpt-4o-mini | $0.15 | $0.60 | Legacy |
| gpt-4o | $2.50 | $10.00 | Legacy |
| claude-sonnet-4 | $3.00 | $15.00 | |
| claude-opus-4 | $15.00 | $75.00 | |
| claude-3-5-haiku | $0.80 | $4.00 | |

Unknown models default to $1.00/$3.00.

## Implementing Usage in New Methods

```python
class MyMethod(Method):
    name = "mymethod"

    async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
        usages = []

        # First LLM call
        response1, usage1 = await llm.call_async_with_usage(
            messages1,
            context={"sample_id": sample.sample_id, "method": self.name, "phase": "step1"},
        )
        usages.append(usage1)

        # Second LLM call
        response2, usage2 = await llm.call_async_with_usage(
            messages2,
            context={"sample_id": sample.sample_id, "method": self.name, "phase": "step2"},
        )
        usages.append(usage2)

        # Aggregate usage
        total_usage = self._accumulate_usage(usages)

        return {
            "sample_id": sample.sample_id,
            "output": response2,
            **total_usage,  # Spreads prompt_tokens, completion_tokens, etc.
        }
```
