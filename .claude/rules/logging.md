# Logging Conventions

## Console Output

Use `output` singleton from `src/addm/utils/output.py`, **not raw `print()`**:

```python
from addm.utils.output import output

output.status("Processing...")   # Suppressed in quiet mode
output.info("Important info")    # Always shown (blue ℹ)
output.success("Done!")          # Always shown (green ✓)
output.warn("Warning")           # Always shown (yellow ⚠)
output.error("Failed")           # Always shown (red ✗)
output.print("Plain text")       # Always shown (no prefix)
```

## Result Files

```
results/dev/{timestamp}_{name}/
└── results.json     # Run metadata + per-sample results array
```

## Usage Tracking

```python
# Call with usage tracking
response, usage = await llm.call_async_with_usage(messages, context={"sample_id": id})

# Accumulate multiple calls
total_usage = self._accumulate_usage([usage1, usage2, ...])
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `src/addm/utils/output.py` | OutputManager, Rich console |
| `src/addm/utils/logging.py` | ResultLogger for experiments |
| `src/addm/utils/usage.py` | UsageTracker, MODEL_PRICING |
| `src/addm/utils/debug_logger.py` | Prompt/response capture |

See `docs/specs/usage_tracking.md` for model pricing.
