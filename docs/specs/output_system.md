# Output System Architecture

## Overview

ADDM uses a centralized output system with three components:

1. **OutputManager** - Rich-formatted console output
2. **ResultLogger** - Experiment result capture
3. **DebugLogger** - LLM call logging (optional)

These are **separate and complementary**:
- `OutputManager` displays to terminal (user experience)
- `ResultLogger` writes experiment data to files (analysis)
- `DebugLogger` captures full LLM interactions (debugging)

## OutputManager

**Location:** `src/addm/utils/output.py`

**Purpose:** Singleton providing Rich-formatted console output for CLI tools.

### Features

- **Rich formatting**: Colors, tables, progress bars, rules
- **Severity levels**: info, success, warn, error, status
- **Mode-aware**: Adjusts verbosity for ondemand vs 24hrbatch
- **Quiet mode**: Suppress verbose output with `--quiet`

### Usage

```python
from addm.utils.output import output

# Configure (usually done once in CLI entrypoint)
output.configure(quiet=False, mode="ondemand")

# Display messages
output.info("Starting experiment...")
output.success("Completed successfully")
output.warn("Using fallback configuration")
output.error("Failed to load dataset")

# Headers and sections
output.header("Experiment Configuration", "Running baseline on G1a")
output.rule("Results")

# Tables
output.print_table(
    title="Model Performance",
    columns=["Model", "Accuracy", "Cost"],
    rows=[
        ["gpt-5-nano", "0.85", "$0.12"],
        ["gpt-4o-mini", "0.88", "$0.45"],
    ]
)

# Configuration display
output.print_config({
    "method": "direct",
    "model": "gpt-5-nano",
    "k": 50,
    "n": 100,
})

# Progress bars
with output.progress("Processing samples") as progress:
    task = progress.add_task("Running...", total=100)
    for i in range(100):
        # do work
        progress.update(task, advance=1)

# Result display
output.print_result(
    name="Restaurant ABC",
    predicted="RISKY",
    ground_truth="RISKY",
    score=75.0,
    correct=True,
)

# Accuracy summary
output.print_accuracy(correct=85, total=100)
```

### Batch Mode Methods

Special methods for batch execution feedback:

```python
output.batch_submitted(batch_id="batch_abc123")
# Batch submitted: batch_abc123
# Cron job will check for completion every 5 minutes

output.batch_status(batch_id="batch_abc123", status="in_progress")
# Batch batch_abc123 status: in_progress

output.batch_completed(batch_id="batch_abc123", n_results=100)
# Batch batch_abc123 completed with 100 results

output.cron_installed(marker="addm_G1a_20260116")
# Cron job installed (marker: addm_G1a_20260116)

output.cron_removed(marker="addm_G1a_20260116")
# Cron job removed (marker: addm_G1a_20260116)
```

### Design Notes

- **Singleton pattern**: One instance shared across all modules
- **Rich Console**: Uses [rich](https://github.com/Textualize/rich) library for formatting
- **No logging**: This is for display only, not persistence

## ResultLogger

**Location:** `src/addm/utils/logging.py`

**Purpose:** Capture structured experiment results for analysis.

### Features

- **Per-sample logging**: Record verdicts, scores, ground truth
- **Metrics logging**: Aggregate metrics at run completion
- **Immediate persistence**: Writes append to file after each result
- **Optional**: Can be disabled if only display is needed

### Usage

```python
from addm.utils.logging import ResultLogger

# Initialize (provide output file path)
logger = ResultLogger(output_file=Path("results/dev/run1/results.json"))

# Enable/disable dynamically
logger.enable(output_file=Path("results.json"))
logger.disable()

# Log a single result
logger.log_result(
    sample_id="restaurant_001",
    verdict="RISKY",
    ground_truth="SAFE",
    risk_score=75.0,
    correct=False,
    metadata={
        "prompt_tokens": 1234,
        "completion_tokens": 567,
        "cost_usd": 0.0045,
    }
)

# Log aggregate metrics
logger.log_metrics(
    metrics={
        "accuracy": 0.85,
        "auprc": 0.78,
        "total_cost": 12.34,
    },
    metadata={
        "run_id": "baseline_G1a",
        "model": "gpt-5-nano",
    }
)

# Clear in-memory cache
logger.clear()
```

### Design Notes

- **Separate from display**: Does not print to console
- **Append mode**: Safe for concurrent writes
- **Metadata support**: Extensible with arbitrary fields
- **Global instance available**: `get_result_logger()` / `set_result_logger()`

## Result File Schema

Results are saved to `results/{mode}/{run_id}/results.json` as a **single JSON object** (not JSONL).

**Directory structure:**
```
results/
├── dev/{timestamp}_{name}/results.json       # --dev mode runs
├── baseline/{task_id}/run_{timestamp}.json   # --benchmark mode
└── test_queries/{name}/results.json          # query tests
```

### Run Metadata

Top-level fields describing the experiment run:

- `run_id`: Unique run identifier (e.g., "G1_allergy_V2")
- `task_id`: Legacy task ID if used (e.g., "G1a") or `null`
- `policy_id`: Policy ID if used (e.g., "G1_allergy_V2") or `null`
- `gt_task_id`: Ground truth task ID for evaluation (e.g., "G1a")
- `domain`: Dataset domain (e.g., "yelp")
- `method`: Method name (e.g., "direct", "rlm")
- `model`: LLM model (e.g., "gpt-5-nano")
- `k`: Number of reviews per sample (25, 50, 100, 200)
- `n`: Number of samples processed
- `timestamp`: Run timestamp (YYYYMMDD_HHMMSS)
- `accuracy`: Overall accuracy (0.0-1.0)
- `correct`: Number of correct predictions
- `total`: Number of samples with ground truth
- `auprc`: Area under precision-recall curve metrics (dict)

### Per-Sample Results

`results` array with one entry per sample:

**Core fields:**
- `business_id`: Sample identifier (restaurant ID)
- `name`: Restaurant name
- `response`: Raw LLM response string
- `parsed`: Parsed response object (verdict, risk_score, reasoning)
- `verdict`: Extracted verdict ("SAFE", "RISKY", etc.)
- `risk_score`: Numeric risk score (0-100) if applicable

**Usage tracking:**
- `prompt_tokens`: Input tokens consumed
- `completion_tokens`: Output tokens generated
- `total_tokens`: Sum of prompt + completion
- `cost_usd`: Estimated cost in USD
- `latency_ms`: Response time in milliseconds
- `llm_calls`: Number of LLM API calls made
- `prompt_chars`: Character count of prompt (debugging)

**Evaluation (if ground truth available):**
- `gt_verdict`: Ground truth verdict
- `correct`: Boolean, whether prediction matches GT

**Example:**
```json
{
  "run_id": "G1_allergy_V2",
  "policy_id": "G1_allergy_V2",
  "method": "direct",
  "model": "gpt-5-nano",
  "k": 50,
  "n": 5,
  "accuracy": 0.8,
  "correct": 4,
  "total": 5,
  "results": [
    {
      "business_id": "abc123",
      "name": "Restaurant Name",
      "response": "VERDICT: RISKY\nSCORE: 75\n...",
      "parsed": {
        "verdict": "RISKY",
        "risk_score": 75,
        "reasoning": "..."
      },
      "verdict": "RISKY",
      "risk_score": 75,
      "prompt_tokens": 1234,
      "completion_tokens": 567,
      "total_tokens": 1801,
      "cost_usd": 0.0045,
      "latency_ms": 892.5,
      "llm_calls": 1,
      "gt_verdict": "RISKY",
      "correct": true
    }
  ]
}
```

## DebugLogger

**Location:** `src/addm/utils/debug_logger.py`

**Purpose:** Capture full LLM prompts and responses for debugging.

### Features

- **Full capture**: Entire prompt text and response text
- **Per-sample files**: Separate file per sample (or JSONL)
- **Phase tracking**: Label different LLM call phases (e.g., "main", "clarification")
- **Optional**: Only used when debugging, not in production runs

### Usage

```python
from addm.utils.debug_logger import DebugLogger

# Initialize with output directory
logger = DebugLogger(output_dir=Path("results/dev/run1"))

# Log an LLM call
logger.log_llm_call(
    sample_id="restaurant_001",
    phase="main",
    prompt="You are a restaurant analyst...\n\nReviews:\n...",
    response="VERDICT: RISKY\nSCORE: 75\n...",
    model="gpt-5-nano",
    latency_ms=892.5,
)

# Flush to disk
logger.flush()
```

### Design Notes

- **Heavy output**: Can generate large files with full prompts
- **Not for production**: Enable only when debugging specific issues
- **Complements UsageTracker**: Debug logger has text, UsageTracker has metrics

## Integration in CLI

Typical CLI setup:

```python
from addm.utils.output import output
from addm.utils.logging import ResultLogger, set_result_logger
from addm.utils.usage import usage_tracker

# Configure output
output.configure(quiet=args.quiet, mode=args.mode)

# Set up result logger
result_logger = ResultLogger(output_file=run_paths.root / "results.json")
set_result_logger(result_logger)

# UsageTracker is global singleton (always active)
# DebugLogger is created per-method if needed

# Display to user
output.header("Experiment Configuration")
output.print_config(config_dict)

# Run experiment
for sample in dataset:
    # ... LLM calls (automatically tracked by UsageTracker) ...

    # Log result
    result_logger.log_result(...)

    # Display to user
    output.print_result(...)

# Final metrics
output.print_accuracy(correct, total)
result_logger.log_metrics(metrics)

# Save usage summary
usage_summary = usage_tracker.get_summary()
# ... write to file ...
```

## Summary

| Component | Purpose | Output | When Used |
|-----------|---------|--------|-----------|
| OutputManager | Display to user | Terminal (Rich) | Always |
| ResultLogger | Record results | File (JSON/JSONL) | Experiments |
| DebugLogger | Capture LLM calls | File (full text) | Debugging |
| UsageTracker | Token/cost metrics | In-memory → file | Always |

**Key principle:** Each component has a single, clear responsibility. They work together but remain independent.
