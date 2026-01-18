# Output Schema

## File Structure

Results are saved to `results/{mode}/{run_id}/results.json` as a **single JSON object** (not JSONL).

**Directory structure:**
```
results/
├── dev/{timestamp}_{name}/results.json       # --dev mode runs
├── baseline/{task_id}/run_{timestamp}.json   # --benchmark mode
└── test_queries/{name}/results.json          # query tests
```

## results.json Format

Single JSON object with two sections:

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

## Legacy Formats

**Note:** Some older documentation references `results.jsonl` (JSONL format). This has been replaced with `results.json` (single JSON object). The JSONL format is no longer used.

## Debug Output

The `DebugLogger` can optionally capture full prompts and responses, but this is not enabled by default in production runs. When enabled, debug logs are written to separate files (implementation varies by method).
