# Output Schema

Results are saved to `results/.../results.jsonl` with fields:

- `sample_id`: input sample id
- `output`: model output string
- `method`: method name
- `provider`: provider name
- `model`: model name

Metrics (if `--eval`) are saved to `metrics.json`:

- `total`: number of evaluated samples
- `correct`: count of correct predictions
- `accuracy`: accuracy value
- `details`: per-sample correctness list
