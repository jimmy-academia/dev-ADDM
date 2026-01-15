# Dataset Schema

The loader accepts JSONL or JSON. Each sample should include:

- `id` or `sample_id`: unique identifier (string/int)
- `query` or `prompt`: query string
- `context`: optional context string
- `metadata`: optional object
- `expected`: optional ground truth output for evaluation

Example JSONL:

```json
{"id": "1", "query": "...", "context": "...", "expected": "..."}
```
