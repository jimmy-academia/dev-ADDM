# CLI Specification

Required:
- `--data`: dataset path (jsonl/json)

Core:
- `--method`: method name (default: `direct`)
- `--provider`: LLM provider (default: `openai`)
- `--model`: LLM model (default: `gpt-4o-mini`)
- `--temperature`: sampling temperature
- `--max-tokens`: max output tokens
- `--max-concurrent`: max concurrent LLM calls
- `--batch-size`: batch size for async calls
- `--sequential`: disable async batching
- `--mode`: LLM execution mode (`ondemand` or `24hrbatch`)
- `--batch-id`: batch ID for fetch-only runs (used by cron)

Run control:
- `--limit`: limit number of samples
- `--run-name`: output folder suffix
- `--benchmark`: use benchmark output path
- `--results-dir`: base output directory
- `--eval`: run evaluation step
- `--validator`: validator name (default: `exact`)

Reliability:
- `--request-timeout`: per-request timeout (seconds)
- `--max-retries`: retry attempts
- `--seed`: random seed
