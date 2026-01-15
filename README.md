# ADDM Experiment Framework

Lightweight scaffold for running LLM experiments with dataset loading, method runners, async batching, evaluation, and results management.

## Quick Start

```bash
uv pip install -e .
python main.py --data path/to/dataset.jsonl --method direct --provider openai --model gpt-4o-mini --eval
```

## Project Structure

```
.
├── src/addm/          # Core package
├── tests/             # Contract tests
├── docs/              # Specs and architecture notes
└── main.py            # CLI entrypoint
```

## CLI Basics

```bash
uv pip install -e .
python main.py --data data.jsonl --method direct --eval
```

Use `--help` to see all arguments.

## Data Creation (Yelp)

```bash
# Select core 100 businesses
.venv/bin/python scripts/select_contexts.py --data yelp

# Build review datasets (K=25/50/100/200)
.venv/bin/python scripts/build_dataset.py --data yelp
```

See `docs/specs/data_creation.md` for the full workflow.

## Documentation

- `docs/specs/cli.md` for arguments
- `docs/specs/datasets.md` for dataset schema
- `docs/specs/outputs.md` for results schema
- `docs/architecture.md` for the pipeline overview
