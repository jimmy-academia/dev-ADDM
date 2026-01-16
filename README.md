# ADDM Experiment Framework

Lightweight scaffold for running LLM experiments with dataset loading, method runners, async batching, evaluation, and results management.

## Quick Start

```bash
uv pip install -e .
python main.py --data path/to/dataset.jsonl --method direct --provider openai --model gpt-4o-mini --eval
```

```bash
# Yelp keyword search -> topic-aware selection
.venv/bin/python scripts/search_restaurants.py --group G1
.venv/bin/python scripts/select_topic_restaurants.py --target 100

# Build review datasets (K=25/50/100/200)
.venv/bin/python scripts/build_dataset.py --data yelp --selection data/selected/yelp/topic_100.json
```

```bash
# Extract L0 judgments and compute ground truth
.venv/bin/python -m addm.tasks.cli.extract --task G1a --domain yelp --k 50
.venv/bin/python -m addm.tasks.cli.compute_gt --task G1a --domain yelp --k 50

# Run a direct LLM baseline on a task (Yelp)
.venv/bin/python -m addm.tasks.cli.run_baseline --task G1a --k 50 -n 5
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

Use `--help` to see all arguments.
See [Data Creation Workflow (docs/specs/data_creation.md)](docs/specs/data_creation.md) for the full workflow.

## Documentation

- [CLI Reference (docs/specs/cli.md)](docs/specs/cli.md) for arguments
- [Dataset Schema (docs/specs/datasets.md)](docs/specs/datasets.md) for dataset schema
- [Results Schema (docs/specs/outputs.md)](docs/specs/outputs.md) for results schema
- [Architecture Overview (docs/architecture.md)](docs/architecture.md) for the pipeline overview
