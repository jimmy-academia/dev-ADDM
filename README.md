# ADDM Experiment Framework

Lightweight scaffold for running LLM experiments with dataset loading, method runners, async batching, evaluation, and results management.

## Quick Start

```bash
source .venv/bin/activate
uv pip install -e .
python main.py --data path/to/dataset.jsonl --method direct --provider openai --model gpt-4o-mini --eval
```

### Data Pipeline

```bash
# Step 1: Topic Analysis - scan 7M reviews for 18 topics
.venv/bin/python scripts/build_topic_selection.py --data yelp

# Step 2: Restaurant Selection - layered greedy by 36 cells (topic × severity)
.venv/bin/python scripts/select_topic_restaurants.py --data yelp

# Step 3: Build Datasets - K reviews per restaurant with full text
.venv/bin/python scripts/build_dataset.py --data yelp

# Extract L0 judgments and compute ground truth
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 50 --mode ondemand
.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V2 --k 50

# Run a direct LLM baseline on a task
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 --k 50 -n 5
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

**New to ADDM?** Start with the **[Quickstart Guide](docs/quickstart.md)** - run your first evaluation in 5 steps.

See **[docs/README.md](docs/README.md)** for the full documentation index including:
- Architecture overview and CLI reference
- Data pipeline (raw data → datasets → queries)
- 72 benchmark tasks (6 groups × 3 topics × 4 variants)
- Baseline methods and evaluation
- [Troubleshooting](docs/troubleshooting.md) - common issues and solutions
