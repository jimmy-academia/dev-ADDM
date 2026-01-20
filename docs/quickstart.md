# Quickstart: Run Your First Evaluation

This guide walks through a complete evaluation using G1_allergy as an example.

## Prerequisites

- Python 3.10+
- `.venv` activated (`source .venv/bin/activate`)
- OpenAI API key configured (`OPENAI_API_KEY` env var)
- Yelp data in `data/raw/yelp/`

## Step 1: Build Context (if not done)

If datasets don't exist in `data/context/yelp/`, build them:

```bash
.venv/bin/python scripts/data/build_dataset.py --data yelp
```

This creates `K{25,50,100,200}_reviews.json` files in `data/context/yelp/`.

## Step 2: Extract Ground Truth Judgments

Extract L0 judgments using multi-model ensemble:

```bash
# Ondemand mode (immediate, good for testing)
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 50 --mode ondemand

# Or 24hr batch mode (cheaper, for production)
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 50 --mode 24hrbatch
```

Extractions are cached in `data/answers/yelp/judgement_cache.json`.

## Step 3: Compute Ground Truth

Aggregate L0 judgments into final ground truth:

```bash
.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V2 --k 50
```

This creates `data/answers/yelp/G1_allergy_V2_K50_groundtruth.json`.

## Step 4: Run Baseline

Run an evaluation:

```bash
# Dev mode (results go to results/dev/)
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 --k 50 -n 5 --dev

# Production mode (results go to results/prod/)
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 --k 50 -n 5
```

## Step 5: View Results

Results are saved to:

```
results/dev/{timestamp}_G1_allergy_V2/
└── results.json     # Metadata + per-sample results
```

The `results.json` contains:
- Run configuration (method, model, K value)
- Per-sample outputs and usage metrics
- Aggregate statistics

## What's Next?

### Try Different Methods

```bash
# RAG (retrieval-augmented)
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --method rag

# RLM (recursive LLM with code execution)
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --method rlm

# AMOS (proposed method)
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --method amos
```

### Scale Up

```bash
# More samples with batch mode
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 100 --mode 24hrbatch

# Larger context size
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --k 200
```

### Explore Other Policies

The benchmark includes 72 policies (6 groups × 3 topics × 4 variants):

```bash
# G2 social context
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G2_romance_V2 -n 5 --dev

# G3 economic value
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G3_price_worth_V2 -n 5 --dev
```

## Reference

| Topic | Documentation |
|-------|---------------|
| CLI options | [docs/specs/cli.md](specs/cli.md) |
| All 72 tasks | [docs/tasks/TAXONOMY.md](tasks/TAXONOMY.md) |
| Method details | [docs/BASELINES.md](BASELINES.md) |
| Troubleshooting | [docs/troubleshooting.md](troubleshooting.md) |
