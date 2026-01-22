# Methods

Available methods for `--method` flag in `run_experiment.py`:

| Method | Description | Token Cost |
|--------|-------------|------------|
| `direct` | All K reviews in prompt (default) | ~K×200 tokens |
| `rlm` | Recursive LLM - code execution to search | ~50k tokens |
| `rag` | Retrieval-Augmented Generation | ~5k tokens |
| `amos` | Agenda-Driven Mining with Observable Steps | ~5-55k tokens |

## RLM Method

Uses [recursive-llm](https://github.com/ysz/recursive-llm) (forked to `lib/recursive-llm/`)

- Stores reviews as Python variable, LLM writes code to search
- Token budget: `--token-limit 50000` (default)
- **Known issue**: gpt-5-nano produces inconsistent results

```bash
# Basic RLM
.venv/bin/python -m addm.tasks.cli.run_experiment \
    --policy G1_allergy_V2 -n 1 --k 50 --dev --method rlm

# Custom token limit
.venv/bin/python -m addm.tasks.cli.run_experiment \
    --policy G1_allergy_V2 -n 1 --k 50 --dev --method rlm --token-limit 30000
```

## AMOS Method

Two-stage retrieval for comprehensive review analysis:

**Stage 1 (Quick Scan)**: Filter reviews, extract signals, check for early exit
**Stage 2 (Thorough Sweep)**: Process ALL remaining reviews (always on)

### Filter Modes (`--filter-mode`)

| Mode | Description | Stage 1 Reviews |
|------|-------------|-----------------|
| `keyword` | Keyword matching only (default) | ~5-10 |
| `embedding` | Embedding similarity only | ~20 |
| `hybrid` | Keywords + embedding | ~20-30 |

```bash
# Default: keyword filter + thorough sweep
.venv/bin/python -m addm.tasks.cli.run_experiment \
    --policy G1_allergy_V2 -n 10 --method amos --dev

# Embedding filter + thorough sweep
.venv/bin/python -m addm.tasks.cli.run_experiment \
    --policy G1_allergy_V2 -n 10 --method amos --filter-mode embedding --dev

# Hybrid filter + thorough sweep
.venv/bin/python -m addm.tasks.cli.run_experiment \
    --policy G1_allergy_V2 -n 10 --method amos --filter-mode hybrid --dev
```

### Two-Stage Flow

```
┌─────────────────────────────────────────────────────┐
│ Stage 1: Quick Scan                                 │
│   Filter by --filter-mode (keyword/embedding/hybrid)│
│   Extract signals from filtered reviews             │
│   IF severe evidence → EARLY EXIT (done)            │
└─────────────────────────────────────────────────────┘
                        │
                        │ (no early exit)
                        ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: Thorough Sweep (always on)                 │
│   Process ALL remaining reviews                     │
│   Recompute final verdict                           │
└─────────────────────────────────────────────────────┘
```

## AMOS File Structure

AMOS lives entirely in `src/addm/methods/amos/`:

| File | Purpose |
|------|---------|
| `amos/__init__.py` | `AMOSMethod` class + module exports |
| `amos/phase1.py` | Entry point for Formula Seed generation |
| `amos/phase1_plan_and_act.py` | PLAN_AND_ACT approach (primary) |
| `amos/phase2.py` | Formula Seed interpreter (two-stage retrieval) |
| `amos/config.py` | `AMOSConfig`, `FilterMode` configuration |

**Import:** `from addm.methods.amos import AMOSMethod, FilterMode`

---

**Full documentation:** [docs/BASELINES.md](../../docs/BASELINES.md) | **Add new method:** [docs/developer/add-method.md](../../docs/developer/add-method.md)
