# Methods

Available methods for `--method` flag in `run_experiment.py`:

| Method | Description | Token Cost |
|--------|-------------|------------|
| `direct` | All K reviews in prompt (default) | ~K×200 tokens |
| `cot` | Chain-of-Thought step-by-step reasoning | ~K×250 tokens |
| `react` | Reasoning + Acting with tool use | ~K×300+ tokens |
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
    --policy G1_allergy_V3 -n 1 --k 50 --dev --method rlm

# Custom token limit
.venv/bin/python -m addm.tasks.cli.run_experiment \
    --policy G1_allergy_V3 -n 1 --k 50 --dev --method rlm --token-limit 30000
```

## AMOS Method

Two-phase method for comprehensive review analysis:

**Phase 1**: Generate Formula Seed (extraction spec) from agenda
**Phase 2**: Process ALL reviews in parallel batches, compute verdict

```bash
# Run AMOS (both phases)
.venv/bin/python -m addm.tasks.cli.run_experiment \
    --policy G1_allergy_V3 -n 10 --method amos --dev

# Phase 1 only (generate seed)
.venv/bin/python -m addm.tasks.cli.run_experiment \
    --policy G1_allergy_V3 --phase 1 --method amos --dev

# Phase 2 only (use pre-generated seed)
.venv/bin/python -m addm.tasks.cli.run_experiment \
    --policy G1_allergy_V3 --phase 2 --seed results/dev/seeds/ -n 5 --method amos --dev
```

### AMOS Flow

```
┌─────────────────────────────────────────────────────┐
│ Phase 1: Generate Formula Seed                      │
│   OBSERVE → PLAN → ACT pipeline                     │
│   Produces extraction spec (keywords, fields, rules)│
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ Phase 2: Execute Extraction                         │
│   Process ALL reviews in parallel batches           │
│   Extract signals, compute verdict                  │
└─────────────────────────────────────────────────────┘
```

## AMOS File Structure

AMOS lives entirely in `src/addm/methods/amos/`:

| File | Purpose |
|------|---------|
| `amos/__init__.py` | `AMOSMethod` class + module exports |
| `amos/phase1.py` | Entry point for Formula Seed generation |
| `amos/phase1_plan_and_act.py` | PLAN_AND_ACT approach (primary) |
| `amos/phase2.py` | Formula Seed interpreter (extraction) |
| `amos/config.py` | `AMOSConfig` configuration |

**Import:** `from addm.methods.amos import AMOSMethod, AMOSConfig`

---

**Full documentation:** [docs/BASELINES.md](../../docs/BASELINES.md) | **Add new method:** [docs/developer/add-method.md](../../docs/developer/add-method.md)
