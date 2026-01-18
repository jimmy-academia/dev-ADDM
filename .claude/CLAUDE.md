# ADDM Project Memory

## Git Rules

### NEVER do these:
- **NEVER add "Co-Authored-By: Claude" or similar co-author lines to commits**
- Never use vague commit messages like "just another push for progress"
- **NEVER switch LLM models without asking** - Always use `gpt-5-nano` as default. If it fails, ASK before trying another model.

### Commit message format:
```
<type>: <short description>

[optional body explaining WHY]
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `perf`, `chore`

## Python Environment

- **Always use `.venv`** - Run Python with `.venv/bin/python`
- Example: `.venv/bin/python -m pytest` or `.venv/bin/python src/addm/main.py`

## Project Structure

```
data/
├── raw/{dataset}/          # Raw academic dataset
├── hits/{dataset}/         # Topic analysis results (~1.2GB)
├── selection/{dataset}/    # Restaurant selections (topic_100.json)
├── context/{dataset}/      # Built datasets (K=25/50/100/200)
└── query/{dataset}/        # Task prompts

docs/
├── README.md           # Doc index
├── architecture.md     # System overview
├── tasks/TAXONOMY.md   # 72 task definitions (6 groups × 12 tasks)
└── specs/              # Detailed specifications

src/addm/
├── methods/            # LLM methods (direct, etc.)
├── tasks/formulas/     # Formula modules per task
├── tasks/              # Extraction, execution, CLI
├── query/              # Query construction system
│   ├── models/         # PolicyIR, Term, Operator
│   ├── libraries/      # Term & operator YAML files
│   ├── policies/       # Policy definitions (V0-V3)
│   └── generators/     # Prompt generation
├── data/               # Dataset loaders
├── eval/               # Evaluation metrics
└── llm.py              # LLM service

.claude/                # Claude Code configuration
```

## Data Notes

**Hits lists** (`data/hits/yelp/G1_*.json`):
- `critical_list` and `high_list` are regex-based, not 100% accurate
- Use as starting point for finding relevant restaurants
- Critical reviews may not be in K=25 subset (sampling effect)
- Always verify actual review content in the dataset

**Research goal**: Find complexity that causes baseline LLM to make errors
- Test across K=25, 50, 100, 200 to observe scaling behavior
- Look for cases where more context leads to incorrect verdicts

## Context Pipeline

```bash
# Full pipeline (analyze → select → build)
.venv/bin/python scripts/build_topic_selection.py --data yelp
.venv/bin/python scripts/select_topic_restaurants.py --data yelp
.venv/bin/python scripts/build_dataset.py --data yelp

# Rebuild datasets only (if topic_100.json exists)
.venv/bin/python scripts/build_dataset.py --data yelp
```

**Selection algorithm**: Layered greedy by cell coverage
- 18 topics × 2 severities (critical/high) = 36 cells
- Fills level n (all cells ≥n) before level n+1
- Multi-topic restaurants prioritized

**Query Pipeline**: PolicyIR → NL prompts
```bash
# Generate prompt from policy
.venv/bin/python -m addm.query.cli.generate \
    --policy src/addm/query/policies/G1/allergy/V2.yaml \
    --output data/query/yelp/G1_allergy_V2_prompt.txt
```

See `docs/specs/query_construction.md` for details.

## Usage Tracking

**Output files per run:**
```
results/dev/{timestamp}_{name}/
├── results.jsonl    # Per-sample with usage fields
├── usage.json       # Run-level aggregation
└── debug/           # Full prompts/responses
    └── {sample_id}.jsonl
```

**Key modules:**
- `src/addm/utils/usage.py` - UsageTracker singleton, MODEL_PRICING, compute_cost
- `src/addm/utils/debug_logger.py` - DebugLogger for prompt/response capture

**Usage in methods:**
```python
# Call with usage tracking
response, usage = await llm.call_async_with_usage(messages, context={"sample_id": id})

# Accumulate multiple calls
total_usage = self._accumulate_usage([usage1, usage2, ...])
```

**Model pricing**: See `MODEL_PRICING` in `src/addm/utils/usage.py` or `docs/specs/usage_tracking.md`

## Benchmark: 72 Tasks

**Structure:** 6 Groups × 3 Topics × 4 Variants

| Group | Perspective | Topics |
|-------|-------------|--------|
| G1 | Customer | Allergy, Dietary, Hygiene |
| G2 | Customer | Romance, Business, Group |
| G3 | Customer | Price-Worth, Hidden Costs, Time-Value |
| G4 | Owner | Server, Kitchen, Environment |
| G5 | Owner | Capacity, Execution, Consistency |
| G6 | Owner | Uniqueness, Comparison, Loyalty |

**Variants:**

*Legacy (a/b/c/d)* - Formula complexity × L1.5:
- **a** = simple formula
- **b** = simple + L1.5 grouping
- **c** = complex formula (credibility weighting)
- **d** = complex + L1.5

*New (V0-V3)* - Policy evolution:
- **V0** = Base (aggregation, multiple incidents required)
- **V1** = +Override (single-instance triggers)
- **V2** = +Scoring (point system, thresholds)
- **V3** = +Recency (time decay, exceptions)

## Quick Reference

**Run baseline:**
```bash
# Legacy task-based
.venv/bin/python -m addm.tasks.cli.run_baseline --task G1a -n 5

# New policy-based
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5

# Dev mode (results/dev/{timestamp}_{id}/)
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --dev

# RLM method (code-execution baseline)
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 1 --method rlm

# Batch mode (24hr async execution)
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 100 --mode 24hrbatch
```

**Other commands:**
- Compute GT: `.venv/bin/python -m addm.tasks.cli.compute_gt --task G1a --domain yelp --k 50`
- Extract judgments: `.venv/bin/python -m addm.tasks.cli.extract --task G1a`
- Verify formulas: `.venv/bin/python scripts/verify_formulas.py`
- Generate prompt: `.venv/bin/python -m addm.query.cli.generate --policy G1/allergy/V2`

## Session Workflow

**Starting a session**: Read this file (done automatically via session-startup.md)

**Ending a session**: Use `/bye` to:
- Sync documentation (automatic)
- Check git status and uncommitted work
- Document incomplete todos
- Capture session notes
- Get clean exit instructions

**Exit cleanly**: After `/bye`, use `Ctrl+C` or `Ctrl+D` (safest for MCP servers).

## Methods

Available methods for `--method` flag in `run_baseline.py`:

| Method | Description | Token Cost |
|--------|-------------|------------|
| `direct` | Send all K reviews in prompt (default) | ~K×200 tokens |
| `rlm` | Recursive LLM - code execution to search reviews | ~50k tokens (16 iters) |

**RLM Method** (`src/addm/methods/rlm.py`):
- Uses [recursive-llm](https://github.com/ysz/recursive-llm) library (forked to `lib/recursive-llm/`)
- Stores reviews as Python variable, LLM writes code to search
- Token budget: `--token-limit 50000` (default), ~3000 tokens/iteration
- Known limitation: gpt-5-nano produces inconsistent results with RLM

**Running RLM:**
```bash
# Basic RLM run
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 1 --k 50 --dev --method rlm

# With custom token limit
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 1 --k 50 --dev --method rlm --token-limit 30000
```

**Token budget comparison:**
- ANoT (user's method): ~5k tokens/restaurant
- RLM: ~50k tokens/restaurant (10x, accepted as reasonable for comparison)

## Current Status

- **Formula modules**: ✅ All 72 complete (G1a-G6l)
- **Verification**: ✅ All pass - see `scripts/verify_formulas.py`
- **Manual review**: See `scripts/manual_review.txt`
- **Ground truth**: Pending
- **Query construction**: In progress
  - ✅ ALL 72 policy definitions complete (G1-G6, all topics, V0-V3)
  - ✅ Experiment code updated (`--policy`, `--dev` flags)
  - ⏳ Prompt generation: Only G1_allergy_V0-V3 generated, others pending
- **Baselines**: See `docs/BASELINES.md` for full details
- **RLM Method**: ⚠️ Implemented but unreliable with gpt-5-nano
  - ✅ `src/addm/methods/rlm.py` created
  - ✅ `--method` and `--token-limit` CLI flags added
  - ✅ recursive-llm forked to `lib/recursive-llm/`
  - ⚠️ gpt-5-nano outputs inconsistent results (sometimes literal placeholders)
  - ⏳ Decision pending: accept unreliability, try other model, or document limitation
