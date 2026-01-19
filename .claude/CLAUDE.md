# ADDM Project Memory

## Active Work

**Current Phase**: Ground Truth Generation (Phase 3)
**Status**: Multi-model extraction framework complete, extracting 18 topics
**Next**: Complete topic extraction, compute GT for all 72 policies

See `docs/ROADMAP.md` for full project progress.

---

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
├── query/{dataset}/        # Task prompts
└── tasks/{dataset}/        # Extraction cache & batch files
    ├── policy_cache.json       # L0 judgment cache (raw + aggregated)
    ├── batch_manifest_*.json   # Multi-batch tracking
    ├── batch_manifest_*.log    # Manifest sidecar log (auto-deleted on completion)
    └── batch_*.log             # Single-batch log (auto-deleted on completion)

results/
├── dev/{timestamp}_{name}/     # Dev run outputs
│   ├── results.json            # Run results
│   └── debug/                  # Debug logs (prompt/response capture)
└── prod/                       # Production runs

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

## Output & Logging

**Console output**: Use `output` singleton from `src/addm/utils/output.py`, not raw `print()`:
```python
from addm.utils.output import output

output.status("Processing...")   # Suppressed in quiet mode (default for cron)
output.info("Important info")    # Always shown (blue ℹ)
output.success("Done!")          # Always shown (green ✓)
output.warn("Warning")           # Always shown (yellow ⚠)
output.error("Failed")           # Always shown (red ✗)
output.print("Plain text")       # Always shown (no prefix)
```

**Output files per run:**
```
results/dev/{timestamp}_{name}/
└── results.json     # Single JSON with run metadata + per-sample results array
```

**Key modules:**
- `src/addm/utils/output.py` - OutputManager singleton, Rich console output
- `src/addm/utils/logging.py` - ResultLogger for experiment result capture
- `src/addm/utils/usage.py` - UsageTracker singleton, MODEL_PRICING, compute_cost
- `src/addm/utils/debug_logger.py` - DebugLogger for prompt/response capture (if enabled)

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
- Extract L0: `.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --mode 24hrbatch`
- Compute GT: `.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V2 --k 200`
- Verify formulas: `.venv/bin/python scripts/verify_formulas.py`
- Generate prompt: `.venv/bin/python -m addm.query.cli.generate --policy G1/allergy/V2`

**Useful flags:**
- `--show-status` (extract): Show progress during cron runs
- `--dry-run` (extract): Test without API calls
- `--verbose` (extract, compute_gt): Detailed output

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
- **Query construction**: ✅ Complete
  - ✅ ALL 72 policy definitions complete (G1-G6, all topics, V0-V3)
  - ✅ Term libraries for all 18 topics (`src/addm/query/libraries/terms/`)
  - ✅ Experiment code updated (`--policy`, `--dev` flags)
- **Ground Truth Generation**: ✅ Complete - Two-step policy-based flow
  - ✅ `src/addm/tasks/policy_gt.py` - Core GT computation logic
  - ✅ `src/addm/tasks/extraction.py` - PolicyJudgmentCache with dual cache
  - ✅ `src/addm/tasks/cli/extract.py` - Multi-model batch extraction (`--topic`, `--policy`, `--all`)
  - ✅ `src/addm/tasks/cli/compute_gt.py` - Policy-based GT computation
  - ✅ `docs/specs/ground_truth.md` - Full documentation
- **Baselines**: See `docs/BASELINES.md` for full details
- **RLM Method**: ⚠️ Implemented but unreliable with gpt-5-nano
  - ✅ `src/addm/methods/rlm.py` created
  - ✅ `--method` and `--token-limit` CLI flags added
  - ✅ recursive-llm forked to `lib/recursive-llm/`
  - ⚠️ gpt-5-nano outputs inconsistent results (sometimes literal placeholders)

## Ground Truth Generation

Two-step flow for policy-based GT:

**Step 1: Extract L0 Judgments** (multi-model ensemble)
```bash
# Extract for all topics (default)
.venv/bin/python -m addm.tasks.cli.extract --k 200 --mode 24hrbatch

# Single topic
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --mode 24hrbatch

# Custom model config (default: gpt-5-mini:1, gpt-5-nano:3)
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --mode 24hrbatch \
    --models "gpt-5-nano:5,gpt-5-mini:3,gpt-5.1:1"
```

**Step 2: Compute Ground Truth**
```bash
# All 72 policies (default)
.venv/bin/python -m addm.tasks.cli.compute_gt --k 200

# Single policy
.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V2 --k 200

# Multiple policies (same topic)
.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V0,G1_allergy_V1,G1_allergy_V2,G1_allergy_V3 --k 200
```

**Key files:**
- `src/addm/tasks/policy_gt.py` - Aggregation, scoring, qualitative evaluation
- `src/addm/tasks/extraction.py` - `PolicyJudgmentCache` with raw/aggregated dual cache
- `data/tasks/yelp/policy_cache.json` - Cached L0 judgments
- `data/tasks/yelp/{policy}_K{k}_groundtruth.json` - GT outputs

See `docs/specs/ground_truth.md` for full details.
