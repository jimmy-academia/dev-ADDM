# ADDM Project Memory

## Active Work

**Current Phase**: Phase I - G1_allergy Pipeline Validation
**Status**: G1_allergy extracted (raw), ready for aggregation
**Next**: Aggregate G1_allergy GT → implement baselines → AMOS development

**Strategy**: Two-phase approach
- **Week 1 (Phase I)**: Validate full pipeline on G1_allergy before scaling
- **Week 2 (Phase II)**: Scale to all 18 topics with batch runs

**Today's Focus** (Jan 18):
- [ ] A1: Aggregate G1_allergy raw judgments → consensus L0
- [ ] B1: Polish Introduction, define key claims for Discussion

**Deadlines**:
- Goal: Feb 1 (14 days)
- Hard: Feb 8 (21 days)

See `docs/ROADMAP.md` for detailed operational roadmap with daily tasks.

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
└── answers/{dataset}/      # Ground truth & caches
    ├── judgement_cache.json        # L0 judgement cache (raw + aggregated)
    ├── *_K*_groundtruth.json       # Ground truth files
    ├── batch_manifest_*.json       # Multi-batch tracking (gitignored)
    └── batch_errors_*.jsonl        # Batch API errors for diagnostics (gitignored)

results/
├── dev/{timestamp}_{name}/     # Dev run outputs (results.json)
├── prod/                       # Production runs
├── cache/                      # Method caches (gitignored)
│   └── rag_embeddings.json     # RAG embedding/retrieval cache
└── logs/                       # Debug & pipeline logs (gitignored)
    ├── extraction/             # Extraction pipeline logs
    └── debug/{run_id}/         # LLM prompt/response capture

docs/
├── sessions/           # Claude session logs (written by /bye)
├── README.md           # Doc index
├── architecture.md     # System overview
├── tasks/TAXONOMY.md   # 72 task definitions (6 groups × 12 tasks)
└── specs/              # Detailed specifications

scripts/
├── data/               # Data preparation scripts
│   ├── build_dataset.py
│   ├── build_topic_selection.py
│   ├── select_topic_restaurants.py
│   └── download_amazon.sh
├── utils/              # Utility scripts
│   ├── verify_formulas.py
│   └── test_allergy_query.py
├── run_g1_allergy.sh   # G1_allergy extraction pipeline
└── manual_review.txt   # Reference doc

src/addm/
├── methods/            # LLM methods (direct, rlm, rag, amos)
├── tasks/              # Extraction, execution, CLI
├── query/              # Query construction system
│   ├── models/         # PolicyIR, Term, Operator
│   ├── libraries/      # Term & operator YAML files
│   ├── policies/       # Policy definitions (G1-G6, V0-V3)
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
.venv/bin/python scripts/data/build_topic_selection.py --data yelp
.venv/bin/python scripts/data/select_topic_restaurants.py --data yelp
.venv/bin/python scripts/data/build_dataset.py --data yelp

# Rebuild datasets only (if topic_100.json exists)
.venv/bin/python scripts/data/build_dataset.py --data yelp
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
- Generate prompt: `.venv/bin/python -m addm.query.cli.generate --policy G1/allergy/V2`

**Useful flags:**
- `--dry-run` (extract): Test without API calls
- `--verbose` (extract, compute_gt): Detailed output

**Pipeline script:**
```bash
# Run G1_allergy GT extraction (waits for batches, Ctrl+C to stop)
./scripts/run_g1_allergy.sh

# Logs: results/logs/extraction/g1_allergy.log
```

## Session Workflow

**Starting a session**: Use `/hello` (runs automatically via session-startup.md)
- Reads `docs/ROADMAP.md` for project status and timeline
- Reads recent session logs from `docs/sessions/`
- Shows project phase, today's focus, and suggested todos

**Ending a session**: Use `/bye` to:
- Sync documentation (automatic)
- Check git status and uncommitted work
- Document incomplete todos
- Capture session notes to `docs/sessions/`
- Get clean exit instructions

**Tracking progress**: Use `/roadmap` to update project milestones

**Exit cleanly**: After `/bye`, use `Ctrl+C` or `Ctrl+D` (safest for MCP servers).

## Methods

Available methods for `--method` flag in `run_baseline.py`:

| Method | Description | Token Cost |
|--------|-------------|------------|
| `direct` | Send all K reviews in prompt (default) | ~K×200 tokens |
| `rlm` | Recursive LLM - code execution to search reviews | ~50k tokens (16 iters) |
| `rag` | Retrieval-Augmented Generation - embed & retrieve relevant reviews | ~5k tokens (varies by --top-k) |
| `amos` | Adaptive Multi-Output Sampling (proposed method) | ~5k tokens (seed + sampling) |

See `docs/BASELINES.md` for detailed specifications of each method.

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

- **Policy system**: ✅ Complete - Transitioned from formula modules to policy-based approach
  - ✅ ALL 72 policy definitions complete (G1-G6, all topics, V0-V3)
  - ✅ Term libraries for all 18 topics (`src/addm/query/libraries/terms/`)
  - ✅ Experiment code updated (`--policy`, `--dev` flags)
  - 📝 Legacy task IDs (G1a-G6l) still supported for backward compatibility
- **Ground Truth Generation**: ✅ Infrastructure complete, 🔄 G1_allergy in progress
  - ✅ `src/addm/tasks/policy_gt.py` - Core GT computation logic
  - ✅ `src/addm/tasks/extraction.py` - PolicyJudgmentCache with dual cache
  - ✅ `src/addm/tasks/cli/extract.py` - Multi-model batch extraction (`--topic`, `--policy`, `--all`)
  - ✅ `src/addm/tasks/cli/compute_gt.py` - Policy-based GT computation
  - ✅ `docs/specs/ground_truth.md` - Full documentation
  - ✅ G1_allergy: 16 GT files complete (V0-V3 × K=25/50/100/200)
  - 🔄 G2-G6 topics: Pending (17 topics remaining)
- **Methods**: See `docs/BASELINES.md` for full details
  - ✅ Direct baseline (default)
  - ✅ RLM baseline (code-execution approach)
  - ✅ RAG baseline (retrieval-based approach)
  - ✅ AMOS (proposed method - Adaptive Multi-Output Sampling)
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
- `data/answers/yelp/judgement_cache.json` - Cached L0 judgements
- `results/cache/rag_embeddings.json` - RAG embedding/retrieval cache (gitignored)
- `data/answers/yelp/{policy}_K{k}_groundtruth.json` - GT outputs

See `docs/specs/ground_truth.md` for full details.
