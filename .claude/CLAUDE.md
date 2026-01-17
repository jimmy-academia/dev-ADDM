# ADDM Project Memory

## Git Rules

### NEVER do these:
- **NEVER add "Co-Authored-By: Claude" or similar co-author lines to commits**
- Never use vague commit messages like "just another push for progress"

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
├── keyword_hits/{dataset}/ # Keyword search results
├── selected/{dataset}/     # Restaurant selections (topic_100.json)
├── processed/{dataset}/    # Built datasets (K=25/50/100/200)
└── tasks/{dataset}/        # Task prompts, cache, ground truth

docs/
├── README.md           # Doc index
├── architecture.md     # System overview
├── tasks/TAXONOMY.md   # 72 task definitions (6 groups × 12 tasks)
└── specs/              # Detailed specifications

src/addm/
├── methods/            # LLM methods (direct, etc.)
├── tasks/formulas/     # Formula modules per task
├── tasks/              # Extraction, execution, CLI
├── data/               # Dataset loaders
├── eval/               # Evaluation metrics
└── llm.py              # LLM service

.claude/                # Claude Code configuration
```

## Data Pipeline

```bash
# Full pipeline (search → select → build)
.venv/bin/python scripts/search_restaurants.py --data yelp --all
.venv/bin/python scripts/select_topic_restaurants.py --data yelp
.venv/bin/python scripts/build_dataset.py --data yelp

# Rebuild datasets only (if topic_100.json exists)
.venv/bin/python scripts/build_dataset.py --data yelp
```

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

**Variants (all produce different verdicts):**
- **a** = simple formula
- **b** = simple + L1.5 grouping
- **c** = complex formula (credibility weighting)
- **d** = complex + L1.5

## Quick Reference

- Run baseline: `.venv/bin/python -m addm.tasks.cli.run_baseline --task G1a -n 5`
- Compute GT: `.venv/bin/python -m addm.tasks.cli.compute_gt --task G1a --domain yelp --k 50`
- Extract judgments: `.venv/bin/python -m addm.tasks.cli.extract --task G1a`
- Verify formulas: `.venv/bin/python scripts/verify_formulas.py`

## Current Status

- **Formula modules**: ✅ All 72 complete (G1a-G6l)
- **Verification**: ✅ All pass - see `scripts/verify_formulas.py`
- **Manual review**: See `scripts/manual_review.txt`
- **Ground truth**: Pending
