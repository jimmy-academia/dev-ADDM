# ADDM Repository Guidelines

## Git Rules

### NEVER do these:
- **NEVER add "Co-Authored-By:" or similar co-author lines to commits**
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
- Create venv: `python -m venv .venv`
- Install deps: `.venv/bin/pip install -r requirements.txt`

## Project Structure

```
data/
├── raw/yelp/           # Raw Yelp academic dataset
├── processed/yelp/     # Built datasets (K=25/50/100/200)
└── tasks/yelp/         # Task prompts, cache, ground truth

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
```

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
- Run tests: `.venv/bin/python -m pytest`

## Coding Style

- Indentation: 4 spaces; UTF-8; keep lines <= 100 chars
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants

## Testing Guidelines

- Place tests in `tests/` with names like `test_<module>.py`
- Name test functions `test_<behavior>()`
- Keep unit tests fast and deterministic; isolate external I/O behind mocks

## Current Status

- G1a: Complete (formula + cache + ground truth)
- All other 71 tasks: Prompts exist, need formula + ground truth
