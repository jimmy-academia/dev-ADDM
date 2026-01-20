# ADDM - Adaptive Decision-Making with LLMs

Research benchmark: 72 policy evaluation tasks across restaurant reviews (Yelp).

## Critical Rules

- **Python**: Always use `.venv/bin/python`
- **LLM**: Default `gpt-5-nano`. ASK before switching models.
- **Console**: Use `output` singleton from `src/addm/utils/output.py`, not `print()`
- **Git**: See global rules. Additional: NEVER add "Co-Authored-By: Claude" lines.
- **Docs**: Check `docs/` FIRST - before writing code, creating docs, or answering questions. Link, don't duplicate.

## Quick Start

**First time?** See the full [Quickstart Guide](../docs/quickstart.md) for step-by-step setup.

```bash
# Run experiment
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 5 --dev

# Extract ground truth
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --mode 24hrbatch

# Compute GT from extractions
.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V2 --k 200
```

## Documentation Index

| Topic | Location |
|-------|----------|
| Project Status & Timeline | [docs/ROADMAP.md](../docs/ROADMAP.md) |
| Architecture & Structure | [docs/architecture.md](../docs/architecture.md) |
| Benchmark (72 Tasks) | [.claude/rules/benchmark.md](rules/benchmark.md) |
| CLI Commands | [.claude/rules/cli-commands.md](rules/cli-commands.md) |
| Ground Truth Pipeline | [.claude/rules/ground-truth.md](rules/ground-truth.md) |
| Methods (direct/rlm/rag/amos) | [.claude/rules/methods.md](rules/methods.md) |
| Data Pipeline | [.claude/rules/data-pipeline.md](rules/data-pipeline.md) |
| Logging Conventions | [.claude/rules/logging.md](rules/logging.md) |
