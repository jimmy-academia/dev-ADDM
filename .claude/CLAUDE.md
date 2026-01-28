# ADDM - Adaptive Decision-Making with LLMs

Research benchmark for policy evaluation tasks across restaurant reviews (Yelp).

**T* System**: 5 tiers × 7 variants = 35 policies (T1P1, T1P2, ..., T5P7)

---

## Critical Rules

- **Python**: Always use `.venv/bin/python`
- **LLM**: Default `gpt-5-nano`. ASK before switching models.
- **Console**: Use `output` singleton from `src/addm/utils/output.py`, not `print()`
- **Git**: See global rules. Additional: NEVER add "Co-Authored-By: Claude" lines.
- **Docs**: Two-tier system. Details go in `docs/`, `.claude/` links to them. See below.
- **AMOS Workflow**: Do NOT run experiments directly unless explicitly authorized by the user. Otherwise, prepare the command and hand it to the user to run. This applies to all AMOS debugging/fixing work.

## Documentation Strategy

| Layer | Purpose | Content |
|-------|---------|---------|
| `.claude/` | Claude memory | Concise rules, quick reference, links to docs |
| `docs/` | Full documentation | Detailed specs, shared by humans & Claude |

**Principle**: Put details in `docs/`. Use `.claude/rules/` as a streamlined index that points Claude to the right doc. Check `docs/` FIRST before writing code or answering questions.

## Quick Start

**First time?** See the full [Quickstart Guide](../docs/quickstart.md) for step-by-step setup.

```bash
# Run experiment (single policy)
.venv/bin/python -m addm.tasks.cli.run_experiment --policy T1P1 -n 5 --dev

# Run experiment (all variants for tier)
.venv/bin/python -m addm.tasks.cli.run_experiment --tier T1 --dev

# Extract ground truth
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --mode batch

# Compute GT from extractions
.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V3 --k 200
```

## Documentation Index

| Topic | Location |
|-------|----------|
| Project Status & Timeline | [docs/ROADMAP.md](../docs/ROADMAP.md) |
| Architecture & Structure | [docs/architecture.md](../docs/architecture.md) |
| **Benchmark (T* System)** | [.claude/rules/benchmark.md](rules/benchmark.md) |
| CLI Commands | [.claude/rules/cli-commands.md](rules/cli-commands.md) |
| Ground Truth Pipeline | [.claude/rules/ground-truth.md](rules/ground-truth.md) |
| **Evaluation Metrics (7)** | [.claude/rules/evaluation.md](rules/evaluation.md) |
| Methods (direct/rlm/rag/amos) | [.claude/rules/methods.md](rules/methods.md) |
| AMOS Phase 1 | [docs/specs/phase1_agenda_spec_generation.md](../docs/specs/phase1_agenda_spec_generation.md) |
| Data Pipeline | [.claude/rules/data-pipeline.md](rules/data-pipeline.md) |
| Logging Conventions | [.claude/rules/logging.md](rules/logging.md) |
