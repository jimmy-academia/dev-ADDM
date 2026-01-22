# ADDM - Adaptive Decision-Making with LLMs

Research benchmark: 72 policy evaluation tasks across restaurant reviews (Yelp).

---

## ⚠️ MULTI-AGENT COORDINATION (2026-01-22)

**5 Claude agents are currently active.** Multiple agents are working on AMOS fixes.

**Before editing AMOS files**, check this coordination log:

| Agent | Terminal | Working On | Status |
|-------|----------|------------|--------|
| Agent 1 | s005 | Coordinator (this note) | Active |
| Agent 2 | s001 | eval/metrics.py type bug fix | Active |
| Agent 3 | s007 | ? | Active |
| Agent 4 | s000 | ? | Sleeping |
| Agent 5 | s006 | ? | Sleeping |

**Files at risk of conflict:**
- `src/addm/methods/amos/phase2.py` - 230+ uncommitted changes
- `src/addm/methods/amos/seed_transform.py` - new untracked file

**BEFORE editing AMOS**: Update this table with what you're working on. ASK the user if unsure.

---

## Critical Rules

- **Python**: Always use `.venv/bin/python`
- **LLM**: Default `gpt-5-nano`. ASK before switching models.
- **Console**: Use `output` singleton from `src/addm/utils/output.py`, not `print()`
- **Git**: See global rules. Additional: NEVER add "Co-Authored-By: Claude" lines.
- **Docs**: Two-tier system. Details go in `docs/`, `.claude/` links to them. See below.

## Documentation Strategy

| Layer | Purpose | Content |
|-------|---------|---------|
| `.claude/` | Claude memory | Concise rules, quick reference, links to docs |
| `docs/` | Full documentation | Detailed specs, shared by humans & Claude |

**Principle**: Put details in `docs/`. Use `.claude/rules/` as a streamlined index that points Claude to the right doc. Check `docs/` FIRST before writing code or answering questions.

## Quick Start

**First time?** See the full [Quickstart Guide](../docs/quickstart.md) for step-by-step setup.

```bash
# Run experiment
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 5 --dev

# Extract ground truth
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --mode batch

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
| AMOS Phase 1 | [docs/specs/phase1_formula_seed_generation.md](../docs/specs/phase1_formula_seed_generation.md) |
| Data Pipeline | [.claude/rules/data-pipeline.md](rules/data-pipeline.md) |
| Logging Conventions | [.claude/rules/logging.md](rules/logging.md) |
