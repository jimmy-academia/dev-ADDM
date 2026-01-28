# CLI Commands Reference

## Run Experiments

```bash
# Default: benchmark mode (quota-controlled)
.venv/bin/python -m addm.tasks.cli.run_experiment --policy T1P1 -n 100

# Dev mode (no quota, saves to results/dev/)
.venv/bin/python -m addm.tasks.cli.run_experiment --policy T1P1 -n 5 --dev

# All variants for a tier (7 policies)
.venv/bin/python -m addm.tasks.cli.run_experiment --tier T1 --dev

# Multiple tiers (14 policies)
.venv/bin/python -m addm.tasks.cli.run_experiment --tier T1,T2 --dev

# All 35 policies (omit --policy and --tier)
.venv/bin/python -m addm.tasks.cli.run_experiment --dev

# Force run even if quota is met
.venv/bin/python -m addm.tasks.cli.run_experiment --policy T1P1 -n 100 --force

# Specific method
.venv/bin/python -m addm.tasks.cli.run_experiment --policy T1P1 -n 1 --method rlm

# AMOS Phase Control
# Phase 1 only: generate Agenda Spec
.venv/bin/python -m addm.tasks.cli.run_experiment --policy T1P1 --phase 1 --method amos --dev

# Phase 2 only: use pre-generated seed (file path)
.venv/bin/python -m addm.tasks.cli.run_experiment --policy T1P1 --phase 2 --seed path/to/T1P1.json -n 5 --method amos --dev

# Phase 2 with seed directory (auto-finds {policy_id}.json)
.venv/bin/python -m addm.tasks.cli.run_experiment --policy T1P1 --phase 2 --seed results/dev/seeds/ -n 5 --method amos --dev

# Batch mode (24hr async)
.venv/bin/python -m addm.tasks.cli.run_experiment --policy T1P1 -n 100 --mode batch
```

## Ground Truth Pipeline

```bash
# Step 1: Extract L0 judgments (uses G* topics for raw extractions)
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --mode batch

# Step 2: Compute GT for T* policies
.venv/bin/python -m addm.tasks.cli.compute_gt --policy T1P3 --k 200

# Compute GT for all K values
for k in 25 50 100 200; do
  .venv/bin/python -m addm.tasks.cli.compute_gt --policy T1P3 --k $k
done
```

## Prompt Generation

```bash
# Generate all 35 policy prompts (default)
.venv/bin/python -m addm.query.cli.generate

# Generate single policy
.venv/bin/python -m addm.query.cli.generate --policy T1P1
```

## Useful Flags

| Flag | Commands | Description |
|------|----------|-------------|
| `--dev` | run_experiment | Dev mode (saves to results/dev/, no quota) |
| `--force` | run_experiment | Override benchmark quota |
| `--method` | run_experiment | Method (direct/cot/react/rlm/rag/amos) |
| `--phase` | run_experiment | AMOS phase: '1' (seed only), '2' (use seed), '1,2' or omit (both) |
| `--batch-size` | run_experiment | AMOS: Reviews per LLM call (default: 10) |
| `--seed` | run_experiment | Path to Agenda Spec file/dir for --phase 2 |
| `--tier` | run_experiment | Run all P1-P7 variants for tier(s) (e.g., T1 or T1,T2) |
| `--k` | all | Context size (25/50/100/200) |
| `-n` | run_experiment | Number of samples |
| `--token-limit` | run_experiment | RLM token budget |

## Benchmark Quota

Default quota: 5 runs per policy/K combination (1 ondemand + 4 batch).

**Configuration:** `src/addm/utils/results_manager.py:27`
```python
DEFAULT_QUOTA = 5  # Adjust this constant to change quota
```

## Scripts Directory

```
scripts/
├── data/                           # Data pipeline (production)
│   ├── build_topic_selection.py
│   ├── select_topic_restaurants.py
│   └── build_dataset.py
├── select_diverse_samples.py       # Sample selection utility
├── generate_evidence_fields.py     # Sync EVIDENCE_FIELDS with term libraries
└── debug/                          # One-off debugging scripts
```

### Utility Scripts

```bash
# Update verdict samples after GT changes
.venv/bin/python scripts/select_diverse_samples.py --all --output data/answers/yelp/verdict_sample_ids.json

# Check EVIDENCE_FIELDS matches term libraries (run before releases)
.venv/bin/python scripts/generate_evidence_fields.py --check

# Regenerate EVIDENCE_FIELDS from term libraries
.venv/bin/python scripts/generate_evidence_fields.py --write
```

---

**Full documentation:** [docs/specs/cli.md](../../docs/specs/cli.md) | **Troubleshooting:** [docs/troubleshooting.md](../../docs/troubleshooting.md)
