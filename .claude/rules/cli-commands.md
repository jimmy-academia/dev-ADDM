# CLI Commands Reference

## Run Experiments

```bash
# Default: benchmark mode (quota-controlled)
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 100

# Dev mode (no quota, saves to results/dev/)
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 5 --dev

# Force run even if quota is met
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 100 --force

# Specific method
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 1 --method rlm

# Batch mode (24hr async)
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 100 --mode 24hrbatch

# Legacy task ID (still works)
.venv/bin/python -m addm.tasks.cli.run_experiment --task G1a -n 5
```

## Ground Truth Pipeline

```bash
# Step 1: Extract L0 judgments
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --mode 24hrbatch

# Step 2: Compute GT from extractions
.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V2 --k 200

# All 72 policies at once
.venv/bin/python -m addm.tasks.cli.compute_gt --k 200
```

## Prompt Generation

```bash
# Generate all 72 policy prompts (default)
.venv/bin/python -m addm.query.cli.generate

# Generate single policy
.venv/bin/python -m addm.query.cli.generate --policy G1/allergy/V2
```

## Useful Flags

| Flag | Commands | Description |
|------|----------|-------------|
| `--dev` | run_experiment | Dev mode (saves to results/dev/, no quota) |
| `--force` | run_experiment | Override benchmark quota |
| `--method` | run_experiment | Method (direct/rlm/rag/amos) |
| `--models` | extract | Custom model config (e.g., "gpt-5-nano:3,gpt-5-mini:1") |
| `--k` | all | Context size (25/50/100/200) |
| `-n` | run_experiment | Number of samples |
| `--token-limit` | run_experiment | RLM token budget |
| `--dry-run` | extract | Test without API calls |
| `--verbose` | extract, compute_gt | Detailed output |

## Benchmark Quota

Default quota: 5 runs per policy/K combination (1 ondemand + 4 batch).

**Configuration:** `src/addm/utils/results_manager.py:27`
```python
DEFAULT_QUOTA = 5  # Adjust this constant to change quota
```

## Pipeline Script

```bash
# Full G1_allergy extraction (waits for batches)
./scripts/run_g1_allergy.sh
# Logs: results/logs/extraction/g1_allergy.log
```

---

**Full documentation:** [docs/specs/cli.md](../../docs/specs/cli.md) | **Troubleshooting:** [docs/troubleshooting.md](../../docs/troubleshooting.md)
