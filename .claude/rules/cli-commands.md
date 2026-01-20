# CLI Commands Reference

## Run Experiments

```bash
# Basic run (policy-based)
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5

# Dev mode (saves to results/dev/)
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --dev

# Specific method
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 1 --method rlm

# Batch mode (24hr async)
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 100 --mode 24hrbatch

# Legacy task ID (still works)
.venv/bin/python -m addm.tasks.cli.run_baseline --task G1a -n 5
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
.venv/bin/python -m addm.query.cli.generate --policy G1/allergy/V2
```

## Useful Flags

| Flag | Commands | Description |
|------|----------|-------------|
| `--dry-run` | extract | Test without API calls |
| `--verbose` | extract, compute_gt | Detailed output |
| `--k` | all | Context size (25/50/100/200) |
| `-n` | run_baseline | Number of samples |
| `--dev` | run_baseline | Save to results/dev/ |
| `--method` | run_baseline | Method (direct/rlm/rag/amos) |
| `--token-limit` | run_baseline | RLM token budget |

## Pipeline Script

```bash
# Full G1_allergy extraction (waits for batches)
./scripts/run_g1_allergy.sh
# Logs: results/logs/extraction/g1_allergy.log
```

---

**Full documentation:** [docs/specs/cli.md](../../docs/specs/cli.md) | **Troubleshooting:** [docs/troubleshooting.md](../../docs/troubleshooting.md)
