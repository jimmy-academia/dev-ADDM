# CLI Reference

Complete documentation for all ADDM command-line interfaces.

## Table of Contents

- [run_baseline](#run_baseline) - Run baseline methods on benchmark tasks
- [extract](#extract) - Extract L0 judgments from reviews
- [compute_gt](#compute_gt) - Compute ground truth from cached judgments
- [generate](#generate) - Generate prompts from policy definitions

---

## run_baseline

Run baseline methods (direct, RLM, RAG, AMOS) on benchmark tasks.

### Basic Usage

```bash
# Policy-based (recommended)
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5

# Legacy task-based
.venv/bin/python -m addm.tasks.cli.run_baseline --task G1a -n 5
```

### Target Selection

**One of these is required:**

| Flag | Description | Example |
|------|-------------|---------|
| `--task` | Legacy task ID | `--task G1a` |
| `--policy` | Policy ID (full or short format) | `--policy G1_allergy_V2` or `--policy G1/allergy/V2` |

### Core Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--domain` | Domain (yelp/amazon) | `yelp` |
| `--k` | Reviews per restaurant | `200` |
| `-n` | Number of restaurants (0=all) | `0` |
| `--skip` | Skip first N restaurants | `0` |
| `--model` | LLM model override | `gpt-5-nano` |
| `--quiet` | Suppress status messages | `False` |
| `--dev` | Save to `results/dev/{timestamp}_{id}/` | `False` |

### Method Selection

| Flag | Description | Default |
|------|-------------|---------|
| `--method` | Method: `direct`, `rlm`, `rag`, `amos` | `direct` |

#### Method-Specific Flags

**RLM (Recursive LLM):**
- `--token-limit` - Token budget per restaurant (default: `50000`)
  - Example: `--method rlm --token-limit 30000`
  - ~3000 tokens/iteration, so 50000 = ~16 iterations

**RAG (Retrieval-Augmented Generation):**
- `--top-k` - Number of reviews to retrieve (default: `20`)
  - Example: `--method rag --top-k 30`
  - Typical: 10% of K (e.g., 20 for K=200)

**AMOS (Adaptive Multi-Output Sampling):**
- `--regenerate-seed` - Force regenerate Formula Seed even if cached
  - Example: `--method amos --regenerate-seed`

### Execution Mode

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | Execution mode: `ondemand` or `24hrbatch` | `ondemand` |
| `--batch-id` | Batch ID for fetch-only runs | `None` |

**Modes:**
- `ondemand`: Synchronous API calls, immediate results
- `24hrbatch`: Async batch API, 24-hour processing window (~50% cheaper)

### Examples

```bash
# Basic run (5 restaurants, default method)
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5

# Dev mode run
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --dev

# RLM method with custom token limit
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 1 --method rlm --token-limit 30000

# RAG method with custom retrieval count
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 10 --method rag --top-k 30

# AMOS method with seed regeneration
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 10 --method amos --regenerate-seed

# Batch mode (async, 24hr)
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 100 --mode 24hrbatch

# Different K value
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --k 50

# Skip first 10 restaurants
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --skip 10
```

### Output

Results saved to:
- Production: `results/{timestamp}_{policy}/results.json`
- Dev mode: `results/dev/{timestamp}_{policy}/results.json`

---

## extract

Extract L0 (review-level) judgments using multi-model ensemble. First step of ground truth generation.

### Basic Usage

```bash
# Single topic (recommended for Phase I)
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --mode 24hrbatch

# All topics (for full benchmark)
.venv/bin/python -m addm.tasks.cli.extract --all --k 200 --mode 24hrbatch

# Derive topic from policy
.venv/bin/python -m addm.tasks.cli.extract --policy G1_allergy_V2 --k 200 --mode 24hrbatch
```

### Target Selection

**One of these (or default to --all):**

| Flag | Description | Example |
|------|-------------|---------|
| `--task` | Legacy task ID | `--task G1a` |
| `--topic` | Topic ID (shared by V0-V3) | `--topic G1_allergy` |
| `--policy` | Policy ID (derives topic) | `--policy G1_allergy_V2` |
| `--all` | Extract all 18 topics | `--all` |

**Default**: If no target specified, defaults to `--all`.

### Core Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--domain` | Domain (yelp/amazon) | `yelp` |
| `--k` | Reviews per restaurant | `200` |
| `--limit` | Limit number of restaurants | `None` |

### Multi-Model Configuration

| Flag | Description | Default |
|------|-------------|---------|
| `--models` | Model config as `model:runs,...` | `gpt-5-mini:1,gpt-5-nano:3` |

**Format**: `model:runs,model:runs,...`

**Examples:**
- Single model: `--models "gpt-5-nano:1"`
- Default (fast): `--models "gpt-5-mini:1,gpt-5-nano:3"` (4 runs total)
- High-quality: `--models "gpt-5.1:1,gpt-5-mini:3,gpt-5-nano:5"` (9 runs total)

**Weighted voting**: Models have different weights for aggregation:
- `gpt-5.1`: weight 3
- `gpt-5-mini`: weight 2
- `gpt-5-nano`: weight 1

### Execution Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--provider` | LLM provider | `openai` |
| `--model` | LLM model (task mode only) | `gpt-5-nano` |
| `--concurrency` | Max concurrent requests | `10` |
| `--mode` | Execution mode: `ondemand` or `24hrbatch` | `24hrbatch` |

### Status & Control

| Flag | Description | Default |
|------|-------------|---------|
| `--verbose` | Verbose output | `False` |
| `--show-status` | Show in-progress status (quiet by default) | `False` |
| `--dry-run` | Dry run (no API calls) | `False` |

**Note**: Extract defaults to **quiet mode**. Use `--show-status` to see progress messages.

### Cache Management

| Flag | Description | Default |
|------|-------------|---------|
| `--invalidate` | Invalidate cache if term hash changed | `False` |

Use `--invalidate` if term definitions changed and you need to re-extract.

### Multi-Batch Tracking

| Flag | Description | Default |
|------|-------------|---------|
| `--batch-id` | Batch ID for legacy single-batch runs | `None` |
| `--manifest-id` | Manifest ID for multi-batch tracking | `None` |

**Note**: Multi-batch mode is automatic when using multi-model config. The system:
1. Creates a manifest tracking all batches
2. Submits multiple batches (one per model×run)
3. Installs a cron job to poll for completion
4. Aggregates results when all batches complete

### Examples

```bash
# Single topic, default config (4 runs)
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --mode 24hrbatch

# High-quality extraction (9 runs)
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --mode 24hrbatch \
    --models "gpt-5.1:1,gpt-5-mini:3,gpt-5-nano:5"

# Single model (fastest, no ensemble)
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --mode 24hrbatch \
    --models "gpt-5-nano:1"

# All topics
.venv/bin/python -m addm.tasks.cli.extract --all --k 200 --mode 24hrbatch

# Dry run (test without API calls)
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --dry-run

# With progress status
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --mode 24hrbatch --show-status

# Check specific manifest status
.venv/bin/python -m addm.tasks.cli.extract --manifest-id manifest_G1_allergy_K200_20240118

# Invalidate stale cache
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --invalidate
```

### Output

- Cache: `data/answers/yelp/judgement_cache.json`
- Logs: `results/logs/extraction/{topic}.log`
- Manifests: `data/answers/yelp/batch_manifest_*.json` (gitignored)
- Errors: `data/answers/yelp/batch_errors_*.jsonl` (gitignored)

---

## compute_gt

Compute ground truth from cached L0 judgments. Second step of ground truth generation.

### Basic Usage

```bash
# Single policy
.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V2 --k 200

# Multiple policies (same topic)
.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V0,G1_allergy_V1,G1_allergy_V2,G1_allergy_V3 --k 200

# All policies for a topic (V0-V3 × K=25,50,100,200)
.venv/bin/python -m addm.tasks.cli.compute_gt --topic G1_allergy

# All 72 policies (default)
.venv/bin/python -m addm.tasks.cli.compute_gt
```

### Target Selection

**One of these (or default to --all):**

| Flag | Description | Example |
|------|-------------|---------|
| `--task` | Legacy task ID | `--task G1a` |
| `--topic` | Topic (computes all V0-V3 variants for all K values) | `--topic G1_allergy` |
| `--policy` | Policy ID(s), comma-separated | `--policy G1_allergy_V2` or `--policy G1_allergy_V0,V1,V2,V3` |
| `--all` | Compute all 72 policies | `--all` |

**Default**: If no target specified, defaults to `--all`.

### Core Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--domain` | Domain (yelp/amazon) | `yelp` |
| `--k` | Dataset K value (ignored when using `--topic`) | `200` |
| `--verbose` | Verbose output | `False` |

**Note**: When using `--topic`, GT is computed for **all K values** (25, 50, 100, 200), regardless of `--k` flag.

### Examples

```bash
# Single policy, single K
.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V2 --k 200

# All variants for a topic (V0-V3 × K=25,50,100,200 = 16 files)
.venv/bin/python -m addm.tasks.cli.compute_gt --topic G1_allergy

# Multiple policies explicitly
.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V0,G1_allergy_V1,G1_allergy_V2,G1_allergy_V3 --k 200

# All 72 policies (full benchmark)
.venv/bin/python -m addm.tasks.cli.compute_gt --all

# Verbose output
.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V2 --k 200 --verbose
```

### Output

Ground truth files saved to:
```
data/answers/{domain}/{policy_id}_K{k}_groundtruth.json
```

Example: `data/answers/yelp/G1_allergy_V2_K200_groundtruth.json`

### Requirements

- L0 judgments must be cached in `data/answers/{domain}/judgement_cache.json`
- Run `extract` first if cache is empty
- Cache must have aggregated judgments for the topic

---

## generate

Generate natural language prompts from policy definitions.

### Basic Usage

```bash
# Single policy (short format)
.venv/bin/python -m addm.query.cli.generate --policy G1/allergy/V2

# Single policy (explicit output path)
.venv/bin/python -m addm.query.cli.generate --policy G1/allergy/V2 \
    --output data/query/yelp/G1_allergy_V2_prompt.txt

# All 72 policies
.venv/bin/python -m addm.query.cli.generate --all

# Print to stdout (don't save)
.venv/bin/python -m addm.query.cli.generate --policy G1/allergy/V2 --no-save
```

### Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--policy` | Policy path (e.g., `G1/allergy/V1`) | `None` |
| `--all` | Generate all 72 policies | `False` |
| `--output` | Output directory | `data/query/yelp/` |
| `--no-save` | Print to stdout, don't save to file | `False` |

**Policy path formats:**
- Short: `G1/allergy/V2`
- Full: `src/addm/query/policies/G1/allergy/V2.yaml`

### Examples

```bash
# Generate single policy
.venv/bin/python -m addm.query.cli.generate --policy G1/allergy/V2

# Generate all policies
.venv/bin/python -m addm.query.cli.generate --all

# View prompt without saving
.venv/bin/python -m addm.query.cli.generate --policy G1/allergy/V2 --no-save

# Custom output directory
.venv/bin/python -m addm.query.cli.generate --policy G1/allergy/V2 --output /tmp/prompts/

# Full policy path
.venv/bin/python -m addm.query.cli.generate --policy src/addm/query/policies/G1/allergy/V2.yaml
```

### Output

Prompts saved to:
```
{output_dir}/{policy_id}_prompt.txt
```

Example: `data/query/yelp/G1_allergy_V2_prompt.txt`

---

## Default Values Summary

Quick reference for default values across all CLIs:

| Flag | run_baseline | extract | compute_gt | generate |
|------|--------------|---------|------------|----------|
| `--domain` | `yelp` | `yelp` | `yelp` | N/A |
| `--k` | `200` | `200` | `200` | N/A |
| `--mode` | `ondemand` | `24hrbatch` | N/A | N/A |
| `--model` | `gpt-5-nano` | `gpt-5-nano` | N/A | N/A |
| `--method` | `direct` | N/A | N/A | N/A |
| `--models` | N/A | `gpt-5-mini:1,gpt-5-nano:3` | N/A | N/A |
| `--output` | N/A | N/A | N/A | `data/query/yelp/` |

---

## Common Workflows

### Phase I: Validate G1_allergy Pipeline

```bash
# Step 1: Extract L0 judgments (multi-model, 24hr batch)
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --mode 24hrbatch

# Wait for batches to complete (cron job polls automatically)
# Or monitor: tail -f results/logs/extraction/g1_allergy.log

# Step 2: Compute ground truth (all K values, all variants)
.venv/bin/python -m addm.tasks.cli.compute_gt --topic G1_allergy

# Step 3: Run baselines
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 10 --method direct
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 10 --method rlm
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 10 --method rag
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 10 --method amos
```

### Phase II: Scale to All Topics

```bash
# Step 1: Extract all topics
.venv/bin/python -m addm.tasks.cli.extract --all --k 200 --mode 24hrbatch

# Step 2: Compute all ground truths
.venv/bin/python -m addm.tasks.cli.compute_gt --all

# Step 3: Run full benchmark (72 policies × 4 methods × 100 restaurants each)
# Use batch scripts for parallel execution
```

### Generate All Prompts

```bash
# Generate all 72 prompts
.venv/bin/python -m addm.query.cli.generate --all

# Prompts saved to: data/query/yelp/*_prompt.txt
```

---

## Troubleshooting

### Batch Status Checks

```bash
# Check extraction status
tail -f results/logs/extraction/g1_allergy.log

# List batch manifests
ls -lah data/answers/yelp/batch_manifest_*.json

# Check specific manifest
.venv/bin/python -m addm.tasks.cli.extract --manifest-id manifest_G1_allergy_K200_20240118

# View batch errors
cat data/answers/yelp/batch_errors_*.jsonl
```

### Cache Issues

```bash
# Invalidate stale cache
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --invalidate

# View cache directly
python -c "import json; print(json.dumps(json.load(open('data/answers/yelp/judgement_cache.json'))['_metadata'], indent=2))"
```

### Dry Runs

```bash
# Test extraction without API calls
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 200 --dry-run
```

---

## Legacy Support

### Task-to-Policy Mapping

Legacy task IDs (G1a-G6l) are supported but deprecated. Mapping:

| Task | Topic | Policy (V2) |
|------|-------|-------------|
| G1a-d | G1_allergy | G1_allergy_V0-V3 |
| G1e-h | G1_dietary | G1_dietary_V0-V3 |
| G1i-l | G1_hygiene | G1_hygiene_V0-V3 |

**Recommendation**: Use policy-based flags (`--policy`, `--topic`) for new work.
