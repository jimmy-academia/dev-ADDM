# Troubleshooting

Common issues and solutions for the ADDM framework.

---

## Ground Truth Issues

### "No cached judgments found"

**Symptom:** `compute_gt` fails with missing judgments.

**Cause:** The extract step was skipped or used a different topic name.

**Fix:** Run extraction first:
```bash
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 50 --mode ondemand
```

### "Insufficient votes for consensus"

**Symptom:** `compute_gt` warns about low vote counts.

**Cause:** Batch extraction is incomplete or some models failed.

**Fix:**
1. Check batch status in OpenAI dashboard
2. Wait for batch completion
3. Re-run `compute_gt`

### Wrong topic/policy name

**Symptom:** No results or "policy not found" errors.

**Cause:** Mismatch between extract topic and compute_gt policy.

**Fix:** Topic and policy names follow specific patterns:
- Extract uses topic: `G1_allergy`
- Compute uses policy: `G1_allergy_V2`

```bash
# Correct pairing
.venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 50
.venv/bin/python -m addm.tasks.cli.compute_gt --policy G1_allergy_V2 --k 50
```

---

## Baseline Run Issues

### "Model rate limited"

**Symptom:** 429 errors, requests failing.

**Cause:** Too many parallel requests to API.

**Fix:** Use batch mode for large runs:
```bash
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 100 --mode 24hrbatch
```

### "Token limit exceeded"

**Symptom:** API rejects requests, context truncated.

**Cause:** K=200 with verbose method exceeds context window.

**Fix:**
- Reduce K: `--k 50` or `--k 100`
- For RLM, set token budget: `--token-limit 30000`
- Use RAG or AMOS which handle large K better

### Empty or placeholder outputs

**Symptom:** RLM returns literal `{verdict}` or empty strings.

**Cause:** gpt-5-nano inconsistency with code execution.

**Fix:** RLM works better with more capable models. Consider:
- Using `--method direct` or `--method rag` instead
- Testing with a different model if available

---

## Batch Mode Issues

### "Batch stuck in_progress"

**Symptom:** Batch never completes, status stays `in_progress`.

**Cause:** OpenAI batch processing delay or failure.

**Fix:**
1. Check OpenAI batch dashboard for status
2. If stuck >24h, cancel and resubmit:
   ```bash
   # Cancel via OpenAI dashboard, then resubmit
   .venv/bin/python -m addm.tasks.cli.extract --topic G1_allergy --k 50 --mode 24hrbatch
   ```

### "Cron job not running"

**Symptom:** Batch submitted but never processed.

**Check:**
```bash
crontab -l | grep addm
```

**Fix:** Batch mode relies on OpenAI's batch API, not local cron. Ensure batch was submitted correctly:
```bash
# Check batch manifest
ls data/answers/yelp/batch_manifest_*.json
```

### Batch errors

**Symptom:** Some samples failed in batch.

**Check:** Look at error logs:
```bash
cat data/answers/yelp/batch_errors_*.jsonl
```

**Fix:** Resubmit with `--mode ondemand` for failed samples.

---

## Data Issues

### "Dataset not found"

**Symptom:** `FileNotFoundError` for context files.

**Cause:** `build_dataset.py` not run for this K value.

**Fix:**
```bash
.venv/bin/python scripts/data/build_dataset.py --data yelp
```

This creates `K{25,50,100,200}_reviews.json` in `data/context/yelp/`.

### "No restaurants selected"

**Symptom:** Empty dataset or missing restaurants.

**Cause:** Selection step was skipped.

**Fix:** Run full pipeline:
```bash
# Step 1: Topic analysis
.venv/bin/python scripts/data/build_topic_selection.py --data yelp

# Step 2: Restaurant selection
.venv/bin/python scripts/data/select_topic_restaurants.py --data yelp

# Step 3: Build datasets
.venv/bin/python scripts/data/build_dataset.py --data yelp
```

### "Raw data not found"

**Symptom:** Scripts fail looking for `data/raw/yelp/`.

**Cause:** Yelp dataset not downloaded/extracted.

**Fix:** Download Yelp dataset and extract to `data/raw/yelp/`:
- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`

---

## Import/Module Issues

### "ModuleNotFoundError: No module named 'addm'"

**Cause:** Package not installed or wrong Python.

**Fix:**
```bash
source .venv/bin/activate
uv pip install -e .
```

### Wrong Python version

**Symptom:** Syntax errors, missing features.

**Fix:** Ensure Python 3.10+:
```bash
.venv/bin/python --version
```

---

## Results Issues

### "Results directory not found"

**Symptom:** Can't find run outputs.

**Cause:** Different mode or missing `--dev` flag.

**Fix:** Check both locations:
```bash
ls results/dev/      # --dev mode
ls results/prod/     # Production mode
```

### Results not saved

**Symptom:** Run completes but no files created.

**Cause:** Run crashed before save, or permissions issue.

**Fix:**
1. Check for errors in output
2. Verify write permissions: `touch results/test && rm results/test`
3. Re-run with `--dev` for debugging

---

## Getting Help

If issues persist:

1. Check debug logs: `results/logs/debug/{run_id}/`
2. Enable verbose mode: `--verbose`
3. Check recent session logs: `docs/sessions/`

## See Also

- [CLI Reference](specs/cli.md) - All command options
- [Architecture](architecture.md) - System overview
- [Ground Truth System](specs/ground_truth.md) - GT pipeline details
