# [2026-01-20] Results Directory Restructuring ✅

**What**: Implemented new results storage system with dev vs benchmark separation, quota-based runs, and per-run artifact organization.

**Impact**: Benchmark runs now tracked in git (`results/{method}/{policy}_K{k}/run_N/results.json`), with 5-run quota per configuration (1 ondemand + 4 batch).

**Files Created**:
- `src/addm/utils/results_manager.py` - Core ResultsManager class with quota logic
- `src/addm/utils/item_logger.py` - Per-restaurant LLM logging to item_logs/

**Files Modified**:
- `src/addm/tasks/cli/run_baseline.py` - Added --benchmark, --force flags; K in path
- `src/addm/utils/debug_logger.py` - Consolidated mode (single debug.jsonl)
- `src/addm/methods/amos_method.py` - save_formula_seed_to_run_dir()
- `src/addm/methods/rag.py` - Uses results/shared/yelp_embeddings.json
- `.gitignore` - Tracks benchmark results.json, ignores debug/item_logs

**Structure**:
```
results/
├── dev/{YYYYMMDD}_{name}/          # Not tracked
├── {method}/{policy}_K{k}/run_N/   # Tracked (results.json, formula_seed.json)
└── shared/                          # Not tracked (embeddings cache)
```

**Status**: Uncommitted: 20+ files (includes doc cleanup from other work)

**Next**:
- Commit the results restructuring changes
- Test with actual benchmark run: `--policy G1_allergy_V2 -n 100 --benchmark`
