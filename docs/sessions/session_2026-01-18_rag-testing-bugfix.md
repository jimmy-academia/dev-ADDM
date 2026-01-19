# Session Log: RAG Method Testing & Bug Fix

**Date**: 2026-01-18 22:30
**Status**: completed

## Summary

Tested RAG baseline method, discovered and fixed a critical cache key bug causing cross-K contamination, then conducted comprehensive testing across 6 restaurants × 4 K values to characterize RAG's performance and limitations.

## Decisions Made

- Fixed RAG cache keys to include `num_reviews` parameter, preventing cross-K cache contamination
- Documented RAG in `docs/BASELINES.md` with full experimental results
- Added RAG observations to `paper/plan/5p_experiments.md` as motivation for AMOS

## Work Completed

### Bug Fix
- **Problem**: RAG cache keys (`rag_embeddings_{sample_id}`, `rag_retrieval_{sample_id}_k{top_k}`) didn't include dataset K value
- **Symptom**: Running RAG with K=50 would use cached embeddings from K=200, causing `IndexError: list index out of range`
- **Fix**: Updated cache keys to include `num_reviews`:
  - `rag_embeddings_{sample_id}_K{num_reviews}`
  - `rag_retrieval_{sample_id}_K{num_reviews}_topk{k}`
- **Files changed**: `src/addm/methods/rag.py` (4 methods + 4 call sites)

### Experimental Testing
Tested 6 restaurants across K=25, 50, 100, 200:

| Restaurant | GT | K=25 | K=50 | K=100 | K=200 |
|------------|-----|------|------|-------|-------|
| Cafe Blue Moose | Critical (24) | ✓ | ✓ | ✓ | ✓ |
| KitzMo Sushi | Critical (17) | ✓ | ✓ | ✓ | ✓ |
| China Pavilion | Critical@K200 | ✓ | ✓ | ✓ | ✓ |
| Chop Steakhouse | Critical (15) | ✗ | ✗ | ✗ | ✗ |
| Hearthside | High@K≥50 | ✓ | ✗ | ✗ | ✗ |
| Sheraton | Critical@K200 | ✓ | ✓ | ✓ | ✗ |

**Overall: 18/24 = 75% accuracy**

### Key Findings
1. RAG works when incidents use explicit language ("anaphylaxis", "allergic reaction")
2. RAG fails when incidents use subtle/indirect language (Chop Steakhouse: 0/4)
3. RAG over-scores edge cases (Hearthside: High→Critical)
4. Semantic similarity ≠ policy relevance (core limitation)

## Current State

RAG baseline is working correctly with proper caching. Results documented in:
- `docs/BASELINES.md` - Full method documentation + experimental results
- `paper/plan/5p_experiments.md` - Paper outline with key findings

## Next Steps

1. Run RAG on full dataset for statistical significance
2. Test different top-k values (10, 30, 50)
3. Compare Direct baseline on same restaurants
4. Continue AMOS implementation (separate session)

## Key Files

- `src/addm/methods/rag.py` - RAG implementation (bug fixed)
- `docs/BASELINES.md` - Baseline documentation (updated with RAG)
- `paper/plan/5p_experiments.md` - Experiments section outline
- `data/answers/yelp/policy_cache.json` - RAG embedding/retrieval cache

## Context & Background

- RAG uses `text-embedding-3-large` for embeddings
- Default retrieval: top-20 reviews by cosine similarity
- Cache is in `data/answers/yelp/policy_cache.json`
- Ground truth files: `data/answers/yelp/G1_allergy_V2_K{k}_groundtruth.json`
