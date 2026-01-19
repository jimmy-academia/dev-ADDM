# Session Log: RAG Cache File Relocation

**Date**: 2026-01-18 22:30
**Status**: completed

## Summary

Relocated the RAG embedding cache from a confusing location (`data/answers/yelp/policy_cache.json`) to a proper location (`results/cache/rag_embeddings.json`). The old file had a misleading name and was mixed with input data.

## Decisions Made

- Cache location: `results/cache/rag_embeddings.json` (not `data/cache/` because `data/` is for inputs, cache is method output)
- Added `results/cache/` to `.gitignore` (285MB file shouldn't be in repo)
- Renamed from `policy_cache.json` to `rag_embeddings.json` for clarity

## Changes Made

| File | Change |
|------|--------|
| `src/addm/methods/rag.py:52` | Updated default cache path |
| `.gitignore` | Added `results/cache/` |
| `.claude/CLAUDE.md` | Updated project structure docs |

## Current State

- Old cache moved to new location (preserved data)
- Code updated to use new path
- Docs updated to reflect change
- Gitignore updated

## Key Files

- `src/addm/methods/rag.py` - RAG method with embedding cache
- `results/cache/rag_embeddings.json` - New cache location (285MB)

## Context

The RAG method caches embeddings to avoid re-computing them. The cache stores:
- Query embeddings (per sample)
- Review embeddings (per sample × K)
- Retrieval results (top-k indices)

`judgement_cache.json` (in `data/answers/yelp/`) is a separate cache for L0 extraction judgments used in ground truth generation.
