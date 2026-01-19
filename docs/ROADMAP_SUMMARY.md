# ADDM Project Roadmap - Quick Summary

**Last Updated**: 2026-01-19

---

## Timeline

| Milestone | Date | Days Remaining |
|-----------|------|----------------|
| Goal | Feb 1 | 13 days |
| Hard Deadline | Feb 8 | 20 days |

---

## Current Phase

**Phase I: G1_allergy Pipeline Validation** (Week 1: Jan 18-25)

Validate entire pipeline on G1_allergy before scaling to 18 topics.

---

## Today's Focus (Jan 19)

**Coding (Track A)**:
- A4: AMOS implementation (draft basic functionality)
- Run AMOS evaluation on G1_allergy_V2 (K=50, N=5-10)
- Validate AMOS generalization on diverse policies

**Writing (Track B)**:
- B2: Related Work draft (informs baseline selection)

---

## Critical Path Status

**✅ Completed**:
- A1: G1_allergy GT aggregation (all 16 files)
- Infrastructure: Methods, logging, eval metrics
- AMOS: Generalization fixes (V2/V3 support)

**🔄 In Progress**:
- A4: AMOS implementation & validation

**⏳ Next**:
- A2: Experiment infrastructure (after AMOS draft)
- A3: Initial baselines (CoT, ReACT, RAG)
- B2: Related Work (informs A3/A5)

---

## Week 1 Checkpoint (Jan 25)

**Goal**: AMOS beats all baselines on G1_allergy ✓

If checkpoint passes → proceed to Phase II (scale to 18 topics)

---

## Critical Blockers

None currently.

**Risks**:
- AMOS untested on 71/72 tasks (validation needed)
- Uncommitted AMOS fixes (8 modified, 2 new files)

---

## Quick Reference

Full details in `docs/ROADMAP.md`
