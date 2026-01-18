# ADDM Project Roadmap

## Overview

This document tracks project milestones, current progress, and future work.
For recent session details, see `docs/logs/` or use `/hello`.

---

## Milestones

### Phase 1: Data Pipeline ✓

- [x] Topic analysis (18 topics)
- [x] Restaurant selection (K=200)
- [x] Dataset building (K=25/50/100/200)

### Phase 2: Query Construction ✓

- [x] PolicyIR model design (Term, Operator, Policy)
- [x] Term libraries for all 18 topics
- [x] Operator library (comparisons, aggregations)
- [x] All 72 policy definitions (G1-G6, V0-V3)
- [x] Prompt generation from PolicyIR

### Phase 3: Ground Truth Generation (Current)

- [x] Multi-model extraction framework
- [x] PolicyJudgmentCache with dual cache (raw + aggregated)
- [x] Batch mode infrastructure (`--mode 24hrbatch`)
- [ ] Extract all 18 topics
  - [ ] G1_allergy
  - [ ] G1_dietary
  - [ ] G1_hygiene
  - [ ] G2_romance
  - [ ] G2_business
  - [ ] G2_group
  - [ ] G3_price_worth
  - [ ] G3_hidden_costs
  - [ ] G3_time_value
  - [ ] G4_server
  - [ ] G4_kitchen
  - [ ] G4_environment
  - [ ] G5_capacity
  - [ ] G5_execution
  - [ ] G5_consistency
  - [ ] G6_uniqueness
  - [ ] G6_comparison
  - [ ] G6_loyalty
- [ ] Compute GT for all 72 policies

### Phase 4: Baseline Evaluation

- [ ] Direct method evaluation across K sizes (25/50/100/200)
- [ ] RLM method comparison
- [ ] Results analysis and paper figures

---

## Technical Debt

- [ ] RLM reliability with gpt-5-nano (sometimes outputs placeholders)
- [x] Doc commands cleanup (consolidated /doc, /sync, /audit, /roadmap)

---

## Future Work

- [ ] Higher quality GT config (more models, stricter thresholds)
- [ ] Additional baseline methods (RAG, iterative refinement)
- [ ] Cross-policy analysis (how do V0-V3 differ in practice?)

---

## Quick Reference

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Data Pipeline | ✓ Complete | Topic analysis, selection, dataset building |
| 2. Query Construction | ✓ Complete | PolicyIR, term libraries, 72 policies |
| 3. Ground Truth | In Progress | Multi-model extraction, GT computation |
| 4. Baseline Eval | Pending | Direct + RLM evaluation |

---

*Last updated: 2026-01-18*
