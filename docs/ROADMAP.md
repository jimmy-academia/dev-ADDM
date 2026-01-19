# ADDM Project Roadmap

## Timeline

| Deadline | Date | Days Left |
|----------|------|-----------|
| TODAY | Jan 18 | - |
| Goal | Feb 1 | 14 days |
| Hard | Feb 8 | 21 days |

```
Jan 18 ──────────────────────────────────────────────────── Feb 1 ──────── Feb 8
   │                                                           │              │
   │  WEEK 1 (Jan 18-25)         WEEK 2 (Jan 26-31)           │  BUFFER      │
   │  PHASE I: G1_allergy        PHASE II: Scale to All       │  ├─ Polish   │
   │  ├─ GT aggregation          ├─ 17 topic extractions      │  ├─ Revise   │
   │  ├─ Baselines impl          ├─ Full experiment runs      │  └─ Submit   │
   │  ├─ AMOS development        ├─ Paper: Experiments        │              │
   │  └─ Paper: Method           └─ Paper: Discussion         │              │
   │                                                           │              │
   └─────────────────────────────────────────────────────────────────────────┘
                              GOAL                           HARD DEADLINE
```

---

## Strategy: Two-Phase Approach

**Phase I (Week 1)**: Use G1_allergy to validate entire pipeline
- GT aggregation → baselines → AMOS → eval
- Validate infrastructure before expensive batch runs

**Phase II (Week 2)**: Scale to all 18 topics
- Submit 17 remaining topic extractions in batch
- Run ALL methods × 18 topics × K values

---

## Current State

### Infrastructure Ready
- **Methods**: Registry-based system in `src/addm/methods/` with clear pattern for adding new methods
- **Logging**: Full support for multi-run variance, token usage (input/output), latency, per-restaurant results
- **Eval**: Ordinal AUPRC, accuracy, primitive accuracy in `src/addm/eval/metrics.py`

### Completed (Phases 1-2)
- [x] Data pipeline (18 topics, K=25/50/100/200)
- [x] Query construction (72 policies, term libraries)
- [x] Paper: Abstract, Introduction (✅ first draft complete), Appendix A & C

### Completed (Phase I)
- G1_allergy extraction and aggregation (raw + consensus L0 + GT V0-V3)

### Draft (needs revision)
- Paper: Problem (Sec 3) - needs alignment with GT approach

### Completed (Infrastructure)
- **AMOS method**: ✅ Fully implemented (Phase 1 + Phase 2, tested)
- **RAG baseline**: ✅ Implemented with caching (experimental results: 75% accuracy)
- **RLM baseline**: ✅ Implemented (known issue: unreliable with gpt-5-nano)

### Not Started
- Other topic extractions, GT computation (17 topics)
- AMOS validation on G1_allergy (Phase I)
- Paper: Related Work, Method, Experiments, Discussion, Conclusion

---

## TRACK A: Coding & Experiments

### A1. Ground Truth Completion

**Phase I (G1_allergy validation)**: ✅ COMPLETE
- [x] Aggregate G1_allergy raw judgments → consensus L0
- [x] Compute GT for G1_allergy V0-V3 policies
- [ ] Verify GT quality (spot-check 5-10 samples) - optional

**Phase II (Scale to all topics)**:
- [ ] Submit batch extraction for remaining 17 topics (parallel)
- [ ] Aggregate all topics as batches complete
- [ ] Compute GT for all 72 policies (18 topics × 4 variants)
- [ ] Verify GT quality across topics

### A2. Experiment Infrastructure

**Critical**: Build infrastructure during Phase I that supports ALL Phase II experiments.

**Result Storage Requirements**:
- [ ] Per-restaurant: verdict, GT, risk_score, correct, parsed primitives
- [ ] Usage: prompt_tokens, completion_tokens (separate), cost_usd
- [ ] Timing: latency_ms per call
- [ ] Multi-run: timestamp-based directories for variance analysis
- [ ] Run metadata: K, N, method, model, policy, topic (for filtering)

**Ablation Support** (for A6):
- [ ] Store AMOS intermediate results (Phase 1 extraction output, Phase 2 aggregation input)
- [ ] Allow running AMOS with GT-substituted phases (Phase1→GT, Phase2→GT)
- [ ] Support per-policy-variant runs (V0, V1, V2, V3 separately)

**Sensitivity Support** (for A7):
- [ ] Parameterize K at runtime (25/50/100/200)
- [ ] Ensure GT available for all K values (or can derive from K=200)
- [ ] Store per-restaurant results (allows varying N post-hoc by subsetting)

**Variance Support**:
- [ ] Multi-run tracking with seed/run_id
- [ ] Easy aggregation across runs (mean, std)

**Results Structure**:
```python
{
    "run_id": str,           # Unique run identifier
    "method": str,           # "direct", "cot", "amos", etc.
    "method_config": dict,   # Method-specific params (e.g., amos_phase1="gt")
    "policy": str,           # "G1_allergy_V2"
    "topic": str,            # "G1_allergy"
    "variant": str,          # "V0", "V1", "V2", "V3"
    "k": int,                # Context size
    "n": int,                # Sample count
    "model": str,            # LLM model
    "seed": int,             # For reproducibility
    "results": [             # Per-restaurant
        {
            "business_id": str,
            "verdict": str,
            "gt_verdict": str,
            "correct": bool,
            "risk_score": float,
            "primitives": dict,        # Extracted L0 values
            "intermediate": {          # For ablations
                "phase1_output": dict, # AMOS extraction result
                "phase2_input": dict,  # What went into aggregation
            },
            "usage": {
                "prompt_tokens": int,
                "completion_tokens": int,
                "cost_usd": float,
                "latency_ms": float,
                "llm_calls": int,
            }
        }
    ],
    "metrics": {
        "accuracy": float,
        "ordinal_auprc": float,
        "total_cost": float,
        "total_latency": float,
    }
}
```

### A3. Initial Baselines

Methods to implement in `src/addm/methods/`:

| Method | File | Description | Priority |
|--------|------|-------------|----------|
| `direct` | direct.py | Zero-shot, all context in prompt | ✅ Exists |
| `cot` | cot.py | Chain-of-Thought prompting | P1 |
| `react` | react.py | ReACT (Reason + Act) | P1 |
| `rag` | rag.py | Naive RAG (embedding retrieval) | P1 |

**Phase I (G1_allergy validation)**:
- [ ] Implement `cot.py` - add "Let's think step by step" prompting
- [ ] Implement `react.py` - reasoning + action loop
- [ ] Implement `rag.py` - embedding-based retrieval + LLM
- [ ] Run each method on G1_allergy (3-5 runs for variance)
- [ ] Validate results storage, metrics, multi-run comparison works

**Phase II (Scale)**:
- [ ] Run all initial baselines × 18 topics × K=25/50/100/200

### A4. AMOS Method Development

**Phase I (G1_allergy validation)**:
- [ ] Design AMOS architecture:
  - Phase 1: Extraction (L0 primitives from reviews)
  - Phase 2: Aggregation (L1→L2→FL computation)
- [ ] Implement `src/addm/methods/amos.py`
- [ ] Tune on G1_allergy to achieve strong performance
- [ ] **Validate AMOS beats baselines on G1_allergy** (critical checkpoint)
- [ ] Document prompts and algorithm in appendix

**Phase II (Scale)**:
- [ ] Run AMOS × 18 topics × K=25/50/100/200

### A5. Advanced Baselines

DDDM+LLM approaches to implement:

| Method | Category | Reference | Priority |
|--------|----------|-----------|----------|
| LOTUS | Semantic operators | Stanford | P1 |
| ToT | Tree-of-Thought | Yao et al. | P1 |
| GoT | Graph-of-Thought | - | P2 |
| Multi-agent | Agentic retrieval | - | P2 |
| UNIFY | Unified text analytics | - | P2 |

**Phase I (G1_allergy validation)**:
- [ ] Implement or adapt LOTUS integration
- [ ] Implement ToT prompting scheme
- [ ] Implement GoT prompting scheme
- [ ] Implement multi-agent retrieval approach
- [ ] Implement UNIFY-style unified analytics
- [ ] Run all on G1_allergy, compare vs AMOS

**Phase II (Scale)**:
- [ ] Run all advanced baselines × 18 topics × K values

### A6. Ablation Studies

| Ablation | Description |
|----------|-------------|
| V0 vs V1 vs V2 vs V3 | Compare policy complexity impact |
| AMOS Phase 1 → GT | Replace extraction with GT primitives |
| AMOS Phase 2 → GT | Replace aggregation with GT |
| Phase 1 only | Extraction quality without aggregation |

- [ ] Design ablation experiment matrix
- [ ] Run ablations on G1_allergy
- [ ] Extend to other topics if time

### A7. Sensitivity Studies

| Variable | Values | Notes |
|----------|--------|-------|
| K (reviews) | 25, 50, 100, 200 | Check GT handles all K |
| N (restaurants) | 10, 25, 50, 100 | Reuse per-restaurant results |

- [ ] Verify GT generation works for all K values
- [ ] Run sensitivity sweeps
- [ ] Generate K-scaling curves

### A8. Qualitative Analysis

- [ ] Token usage recording (input/output separate per method)
- [ ] Cost comparison table across methods
- [ ] Efficiency analysis (tokens per correct prediction)

### A9. Case Studies

| Study | Purpose |
|-------|---------|
| Latency measurement | Small on-demand mode run |
| Failure case analysis | Find where baseline fails, AMOS succeeds |
| Success case analysis | Why AMOS works |
| LLM judgment verification | Compile policy_cache to human-readable form |

- [ ] Run latency benchmark (on-demand mode, small N)
- [ ] Sample failure cases for baseline methods
- [ ] Sample success cases for AMOS
- [ ] **LLM judgment doc**: Export policy_cache to compact multi-line document format
  - Show raw judgments from 1× gpt-5-mini + 3× gpt-5-nano
  - Highlight cross-model agreement/disagreement
  - Format for human review and appendix inclusion

---

## TRACK B: Paper Writing

### Paper Section Dependencies

```
B1: Introduction (narrative, contributions)
    ↓
B2: Related Work (groups, gaps, positioning)
    ↓
    ├─→ Baseline selection (which methods to compare)
    │       ↓
    │   B5: Experiments (setup, baselines, results)
    │
    └─→ B6: Discussion (circle back to Intro claims, limitations)

B3: Problem (draft, needs revision) ←→ A1: GT approach (must align)
    ↓
B4: Method (after A4 AMOS) → B5: Experiments
```

### B1. Introduction (Sec 1)

Current: ✅ **COMPLETE** (First Draft)

- [x] Adjust narrative for clear story flow
- [x] Define key claims that Discussion will address
- [x] Add concrete motivating example showing versatility
- [x] Expand method preview (AMOS two-phase framework)
- [x] Expand experiment preview (benchmark characteristics)
- [x] Strengthen positioning vs semantic operators
- [ ] Ensure contributions list matches final results (pending experiments)
- [x] Link to Appendix A for extended justification

**Outputs for downstream sections**:
- Key claims → B6 Discussion
- Problem framing → B2 Related Work positioning
- Contributions → B5 Experiments (what to demonstrate)

### B2. Related Work (Sec 2)

**Depends on**: B1 Introduction (to know what gaps to highlight)

Groups to cover:
1. **ABSA/Opinion Mining** - sentiment extraction, aspect-based
2. **Text Summarization** - extractive, abstractive, query-focused
3. **RAG & Retrieval** - dense retrieval, semantic search
4. **Semantic Operators** - LOTUS, PALIMPZEST, text-to-SQL
5. **LLM Reasoning** - CoT, ToT, GoT, ReACT
6. **Multi-Agent Systems** - collaborative LLM agents

- [ ] Write differential argument for each group
- [ ] Keep main paper concise (~0.5 page)
- [ ] Expand in Appendix B

**Outputs for downstream sections**:
- Gap analysis → justifies AMOS design (B4)
- Related methods → baseline selection (A3, A5)
- Positioning → Experiment comparisons (B5)

### B3. Problem Formulation (Sec 3)

Current: **Draft** (requires revision)

- [ ] Revise problem formulation for clarity
- [ ] Explain benchmark setting clearly
- [ ] Ensure alignment with actual GT generation approach
- [ ] Ensure Appendix C has full benchmark details

**Depends on**: Final GT approach (A1) should match problem description

### B4. Method (Sec 4) - AMOS

Structure:
1. Overview diagram/figure
2. Phase 1: Extraction (L0 primitives)
3. Phase 2: Aggregation (L1→L2→FL)
4. LLM integration details

- [ ] Write method description (~1-1.5 pages)
- [ ] Create system diagram
- [ ] Document prompts in Appendix D
- [ ] Include algorithm pseudocode in Appendix D

### B5. Experiments (Sec 5)

**Depends on**: B2 (baseline selection), B3 (benchmark), B4 (AMOS to evaluate), A3-A9 (results)

Structure:
1. Setup (dataset, metrics, baselines from B2)
2. Main results table
3. Scaling analysis (K variation)
4. Ablation results
5. Observations and analysis

- [ ] Write experimental setup (references B2 for baseline justification)
- [ ] Baseline list must match Related Work positioning
- [ ] Create main results table template
- [ ] Fill from experiment results (A3-A9)
- [ ] Write analysis paragraphs
- [ ] Extended experiments in Appendix E

**Inputs from other sections**:
- B2 Related Work → which baselines to include and why
- B3 Problem → benchmark metrics (ordinal AUPRC, etc.)
- B4 Method → AMOS description to evaluate

### B6. Discussion (Sec 6)

**Depends on**: B1 (claims to address), B5 (findings to discuss)

- [ ] **Address claims from Introduction** (close the loop)
- [ ] Summarize key findings from B5
- [ ] Why AMOS works (connect to method design)
- [ ] Limitations acknowledgment
- [ ] Point to Appendix A for extended discussion

**Must address**:
- Each contribution claim from B1 → evidence from B5
- Gaps identified in B2 → how AMOS fills them

### B7. Conclusion (Sec 7)

- [ ] Summarize contributions
- [ ] Future work directions
- [ ] Final takeaway

### B8. Abstract

- [ ] Final revision after experiments complete
- [ ] Ensure numbers match results

---

## Appendix Mapping

| Appendix | Content | Main Section Link |
|----------|---------|-------------------|
| A | Extended justification, real-world policy discussion | Introduction |
| B | Extended related work with detailed comparisons | Related Work |
| C | Benchmark design details, taxonomy, pipeline | Problem |
| D | AMOS prompts, algorithm pseudocode | Method |
| E | Extended experiments, additional results | Experiments |

---

## Weekly Schedule

### Week 1 (Jan 18-25) - PHASE I: G1_allergy Validation

| Day | Coding | Paper |
|-----|--------|-------|
| Sat 18 | ✅ A1: Aggregate G1_allergy GT | B1: Intro polish, define claims |
| Sun 19 | **A4: AMOS draft implementation** | **B2: Related Work draft** (informs baseline selection) |
| Mon 20 | A4: AMOS continued, basic functionality | B2: Related Work complete, **B3: Problem revision** |
| Tue 21 | A2: Experiment infrastructure (informed by AMOS) | B3: Align with GT approach |
| Wed 22 | A3: Implement CoT, ReACT, RAG (per B2) | - |
| Thu 23 | A3: Run initial baselines + AMOS on G1_allergy | B4: Method (after AMOS works) |
| Fri 24 | A4: AMOS tuning, **validate beats baselines** | B4: Method complete |
| Sat 25 | A5: LOTUS, ToT, GoT impl + comparison | - |

**Checkpoint (Sat 25)**: AMOS beats all baselines on G1_allergy ✓

**Key insights**:
- **AMOS first (Sun-Mon)** → informs experiment infrastructure needs (A2)
- B2 Related Work early (Sun 19) → informs which baselines to implement (A3, A5)
- B3 Problem revision (Mon-Tue) → must align with A1 GT approach

### Week 2 (Jan 26-31) - PHASE II: Scale to All 18 Topics

| Day | Coding | Paper |
|-----|--------|-------|
| Sun 26 | A1: Submit 17 topic extractions (batch) | - |
| Mon 27 | A1: Aggregate batches, A6: Ablations | B5: Experiments setup (refs B2 for baselines) |
| Tue 28 | **Phase II runs**: ALL baselines × 18 topics | B5: Results table template |
| Wed 29 | A7: Sensitivity studies (K variation) | B5: Fill results, analysis |
| Thu 30 | A8: Qualitative analysis, token usage | B6: Discussion (refs B1 claims, B5 findings) |
| Fri 31 | A9: Case studies, failure analysis | B7: Conclusion, B8: Abstract revision |

**Paper flow**: B1 claims → B5 evidence → B6 discussion (close the loop)

### Buffer Week (Feb 1-8)

- Feb 1-3: Fill results, complete tables/figures
- Feb 4-6: Review and revision
- Feb 7: Final formatting
- Feb 8: Submit (hard deadline)

**Note**: Phase I validates on G1_allergy before expensive Phase II batch runs.

---

## Key Dependencies

### Phase I Critical Path (Week 1)

```
✅ A1: G1_allergy GT aggregation (COMPLETE)
    ↓
A4: AMOS draft implementation (Sun-Mon)
    ↓
A2: Experiment infrastructure (design informed by AMOS needs)
    ↓
A3: Initial Baselines (Direct, CoT, ReACT, RAG) on G1_allergy
    ↓
A4: AMOS tuning & validation
    ↓
B4: Paper Method section (AFTER AMOS works)
    ↓
A5: Advanced Baselines on G1_allergy
    ↓
✓ CHECKPOINT: AMOS beats baselines on G1_allergy
```

### Phase II Critical Path (Week 2)

```
A1: Submit 17 topic extractions (batch)
    ↓
A1: Aggregate batches as complete
    ↓
Run ALL methods × 18 topics × K values (batch)
    ↓
A6: Ablations + A7: Sensitivity + A8: Qualitative
    ↓
B5: Paper Experiments (fill results)
    ↓
A9: Case Studies → B6: Discussion
```

**Critical Path**: ✅ G1_allergy GT → AMOS draft → infrastructure → baselines → AMOS validated → 17 topics batch → full runs → paper

---

## Risk Factors

| Risk | Impact | Mitigation |
|------|--------|------------|
| Batch API delays | GT blocked | Start extraction immediately, monitor |
| AMOS underperforms | Weak paper | Iterate quickly on G1_allergy before scaling |
| RLM unreliable | Missing baseline | Focus on other methods, note RLM as future work |
| Writing behind | Late submission | Parallelize: write Method while GT runs |

---

## Quick Reference

| Track | Task | Status | Target | Notes |
|-------|------|--------|--------|-------|
| A1 | GT Completion | ✅ Complete | Jan 18 | G1_allergy done (Phase I) |
| A4 | AMOS | In Progress | Jan 20 | Draft first, then tune |
| A2 | Experiment Infra | Not Started | Jan 21 | Design after AMOS draft |
| A3 | Initial Baselines | Not Started | Jan 22 | CoT, ReACT, RAG |
| A5 | Advanced Baselines | Not Started | Jan 25 | LOTUS, ToT, GoT |
| A6 | Ablations | Not Started | Jan 27 | After AMOS |
| A7 | Sensitivity | Not Started | Jan 29 | K variation |
| A8 | Qualitative | Not Started | Jan 30 | Token usage |
| A9 | Case Studies | Not Started | Jan 31 | Failure analysis |
| B1 | Introduction | ~70% | Jan 18 | Define claims |
| B2 | Related Work | Not Started | Jan 20 | Informs baselines |
| B3 | Problem | Draft | Jan 21 | Needs revision |
| B4 | Method | Not Started | Jan 24 | After AMOS |
| B5 | Experiments | Not Started | Jan 29 | Fill results |
| B6 | Discussion | Not Started | Jan 30 | Close the loop |
| B7 | Conclusion | Not Started | Jan 31 | Final |
| B8 | Abstract | Draft | Jan 31 | Update numbers |

---

*Last updated: Jan 18, 2026 (evening - A1 complete, AMOS prioritized)*
