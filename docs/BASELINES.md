# Baselines

This document describes baseline methods and the proposed AMOS method for ADDM experiments.

---

## Overview

ADDM evaluates methods on 72 benchmark tasks requiring analysis of restaurant reviews (K=25/50/100/200).

**Proposed Method:**
- **AMOS** - Adaptive Multi-Output Sampling: Two-phase approach with cached Formula Seed and parallel extraction

**Baseline Methods:**
- **Direct** - Full-context prompting (all K reviews)
- **CoT** - Chain-of-Thought step-by-step reasoning
- **ReACT** - Reasoning + Acting with tool use
- **RAG** - Retrieval-augmented generation (semantic search)
- **RLM** - Recursive LLM with code execution

---

## Context Handling

Methods are categorized by how they handle restaurant review context:

| Mode | Method | Description | Token Cost |
|------|--------|-------------|------------|
| **Proposed** | **`amos`** | **Two-phase: Formula Seed generation + two-stage extraction** | **~5-55k tokens** |
| Full-context | `direct` | All K reviews in prompt | ~K×200 tokens |
| Reasoning | `cot` | Step-by-step reasoning before answer | ~K×250 tokens |
| Tool-use | `react` | Interleaved reasoning and tool actions | ~K×300+ tokens |
| Retrieval | `rag` | Embed query + reviews, retrieve top-k, LLM analyzes subset | ~4k tokens + embedding |
| Code-execution | `rlm` | Reviews as Python variable, LLM writes search code | ~50k tokens |

---

## Proposed Method

### AMOS (Adaptive Multi-Output Sampling)

| Attribute | Value |
|-----------|-------|
| Method | `amos` |
| File | `src/addm/methods/amos/__init__.py` |
| Description | Two-phase method: (1) LLM generates executable Formula Seed from policy, (2) Interpreter executes seed with parallel extraction |
| Context handling | Two-stage retrieval: Quick Scan (filtered) + Thorough Sweep (all remaining) |
| Token cost | ~5-55k tokens per restaurant (depends on review count and early exit) |
| Strengths | Explicit search strategy, parallel extraction, deterministic aggregation, cached compilation |
| Weaknesses | Depends on keyword filtering quality, may miss subtle incidents |

**Key Innovation:** Separates policy understanding (Phase 1, cached) from data analysis (Phase 2, per-restaurant), enabling efficient parallel processing.

**Usage:**
```bash
# Basic run (keyword filter + thorough sweep)
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 10 --method amos

# Embedding filter mode
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 10 --method amos --filter-mode embedding

# Hybrid filter mode
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 10 --method amos --filter-mode hybrid

# Dev mode
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 10 --method amos --dev
```

**How AMOS Works:**

**Phase 1: Formula Seed Generation (once per policy)**
1. LLM reads policy agenda/prompt
2. LLM produces executable JSON specification (Formula Seed) with:
   - **Filter**: Keywords to identify relevant reviews
   - **Extract**: Structured fields to extract (enum/int/float/bool with possible values)
   - **Compute**: Aggregation rules to calculate verdict
3. Formula Seed cached to `data/formula_seeds/{policy_id}.json`
4. Reused for all restaurants in the policy

Example Formula Seed snippet:
```json
{
  "task_name": "G1_allergy_V2",
  "filter": {
    "keywords": ["allergy", "allergic", "anaphylaxis", "epipen", ...]
  },
  "extract": {
    "fields": [
      {
        "name": "incident_severity",
        "type": "enum",
        "values": {
          "none": "No allergy incident",
          "mild": "Minor reaction (itching, rash)",
          "moderate": "Significant reaction (swelling, breathing difficulty)",
          "severe": "Life-threatening (anaphylaxis, hospitalization)"
        }
      }
    ]
  },
  "compute": [
    {"name": "N_SEVERE", "op": "count", "where": {"incident_severity": "severe"}},
    {"name": "SCORE", "op": "expr", "expr": "N_SEVERE * 15 + N_MODERATE * 5"},
    {"name": "VERDICT", "op": "case", "source": "SCORE", "rules": [...]}
  ]
}
```

**Phase 2: Execution (per restaurant)**
1. **Stage 1 - Quick Scan**: Filter reviews by mode (keyword/embedding/hybrid)
2. **Extract**: LLM extracts structured fields from filtered reviews in parallel
3. **Early Exit Check**: If severe evidence found → done
4. **Stage 2 - Thorough Sweep**: Process ALL remaining reviews (always on)
5. **Compute**: Deterministic aggregation of all extractions → final verdict

**Caching:**
- Formula Seeds cached in `data/formula_seeds/{policy_id}.json`
- Seeds auto-regenerate when policy changes (hash-based invalidation)

**Design Rationale:**

1. **Compilation model**: Separates policy understanding (expensive, done once) from data processing (cheaper, per-sample)
2. **Observable steps**: Each phase produces inspectable artifacts (Formula Seed, extracted fields)
3. **Deterministic aggregation**: Computation rules are explicit, not black-box LLM reasoning
4. **Scalability**: Parallel extraction across reviews enables efficient processing at K=200

**Comparison to Baselines:**

| Aspect | AMOS | Direct | RAG | RLM |
|--------|------|--------|-----|-----|
| **Context strategy** | Keyword filter + parallel extract | Full context | Semantic retrieval | Code execution |
| **Policy encoding** | Explicit Formula Seed | Implicit in prompt | Implicit in embedding | Implicit in code |
| **Aggregation** | Deterministic rules | LLM reasoning | LLM reasoning | LLM reasoning |
| **Scalability** | Parallel (32 concurrent) | Single call | Single call | Sequential iterations |
| **Token cost** | ~5-55k | ~K×200 | ~4k + embed | ~50k |
| **Phase 1 cost** | ~2k (amortized) | N/A | N/A | N/A |

**Expected Performance:**
- Should outperform Direct at K=100, 200 (avoids context rot)
- Should match or exceed RAG (explicit keyword search vs. semantic similarity)
- More efficient than RLM (lower token cost, parallel execution)

**Status:** ✅ Fully implemented, ready for Phase I validation on G1_allergy

---

## Baseline Methods

### Direct Baseline

| Attribute | Value |
|-----------|-------|
| Method | `direct` |
| File | `src/addm/methods/direct.py` |
| Description | Send all K reviews directly in prompt, ask LLM to analyze |
| Context handling | Full context in prompt (truncated if needed) |
| Token cost | ~K × 200 tokens per restaurant |
| Strengths | Simple, single LLM call |
| Weaknesses | Context rot at high K (50, 200) - model finds evidence but makes reasoning errors |

**Usage:**
```bash
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 5 --k 50 --method direct
```

---

### Recursive LLM (RLM)

| Attribute | Value |
|-----------|-------|
| Method | `rlm` |
| File | `src/addm/methods/rlm.py` |
| Library | [recursive-llm](https://github.com/ysz/recursive-llm) (forked to `lib/recursive-llm/`) |
| Description | Store context as Python variable, LLM writes code to search/explore |
| Context handling | Variable in sandboxed REPL, LLM controls access via code |
| Token cost | ~3000 tokens/iteration, default 50k budget (~16 iterations) |
| Strengths | Can search for keywords, focus on relevant reviews, avoid context rot |
| Weaknesses | Unreliable with gpt-5-nano (inconsistent outputs) |

**Reference:**
- No formal paper yet
- GitHub: https://github.com/ysz/recursive-llm

**Usage:**
```bash
# Basic run
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 1 --k 50 --method rlm

# Custom token budget
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 1 --k 50 --method rlm --token-limit 30000
```

**How RLM works:**
1. Reviews stored as `context` Python variable (not in prompt)
2. LLM receives task instructions + code execution environment
3. LLM writes Python code to search/filter reviews (e.g., `re.search(r'allerg', context)`)
4. Code executed in sandboxed REPL, results shown to LLM
5. LLM iterates until calling `FINAL(answer)` or hitting iteration limit

**Known Issues:**
- gpt-5-nano produces inconsistent results (sometimes correct verdict, sometimes literal placeholders)
- May need more capable model for reliable code execution

---

### RAG (Retrieval-Augmented Generation)

| Attribute | Value |
|-----------|-------|
| Method | `rag` |
| File | `src/addm/methods/rag.py` |
| Description | Embed query + all reviews, retrieve top-k most similar, LLM analyzes subset |
| Embedding model | `text-embedding-3-large` |
| Retrieval | Top-20 reviews by cosine similarity (default) |
| Token cost | ~4k LLM tokens + embedding cost (~$0.13/1M tokens) |
| Strengths | Reduces context size, focuses on semantically relevant reviews |
| Weaknesses | Misses incidents using subtle/indirect language; retrieval quality depends on query-review semantic similarity |

**Usage:**
```bash
# Basic run (top-20 retrieval)
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 5 --k 200 --method rag

# Custom top-k
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 5 --k 200 --method rag --top-k 30
```

**How RAG works:**
1. Parse restaurant context to extract reviews
2. Embed query (policy prompt) using OpenAI embeddings API
3. Embed all K reviews in batch
4. Compute cosine similarity between query and each review
5. Select top-k most similar reviews
6. Build reduced context with only retrieved reviews
7. LLM analyzes reduced context to produce verdict

**Caching:**
- Embeddings cached in `data/answers/yelp/policy_cache.json`
- Cache keys include K value to prevent cross-K contamination: `rag_embeddings_{sample_id}_K{num_reviews}`
- Retrieval results also cached: `rag_retrieval_{sample_id}_K{num_reviews}_topk{k}`

**Experimental Results (G1_allergy_V2, gpt-5-nano):**

*Tested on 6 restaurants across K=25, 50, 100, 200:*

| Restaurant | K=25 | K=50 | K=100 | K=200 | Pattern |
|------------|------|------|-------|-------|---------|
| Cafe Blue Moose (Critical) | ✓ | ✓ | ✓ | ✓ | Explicit "anaphylaxis" language retrieved |
| KitzMo Sushi (Critical) | ✓ | ✓ | ✓ | ✓ | Explicit "anaphylactic shock" language |
| China Pavilion (Critical@K200) | ✓ | ✓ | ✓ | ✓ | Clear "peanut allergy" incident |
| Chop Steakhouse (Critical) | ✗ | ✗ | ✗ | ✗ | Subtle language, incidents not retrieved |
| Hearthside (High@K≥50) | ✓ | ✗ | ✗ | ✗ | Over-scores: adds +3 dismissive staff |
| Sheraton (Critical@K200) | ✓ | ✓ | ✓ | ✗ | Incidents only in K=200, not retrieved |

**Overall accuracy: 18/24 = 75%**

**Key Observations:**
1. **Works well** when allergy incidents use explicit language ("anaphylaxis", "allergic reaction", "peanut allergy") that semantically matches the policy query
2. **Fails** when incidents use subtle/indirect language that doesn't rank high in embedding similarity
3. **Over-scores** edge cases where model adds modifiers (e.g., "dismissive staff") not warranted by evidence
4. **Retrieval ratio matters**: 20/200 = 10% retrieval at K=200 more likely to miss incidents than 20/50 = 40% at K=50

**Limitation:** Semantic retrieval fundamentally relies on language similarity. If allergy incidents are described using indirect phrasing (e.g., "staff didn't accommodate my dietary needs" vs "allergic reaction"), they won't be retrieved.

---

## Token Budget Comparison

For fair comparison across methods:

| Method | Tokens/Restaurant | Phase 1 Cost | Relative Cost |
|--------|-------------------|--------------|---------------|
| **AMOS (proposed)** | **~5,000** | **~2,000 (amortized)** | **1x** |
| RAG (K=200, top-20) | ~4,000 + embed | N/A | 0.8x + embed |
| Direct (K=50) | ~10,000 | N/A | 2x |
| Direct (K=200) | ~40,000 | N/A | 8x |
| RLM | ~50,000 | N/A | 10x |

**Notes:**
- **AMOS Phase 1** cost is amortized across all samples (e.g., for 100 restaurants, adds ~20 tokens/restaurant)
- RLM is ~10x more expensive than AMOS/RAG. This is accepted for baseline comparison purposes.
- RAG embedding cost: ~$0.13/1M tokens. For K=200 (~40k tokens), embedding costs ~$0.005/restaurant.
- **AMOS is designed to match RAG's token efficiency while providing explicit search strategies**

---

## Adding New Methods

To implement a new baseline method, see the **[Adding a New Method](developer/add-method.md)** guide.

The guide covers:
- Creating a method file from the `direct.py` template
- Implementing the `Method` interface with usage tracking
- Registering the method in `__init__.py`
- Testing and documenting the method

---

### Chain-of-Thought (CoT)

| Attribute | Value |
|-----------|-------|
| Method | `cot` |
| File | `src/addm/methods/cot.py` |
| Reference | Wei et al., NeurIPS 2022 |
| Description | Prompts LLM to think step-by-step before providing final answer |
| Context handling | Full context in prompt with reasoning chain |
| Token cost | ~K × 250 tokens per restaurant |
| Strengths | Improved reasoning through explicit steps |
| Weaknesses | Higher token cost than direct, still subject to context rot |

**Usage:**
```bash
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 5 --method cot
```

---

### ReACT (Reasoning + Acting)

| Attribute | Value |
|-----------|-------|
| Method | `react` |
| File | `src/addm/methods/react.py` |
| Reference | Yao et al., ICLR 2023 |
| Description | Interleaves reasoning and tool-based actions |
| Context handling | Tool-mediated access to reviews |
| Token cost | ~K × 300+ tokens per restaurant (multi-turn) |
| Strengths | Can focus on relevant reviews via tools |
| Weaknesses | Higher latency due to multi-turn execution |

**Usage:**
```bash
.venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 -n 5 --method react
```

---

## To Be Considered

| Method | Paper | Venue | Year | Notes |
|--------|-------|-------|------|-------|
| Program-of-Thoughts | Chen et al. | TMLR | 2023 | Similar to RLM but structured |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-18 | **Added AMOS (Adaptive Multi-Output Sampling) as proposed method** - Two-phase approach with Formula Seed compilation and parallel extraction |
| 2026-01-18 | Reorganized document structure to highlight AMOS as proposed method vs. baselines |
| 2026-01-18 | Added RAG method with experimental results (75% accuracy, semantic retrieval limitations) |
| 2026-01-18 | Fixed RAG cache key bug (cross-K contamination) |
| 2026-01-18 | Added RLM method with 50k token budget |
| 2026-01-18 | Created baselines document |
