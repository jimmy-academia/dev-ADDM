# Baselines

This document tracks baseline methods for ADDM experiments.

---

## Context Handling

Methods are categorized by how they handle restaurant review context:

| Mode | Method | Description | Token Cost |
|------|--------|-------------|------------|
| Full-context | `direct` | All K reviews in prompt | ~K×200 tokens |
| Retrieval | `rag` | Embed query + reviews, retrieve top-k, LLM analyzes subset | ~4k tokens + embedding |
| Code-execution | `rlm` | Reviews as Python variable, LLM writes search code | ~50k tokens |

---

## Implemented Methods

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
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --k 50 --method direct
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
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 1 --k 50 --method rlm

# Custom token budget
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 1 --k 50 --method rlm --token-limit 30000
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
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --k 200 --method rag

# Custom top-k
.venv/bin/python -m addm.tasks.cli.run_baseline --policy G1_allergy_V2 -n 5 --k 200 --method rag --top-k 30
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

| Method | Tokens/Restaurant | Relative Cost |
|--------|-------------------|---------------|
| ANoT (reference) | ~5,000 | 1x |
| RAG (K=200, top-20) | ~4,000 + embed | 0.8x + embed |
| Direct (K=50) | ~10,000 | 2x |
| Direct (K=200) | ~40,000 | 8x |
| RLM | ~50,000 | 10x |

**Notes:**
- RLM is ~10x more expensive than ANoT. This is accepted for baseline comparison purposes.
- RAG embedding cost: ~$0.13/1M tokens. For K=200 (~40k tokens), embedding costs ~$0.005/restaurant.

---

## To Be Considered

| Method | Paper | Venue | Year | Notes |
|--------|-------|-------|------|-------|
| Chain-of-Thought | Wei et al. | NeurIPS | 2022 | Add reasoning steps to direct baseline |
| ReAct | Yao et al. | ICLR | 2023 | Reason + Act paradigm |
| Program-of-Thoughts | Chen et al. | TMLR | 2023 | Similar to RLM but structured |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-18 | Added RAG method with experimental results (75% accuracy, semantic retrieval limitations) |
| 2026-01-18 | Fixed RAG cache key bug (cross-K contamination) |
| 2026-01-18 | Added RLM method with 50k token budget |
| 2026-01-18 | Created baselines document |
