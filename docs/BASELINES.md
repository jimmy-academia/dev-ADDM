# Baselines

This document tracks baseline methods for ADDM experiments.

---

## Context Handling

Methods are categorized by how they handle restaurant review context:

| Mode | Method | Description | Token Cost |
|------|--------|-------------|------------|
| Full-context | `direct` | All K reviews in prompt | ~K×200 tokens |
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

## Token Budget Comparison

For fair comparison across methods:

| Method | Tokens/Restaurant | Relative Cost |
|--------|-------------------|---------------|
| ANoT (reference) | ~5,000 | 1x |
| Direct (K=50) | ~10,000 | 2x |
| Direct (K=200) | ~40,000 | 8x |
| RLM | ~50,000 | 10x |

**Note:** RLM is ~10x more expensive than ANoT. This is accepted for baseline comparison purposes.

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
| 2026-01-18 | Added RLM method with 50k token budget |
| 2026-01-18 | Created baselines document |
