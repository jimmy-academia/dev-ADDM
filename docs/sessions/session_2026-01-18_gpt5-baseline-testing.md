# Session Log: GPT-5 Baseline Testing & Trickiness Analysis

**Date**: 2026-01-18 23:50
**Status**: blocked (needs strategic decision)

## Summary

Tested gpt-5 direct method on 2 "tricky" restaurants (Cafe Blue Moose, Oceana Grill) for G1_allergy_V2 policy. Achieved 100% accuracy (4/4 tests). Deep investigation revealed a fundamental mismatch: restaurant "trickiness" is calculated GLOBALLY (across all 18 topics) but testing was done on ONE specific topic (G1_allergy). This makes globally-tricky restaurants potentially trivial for specific topics.

## Key Discovery

**The trickiness calculation is GLOBAL, not topic-specific:**
- Oceana Grill: Global rank #1 (18/36 cells covered), but only 1 allergy incident → trivial for G1_allergy
- Cafe Blue Moose: Global rank #2211, but has 6 allergy incidents with ambiguous severity → actually trickier for G1_allergy

This undermines the research goal: if baseline LLMs get 100% accuracy on "tricky" cases, it's hard to demonstrate AMOS method's value.

## Decisions Made

1. **Bug fix applied**: Fixed type conversion error in `src/addm/eval/intermediate_metrics.py:354` where `total_score` from LLM JSON was returned as string, causing TypeError in metrics computation.

2. **Baseline model clarification**: Confirmed gpt-5-nano (not gpt-5) should be the baseline:
   - gpt-5-nano is 25x cheaper
   - gpt-5-nano DOES fail on some cases (e.g., Cafe Blue Moose direct@K50: predicted "High Risk" vs GT "Critical Risk")
   - Prior documentation shows gpt-5-nano RAG achieves ~75% accuracy

## Current State

**Test Results (gpt-5 on G1_allergy_V2):**
- Cafe Blue Moose K=25: Critical Risk ✓ (cost: $0.0229, 11k tokens)
- Cafe Blue Moose K=50: Critical Risk ✓ (cost: $0.0279, 16k tokens)
- Oceana Grill K=25: Low Risk ✓ (cost: $0.0108, 7k tokens)
- Oceana Grill K=50: Low Risk ✓ (cost: $0.0188, 14k tokens)
- **Total cost: $0.0803** (vs planned $0.06)
- **Verdict accuracy: 100%**
- **Cross-K consistency: Perfect** (both restaurants maintained identical verdicts across K=25 and K=50)

**Uncommitted changes:**
- `src/addm/eval/intermediate_metrics.py` - bug fix (string→float conversion)
- Several other files with unrelated changes

**Background process running:**
- AMOS method test on 40 samples (skip 25) - still executing

## The Problem

User's reaction: "This is very bad. we don't want it to work... what to do..."

**Why this is problematic:**
1. If baseline (gpt-5) achieves 100% accuracy, there's no clear gap for AMOS to fill
2. The test restaurants were selected based on GLOBAL trickiness, not G1_allergy-specific difficulty
3. Oceana Grill (global #1) is actually TRIVIAL for G1_allergy (only 1 incident)

**Why it happened:**
- Selection algorithm (`topic_selection.py`) optimizes for cell coverage (breadth across topics)
- Single-topic restaurants with severe issues are excluded despite being harder for that topic
- Example: FiddleCakes has max_critical_score=22.5 for G1_allergy but wasn't selected (only 1 cell = low global rank)

## Next Steps (NEEDS USER DECISION)

**Three strategic options:**

### Option 1: Test with gpt-5-nano (intended baseline)
- Use gpt-5-nano instead of gpt-5 (25x cheaper)
- gpt-5-nano DOES make errors (documented ~75% RAG accuracy)
- This would show real baseline weaknesses for AMOS to improve

### Option 2: Find G1_allergy-specific tricky restaurants
- Use restaurants with highest G1_allergy incident scores (not global rank)
- Candidates: FiddleCakes (22.5), More Than Just Ice Cream (15.0), Texas Roadhouse (15.0)
- These have concentrated allergy issues, not diluted across 18 topics

### Option 3: Accept the selection and test at scale
- Current selection is DESIGNED for balanced benchmark across all topics
- Test on full 100 restaurants × 72 policies to see real accuracy distribution
- Small samples (n=2-4) may not show weaknesses that appear at scale

**Recommendation:** Option 1 + Option 3
- Run baseline with gpt-5-nano (the actual comparison baseline)
- Test on larger sample size to find cases where baseline fails
- Use gpt-5 only as "oracle" comparison (best possible single-shot performance)

## Open Questions

1. **What is the research goal?**
   - Show AMOS beats baseline LLM on complex cases?
   - Show AMOS is cost-effective alternative to gpt-5?
   - Show AMOS handles ambiguity better than retrieval-based methods?

2. **What should "tricky" mean?**
   - Globally tricky (current selection): Tests broad understanding across topics
   - Topic-specific tricky: Tests depth on individual policies
   - Both? (tiered evaluation)

3. **What baseline model to use?**
   - gpt-5-nano: Cheaper, documented failures, realistic comparison
   - gpt-5: Expensive, near-perfect, sets upper bound

4. **Continue with current test plan or pivot?**
   - Current plan was to test gpt-5 on 2 restaurants
   - Now revealed that selection may not test what we thought

## Key Files

**Modified:**
- `src/addm/eval/intermediate_metrics.py:354` - Fixed string→float conversion for total_score

**Analyzed:**
- `data/selection/yelp/topic_100.json` - Restaurant selection with global ranks
- `data/hits/yelp/G1_allergy.json` - Raw keyword extraction (522 critical restaurants)
- `src/addm/data/pipelines/topic_selection.py` - Selection algorithm (greedy by cell coverage)
- `results/dev/20260118_232443_G1_allergy_V2/results.json` - Cafe Blue Moose K=25
- `results/dev/20260118_232617_G1_allergy_V2/results.json` - Cafe Blue Moose K=50
- `results/dev/20260118_232757_G1_allergy_V2/results.json` - Oceana Grill K=25
- `results/dev/20260118_232829_G1_allergy_V2/results.json` - Oceana Grill K=50

**Plan file (in progress):**
- `/Users/jimmyyeh/.claude/plans/shiny-giggling-ladybug.md` - Original test plan (now outdated)

## Context & Background

**Selection algorithm design:**
```python
# From topic_selection.py - greedy_select_layered()
score = sum(1 for cell in r["cells"] if cell in needy_cells_set)
# Prioritizes: multi-topic coverage > single-topic severity
```

**Global rank vs topic-specific difficulty:**
- Oceana Grill: 18 cells (all topics) = rank #1, but only 1 G1_allergy incident
- FiddleCakes: 1 cell (G1_allergy only) = very low rank, but 22.5 severity score

**Baseline model comparison:**
- gpt-5: $1.25/1M input, $10/1M output (25x more expensive)
- gpt-5-nano: $0.05/1M input, $0.40/1M output (default baseline)
- Prior testing shows gpt-5-nano ~75% accurate on G1_allergy RAG method

**Project context:**
- Phase I: G1_allergy pipeline validation (current phase)
- Goal: Demonstrate AMOS method improves over baseline LLMs
- Challenge: Need to show clear gap between baseline and proposed method
