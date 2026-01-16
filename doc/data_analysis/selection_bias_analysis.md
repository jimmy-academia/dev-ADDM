# Selection Bias Analysis: Topic-Hit Based Selection

**Date**: 2025-01-16
**Question**: Does selecting restaurants based on keyword hits introduce problematic bias?

## Executive Summary

**Finding**: Topic-hit selection introduces **real but acceptable bias** for this benchmark.

| Bias Type | Severity | Impact | Mitigation |
|-----------|----------|--------|------------|
| Review volume | Low (r=0.038) | Minimal | None needed |
| Star rating | Low | Minimal | Stratification |
| Cuisine type | Moderate | Fine dining over-represented | Stratification helps |
| Review quality | High | Rich reviews favored | Acceptable for benchmark |
| Geographic | Moderate | Philadelphia-heavy | Accept for consistency |

## Detailed Analysis

### 1. Review Count vs Topic Coverage

**Question**: Do restaurants with more reviews get more topic coverage simply due to volume?

**Finding**: Correlation is **0.038** (essentially zero).

| Review Count Bucket | Avg Topic Count | N |
|--------------------|-----------------|---|
| Low (<400) | 1.14 | 374 |
| Mid (400-800) | 1.59 | 592 |
| High (800+) | 1.52 | 234 |

**Conclusion**: Review volume does NOT strongly predict topic coverage. A restaurant with 2000 reviews is NOT automatically in more topics than one with 400.

### 2. Star Rating vs Topic Coverage

**Question**: Are higher-rated restaurants over-represented in multi-topic selection?

**Finding**: Correlation is minimal.

| Stars | Avg Topic Count | N |
|-------|-----------------|---|
| 2.5 | 1.41 | 46 |
| 3.0 | 1.41 | 101 |
| 3.5 | 1.39 | 259 |
| 4.0 | 1.43 | 467 |
| 4.5 | 1.53 | 303 |

**Conclusion**: Star rating has minimal impact on topic coverage. The 3.5-star to 4.5-star range is nearly flat.

### 3. Review Quality/Length Bias

**Question**: Do restaurants with "better" (longer, more detailed) reviews get selected?

**Finding**: YES - this is the dominant bias.

**Top 10 by hits-per-review (normalized efficiency):**
1. Vetri Cucina - 1.63 hits/review (fine dining)
2. Atlantis Steakhouse - 1.39 hits/review (upscale)
3. Atlantis Bistro Napa - 1.30 hits/review (upscale)
4. Bistrot La Minette - 1.26 hits/review (French fine dining)
5. Marrakesh - 1.24 hits/review (unique experience)

**Bottom 10 by hits-per-review:**
1. Hattie B's Hot Chicken - 0.010 hits/review (casual)
2. Reading Terminal Market - 0.011 hits/review (food hall)
3. Acme Oyster House - 0.012 hits/review (casual seafood)
4. St. Elmo Steak House - 0.012 hits/review (traditional)
5. Coop's Place - 0.013 hits/review (dive bar)

**Interpretation**: Fine dining and "experience" restaurants generate reviews with more topic-relevant content. Casual spots get shorter, simpler reviews.

### 4. Cuisine/Category Bias

**Question**: Are certain restaurant types over-represented in multi-topic selection?

**Finding**: YES - significant over-representation of:

| Category | Multi-Topic % | Single-Topic % | Ratio |
|----------|--------------|----------------|-------|
| Hotels | 10.0% | 2.7% | 3.72x |
| Hotels & Travel | 11.7% | 3.6% | 3.22x |
| Lounges | 10.0% | 3.6% | 2.76x |
| Steakhouses | 17.5% | 7.7% | 2.27x |
| Japanese | 10.0% | 4.6% | 2.19x |
| Italian | 12.5% | 5.8% | 2.14x |
| Venues & Event Spaces | 8.3% | 4.0% | 2.10x |

**Interpretation**: Multi-topic restaurants tend to be:
- Fine dining (steakhouses, Italian, Japanese)
- Event/occasion venues (hotels, lounges, event spaces)
- Places that inspire detailed, multi-faceted reviews

**Under-represented**: Fast food, casual chains, neighborhood spots.

### 5. Geographic Bias

**Question**: Is the selection geographically skewed?

**Finding**: Philadelphia dominates (45% of multi-topic restaurants).

| City | Count | Percentage |
|------|-------|------------|
| Philadelphia | 54 | 45.0% |
| Reno | 9 | 7.5% |
| Tampa | 8 | 6.7% |
| Nashville | 8 | 6.7% |
| New Orleans | 8 | 6.7% |

**Cause**: Likely reflects Yelp dataset distribution or Philadelphia reviewer behavior.

**Mitigation considered**: Enforce geographic diversity caps.

**Decision**: Accept concentration. Philadelphia provides consistent reviewer pool.

## Impact of Stratification

Stratifying by Stars × Volume helps balance the bias:

| Quadrant | Count | Characteristics |
|----------|-------|-----------------|
| High Stars, High Volume | 21 | Steakhouses, upscale American |
| High Stars, Low Volume | 20 | French, fine dining |
| Low Stars, High Volume | 39 | Event spaces, bars |
| Low Stars, Low Volume | 40 | Traditional American, neighborhood |

**Key insight**: Low-rated quadrants bring in different restaurant types (casual American, neighborhood bars) that partially offset the fine-dining bias.

## Implications for Benchmark

### Why the bias is ACCEPTABLE:

1. **Benchmark purpose**: We're testing LLM capability to extract nuanced judgments from review text. Fine dining reviews ARE richer in topic-relevant content - this is the content we WANT to test.

2. **Review complexity**: Casual reviews ("food good, fast service") don't exercise the full complexity of our prompts. Rich reviews from fine dining test L0/L1/L1.5/L2 extraction properly.

3. **Fair comparison**: The same restaurants are used for ALL models. Any bias applies equally, so model comparison is fair.

4. **Stratification helps**: By enforcing Stars × Volume quadrants, we get some casual/neighborhood representation.

### Why the bias might be CONCERNING:

1. **Generalization**: Results may not transfer to casual dining evaluation tasks.

2. **Linguistic patterns**: Fine dining reviews may have different vocabulary and structure than typical restaurant reviews.

3. **Topic applicability**: Event venues naturally span multiple topics (romantic dinners, business lunches, group events) - this may inflate their "multi-topic" appearance artificially.

## Recommendations

1. **Accept the bias** - it's inherent to topic-based selection and serves benchmark goals.

2. **Document explicitly** - users should understand the restaurant type distribution.

3. **Apply stratification** - Stars × Volume quadrants provide some diversity.

4. **Future work**: Consider a separate "casual dining" subset for generalization testing.

## Decision

**Proceed with topic-hit based selection** with Stars × Volume stratification.

The bias toward detailed reviews from fine dining establishments is acceptable because:
- It serves the benchmark's goal of testing complex judgment extraction
- It's systematic and fair across all models being compared
- Stratification provides some category diversity
- The limitation is documented for benchmark users
