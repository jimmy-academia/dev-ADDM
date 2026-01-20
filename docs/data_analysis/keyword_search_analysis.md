# Keyword Search Analysis

**Date**: 2025-01-16
**Purpose**: Document findings from keyword search phase to inform restaurant selection.

## Data Overview

### Keyword Hit Files

Location: `data/selection/yelp/G*_*.json`

| File | Topic | Restaurants | Notes |
|------|-------|-------------|-------|
| G1_allergy.json | Allergy safety | 100 | Peanut mentions dominate |
| G1_dietary.json | Dietary restrictions | 100 | |
| G1_hygiene.json | Hygiene concerns | 100 | |
| G2_romance.json | Romantic dining | 100 | |
| G2_business.json | Business dining | 100 | |
| G2_group.json | Group dining | 100 | |
| G3_price_worth.json | Value assessment | 100 | |
| G3_hidden_costs.json | Hidden costs | **27** | Sparse - keywords too specific |
| G3_time_value.json | Time/wait value | 100 | |
| G4_server.json | Server quality | 100 | |
| G4_kitchen.json | Kitchen/food quality | 100 | |
| G4_environment.json | Environment/ambiance | 100 | |
| G5_capacity.json | Capacity handling | 100 | |
| G5_execution.json | Order execution | 100 | |
| G5_consistency.json | Consistency | 100 | |
| G6_uniqueness.json | Uniqueness | 100 | |
| G6_comparison.json | Competitor comparison | 100 | |
| G6_loyalty.json | Customer loyalty | 100 | |

### Record Structure

Each restaurant record contains:
```json
{
  "business_id": "5CbJgHWgvjyDRw9Sl4BXhg",
  "name": "Yo Mama's Bar & Grill",
  "categories": "Pubs, Dive Bars, Nightlife, Bars, Restaurants, Burgers",
  "city": "New Orleans",
  "state": "LA",
  "stars": 4.0,
  "review_count": 825,
  "keyword_hits": 223,
  "sample_matches": ["Peanut", "peanut", "Peanut", "Peanut", "peanut"]
}
```

## Coverage Analysis

### Topic Coverage Distribution

| Topics Covered | Restaurant Count | Percentage |
|----------------|------------------|------------|
| 1 topic | 856 | 71.3% |
| 2 topics | 224 | 18.7% |
| 3 topics | 74 | 6.2% |
| 4 topics | 32 | 2.7% |
| 5 topics | 11 | 0.9% |
| 6 topics | 3 | 0.3% |
| **Total** | **1,200** | 100% |

**Key insight**: Most restaurants are "topic specialists" - only 10% appear in 3+ topics.

### Multi-Topic Restaurants (3+ coverage)

Top 10 by topic coverage:
1. Rat's Restaurant (Hamilton, NJ) - 6 topics
2. Vetri Cucina (Philadelphia, PA) - 6 topics
3. Bistrot La Minette (Philadelphia, PA) - 6 topics
4. Fleming's Prime Steakhouse & Wine Bar (Tucson, AZ) - 5 topics
5. Ocean Prime (Tampa, FL) - 5 topics
6. Fleming's Prime Steakhouse & Wine Bar (Tampa, FL) - 5 topics
7. Atlantis Steakhouse (Reno, NV) - 5 topics
8. Vernick Food & Drink (Philadelphia, PA) - 5 topics
9. Butcher and Singer (Philadelphia, PA) - 5 topics
10. Fork (Philadelphia, PA) - 5 topics

**Pattern**: High-end restaurants tend to have broader topic coverage (more detailed reviews).

### Within-Group Overlap

| Group | All 3 Topics | Any 2 Topics | Union |
|-------|-------------|--------------|-------|
| G1 (Health) | 1 | 7 | 292 |
| G2 (Social) | 11 | 46 | 243 |
| G3 (Value) | 0 | 4 | 223 |
| G4 (Quality) | 0 | 12 | 288 |
| G5 (Operations) | 0 | 15 | 285 |
| G6 (Strategy) | 0 | 26 | 274 |

**Key insight**: Cross-topic overlap within groups is rare. G2 (social context) has best overlap.

### Cross-Group Overlap

Restaurants covering 2+ topics in multiple groups:
- G2 + G4 + G6: 1 restaurant
- G2 + G6: 4 restaurants
- G4 + G6: 1 restaurant
- G2 + G4: 2 restaurants
- G2 + G5: 1 restaurant

**Key insight**: Cross-group coverage is extremely rare (only 9 restaurants).

## Original Core_100 Comparison

The original `core_100.json` selection (Philadelphia, 5 cuisines, stratified) has:
- **Only 13/100 restaurants appear in ANY keyword hit list**
- Per-topic coverage ranges from 0-5 restaurants

| Topic | core_100 Overlap |
|-------|-----------------|
| G1_allergy | 0 |
| G1_dietary | 1 |
| G1_hygiene | 0 |
| G2_business | 2 |
| G2_group | 4 |
| G2_romance | 4 |
| G3_hidden_costs | 0 |
| G3_price_worth | 2 |
| G3_time_value | 0 |
| G4_environment | 2 |
| G4_kitchen | 1 |
| G4_server | 2 |
| G5_capacity | 1 |
| G5_consistency | 1 |
| G5_execution | 1 |
| G6_comparison | 5 |
| G6_loyalty | 2 |
| G6_uniqueness | 3 |

**Conclusion**: Original selection approach is unusable for topic-based tasks.

## Multi-Topic Restaurant Characteristics

### Star Rating Distribution (120 restaurants with 3+ topics)

| Stars | Count | Percentage |
|-------|-------|------------|
| 2.5 | 5 | 4.2% |
| 3.0 | 12 | 10.0% |
| 3.5 | 21 | 17.5% |
| 4.0 | 41 | 34.2% |
| 4.5 | 41 | 34.2% |

**Observation**: Skewed toward high ratings (68% are 4.0+).

### Review Count Distribution

- Minimum: 326
- Maximum: 2,924
- Median: 578

**Observation**: All have substantial review volume (min 326).

### Geographic Distribution (120 restaurants with 3+ topics)

| City | Count | Percentage |
|------|-------|------------|
| Philadelphia, PA | 54 | 45.0% |
| Reno, NV | 9 | 7.5% |
| Tampa, FL | 8 | 6.7% |
| Nashville, TN | 8 | 6.7% |
| New Orleans, LA | 8 | 6.7% |
| Santa Barbara, CA | 6 | 5.0% |
| Tucson, AZ | 5 | 4.2% |
| Indianapolis, IN | 5 | 4.2% |
| Saint Louis, MO | 4 | 3.3% |
| Other | 13 | 10.8% |

**Observation**: Philadelphia dominates (45%), but 10 cities represented.

### Category Distribution (120 restaurants with 3+ topics)

| Category | Count |
|----------|-------|
| Restaurants | 116 |
| American (New) | 46 |
| Nightlife | 43 |
| Bars | 41 |
| American (Traditional) | 27 |
| Food | 25 |
| Seafood | 22 |
| Steakhouses | 21 |
| Event Planning & Services | 17 |
| Breakfast & Brunch | 17 |
| Italian | 15 |
| Hotels & Travel | 14 |
| Arts & Entertainment | 13 |
| Cocktail Bars | 12 |
| Japanese | 12 |

**Observation**: American cuisine and bars dominate. Many are upscale/event venues.

## Sparse Topic: G3_hidden_costs

Only 27 restaurants found with hidden cost mentions.

**Possible causes**:
1. Keywords too specific (gratuity, service charge, corkage, etc.)
2. Hidden costs not commonly discussed in reviews
3. Yelp reviewers focus on food/service more than pricing surprises

**Recommendation**: Accept sparse coverage for this topic.

## Recommendations for Selection

Based on this analysis:

1. **Abandon core_100**: Only 13% topic overlap - unusable
2. **Use topic coverage as primary criterion**: Select restaurants appearing in most topics
3. **Stratify by stars × volume**: Ensure quality/popularity diversity
4. **Accept geographic concentration**: Philadelphia has best coverage naturally
5. **Accept G3_hidden_costs sparsity**: Only 10-20 restaurants will have signal
6. **Target multi-topic restaurants**: 120 restaurants have 3+ topic coverage

## Files Generated

- `data/selection/yelp/G*_*.json` - Topic-specific restaurant rankings
- `data/selection/yelp/*_partial.json` - Intermediate files (can be cleaned up)
