"""
Ground Truth computation for G3c (Price-Worth - Complex).

Implements the formula from data/tasks/yelp/G3c_prompt.txt.
Complex formula with price-quality ratio calculations.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

# Price scores (lower price = higher score)
PRICE_SCORES = {
    "cheap": 4,
    "moderate": 3,
    "expensive": 2,
    "very_expensive": 1,
    "not_mentioned": 2.5,
}

# Quality scores
QUALITY_SCORES = {
    "exceptional": 4,
    "good": 3,
    "average": 2,
    "poor": 1,
    "not_mentioned": 2.5,
}

# Portion modifiers
PORTION_MODIFIERS = {
    "generous": 1.0,
    "adequate": 0,
    "small": -0.5,
    "tiny": -1.0,
    "not_mentioned": 0,
}

# Sentiment modifiers
SENTIMENT_MODIFIERS = {
    "excellent_value": 2.0,
    "good_value": 1.0,
    "fair": 0,
    "poor_value": -1.5,
    "ripoff": -3.0,
    "not_mentioned": 0,
}

# Comparison modifiers
COMPARISON_MODIFIERS = {
    "better_value": 1.0,
    "similar_value": 0,
    "worse_value": -1.0,
    "not_compared": 0,
}


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    """Compute L1 composites using complex formula."""
    price_level = judgment.get("price_level", "not_mentioned")
    quality_perception = judgment.get("quality_perception", "not_mentioned")
    portion_size = judgment.get("portion_size", "not_mentioned")
    value_sentiment = judgment.get("value_sentiment", "not_mentioned")
    comparison = judgment.get("comparison_to_alternatives", "not_compared")

    price_score = PRICE_SCORES.get(price_level, 2.5)
    quality_score = QUALITY_SCORES.get(quality_perception, 2.5)
    portion_modifier = PORTION_MODIFIERS.get(portion_size, 0)
    sentiment_modifier = SENTIMENT_MODIFIERS.get(value_sentiment, 0)
    comparison_modifier = COMPARISON_MODIFIERS.get(comparison, 0)

    # Price-quality ratio (higher = better value)
    price_quality_ratio = quality_score / price_score

    # L1_VALUE_SCORE
    l1_value_score = (price_quality_ratio * 2) + portion_modifier + sentiment_modifier + comparison_modifier

    # Interaction effects
    exceptional_cheap = 2.0 if (price_level == "cheap" and quality_perception == "exceptional") else 0.0
    expensive_poor = -2.0 if (price_level in ("expensive", "very_expensive") and quality_perception == "poor") else 0.0

    l1_total_score = l1_value_score + exceptional_cheap + expensive_poor

    return {
        "price_score": price_score,
        "quality_score": quality_score,
        "portion_modifier": portion_modifier,
        "sentiment_modifier": sentiment_modifier,
        "comparison_modifier": comparison_modifier,
        "price_quality_ratio": round(price_quality_ratio, 3),
        "l1_value_score": round(l1_value_score, 2),
        "exceptional_cheap": exceptional_cheap,
        "expensive_poor": expensive_poor,
        "l1_total_score": round(l1_total_score, 2),
    }


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute ground truth from extracted judgments.
    """
    value_judgments = [j for j in judgments if j.get("is_value_related", False)]
    n_value_reviews = len(value_judgments)

    if n_value_reviews == 0:
        confidence_level = "none"
    elif n_value_reviews <= 2:
        confidence_level = "low"
    elif n_value_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    sum_l1_score = 0.0
    sum_price_quality_ratio = 0.0
    n_excellent_value = 0
    n_ripoff = 0
    n_better_than_alt = 0
    n_worse_than_alt = 0

    for j in value_judgments:
        l1 = compute_l1_complex(j)

        sum_l1_score += l1["l1_total_score"]
        sum_price_quality_ratio += l1["price_quality_ratio"]

        if j.get("value_sentiment") == "excellent_value":
            n_excellent_value += 1
        if j.get("value_sentiment") == "ripoff":
            n_ripoff += 1
        if j.get("comparison_to_alternatives") == "better_value":
            n_better_than_alt += 1
        if j.get("comparison_to_alternatives") == "worse_value":
            n_worse_than_alt += 1

    # Aggregation
    mean_l1_score = sum_l1_score / max(n_value_reviews, 1)
    avg_price_quality_ratio = sum_price_quality_ratio / max(n_value_reviews, 1)

    # Formulas
    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
    if final_score >= 7.5:
        base_verdict_by_score = "Excellent Value"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good Value"
    elif final_score >= 3.5:
        base_verdict_by_score = "Mixed Value"
    else:
        base_verdict_by_score = "Poor Value"

    override_applied = "none"
    verdict = base_verdict_by_score

    if n_ripoff >= 2:
        override_applied = "ripoff_pattern"
        verdict = "Poor Value"
    elif avg_price_quality_ratio >= 1.5 and n_value_reviews >= 3 and verdict in ("Mixed Value", "Poor Value"):
        override_applied = "high_ratio_min_good"
        verdict = "Good Value"
    elif n_better_than_alt >= 2 and n_worse_than_alt == 0 and verdict in ("Mixed Value", "Poor Value"):
        override_applied = "better_than_alt_min_good"
        verdict = "Good Value"
    elif mean_l1_score < -1:
        override_applied = "low_mean_score"
        verdict = "Poor Value"

    return {
        # L2 Aggregates
        "N_VALUE_REVIEWS": n_value_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_EXCELLENT_VALUE": n_excellent_value,
        "N_RIPOFF": n_ripoff,
        "N_BETTER_THAN_ALT": n_better_than_alt,
        "N_WORSE_THAN_ALT": n_worse_than_alt,
        # Aggregation
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "AVG_PRICE_QUALITY_RATIO": round(avg_price_quality_ratio, 3),
        # Formula results
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_value_reviews": n_value_reviews,
    }
