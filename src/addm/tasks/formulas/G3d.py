"""
Ground Truth computation for G3d (Price-Worth - Complex + L1.5).

Implements the formula from data/tasks/yelp/G3d_prompt.txt.
Complex formula with price-quality ratio + L1.5 meal type grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

PRICE_SCORES = {
    "cheap": 4, "moderate": 3, "expensive": 2, "very_expensive": 1, "not_mentioned": 2.5,
}
QUALITY_SCORES = {
    "exceptional": 4, "good": 3, "average": 2, "poor": 1, "not_mentioned": 2.5,
}
PORTION_MODIFIERS = {
    "generous": 1.0, "adequate": 0, "small": -0.5, "tiny": -1.0, "not_mentioned": 0,
}
SENTIMENT_MODIFIERS = {
    "excellent_value": 2.0, "good_value": 1.0, "fair": 0, "poor_value": -1.5, "ripoff": -3.0, "not_mentioned": 0,
}
COMPARISON_MODIFIERS = {
    "better_value": 1.0, "similar_value": 0, "worse_value": -1.0, "not_compared": 0,
}

VALUE_PATTERN_BONUS = {
    "consistently_good_value": 2.0,
    "lunch_special_value": 1.0,
    "dinner_worth_splurge": 1.5,
    "great_for_casual": 1.0,
    "meal_dependent": 0.0,
    "poor_value_overall": -1.0,
}


# =============================================================================
# Helpers
# =============================================================================


def get_meal_bucket(meal_type: str) -> str:
    if meal_type == "lunch":
        return "lunch"
    elif meal_type == "dinner":
        return "dinner"
    elif meal_type in ("brunch", "drinks_only"):
        return "casual"
    return "other"


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
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

    price_quality_ratio = quality_score / price_score
    l1_value_score = (price_quality_ratio * 2) + portion_modifier + sentiment_modifier + comparison_modifier

    exceptional_cheap = 2.0 if (price_level == "cheap" and quality_perception == "exceptional") else 0.0
    expensive_poor = -2.0 if (price_level in ("expensive", "very_expensive") and quality_perception == "poor") else 0.0

    l1_total_score = l1_value_score + exceptional_cheap + expensive_poor

    return {
        "price_score": price_score,
        "quality_score": quality_score,
        "price_quality_ratio": round(price_quality_ratio, 3),
        "l1_value_score": round(l1_value_score, 2),
        "exceptional_cheap": exceptional_cheap,
        "expensive_poor": expensive_poor,
        "l1_total_score": round(l1_total_score, 2),
    }


def compute_l1_positive(judgment: Dict[str, Any]) -> bool:
    value_sentiment = judgment.get("value_sentiment", "not_mentioned")
    price_level = judgment.get("price_level", "not_mentioned")
    quality_perception = judgment.get("quality_perception", "not_mentioned")
    portion_size = judgment.get("portion_size", "not_mentioned")

    if value_sentiment in ("excellent_value", "good_value"):
        return True
    if price_level in ("cheap", "moderate") and quality_perception in ("exceptional", "good"):
        return True
    if portion_size == "generous" and price_level != "very_expensive":
        return True
    return False


def compute_l1_negative(judgment: Dict[str, Any]) -> bool:
    value_sentiment = judgment.get("value_sentiment", "not_mentioned")
    price_level = judgment.get("price_level", "not_mentioned")
    quality_perception = judgment.get("quality_perception", "not_mentioned")
    portion_size = judgment.get("portion_size", "not_mentioned")

    if value_sentiment in ("poor_value", "ripoff"):
        return True
    if price_level in ("expensive", "very_expensive") and quality_perception in ("average", "poor"):
        return True
    if portion_size in ("small", "tiny") and price_level in ("expensive", "very_expensive"):
        return True
    return False


def compute_l15_buckets(value_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "lunch": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
        "dinner": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
        "casual": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
    }

    for j in value_judgments:
        meal_type = j.get("meal_type", "not_specified")
        bucket = get_meal_bucket(meal_type)
        if bucket not in buckets:
            continue

        buckets[bucket]["n_reviews"] += 1
        if compute_l1_positive(j):
            buckets[bucket]["n_positive"] += 1
        if compute_l1_negative(j):
            buckets[bucket]["n_negative"] += 1

    for key, bucket in buckets.items():
        n_positive = bucket["n_positive"]
        n_negative = bucket["n_negative"]
        bucket["value_rate"] = n_positive / max(n_positive + n_negative, 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_positive"] + v["n_negative"] > 0]

    if buckets_with_data:
        best_value_meal, best_bucket = max(buckets_with_data, key=lambda x: x[1]["value_rate"])
        best_value_rate = best_bucket["value_rate"]
        worst_value_meal, _ = min(buckets_with_data, key=lambda x: x[1]["value_rate"])
    else:
        best_value_meal = None
        best_value_rate = 0.0
        worst_value_meal = None

    n_meals_good_value = sum(1 for k, v in buckets.items() if v["value_rate"] >= 0.7 and v["n_reviews"] > 0)

    if n_meals_good_value >= 3:
        value_pattern = "consistently_good_value"
    elif buckets["lunch"]["value_rate"] >= 0.8 and buckets["lunch"]["n_reviews"] > 0:
        value_pattern = "lunch_special_value"
    elif buckets["dinner"]["value_rate"] >= 0.8 and buckets["dinner"]["n_reviews"] > 0:
        value_pattern = "dinner_worth_splurge"
    elif buckets["casual"]["value_rate"] >= 0.8 and buckets["casual"]["n_reviews"] > 0:
        value_pattern = "great_for_casual"
    elif n_meals_good_value >= 1:
        value_pattern = "meal_dependent"
    else:
        value_pattern = "poor_value_overall"

    return {
        "lunch": buckets["lunch"],
        "dinner": buckets["dinner"],
        "casual": buckets["casual"],
        "best_value_meal": best_value_meal,
        "best_value_rate": round(best_value_rate, 3),
        "worst_value_meal": worst_value_meal,
        "n_meals_good_value": n_meals_good_value,
        "value_pattern": value_pattern,
    }


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
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

    # L1.5 Meal Buckets
    l15_buckets = compute_l15_buckets(value_judgments)
    value_pattern = l15_buckets["value_pattern"]
    value_pattern_bonus = VALUE_PATTERN_BONUS.get(value_pattern, 0.0)

    mean_l1_score = sum_l1_score / max(n_value_reviews, 1)
    avg_price_quality_ratio = sum_price_quality_ratio / max(n_value_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score + value_pattern_bonus
    final_score = max(0.0, min(10.0, raw_score))

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
        "L1_5_meal_buckets": l15_buckets,
        "N_VALUE_REVIEWS": n_value_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_EXCELLENT_VALUE": n_excellent_value,
        "N_RIPOFF": n_ripoff,
        "N_BETTER_THAN_ALT": n_better_than_alt,
        "N_WORSE_THAN_ALT": n_worse_than_alt,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "AVG_PRICE_QUALITY_RATIO": round(avg_price_quality_ratio, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "VALUE_PATTERN_BONUS": value_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_value_reviews": n_value_reviews,
    }
