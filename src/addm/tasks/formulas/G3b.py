"""
Ground Truth computation for G3b (Price-Worth - Simple + L1.5).

Implements the formula from data/tasks/yelp/G3b_prompt.txt.
Simple formula with L1.5 meal type grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

# L1.5 Value Pattern Bonus values
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
    """Map meal type to L1.5 bucket."""
    if meal_type == "lunch":
        return "lunch"
    elif meal_type == "dinner":
        return "dinner"
    elif meal_type in ("brunch", "drinks_only"):
        return "casual"
    else:
        return "other"


def compute_l1_positive(judgment: Dict[str, Any]) -> bool:
    """POSITIVE_VALUE computation."""
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
    """NEGATIVE_VALUE computation."""
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
    """Compute L1.5 meal type buckets."""
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

    n_positive = 0
    n_negative = 0

    for j in value_judgments:
        if compute_l1_positive(j):
            n_positive += 1
        if compute_l1_negative(j):
            n_negative += 1

    # L1.5 Meal Buckets
    l15_buckets = compute_l15_buckets(value_judgments)
    value_pattern = l15_buckets["value_pattern"]
    value_pattern_bonus = VALUE_PATTERN_BONUS.get(value_pattern, 0.0)

    # Formulas
    value_rate = n_positive / max(n_positive + n_negative, 1)
    positive_score = n_positive * 1.5
    negative_score = n_negative * 1.5

    raw_score = BASE_SCORE + positive_score - negative_score + value_pattern_bonus
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
    meal_recommendation = None

    if value_pattern == "consistently_good_value" and verdict in ("Mixed Value", "Poor Value"):
        override_applied = "consistent_value_min_good"
        verdict = "Good Value"
    elif value_pattern == "lunch_special_value":
        meal_recommendation = "Best value at lunch"
    elif value_pattern == "poor_value_overall" and n_negative >= 3:
        override_applied = "poor_overall_with_negatives"
        verdict = "Poor Value"

    # Check for exceptional value at specific meal
    for bucket_name, bucket in [("lunch", l15_buckets["lunch"]), ("dinner", l15_buckets["dinner"]), ("casual", l15_buckets["casual"])]:
        if bucket["value_rate"] >= 0.9 and bucket["n_reviews"] >= 3:
            meal_recommendation = f"Best value at {bucket_name}"
            break

    result = {
        # L1.5 Meal Buckets
        "L1_5_meal_buckets": l15_buckets,
        # L2 Aggregates
        "N_VALUE_REVIEWS": n_value_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "VALUE_RATE": round(value_rate, 3),
        # Formula results
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "VALUE_PATTERN_BONUS": value_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_value_reviews": n_value_reviews,
    }

    if meal_recommendation:
        result["meal_recommendation"] = meal_recommendation

    return result
