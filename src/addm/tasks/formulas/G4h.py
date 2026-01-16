"""
Ground Truth computation for G4h (Kitchen Quality - Complex + L1.5).

Implements the formula from data/tasks/yelp/G4h_prompt.txt.
Complex formula with weighted food factors + L1.5 food aspect grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

TASTE_SCORES = {"exceptional": 4.0, "good": 2.0, "average": 0, "below_average": -2.0, "poor": -4.0, "not_mentioned": 0}
FRESHNESS_SCORES = {"very_fresh": 2.0, "fresh": 1.0, "questionable": -2.0, "stale": -4.0, "not_mentioned": 0}
TEMPERATURE_SCORES = {"perfect": 1.5, "acceptable": 0.5, "lukewarm": -1.0, "cold": -2.5, "not_mentioned": 0}
PRESENTATION_SCORES = {"stunning": 2.0, "nice": 1.0, "adequate": 0, "sloppy": -1.5, "not_mentioned": 0}
COOKING_SCORES = {"perfect": 2.5, "good": 1.5, "acceptable": 0, "overcooked": -2.0, "undercooked": -3.0, "not_mentioned": 0}
SEASONING_SCORES = {"perfectly_seasoned": 1.5, "well_seasoned": 0.5, "under_seasoned": -1.0, "over_seasoned": -1.5, "not_mentioned": 0}
CONSISTENCY_MODIFIERS = {"always_good": 1.5, "usually_good": 0.5, "hit_or_miss": -1.0, "declined": -2.0, "not_mentioned": 0}

KITCHEN_PATTERN_BONUS = {
    "all_around_excellent": 2.0,
    "flavor_forward": 1.5,
    "technically_skilled": 1.5,
    "quality_ingredients": 1.0,
    "partial_strengths": 0.0,
    "needs_work": -1.0,
}


# =============================================================================
# Helpers
# =============================================================================


def get_kitchen_bucket(judgment: Dict[str, Any]) -> str:
    if judgment.get("taste_quality", "not_mentioned") != "not_mentioned" or judgment.get("seasoning", "not_mentioned") != "not_mentioned":
        return "taste"
    elif judgment.get("cooking_execution", "not_mentioned") != "not_mentioned" or judgment.get("temperature", "not_mentioned") != "not_mentioned":
        return "execution"
    elif judgment.get("freshness", "not_mentioned") != "not_mentioned" or judgment.get("presentation", "not_mentioned") != "not_mentioned":
        return "quality"
    return "other"


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    taste_quality = judgment.get("taste_quality", "not_mentioned")
    freshness = judgment.get("freshness", "not_mentioned")
    temperature = judgment.get("temperature", "not_mentioned")
    presentation = judgment.get("presentation", "not_mentioned")
    cooking_execution = judgment.get("cooking_execution", "not_mentioned")
    seasoning = judgment.get("seasoning", "not_mentioned")
    dish_consistency = judgment.get("dish_consistency", "not_mentioned")

    taste_score = TASTE_SCORES.get(taste_quality, 0)
    freshness_score = FRESHNESS_SCORES.get(freshness, 0)
    temperature_score = TEMPERATURE_SCORES.get(temperature, 0)
    presentation_score = PRESENTATION_SCORES.get(presentation, 0)
    cooking_score = COOKING_SCORES.get(cooking_execution, 0)
    seasoning_score = SEASONING_SCORES.get(seasoning, 0)
    consistency_modifier = CONSISTENCY_MODIFIERS.get(dish_consistency, 0)

    l1_kitchen_score = (
        taste_score + freshness_score + temperature_score +
        presentation_score + cooking_score + seasoning_score + consistency_modifier
    )

    stale_and_poor = -3.0 if (freshness == "stale" and taste_quality == "poor") else 0.0
    perfect_execution = 2.0 if (cooking_execution == "perfect" and temperature == "perfect" and taste_quality == "exceptional") else 0.0
    undercooked_penalty = -2.0 if cooking_execution == "undercooked" else 0.0

    l1_total_score = l1_kitchen_score + stale_and_poor + perfect_execution + undercooked_penalty

    return {
        "taste_score": taste_score,
        "freshness_score": freshness_score,
        "l1_kitchen_score": round(l1_kitchen_score, 2),
        "stale_and_poor": stale_and_poor,
        "perfect_execution": perfect_execution,
        "undercooked_penalty": undercooked_penalty,
        "l1_total_score": round(l1_total_score, 2),
    }


def compute_l1_positive_kitchen(judgment: Dict[str, Any]) -> bool:
    taste_quality = judgment.get("taste_quality", "not_mentioned")
    cooking_execution = judgment.get("cooking_execution", "not_mentioned")
    presentation = judgment.get("presentation", "not_mentioned")
    return taste_quality in ("exceptional", "good") or cooking_execution in ("perfect", "good") or presentation == "stunning"


def compute_l1_negative_kitchen(judgment: Dict[str, Any]) -> bool:
    taste_quality = judgment.get("taste_quality", "not_mentioned")
    cooking_execution = judgment.get("cooking_execution", "not_mentioned")
    freshness = judgment.get("freshness", "not_mentioned")
    return taste_quality in ("below_average", "poor") or cooking_execution in ("overcooked", "undercooked") or freshness in ("questionable", "stale")


def compute_l15_buckets(kitchen_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "taste": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
        "execution": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
        "quality": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
    }

    for j in kitchen_judgments:
        bucket = get_kitchen_bucket(j)
        if bucket not in buckets:
            continue

        buckets[bucket]["n_reviews"] += 1

        if compute_l1_positive_kitchen(j):
            buckets[bucket]["n_positive"] += 1
        if compute_l1_negative_kitchen(j):
            buckets[bucket]["n_negative"] += 1

    for key, bucket in buckets.items():
        n_positive = bucket["n_positive"]
        n_negative = bucket["n_negative"]
        bucket["success_rate"] = n_positive / max(n_positive + n_negative, 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_positive"] + v["n_negative"] > 0]

    if buckets_with_data:
        best_aspect, best_bucket = max(buckets_with_data, key=lambda x: x[1]["success_rate"])
        best_success_rate = best_bucket["success_rate"]
        worst_aspect, _ = min(buckets_with_data, key=lambda x: x[1]["success_rate"])
    else:
        best_aspect = None
        best_success_rate = 0.0
        worst_aspect = None

    n_aspects_strong = sum(1 for k, v in buckets.items() if v["success_rate"] >= 0.7 and v["n_reviews"] > 0)

    if n_aspects_strong >= 3:
        kitchen_pattern = "all_around_excellent"
    elif buckets["taste"]["success_rate"] >= 0.8 and buckets["taste"]["n_reviews"] > 0:
        kitchen_pattern = "flavor_forward"
    elif buckets["execution"]["success_rate"] >= 0.8 and buckets["execution"]["n_reviews"] > 0:
        kitchen_pattern = "technically_skilled"
    elif buckets["quality"]["success_rate"] >= 0.8 and buckets["quality"]["n_reviews"] > 0:
        kitchen_pattern = "quality_ingredients"
    elif n_aspects_strong >= 1:
        kitchen_pattern = "partial_strengths"
    else:
        kitchen_pattern = "needs_work"

    return {
        "taste": buckets["taste"],
        "execution": buckets["execution"],
        "quality": buckets["quality"],
        "best_aspect": best_aspect,
        "best_aspect_rate": round(best_success_rate, 3),
        "worst_aspect": worst_aspect,
        "n_aspects_strong": n_aspects_strong,
        "kitchen_pattern": kitchen_pattern,
    }


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    kitchen_judgments = [j for j in judgments if j.get("is_kitchen_related", False)]
    n_kitchen_reviews = len(kitchen_judgments)

    if n_kitchen_reviews == 0:
        confidence_level = "none"
    elif n_kitchen_reviews <= 2:
        confidence_level = "low"
    elif n_kitchen_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    sum_l1_score = 0.0
    n_exceptional = 0
    n_stale = 0
    n_undercooked = 0
    n_declined = 0

    for j in kitchen_judgments:
        l1 = compute_l1_complex(j)
        sum_l1_score += l1["l1_total_score"]

        if j.get("taste_quality") == "exceptional":
            n_exceptional += 1
        if j.get("freshness") == "stale":
            n_stale += 1
        if j.get("cooking_execution") == "undercooked":
            n_undercooked += 1
        if j.get("dish_consistency") == "declined":
            n_declined += 1

    # L1.5 Kitchen Buckets
    l15_buckets = compute_l15_buckets(kitchen_judgments)
    kitchen_pattern = l15_buckets["kitchen_pattern"]
    kitchen_pattern_bonus = KITCHEN_PATTERN_BONUS.get(kitchen_pattern, 0.0)

    mean_l1_score = sum_l1_score / max(n_kitchen_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score + kitchen_pattern_bonus
    final_score = max(0.0, min(10.0, raw_score))

    if final_score >= 7.5:
        base_verdict_by_score = "Excellent Kitchen"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good Kitchen"
    elif final_score >= 3.5:
        base_verdict_by_score = "Mixed Kitchen"
    else:
        base_verdict_by_score = "Poor Kitchen"

    override_applied = "none"
    verdict = base_verdict_by_score
    food_safety_note = None

    if kitchen_pattern == "all_around_excellent" and verdict in ("Mixed Kitchen", "Poor Kitchen"):
        override_applied = "all_around_min_good"
        verdict = "Good Kitchen"
    elif kitchen_pattern == "needs_work" and (n_undercooked >= 1 or n_stale >= 2):
        override_applied = "needs_work_with_issues"
        verdict = "Poor Kitchen"
        if n_undercooked >= 1:
            food_safety_note = "Food safety concern: undercooked food reported"
    elif n_undercooked >= 1 and verdict in ("Excellent Kitchen", "Good Kitchen"):
        override_applied = "undercooked_max_mixed"
        verdict = "Mixed Kitchen"
        food_safety_note = "Food safety concern: undercooked food reported"
    elif mean_l1_score < -2:
        override_applied = "low_mean_score"
        verdict = "Poor Kitchen"

    result = {
        "L1_5_kitchen_buckets": l15_buckets,
        "N_KITCHEN_REVIEWS": n_kitchen_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_EXCEPTIONAL": n_exceptional,
        "N_STALE": n_stale,
        "N_UNDERCOOKED": n_undercooked,
        "N_DECLINED": n_declined,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "KITCHEN_PATTERN_BONUS": kitchen_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_kitchen_reviews": n_kitchen_reviews,
    }

    if food_safety_note:
        result["food_safety_note"] = food_safety_note

    return result
