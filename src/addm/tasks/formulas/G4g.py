"""
Ground Truth computation for G4g (Kitchen Quality - Complex).

Implements the formula from data/tasks/yelp/G4g_prompt.txt.
Complex formula with weighted food factors and interaction effects.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

TASTE_SCORES = {
    "exceptional": 4.0, "good": 2.0, "average": 0, "below_average": -2.0,
    "poor": -4.0, "not_mentioned": 0,
}
FRESHNESS_SCORES = {"very_fresh": 2.0, "fresh": 1.0, "questionable": -2.0, "stale": -4.0, "not_mentioned": 0}
TEMPERATURE_SCORES = {"perfect": 1.5, "acceptable": 0.5, "lukewarm": -1.0, "cold": -2.5, "not_mentioned": 0}
PRESENTATION_SCORES = {"stunning": 2.0, "nice": 1.0, "adequate": 0, "sloppy": -1.5, "not_mentioned": 0}
COOKING_SCORES = {"perfect": 2.5, "good": 1.5, "acceptable": 0, "overcooked": -2.0, "undercooked": -3.0, "not_mentioned": 0}
SEASONING_SCORES = {"perfectly_seasoned": 1.5, "well_seasoned": 0.5, "under_seasoned": -1.0, "over_seasoned": -1.5, "not_mentioned": 0}
CONSISTENCY_MODIFIERS = {"always_good": 1.5, "usually_good": 0.5, "hit_or_miss": -1.0, "declined": -2.0, "not_mentioned": 0}


# =============================================================================
# Helpers
# =============================================================================


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

    # Interaction effects
    stale_and_poor = -3.0 if (freshness == "stale" and taste_quality == "poor") else 0.0
    perfect_execution = 2.0 if (cooking_execution == "perfect" and temperature == "perfect" and taste_quality == "exceptional") else 0.0
    undercooked_penalty = -2.0 if cooking_execution == "undercooked" else 0.0

    l1_total_score = l1_kitchen_score + stale_and_poor + perfect_execution + undercooked_penalty

    return {
        "taste_score": taste_score,
        "freshness_score": freshness_score,
        "temperature_score": temperature_score,
        "presentation_score": presentation_score,
        "cooking_score": cooking_score,
        "seasoning_score": seasoning_score,
        "consistency_modifier": consistency_modifier,
        "l1_kitchen_score": round(l1_kitchen_score, 2),
        "stale_and_poor": stale_and_poor,
        "perfect_execution": perfect_execution,
        "undercooked_penalty": undercooked_penalty,
        "l1_total_score": round(l1_total_score, 2),
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
    n_poor = 0
    n_stale = 0
    n_undercooked = 0
    n_declined = 0

    for j in kitchen_judgments:
        l1 = compute_l1_complex(j)
        sum_l1_score += l1["l1_total_score"]

        if j.get("taste_quality") == "exceptional":
            n_exceptional += 1
        if j.get("taste_quality") == "poor":
            n_poor += 1
        if j.get("freshness") == "stale":
            n_stale += 1
        if j.get("cooking_execution") == "undercooked":
            n_undercooked += 1
        if j.get("dish_consistency") == "declined":
            n_declined += 1

    mean_l1_score = sum_l1_score / max(n_kitchen_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score
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
    decline_warning = None

    if n_undercooked >= 1 and verdict in ("Excellent Kitchen", "Good Kitchen"):
        override_applied = "undercooked_max_mixed"
        verdict = "Mixed Kitchen"
        food_safety_note = "Food safety concern: undercooked food reported"
    elif n_stale >= 2 and verdict in ("Excellent Kitchen", "Good Kitchen"):
        override_applied = "stale_max_mixed"
        verdict = "Mixed Kitchen"
    elif n_declined >= 2:
        decline_warning = "Quality may have declined recently"
    elif mean_l1_score >= 4 and n_exceptional >= 2 and verdict in ("Mixed Kitchen", "Poor Kitchen"):
        override_applied = "exceptional_min_good"
        verdict = "Good Kitchen"
    elif mean_l1_score < -2:
        override_applied = "low_mean_score"
        verdict = "Poor Kitchen"

    result = {
        "N_KITCHEN_REVIEWS": n_kitchen_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_EXCEPTIONAL": n_exceptional,
        "N_POOR": n_poor,
        "N_STALE": n_stale,
        "N_UNDERCOOKED": n_undercooked,
        "N_DECLINED": n_declined,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_kitchen_reviews": n_kitchen_reviews,
    }

    if food_safety_note:
        result["food_safety_note"] = food_safety_note
    if decline_warning:
        result["decline_warning"] = decline_warning

    return result
