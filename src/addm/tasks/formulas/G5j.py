"""
Ground Truth computation for G5j (Consistency - Simple + L1.5).

Implements the formula from data/tasks/yelp/G5j_prompt.txt.
Simple formula with L1.5 consistency-aspect grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

CONSISTENCY_PATTERN_BONUS = {
    "rock_solid": 2.0,
    "reliable_kitchen": 1.5,
    "reliable_service": 1.5,
    "reliable_timing": 1.0,
    "partial_reliability": 0.0,
    "highly_variable": -1.5,
    "inconsistent_experience": -1.0,
}


# =============================================================================
# Helpers
# =============================================================================


def get_consistency_bucket(judgment: Dict[str, Any]) -> str:
    if judgment.get("food_consistency", "not_mentioned") != "not_mentioned":
        return "food"
    elif judgment.get("service_consistency", "not_mentioned") != "not_mentioned":
        return "service"
    elif judgment.get("timing_consistency", "not_mentioned") != "not_mentioned":
        return "timing"
    return "other"


def is_consistent(value: str) -> bool:
    return value in ("always_excellent", "consistent", "improving", "always_prompt")


def is_inconsistent(value: str) -> bool:
    return value in ("variable", "declining")


def compute_l1_positive_consistency(judgment: Dict[str, Any]) -> bool:
    food_consistency = judgment.get("food_consistency", "not_mentioned")
    service_consistency = judgment.get("service_consistency", "not_mentioned")
    overall_change = judgment.get("overall_change", "not_mentioned")

    return (food_consistency in ("always_excellent", "consistent", "improving") or
            service_consistency in ("always_excellent", "consistent", "improving") or
            overall_change == "better_than_before")


def compute_l1_negative_consistency(judgment: Dict[str, Any]) -> bool:
    food_consistency = judgment.get("food_consistency", "not_mentioned")
    service_consistency = judgment.get("service_consistency", "not_mentioned")
    overall_change = judgment.get("overall_change", "not_mentioned")
    recommendation_change = judgment.get("recommendation_change", "not_mentioned")

    return (food_consistency in ("variable", "declining") or
            service_consistency in ("variable", "declining") or
            overall_change == "worse_than_before" or
            recommendation_change in ("wont_return", "lost_customer"))


def compute_l15_buckets(consistency_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "food": {"n_reviews": 0, "n_consistent": 0, "n_inconsistent": 0},
        "service": {"n_reviews": 0, "n_consistent": 0, "n_inconsistent": 0},
        "timing": {"n_reviews": 0, "n_consistent": 0, "n_inconsistent": 0},
    }

    for j in consistency_judgments:
        food_consistency = j.get("food_consistency", "not_mentioned")
        service_consistency = j.get("service_consistency", "not_mentioned")
        timing_consistency = j.get("timing_consistency", "not_mentioned")

        if food_consistency != "not_mentioned":
            buckets["food"]["n_reviews"] += 1
            if is_consistent(food_consistency):
                buckets["food"]["n_consistent"] += 1
            if is_inconsistent(food_consistency):
                buckets["food"]["n_inconsistent"] += 1

        if service_consistency != "not_mentioned":
            buckets["service"]["n_reviews"] += 1
            if is_consistent(service_consistency):
                buckets["service"]["n_consistent"] += 1
            if is_inconsistent(service_consistency):
                buckets["service"]["n_inconsistent"] += 1

        if timing_consistency != "not_mentioned":
            buckets["timing"]["n_reviews"] += 1
            if is_consistent(timing_consistency):
                buckets["timing"]["n_consistent"] += 1
            if is_inconsistent(timing_consistency):
                buckets["timing"]["n_inconsistent"] += 1

    for key, bucket in buckets.items():
        n_consistent = bucket["n_consistent"]
        n_inconsistent = bucket["n_inconsistent"]
        bucket["consistency_rate"] = n_consistent / max(n_consistent + n_inconsistent, 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_consistent"] + v["n_inconsistent"] > 0]

    if buckets_with_data:
        most_consistent_aspect, most_bucket = max(buckets_with_data, key=lambda x: x[1]["consistency_rate"])
        most_consistent_rate = most_bucket["consistency_rate"]
        least_consistent_aspect, _ = min(buckets_with_data, key=lambda x: x[1]["consistency_rate"])
    else:
        most_consistent_aspect = None
        most_consistent_rate = 0.0
        least_consistent_aspect = None

    n_aspects_stable = sum(1 for k, v in buckets.items() if v["consistency_rate"] >= 0.7 and v["n_reviews"] > 0)
    any_highly_variable = any(v["consistency_rate"] < 0.3 and v["n_reviews"] > 0 for v in buckets.values())

    # Determine consistency pattern
    if n_aspects_stable >= 3:
        consistency_pattern = "rock_solid"
    elif buckets["food"]["consistency_rate"] >= 0.8 and buckets["food"]["n_reviews"] > 0:
        consistency_pattern = "reliable_kitchen"
    elif buckets["service"]["consistency_rate"] >= 0.8 and buckets["service"]["n_reviews"] > 0:
        consistency_pattern = "reliable_service"
    elif buckets["timing"]["consistency_rate"] >= 0.8 and buckets["timing"]["n_reviews"] > 0:
        consistency_pattern = "reliable_timing"
    elif n_aspects_stable >= 1:
        consistency_pattern = "partial_reliability"
    elif any_highly_variable:
        consistency_pattern = "highly_variable"
    else:
        consistency_pattern = "inconsistent_experience"

    return {
        "food": buckets["food"],
        "service": buckets["service"],
        "timing": buckets["timing"],
        "most_consistent_aspect": most_consistent_aspect,
        "most_consistent_rate": round(most_consistent_rate, 3),
        "least_consistent_aspect": least_consistent_aspect,
        "n_aspects_stable": n_aspects_stable,
        "consistency_pattern": consistency_pattern,
    }


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    consistency_judgments = [j for j in judgments if j.get("is_consistency_related", False)]
    n_consistency_reviews = len(consistency_judgments)

    if n_consistency_reviews == 0:
        confidence_level = "none"
    elif n_consistency_reviews <= 2:
        confidence_level = "low"
    elif n_consistency_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    n_positive = 0
    n_negative = 0
    n_repeat_visitors = 0

    for j in consistency_judgments:
        if compute_l1_positive_consistency(j):
            n_positive += 1
        if compute_l1_negative_consistency(j):
            n_negative += 1
        if j.get("visit_type") in ("repeat_visit", "regular", "returning_after_gap"):
            n_repeat_visitors += 1

    # L1.5 Aspect Buckets
    l15_buckets = compute_l15_buckets(consistency_judgments)
    consistency_pattern = l15_buckets["consistency_pattern"]
    consistency_pattern_bonus = CONSISTENCY_PATTERN_BONUS.get(consistency_pattern, 0.0)

    # Formulas
    satisfaction_rate = n_positive / max(n_positive + n_negative, 1)
    positive_score = n_positive * 1.5
    negative_score = n_negative * 1.5

    raw_score = BASE_SCORE + positive_score - negative_score + consistency_pattern_bonus
    final_score = max(0.0, min(10.0, raw_score))

    if final_score >= 7.5:
        base_verdict_by_score = "Excellent Consistency"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good Consistency"
    elif final_score >= 3.5:
        base_verdict_by_score = "Variable Consistency"
    else:
        base_verdict_by_score = "Poor Consistency"

    override_applied = "none"
    verdict = base_verdict_by_score
    strength_note = None
    improvement_note = None

    if consistency_pattern == "rock_solid" and verdict in ("Variable Consistency", "Poor Consistency"):
        override_applied = "rock_solid_min_good"
        verdict = "Good Consistency"
    elif consistency_pattern == "highly_variable" and verdict in ("Excellent Consistency", "Good Consistency"):
        override_applied = "highly_variable_max_variable"
        verdict = "Variable Consistency"

    if l15_buckets["most_consistent_aspect"]:
        strength_note = f"Consistent {l15_buckets['most_consistent_aspect']}"

    if l15_buckets["least_consistent_aspect"] and l15_buckets[l15_buckets["least_consistent_aspect"]]["consistency_rate"] < 0.5:
        improvement_note = f"Could improve on {l15_buckets['least_consistent_aspect']} consistency"

    result = {
        "L1_5_aspect_buckets": l15_buckets,
        "N_CONSISTENCY_REVIEWS": n_consistency_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "N_REPEAT_VISITORS": n_repeat_visitors,
        "SATISFACTION_RATE": round(satisfaction_rate, 3),
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "CONSISTENCY_PATTERN_BONUS": consistency_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_consistency_reviews": n_consistency_reviews,
    }

    if strength_note:
        result["strength_note"] = strength_note
    if improvement_note:
        result["improvement_note"] = improvement_note

    return result
