"""
Ground Truth computation for G4f (Kitchen Quality - Simple + L1.5).

Implements the formula from data/tasks/yelp/G4f_prompt.txt.
Simple formula with L1.5 food aspect grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

# L1.5 Kitchen Pattern Bonus values
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
    """Determine which food aspect is primary for this review."""
    if judgment.get("taste_quality", "not_mentioned") != "not_mentioned" or judgment.get("seasoning", "not_mentioned") != "not_mentioned":
        return "taste"
    elif judgment.get("cooking_execution", "not_mentioned") != "not_mentioned" or judgment.get("temperature", "not_mentioned") != "not_mentioned":
        return "execution"
    elif judgment.get("freshness", "not_mentioned") != "not_mentioned" or judgment.get("presentation", "not_mentioned") != "not_mentioned":
        return "quality"
    return "other"


def compute_l1_positive_kitchen(judgment: Dict[str, Any]) -> bool:
    taste_quality = judgment.get("taste_quality", "not_mentioned")
    cooking_execution = judgment.get("cooking_execution", "not_mentioned")
    presentation = judgment.get("presentation", "not_mentioned")

    if taste_quality in ("exceptional", "good"):
        return True
    if cooking_execution in ("perfect", "good"):
        return True
    if presentation == "stunning":
        return True
    return False


def compute_l1_negative_kitchen(judgment: Dict[str, Any]) -> bool:
    taste_quality = judgment.get("taste_quality", "not_mentioned")
    cooking_execution = judgment.get("cooking_execution", "not_mentioned")
    freshness = judgment.get("freshness", "not_mentioned")

    if taste_quality in ("below_average", "poor"):
        return True
    if cooking_execution in ("overcooked", "undercooked"):
        return True
    if freshness in ("questionable", "stale"):
        return True
    return False


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

    # Determine kitchen pattern
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

    n_positive = 0
    n_negative = 0

    for j in kitchen_judgments:
        if compute_l1_positive_kitchen(j):
            n_positive += 1
        if compute_l1_negative_kitchen(j):
            n_negative += 1

    # L1.5 Kitchen Buckets
    l15_buckets = compute_l15_buckets(kitchen_judgments)
    kitchen_pattern = l15_buckets["kitchen_pattern"]
    kitchen_pattern_bonus = KITCHEN_PATTERN_BONUS.get(kitchen_pattern, 0.0)

    # Formulas
    success_rate = n_positive / max(n_positive + n_negative, 1)
    positive_score = n_positive * 1.5
    negative_score = n_negative * 1.5

    raw_score = BASE_SCORE + positive_score - negative_score + kitchen_pattern_bonus
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
    strength_note = None
    weakness_note = None

    if kitchen_pattern == "all_around_excellent" and verdict in ("Mixed Kitchen", "Poor Kitchen"):
        override_applied = "all_around_min_good"
        verdict = "Good Kitchen"
    elif kitchen_pattern == "needs_work" and n_negative >= 3:
        override_applied = "needs_work_with_negative"
        verdict = "Poor Kitchen"

    # Strength/weakness notes
    if kitchen_pattern == "flavor_forward":
        strength_note = "Excellent flavors"
    elif kitchen_pattern == "technically_skilled":
        strength_note = "Technically skilled kitchen"
    elif kitchen_pattern == "quality_ingredients":
        strength_note = "Quality ingredients"

    if l15_buckets["worst_aspect"] and buckets_rate_below_threshold(l15_buckets, 0.5):
        weakness_note = f"Could improve on {l15_buckets['worst_aspect']}"

    result = {
        "L1_5_kitchen_buckets": l15_buckets,
        "N_KITCHEN_REVIEWS": n_kitchen_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "SUCCESS_RATE": round(success_rate, 3),
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "KITCHEN_PATTERN_BONUS": kitchen_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_kitchen_reviews": n_kitchen_reviews,
    }

    if strength_note:
        result["strength_note"] = strength_note
    if weakness_note:
        result["weakness_note"] = weakness_note

    return result


def buckets_rate_below_threshold(l15_buckets: Dict[str, Any], threshold: float) -> bool:
    worst = l15_buckets.get("worst_aspect")
    if worst and worst in l15_buckets:
        return l15_buckets[worst].get("success_rate", 1.0) < threshold
    return False
