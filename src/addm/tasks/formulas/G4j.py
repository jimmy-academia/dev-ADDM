"""
Ground Truth computation for G4j (Environment Quality - Simple + L1.5).

Implements the formula from data/tasks/yelp/G4j_prompt.txt.
Simple formula with L1.5 environment aspect grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

# L1.5 Environment Pattern Bonus values
ENVIRONMENT_PATTERN_BONUS = {
    "complete_package": 2.0,
    "beautiful_space": 1.5,
    "comfortable_dining": 1.0,
    "spotless_venue": 1.5,
    "partial_strengths": 0.0,
    "needs_attention": -1.0,
}


# =============================================================================
# Helpers
# =============================================================================


def get_environment_bucket(judgment: Dict[str, Any]) -> str:
    """Determine which environment aspect is primary for this review."""
    if judgment.get("decor_style", "not_mentioned") != "not_mentioned" or judgment.get("lighting", "not_mentioned") != "not_mentioned":
        return "ambiance"
    elif judgment.get("seating_comfort", "not_mentioned") != "not_mentioned" or judgment.get("temperature", "not_mentioned") != "not_mentioned" or judgment.get("noise_level", "not_mentioned") != "not_mentioned":
        return "comfort"
    elif judgment.get("cleanliness", "not_mentioned") != "not_mentioned":
        return "cleanliness"
    return "other"


def compute_l1_positive_environment(judgment: Dict[str, Any]) -> bool:
    cleanliness = judgment.get("cleanliness", "not_mentioned")
    decor_style = judgment.get("decor_style", "not_mentioned")
    noise_level = judgment.get("noise_level", "not_mentioned")

    if cleanliness in ("spotless", "clean"):
        return True
    if decor_style in ("stunning", "attractive"):
        return True
    if noise_level in ("quiet", "pleasant"):
        return True
    return False


def compute_l1_negative_environment(judgment: Dict[str, Any]) -> bool:
    cleanliness = judgment.get("cleanliness", "not_mentioned")
    noise_level = judgment.get("noise_level", "not_mentioned")
    seating_comfort = judgment.get("seating_comfort", "not_mentioned")

    if cleanliness in ("dirty", "very_dirty"):
        return True
    if noise_level in ("loud", "very_loud"):
        return True
    if seating_comfort == "uncomfortable":
        return True
    return False


def compute_l15_buckets(environment_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "ambiance": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
        "comfort": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
        "cleanliness": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
    }

    for j in environment_judgments:
        bucket = get_environment_bucket(j)
        if bucket not in buckets:
            continue

        buckets[bucket]["n_reviews"] += 1

        if compute_l1_positive_environment(j):
            buckets[bucket]["n_positive"] += 1
        if compute_l1_negative_environment(j):
            buckets[bucket]["n_negative"] += 1

    for key, bucket in buckets.items():
        n_positive = bucket["n_positive"]
        n_negative = bucket["n_negative"]
        bucket["satisfaction_rate"] = n_positive / max(n_positive + n_negative, 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_positive"] + v["n_negative"] > 0]

    if buckets_with_data:
        best_aspect, best_bucket = max(buckets_with_data, key=lambda x: x[1]["satisfaction_rate"])
        best_satisfaction_rate = best_bucket["satisfaction_rate"]
        worst_aspect, _ = min(buckets_with_data, key=lambda x: x[1]["satisfaction_rate"])
    else:
        best_aspect = None
        best_satisfaction_rate = 0.0
        worst_aspect = None

    n_aspects_strong = sum(1 for k, v in buckets.items() if v["satisfaction_rate"] >= 0.7 and v["n_reviews"] > 0)

    # Determine environment pattern
    if n_aspects_strong >= 3:
        environment_pattern = "complete_package"
    elif buckets["ambiance"]["satisfaction_rate"] >= 0.8 and buckets["ambiance"]["n_reviews"] > 0:
        environment_pattern = "beautiful_space"
    elif buckets["comfort"]["satisfaction_rate"] >= 0.8 and buckets["comfort"]["n_reviews"] > 0:
        environment_pattern = "comfortable_dining"
    elif buckets["cleanliness"]["satisfaction_rate"] >= 0.8 and buckets["cleanliness"]["n_reviews"] > 0:
        environment_pattern = "spotless_venue"
    elif n_aspects_strong >= 1:
        environment_pattern = "partial_strengths"
    else:
        environment_pattern = "needs_attention"

    return {
        "ambiance": buckets["ambiance"],
        "comfort": buckets["comfort"],
        "cleanliness": buckets["cleanliness"],
        "best_aspect": best_aspect,
        "best_aspect_rate": round(best_satisfaction_rate, 3),
        "worst_aspect": worst_aspect,
        "n_aspects_strong": n_aspects_strong,
        "environment_pattern": environment_pattern,
    }


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    environment_judgments = [j for j in judgments if j.get("is_environment_related", False)]
    n_environment_reviews = len(environment_judgments)

    if n_environment_reviews == 0:
        confidence_level = "none"
    elif n_environment_reviews <= 2:
        confidence_level = "low"
    elif n_environment_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    n_positive = 0
    n_negative = 0

    for j in environment_judgments:
        if compute_l1_positive_environment(j):
            n_positive += 1
        if compute_l1_negative_environment(j):
            n_negative += 1

    # L1.5 Environment Buckets
    l15_buckets = compute_l15_buckets(environment_judgments)
    environment_pattern = l15_buckets["environment_pattern"]
    environment_pattern_bonus = ENVIRONMENT_PATTERN_BONUS.get(environment_pattern, 0.0)

    # Formulas
    satisfaction_rate = n_positive / max(n_positive + n_negative, 1)
    positive_score = n_positive * 1.5
    negative_score = n_negative * 1.5

    raw_score = BASE_SCORE + positive_score - negative_score + environment_pattern_bonus
    final_score = max(0.0, min(10.0, raw_score))

    if final_score >= 7.5:
        base_verdict_by_score = "Excellent Environment"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good Environment"
    elif final_score >= 3.5:
        base_verdict_by_score = "Average Environment"
    else:
        base_verdict_by_score = "Poor Environment"

    override_applied = "none"
    verdict = base_verdict_by_score
    strength_note = None
    improvement_note = None

    if environment_pattern == "complete_package" and verdict in ("Average Environment", "Poor Environment"):
        override_applied = "complete_min_good"
        verdict = "Good Environment"
    elif l15_buckets["cleanliness"]["satisfaction_rate"] < 0.5 and l15_buckets["cleanliness"]["n_reviews"] > 0 and verdict in ("Excellent Environment", "Good Environment"):
        override_applied = "cleanliness_low_max_average"
        verdict = "Average Environment"

    # Strength/improvement notes
    if environment_pattern == "beautiful_space":
        strength_note = "Beautiful ambiance"
    elif environment_pattern == "comfortable_dining":
        strength_note = "Comfortable dining experience"
    elif environment_pattern == "spotless_venue":
        strength_note = "Spotless venue"

    if l15_buckets["worst_aspect"] and buckets_rate_below_threshold(l15_buckets, 0.5):
        improvement_note = f"Could improve on {l15_buckets['worst_aspect']}"

    result = {
        "L1_5_environment_buckets": l15_buckets,
        "N_ENVIRONMENT_REVIEWS": n_environment_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "SATISFACTION_RATE": round(satisfaction_rate, 3),
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "ENVIRONMENT_PATTERN_BONUS": environment_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_environment_reviews": n_environment_reviews,
    }

    if strength_note:
        result["strength_note"] = strength_note
    if improvement_note:
        result["improvement_note"] = improvement_note

    return result


def buckets_rate_below_threshold(l15_buckets: Dict[str, Any], threshold: float) -> bool:
    worst = l15_buckets.get("worst_aspect")
    if worst and worst in l15_buckets:
        return l15_buckets[worst].get("satisfaction_rate", 1.0) < threshold
    return False
