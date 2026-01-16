"""
Ground Truth computation for G4l (Environment Quality - Complex + L1.5).

Implements the formula from data/tasks/yelp/G4l_prompt.txt.
Complex formula with weighted environment factors + L1.5 environment aspect grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

CLEANLINESS_SCORES = {"spotless": 3.0, "clean": 1.5, "adequate": 0, "dirty": -3.0, "very_dirty": -5.0, "not_mentioned": 0}
DECOR_SCORES = {"stunning": 2.5, "attractive": 1.5, "average": 0, "dated": -1.0, "unappealing": -2.0, "not_mentioned": 0}
NOISE_SCORES = {"quiet": 2.0, "pleasant": 1.0, "moderate": 0, "loud": -1.5, "very_loud": -3.0, "not_mentioned": 0}
LIGHTING_SCORES = {"perfect": 1.5, "good": 0.5, "adequate": 0, "too_dark": -1.0, "too_bright": -1.0, "not_mentioned": 0}
COMFORT_SCORES = {"very_comfortable": 2.0, "comfortable": 1.0, "adequate": 0, "uncomfortable": -2.0, "not_mentioned": 0}
TEMPERATURE_SCORES = {"perfect": 1.0, "comfortable": 0.5, "too_hot": -1.5, "too_cold": -1.5, "not_mentioned": 0}
BATHROOM_MODIFIERS = {"excellent": 1.0, "acceptable": 0, "poor": -2.0, "not_mentioned": 0}
OUTDOOR_MODIFIERS = {"excellent": 1.5, "available": 0.5, "limited": 0, "none": 0, "not_mentioned": 0}

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
    if judgment.get("decor_style", "not_mentioned") != "not_mentioned" or judgment.get("lighting", "not_mentioned") != "not_mentioned":
        return "ambiance"
    elif judgment.get("seating_comfort", "not_mentioned") != "not_mentioned" or judgment.get("temperature", "not_mentioned") != "not_mentioned" or judgment.get("noise_level", "not_mentioned") != "not_mentioned":
        return "comfort"
    elif judgment.get("cleanliness", "not_mentioned") != "not_mentioned":
        return "cleanliness"
    return "other"


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    cleanliness = judgment.get("cleanliness", "not_mentioned")
    decor_style = judgment.get("decor_style", "not_mentioned")
    noise_level = judgment.get("noise_level", "not_mentioned")
    lighting = judgment.get("lighting", "not_mentioned")
    seating_comfort = judgment.get("seating_comfort", "not_mentioned")
    temperature = judgment.get("temperature", "not_mentioned")
    bathroom_condition = judgment.get("bathroom_condition", "not_mentioned")
    outdoor_space = judgment.get("outdoor_space", "not_mentioned")

    cleanliness_score = CLEANLINESS_SCORES.get(cleanliness, 0)
    decor_score = DECOR_SCORES.get(decor_style, 0)
    noise_score = NOISE_SCORES.get(noise_level, 0)
    lighting_score = LIGHTING_SCORES.get(lighting, 0)
    comfort_score = COMFORT_SCORES.get(seating_comfort, 0)
    temperature_score = TEMPERATURE_SCORES.get(temperature, 0)
    bathroom_modifier = BATHROOM_MODIFIERS.get(bathroom_condition, 0)
    outdoor_modifier = OUTDOOR_MODIFIERS.get(outdoor_space, 0)

    l1_environment_score = (
        cleanliness_score + decor_score + noise_score + lighting_score +
        comfort_score + temperature_score + bathroom_modifier + outdoor_modifier
    )

    dirty_and_loud = -2.0 if (cleanliness in ("dirty", "very_dirty") and noise_level in ("loud", "very_loud")) else 0.0
    perfect_ambiance = 2.0 if (decor_style == "stunning" and lighting == "perfect" and noise_level in ("quiet", "pleasant")) else 0.0

    l1_total_score = l1_environment_score + dirty_and_loud + perfect_ambiance

    return {
        "cleanliness_score": cleanliness_score,
        "decor_score": decor_score,
        "l1_environment_score": round(l1_environment_score, 2),
        "dirty_and_loud": dirty_and_loud,
        "perfect_ambiance": perfect_ambiance,
        "l1_total_score": round(l1_total_score, 2),
    }


def compute_l1_positive_environment(judgment: Dict[str, Any]) -> bool:
    cleanliness = judgment.get("cleanliness", "not_mentioned")
    decor_style = judgment.get("decor_style", "not_mentioned")
    noise_level = judgment.get("noise_level", "not_mentioned")
    return cleanliness in ("spotless", "clean") or decor_style in ("stunning", "attractive") or noise_level in ("quiet", "pleasant")


def compute_l1_negative_environment(judgment: Dict[str, Any]) -> bool:
    cleanliness = judgment.get("cleanliness", "not_mentioned")
    noise_level = judgment.get("noise_level", "not_mentioned")
    seating_comfort = judgment.get("seating_comfort", "not_mentioned")
    return cleanliness in ("dirty", "very_dirty") or noise_level in ("loud", "very_loud") or seating_comfort == "uncomfortable"


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

    sum_l1_score = 0.0
    n_dirty = 0
    n_very_dirty = 0
    n_stunning = 0
    n_poor_bathroom = 0

    for j in environment_judgments:
        l1 = compute_l1_complex(j)
        sum_l1_score += l1["l1_total_score"]

        if j.get("cleanliness") == "very_dirty":
            n_very_dirty += 1
        if j.get("cleanliness") in ("dirty", "very_dirty"):
            n_dirty += 1
        if j.get("decor_style") == "stunning":
            n_stunning += 1
        if j.get("bathroom_condition") == "poor":
            n_poor_bathroom += 1

    # L1.5 Environment Buckets
    l15_buckets = compute_l15_buckets(environment_judgments)
    environment_pattern = l15_buckets["environment_pattern"]
    environment_pattern_bonus = ENVIRONMENT_PATTERN_BONUS.get(environment_pattern, 0.0)

    mean_l1_score = sum_l1_score / max(n_environment_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score + environment_pattern_bonus
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
    bathroom_warning = None

    if environment_pattern == "complete_package" and verdict in ("Average Environment", "Poor Environment"):
        override_applied = "complete_min_good"
        verdict = "Good Environment"
    elif environment_pattern == "needs_attention" and (n_dirty >= 2 or n_very_dirty >= 1):
        override_applied = "needs_attention_with_dirty"
        verdict = "Poor Environment"
    elif l15_buckets["cleanliness"]["satisfaction_rate"] < 0.5 and l15_buckets["cleanliness"]["n_reviews"] > 0 and verdict in ("Excellent Environment", "Good Environment"):
        override_applied = "cleanliness_low_max_average"
        verdict = "Average Environment"
    elif n_poor_bathroom >= 2:
        bathroom_warning = "Multiple reports of poor bathroom conditions"
    elif mean_l1_score < -2:
        override_applied = "low_mean_score"
        verdict = "Poor Environment"

    result = {
        "L1_5_environment_buckets": l15_buckets,
        "N_ENVIRONMENT_REVIEWS": n_environment_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_DIRTY": n_dirty,
        "N_VERY_DIRTY": n_very_dirty,
        "N_STUNNING": n_stunning,
        "N_POOR_BATHROOM": n_poor_bathroom,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "ENVIRONMENT_PATTERN_BONUS": environment_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_environment_reviews": n_environment_reviews,
    }

    if bathroom_warning:
        result["bathroom_warning"] = bathroom_warning

    return result
