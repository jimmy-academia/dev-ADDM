"""
Ground Truth computation for G4k (Environment Quality - Complex).

Implements the formula from data/tasks/yelp/G4k_prompt.txt.
Complex formula with weighted environment factors.

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


# =============================================================================
# Helpers
# =============================================================================


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

    # Interaction effects
    dirty_and_loud = -2.0 if (cleanliness in ("dirty", "very_dirty") and noise_level in ("loud", "very_loud")) else 0.0
    perfect_ambiance = 2.0 if (decor_style == "stunning" and lighting == "perfect" and noise_level in ("quiet", "pleasant")) else 0.0

    l1_total_score = l1_environment_score + dirty_and_loud + perfect_ambiance

    return {
        "cleanliness_score": cleanliness_score,
        "decor_score": decor_score,
        "noise_score": noise_score,
        "lighting_score": lighting_score,
        "comfort_score": comfort_score,
        "temperature_score": temperature_score,
        "bathroom_modifier": bathroom_modifier,
        "outdoor_modifier": outdoor_modifier,
        "l1_environment_score": round(l1_environment_score, 2),
        "dirty_and_loud": dirty_and_loud,
        "perfect_ambiance": perfect_ambiance,
        "l1_total_score": round(l1_total_score, 2),
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
    n_very_dirty = 0
    n_dirty = 0
    n_very_loud = 0
    n_stunning = 0
    n_poor_bathroom = 0

    for j in environment_judgments:
        l1 = compute_l1_complex(j)
        sum_l1_score += l1["l1_total_score"]

        if j.get("cleanliness") == "very_dirty":
            n_very_dirty += 1
        if j.get("cleanliness") in ("dirty", "very_dirty"):
            n_dirty += 1
        if j.get("noise_level") == "very_loud":
            n_very_loud += 1
        if j.get("decor_style") == "stunning":
            n_stunning += 1
        if j.get("bathroom_condition") == "poor":
            n_poor_bathroom += 1

    mean_l1_score = sum_l1_score / max(n_environment_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score
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

    if n_very_dirty >= 1 and verdict in ("Excellent Environment", "Good Environment"):
        override_applied = "very_dirty_max_average"
        verdict = "Average Environment"
    elif n_dirty >= 2:
        override_applied = "dirty_pattern"
        verdict = "Poor Environment"
    elif n_poor_bathroom >= 2:
        bathroom_warning = "Multiple reports of poor bathroom conditions"
    elif mean_l1_score >= 4 and n_stunning >= 2 and verdict in ("Average Environment", "Poor Environment"):
        override_applied = "stunning_min_good"
        verdict = "Good Environment"
    elif mean_l1_score < -2:
        override_applied = "low_mean_score"
        verdict = "Poor Environment"

    result = {
        "N_ENVIRONMENT_REVIEWS": n_environment_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_VERY_DIRTY": n_very_dirty,
        "N_DIRTY": n_dirty,
        "N_VERY_LOUD": n_very_loud,
        "N_STUNNING": n_stunning,
        "N_POOR_BATHROOM": n_poor_bathroom,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
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
