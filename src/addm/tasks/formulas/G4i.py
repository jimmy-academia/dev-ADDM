"""
Ground Truth computation for G4i (Environment Quality - Simple).

Implements the formula from data/tasks/yelp/G4i_prompt.txt.
Simple formula without credibility weighting.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_positive_environment(judgment: Dict[str, Any]) -> bool:
    """
    POSITIVE_ENVIRONMENT = true iff ANY:
      - CLEANLINESS in {spotless, clean}
      - DECOR_STYLE in {stunning, attractive}
      - NOISE_LEVEL in {quiet, pleasant}
      - LIGHTING = perfect
    """
    cleanliness = judgment.get("cleanliness", "not_mentioned")
    decor_style = judgment.get("decor_style", "not_mentioned")
    noise_level = judgment.get("noise_level", "not_mentioned")
    lighting = judgment.get("lighting", "not_mentioned")

    if cleanliness in ("spotless", "clean"):
        return True
    if decor_style in ("stunning", "attractive"):
        return True
    if noise_level in ("quiet", "pleasant"):
        return True
    if lighting == "perfect":
        return True

    return False


def compute_l1_negative_environment(judgment: Dict[str, Any]) -> bool:
    """
    NEGATIVE_ENVIRONMENT = true iff ANY:
      - CLEANLINESS in {dirty, very_dirty}
      - NOISE_LEVEL in {loud, very_loud}
      - SEATING_COMFORT = uncomfortable
      - TEMPERATURE in {too_hot, too_cold}
    """
    cleanliness = judgment.get("cleanliness", "not_mentioned")
    noise_level = judgment.get("noise_level", "not_mentioned")
    seating_comfort = judgment.get("seating_comfort", "not_mentioned")
    temperature = judgment.get("temperature", "not_mentioned")

    if cleanliness in ("dirty", "very_dirty"):
        return True
    if noise_level in ("loud", "very_loud"):
        return True
    if seating_comfort == "uncomfortable":
        return True
    if temperature in ("too_hot", "too_cold"):
        return True

    return False


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute ground truth from extracted judgments."""
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
    n_spotless = 0
    n_dirty = 0
    n_stunning_decor = 0
    n_very_loud = 0

    for j in environment_judgments:
        if compute_l1_positive_environment(j):
            n_positive += 1
        if compute_l1_negative_environment(j):
            n_negative += 1

        if j.get("cleanliness") == "spotless":
            n_spotless += 1
        if j.get("cleanliness") in ("dirty", "very_dirty"):
            n_dirty += 1
        if j.get("decor_style") == "stunning":
            n_stunning_decor += 1
        if j.get("noise_level") == "very_loud":
            n_very_loud += 1

    # Formulas
    positive_score = (n_positive * 1.5) + (n_spotless * 0.5) + (n_stunning_decor * 0.5)
    negative_score = (n_negative * 1.5) + (n_dirty * 2) + (n_very_loud * 1)

    raw_score = BASE_SCORE + positive_score - negative_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
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

    if n_dirty >= 2:
        override_applied = "dirty_pattern"
        verdict = "Poor Environment"
    elif n_very_loud >= 2 and verdict in ("Excellent Environment", "Good Environment"):
        override_applied = "very_loud_max_average"
        verdict = "Average Environment"
    elif n_stunning_decor >= 2 and n_negative == 0 and verdict in ("Average Environment", "Poor Environment"):
        override_applied = "stunning_decor_min_good"
        verdict = "Good Environment"

    return {
        "N_ENVIRONMENT_REVIEWS": n_environment_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "N_SPOTLESS": n_spotless,
        "N_DIRTY": n_dirty,
        "N_STUNNING_DECOR": n_stunning_decor,
        "N_VERY_LOUD": n_very_loud,
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_environment_reviews": n_environment_reviews,
    }
