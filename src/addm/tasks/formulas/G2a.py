"""
Ground Truth computation for G2a (Romance - Simple).

Implements the formula from data/tasks/yelp/G2a_prompt.txt.
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


def compute_l1_positive(judgment: Dict[str, Any]) -> bool:
    """
    POSITIVE_ROMANTIC_EXPERIENCE = true iff ALL:
      - OCCASION_TYPE != not_romantic
      - AMBIANCE_RATING in {excellent, good}
      - At least one of: PRIVACY_LEVEL = intimate, NOISE_LEVEL = quiet, ROMANTIC_ELEMENTS = present
    """
    occasion_type = judgment.get("occasion_type", "not_romantic")
    ambiance_rating = judgment.get("ambiance_rating", "not_mentioned")
    privacy_level = judgment.get("privacy_level", "not_mentioned")
    noise_level = judgment.get("noise_level", "not_mentioned")
    romantic_elements = judgment.get("romantic_elements", "not_mentioned")

    if occasion_type == "not_romantic":
        return False

    if ambiance_rating not in ("excellent", "good"):
        return False

    has_positive_feature = (
        privacy_level == "intimate"
        or noise_level == "quiet"
        or romantic_elements == "present"
    )

    return has_positive_feature


def compute_l1_negative(judgment: Dict[str, Any]) -> bool:
    """
    NEGATIVE_ROMANTIC_EXPERIENCE = true iff ALL:
      - OCCASION_TYPE != not_romantic
      - Any of: AMBIANCE_RATING = poor, NOISE_LEVEL = loud, PRIVACY_LEVEL = lacking
    """
    occasion_type = judgment.get("occasion_type", "not_romantic")
    ambiance_rating = judgment.get("ambiance_rating", "not_mentioned")
    privacy_level = judgment.get("privacy_level", "not_mentioned")
    noise_level = judgment.get("noise_level", "not_mentioned")

    if occasion_type == "not_romantic":
        return False

    has_negative = (
        ambiance_rating == "poor"
        or noise_level == "loud"
        or privacy_level == "lacking"
    )

    return has_negative


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
    # Filter to romantic-related judgments only
    romantic_judgments = [j for j in judgments if j.get("is_romantic_related", False)]
    n_romantic_reviews = len(romantic_judgments)

    # CONFIDENCE_LEVEL
    if n_romantic_reviews == 0:
        confidence_level = "none"
    elif n_romantic_reviews <= 2:
        confidence_level = "low"
    elif n_romantic_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    # Compute L1 for each judgment
    n_positive = 0
    n_negative = 0
    n_excellent_ambiance = 0
    n_intimate = 0
    n_quiet = 0
    n_loud = 0

    for j in romantic_judgments:
        if compute_l1_positive(j):
            n_positive += 1
        if compute_l1_negative(j):
            n_negative += 1

        if j.get("ambiance_rating") == "excellent":
            n_excellent_ambiance += 1
        if j.get("privacy_level") == "intimate":
            n_intimate += 1
        if j.get("noise_level") == "quiet":
            n_quiet += 1
        if j.get("noise_level") == "loud":
            n_loud += 1

    # Formulas
    positive_score = (n_positive * 2) + (n_excellent_ambiance * 1) + (n_intimate * 1) + (n_quiet * 0.5)
    negative_score = (n_negative * 2) + (n_loud * 1)

    raw_score = BASE_SCORE + positive_score - negative_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy: Base verdict by score
    if final_score >= 7.5:
        base_verdict_by_score = "Highly Romantic"
    elif final_score >= 5.5:
        base_verdict_by_score = "Romantic"
    elif final_score >= 3.5:
        base_verdict_by_score = "Somewhat Romantic"
    else:
        base_verdict_by_score = "Not Romantic"

    # Decision Policy: Overrides
    override_applied = "none"
    verdict = base_verdict_by_score

    # Override 1: If N_EXCELLENT_AMBIANCE >= 3 AND N_NEGATIVE == 0 => min Romantic
    if n_excellent_ambiance >= 3 and n_negative == 0 and verdict in ("Somewhat Romantic", "Not Romantic"):
        override_applied = "excellent_ambiance_min_romantic"
        verdict = "Romantic"
    # Override 2: If N_LOUD >= 2 => max Somewhat Romantic
    elif n_loud >= 2 and verdict in ("Highly Romantic", "Romantic"):
        override_applied = "loud_max_somewhat"
        verdict = "Somewhat Romantic"
    # Override 3: If N_NEGATIVE >= 3 => Not Romantic
    elif n_negative >= 3:
        override_applied = "many_negative"
        verdict = "Not Romantic"

    return {
        # L2 Aggregates
        "N_ROMANTIC_REVIEWS": n_romantic_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "N_EXCELLENT_AMBIANCE": n_excellent_ambiance,
        "N_INTIMATE": n_intimate,
        "N_QUIET": n_quiet,
        "N_LOUD": n_loud,
        # Formula results
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_romantic_reviews": n_romantic_reviews,
    }
