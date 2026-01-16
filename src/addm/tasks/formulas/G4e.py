"""
Ground Truth computation for G4e (Kitchen Quality - Simple).

Implements the formula from data/tasks/yelp/G4e_prompt.txt.
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


def compute_l1_positive_kitchen(judgment: Dict[str, Any]) -> bool:
    """
    POSITIVE_KITCHEN = true iff ANY:
      - TASTE_QUALITY in {exceptional, good}
      - FRESHNESS in {very_fresh, fresh}
      - COOKING_EXECUTION in {perfect, good}
      - PRESENTATION = stunning
    """
    taste_quality = judgment.get("taste_quality", "not_mentioned")
    freshness = judgment.get("freshness", "not_mentioned")
    cooking_execution = judgment.get("cooking_execution", "not_mentioned")
    presentation = judgment.get("presentation", "not_mentioned")

    if taste_quality in ("exceptional", "good"):
        return True
    if freshness in ("very_fresh", "fresh"):
        return True
    if cooking_execution in ("perfect", "good"):
        return True
    if presentation == "stunning":
        return True

    return False


def compute_l1_negative_kitchen(judgment: Dict[str, Any]) -> bool:
    """
    NEGATIVE_KITCHEN = true iff ANY:
      - TASTE_QUALITY in {below_average, poor}
      - FRESHNESS in {questionable, stale}
      - COOKING_EXECUTION in {overcooked, undercooked}
      - TEMPERATURE = cold
    """
    taste_quality = judgment.get("taste_quality", "not_mentioned")
    freshness = judgment.get("freshness", "not_mentioned")
    cooking_execution = judgment.get("cooking_execution", "not_mentioned")
    temperature = judgment.get("temperature", "not_mentioned")

    if taste_quality in ("below_average", "poor"):
        return True
    if freshness in ("questionable", "stale"):
        return True
    if cooking_execution in ("overcooked", "undercooked"):
        return True
    if temperature == "cold":
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
    n_exceptional_taste = 0
    n_poor_taste = 0
    n_stale = 0
    n_undercooked = 0

    for j in kitchen_judgments:
        if compute_l1_positive_kitchen(j):
            n_positive += 1
        if compute_l1_negative_kitchen(j):
            n_negative += 1

        if j.get("taste_quality") == "exceptional":
            n_exceptional_taste += 1
        if j.get("taste_quality") == "poor":
            n_poor_taste += 1
        if j.get("freshness") == "stale":
            n_stale += 1
        if j.get("cooking_execution") == "undercooked":
            n_undercooked += 1

    # Formulas
    positive_score = (n_positive * 1.5) + (n_exceptional_taste * 1)
    negative_score = (n_negative * 1.5) + (n_poor_taste * 1) + (n_stale * 1.5) + (n_undercooked * 1)

    raw_score = BASE_SCORE + positive_score - negative_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
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

    if n_undercooked >= 2 and verdict in ("Excellent Kitchen", "Good Kitchen"):
        override_applied = "undercooked_max_mixed"
        verdict = "Mixed Kitchen"
    elif n_stale >= 2 and verdict in ("Excellent Kitchen", "Good Kitchen"):
        override_applied = "stale_max_mixed"
        verdict = "Mixed Kitchen"
    elif n_exceptional_taste >= 3 and n_negative == 0 and verdict in ("Mixed Kitchen", "Poor Kitchen"):
        override_applied = "exceptional_taste_min_good"
        verdict = "Good Kitchen"

    return {
        "N_KITCHEN_REVIEWS": n_kitchen_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "N_EXCEPTIONAL_TASTE": n_exceptional_taste,
        "N_POOR_TASTE": n_poor_taste,
        "N_STALE": n_stale,
        "N_UNDERCOOKED": n_undercooked,
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_kitchen_reviews": n_kitchen_reviews,
    }
