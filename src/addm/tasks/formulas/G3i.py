"""
Ground Truth computation for G3i (Time-Value - Simple).

Implements the formula from data/tasks/yelp/G3i_prompt.txt.
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


def compute_l1_good_time(judgment: Dict[str, Any]) -> bool:
    """
    GOOD_TIME_EXPERIENCE = true iff ANY:
      - OVERALL_TIME_SENTIMENT in {efficient, acceptable}
      - WAIT_FOR_TABLE in {none, short} AND FOOD_DELIVERY_SPEED in {fast, reasonable}
      - SERVICE_PACING = perfect
    """
    overall_sentiment = judgment.get("overall_time_sentiment", "not_mentioned")
    wait_for_table = judgment.get("wait_for_table", "not_mentioned")
    food_delivery = judgment.get("food_delivery_speed", "not_mentioned")
    service_pacing = judgment.get("service_pacing", "not_mentioned")

    if overall_sentiment in ("efficient", "acceptable"):
        return True

    if wait_for_table in ("none", "short") and food_delivery in ("fast", "reasonable"):
        return True

    if service_pacing == "perfect":
        return True

    return False


def compute_l1_bad_time(judgment: Dict[str, Any]) -> bool:
    """
    BAD_TIME_EXPERIENCE = true iff ANY:
      - OVERALL_TIME_SENTIMENT in {frustrating, unacceptable}
      - WAIT_FOR_TABLE in {long, excessive}
      - RESERVATION_ACCURACY in {significant_delay, lost}
      - SERVICE_PACING in {forgotten, very_rushed}
    """
    overall_sentiment = judgment.get("overall_time_sentiment", "not_mentioned")
    wait_for_table = judgment.get("wait_for_table", "not_mentioned")
    reservation_accuracy = judgment.get("reservation_accuracy", "not_mentioned")
    service_pacing = judgment.get("service_pacing", "not_mentioned")

    if overall_sentiment in ("frustrating", "unacceptable"):
        return True

    if wait_for_table in ("long", "excessive"):
        return True

    if reservation_accuracy in ("significant_delay", "lost"):
        return True

    if service_pacing in ("forgotten", "very_rushed"):
        return True

    return False


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
    time_judgments = [j for j in judgments if j.get("is_time_related", False)]
    n_time_reviews = len(time_judgments)

    if n_time_reviews == 0:
        confidence_level = "none"
    elif n_time_reviews <= 2:
        confidence_level = "low"
    elif n_time_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    n_good = 0
    n_bad = 0
    n_excessive_wait = 0
    n_lost_reservation = 0
    n_forgotten = 0
    n_efficient = 0

    for j in time_judgments:
        if compute_l1_good_time(j):
            n_good += 1
        if compute_l1_bad_time(j):
            n_bad += 1

        if j.get("wait_for_table") == "excessive":
            n_excessive_wait += 1
        if j.get("reservation_accuracy") == "lost":
            n_lost_reservation += 1
        if j.get("service_pacing") == "forgotten":
            n_forgotten += 1
        if j.get("overall_time_sentiment") == "efficient":
            n_efficient += 1

    # Formulas
    positive_score = (n_good * 1.5) + (n_efficient * 1)
    negative_score = (n_bad * 1.5) + (n_excessive_wait * 1) + (n_lost_reservation * 2) + (n_forgotten * 1)

    raw_score = BASE_SCORE + positive_score - negative_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
    if final_score >= 7.5:
        base_verdict_by_score = "Excellent Timing"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good Timing"
    elif final_score >= 3.5:
        base_verdict_by_score = "Variable Timing"
    else:
        base_verdict_by_score = "Poor Timing"

    override_applied = "none"
    verdict = base_verdict_by_score

    if n_lost_reservation >= 1 and verdict in ("Excellent Timing", "Good Timing"):
        override_applied = "lost_reservation_max_variable"
        verdict = "Variable Timing"
    elif n_excessive_wait >= 2:
        override_applied = "excessive_wait_pattern"
        verdict = "Poor Timing"
    elif n_efficient >= 3 and n_bad == 0 and verdict in ("Variable Timing", "Poor Timing"):
        override_applied = "efficient_min_good"
        verdict = "Good Timing"

    return {
        "N_TIME_REVIEWS": n_time_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_GOOD": n_good,
        "N_BAD": n_bad,
        "N_EXCESSIVE_WAIT": n_excessive_wait,
        "N_LOST_RESERVATION": n_lost_reservation,
        "N_FORGOTTEN": n_forgotten,
        "N_EFFICIENT": n_efficient,
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_time_reviews": n_time_reviews,
    }
