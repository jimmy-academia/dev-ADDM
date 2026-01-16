"""
Ground Truth computation for G5a (Capacity Management - Simple).

Implements the formula from data/tasks/yelp/G5a_prompt.txt.
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


def compute_l1_positive_capacity(judgment: Dict[str, Any]) -> bool:
    crowd_level = judgment.get("crowd_level", "not_mentioned")
    wait_time = judgment.get("wait_time", "not_mentioned")
    staff_coverage = judgment.get("staff_coverage", "not_mentioned")
    reservation_system = judgment.get("reservation_system", "not_mentioned")

    if crowd_level in ("empty", "comfortable"):
        return True
    if wait_time in ("none", "short"):
        return True
    if staff_coverage in ("excellent", "adequate"):
        return True
    if reservation_system in ("excellent", "good"):
        return True
    return False


def compute_l1_negative_capacity(judgment: Dict[str, Any]) -> bool:
    crowd_level = judgment.get("crowd_level", "not_mentioned")
    wait_time = judgment.get("wait_time", "not_mentioned")
    staff_coverage = judgment.get("staff_coverage", "not_mentioned")
    reservation_system = judgment.get("reservation_system", "not_mentioned")

    if crowd_level in ("packed", "overwhelming"):
        return True
    if wait_time in ("long", "excessive"):
        return True
    if staff_coverage in ("stretched", "insufficient"):
        return True
    if reservation_system in ("problematic", "failed"):
        return True
    return False


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    capacity_judgments = [j for j in judgments if j.get("is_capacity_related", False)]
    n_capacity_reviews = len(capacity_judgments)

    if n_capacity_reviews == 0:
        confidence_level = "none"
    elif n_capacity_reviews <= 2:
        confidence_level = "low"
    elif n_capacity_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    n_positive = 0
    n_negative = 0
    n_excessive_wait = 0
    n_understaffed = 0
    n_reservation_failed = 0

    for j in capacity_judgments:
        if compute_l1_positive_capacity(j):
            n_positive += 1
        if compute_l1_negative_capacity(j):
            n_negative += 1

        if j.get("wait_time") == "excessive":
            n_excessive_wait += 1
        if j.get("staff_coverage") == "insufficient":
            n_understaffed += 1
        if j.get("reservation_system") == "failed":
            n_reservation_failed += 1

    # Formulas
    positive_score = n_positive * 1.5
    negative_score = (n_negative * 1.5) + (n_excessive_wait * 1.0) + (n_understaffed * 1.0)

    raw_score = BASE_SCORE + positive_score - negative_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
    if final_score >= 7.5:
        base_verdict_by_score = "Excellent Capacity Management"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good Capacity Management"
    elif final_score >= 3.5:
        base_verdict_by_score = "Average Capacity Management"
    else:
        base_verdict_by_score = "Poor Capacity Management"

    override_applied = "none"
    verdict = base_verdict_by_score
    reservation_warning = None

    if n_excessive_wait >= 2 and verdict in ("Excellent Capacity Management", "Good Capacity Management"):
        override_applied = "excessive_wait_max_average"
        verdict = "Average Capacity Management"
    elif n_understaffed >= 2 and verdict in ("Excellent Capacity Management", "Good Capacity Management"):
        override_applied = "understaffed_max_average"
        verdict = "Average Capacity Management"

    if n_reservation_failed >= 1:
        reservation_warning = "Reservation system has failed customers"

    result = {
        "N_CAPACITY_REVIEWS": n_capacity_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "N_EXCESSIVE_WAIT": n_excessive_wait,
        "N_UNDERSTAFFED": n_understaffed,
        "N_RESERVATION_FAILED": n_reservation_failed,
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_capacity_reviews": n_capacity_reviews,
    }

    if reservation_warning:
        result["reservation_warning"] = reservation_warning

    return result
