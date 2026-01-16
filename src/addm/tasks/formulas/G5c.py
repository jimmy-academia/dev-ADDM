"""
Ground Truth computation for G5c (Capacity Management - Complex).

Implements the formula from data/tasks/yelp/G5c_prompt.txt.
Complex formula with weighted capacity factors.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

CROWD_SCORES = {"empty": 1.0, "comfortable": 2.0, "busy": 0.5, "packed": -1.5, "overwhelming": -3.0, "not_mentioned": 0}
WAIT_SCORES = {"none": 2.5, "short": 1.5, "moderate": 0, "long": -2.0, "excessive": -4.0, "not_mentioned": 0}
TABLE_SCORES = {"abundant": 1.5, "adequate": 0.5, "limited": -1.0, "none": -2.0, "not_mentioned": 0}
RESERVATION_SCORES = {"excellent": 2.0, "good": 1.0, "problematic": -2.0, "failed": -4.0, "not_mentioned": 0}
STAFF_SCORES = {"excellent": 2.5, "adequate": 1.0, "stretched": -1.5, "insufficient": -3.5, "not_mentioned": 0}
PEAK_SCORES = {"excellent": 2.0, "good": 1.0, "struggling": -1.5, "failing": -3.0, "not_mentioned": 0}
SPACE_SCORES = {"spacious": 1.5, "comfortable": 0.5, "tight": -0.5, "cramped": -1.5, "not_mentioned": 0}


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    crowd_level = judgment.get("crowd_level", "not_mentioned")
    wait_time = judgment.get("wait_time", "not_mentioned")
    table_availability = judgment.get("table_availability", "not_mentioned")
    reservation_system = judgment.get("reservation_system", "not_mentioned")
    staff_coverage = judgment.get("staff_coverage", "not_mentioned")
    peak_handling = judgment.get("peak_handling", "not_mentioned")
    space_comfort = judgment.get("space_comfort", "not_mentioned")

    crowd_score = CROWD_SCORES.get(crowd_level, 0)
    wait_score = WAIT_SCORES.get(wait_time, 0)
    table_score = TABLE_SCORES.get(table_availability, 0)
    reservation_score = RESERVATION_SCORES.get(reservation_system, 0)
    staff_score = STAFF_SCORES.get(staff_coverage, 0)
    peak_score = PEAK_SCORES.get(peak_handling, 0)
    space_score = SPACE_SCORES.get(space_comfort, 0)

    l1_capacity_score = (
        crowd_score + wait_score + table_score + reservation_score +
        staff_score + peak_score + space_score
    )

    # Interaction effects
    chaos_penalty = -2.0 if (crowd_level in ("packed", "overwhelming") and staff_coverage in ("stretched", "insufficient")) else 0.0
    smooth_operation_bonus = 2.0 if (wait_time in ("none", "short") and staff_coverage == "excellent" and peak_handling in ("excellent", "good")) else 0.0

    l1_total_score = l1_capacity_score + chaos_penalty + smooth_operation_bonus

    return {
        "crowd_score": crowd_score,
        "wait_score": wait_score,
        "table_score": table_score,
        "reservation_score": reservation_score,
        "staff_score": staff_score,
        "peak_score": peak_score,
        "space_score": space_score,
        "l1_capacity_score": round(l1_capacity_score, 2),
        "chaos_penalty": chaos_penalty,
        "smooth_operation_bonus": smooth_operation_bonus,
        "l1_total_score": round(l1_total_score, 2),
    }


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

    sum_l1_score = 0.0
    n_excessive_wait = 0
    n_understaffed = 0
    n_reservation_failed = 0
    n_smooth_operation = 0

    for j in capacity_judgments:
        l1 = compute_l1_complex(j)
        sum_l1_score += l1["l1_total_score"]

        if j.get("wait_time") == "excessive":
            n_excessive_wait += 1
        if j.get("staff_coverage") == "insufficient":
            n_understaffed += 1
        if j.get("reservation_system") == "failed":
            n_reservation_failed += 1
        if l1["smooth_operation_bonus"] > 0:
            n_smooth_operation += 1

    mean_l1_score = sum_l1_score / max(n_capacity_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score
    final_score = max(0.0, min(10.0, raw_score))

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
    elif n_reservation_failed >= 1:
        reservation_warning = "Reservation system has failed customers"
    elif n_smooth_operation >= 2 and mean_l1_score >= 3 and verdict in ("Average Capacity Management", "Poor Capacity Management"):
        override_applied = "smooth_min_good"
        verdict = "Good Capacity Management"
    elif mean_l1_score < -3:
        override_applied = "low_mean_score"
        verdict = "Poor Capacity Management"

    result = {
        "N_CAPACITY_REVIEWS": n_capacity_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_EXCESSIVE_WAIT": n_excessive_wait,
        "N_UNDERSTAFFED": n_understaffed,
        "N_RESERVATION_FAILED": n_reservation_failed,
        "N_SMOOTH_OPERATION": n_smooth_operation,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
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
