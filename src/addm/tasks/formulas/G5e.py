"""
Ground Truth computation for G5e (Order Execution - Simple).

Implements the formula from data/tasks/yelp/G5e_prompt.txt.
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


def compute_l1_positive_execution(judgment: Dict[str, Any]) -> bool:
    order_accuracy = judgment.get("order_accuracy", "not_mentioned")
    special_request = judgment.get("special_request_handling", "not_mentioned")
    timing = judgment.get("timing_coordination", "not_mentioned")
    error_recovery = judgment.get("error_recovery", "not_mentioned")

    if order_accuracy in ("perfect", "correct"):
        return True
    if special_request in ("exceeded", "honored"):
        return True
    if timing in ("perfectly_timed", "well_coordinated"):
        return True
    if error_recovery in ("excellent", "good"):
        return True
    return False


def compute_l1_negative_execution(judgment: Dict[str, Any]) -> bool:
    order_accuracy = judgment.get("order_accuracy", "not_mentioned")
    special_request = judgment.get("special_request_handling", "not_mentioned")
    timing = judgment.get("timing_coordination", "not_mentioned")
    error_recovery = judgment.get("error_recovery", "not_mentioned")

    if order_accuracy in ("major_error", "completely_wrong"):
        return True
    if special_request in ("ignored", "refused"):
        return True
    if timing in ("poorly_timed", "chaotic"):
        return True
    if error_recovery in ("poor", "none"):
        return True
    return False


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    execution_judgments = [j for j in judgments if j.get("is_execution_related", False)]
    n_execution_reviews = len(execution_judgments)

    if n_execution_reviews == 0:
        confidence_level = "none"
    elif n_execution_reviews <= 2:
        confidence_level = "low"
    elif n_execution_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    n_positive = 0
    n_negative = 0
    n_major_error = 0
    n_ignored_request = 0
    n_poor_recovery = 0

    for j in execution_judgments:
        if compute_l1_positive_execution(j):
            n_positive += 1
        if compute_l1_negative_execution(j):
            n_negative += 1

        if j.get("order_accuracy") in ("major_error", "completely_wrong"):
            n_major_error += 1
        if j.get("special_request_handling") in ("ignored", "refused"):
            n_ignored_request += 1
        if j.get("error_recovery") in ("poor", "none"):
            n_poor_recovery += 1

    # Formulas
    positive_score = n_positive * 1.5
    negative_score = (n_negative * 1.5) + (n_major_error * 1.5) + (n_poor_recovery * 1.0)

    raw_score = BASE_SCORE + positive_score - negative_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
    if final_score >= 7.5:
        base_verdict_by_score = "Excellent Execution"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good Execution"
    elif final_score >= 3.5:
        base_verdict_by_score = "Average Execution"
    else:
        base_verdict_by_score = "Poor Execution"

    override_applied = "none"
    verdict = base_verdict_by_score
    recovery_warning = None
    special_request_warning = None

    if n_major_error >= 2 and verdict in ("Excellent Execution", "Good Execution"):
        override_applied = "major_error_max_average"
        verdict = "Average Execution"

    if n_poor_recovery >= 2:
        recovery_warning = "Poor error recovery reported multiple times"
    if n_ignored_request >= 2:
        special_request_warning = "Special requests often ignored"

    result = {
        "N_EXECUTION_REVIEWS": n_execution_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "N_MAJOR_ERROR": n_major_error,
        "N_IGNORED_REQUEST": n_ignored_request,
        "N_POOR_RECOVERY": n_poor_recovery,
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_execution_reviews": n_execution_reviews,
    }

    if recovery_warning:
        result["recovery_warning"] = recovery_warning
    if special_request_warning:
        result["special_request_warning"] = special_request_warning

    return result
