"""
Ground Truth computation for G5g (Order Execution - Complex).

Implements the formula from data/tasks/yelp/G5g_prompt.txt.
Complex formula with weighted execution factors.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

ACCURACY_SCORES = {"perfect": 4.0, "correct": 2.0, "minor_error": -0.5, "major_error": -3.0, "completely_wrong": -5.0, "not_mentioned": 0}
REQUEST_SCORES = {"exceeded": 3.0, "honored": 1.5, "partial": -0.5, "ignored": -2.5, "refused": -3.0, "not_mentioned": 0}
TIMING_SCORES = {"perfectly_timed": 2.5, "well_coordinated": 1.5, "acceptable": 0, "poorly_timed": -2.0, "chaotic": -4.0, "not_mentioned": 0}
COMMUNICATION_SCORES = {"excellent": 2.0, "good": 1.0, "unclear": -1.0, "poor": -2.5, "not_mentioned": 0}
RECOVERY_SCORES = {"excellent": 2.5, "good": 1.5, "slow": 0, "poor": -2.0, "none": -3.0, "not_applicable": 0}
MODIFICATION_SCORES = {"easy": 1.5, "accommodated": 0.5, "difficult": -1.0, "refused": -2.0, "not_mentioned": 0}


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    order_accuracy = judgment.get("order_accuracy", "not_mentioned")
    special_request = judgment.get("special_request_handling", "not_mentioned")
    timing = judgment.get("timing_coordination", "not_mentioned")
    communication = judgment.get("communication_clarity", "not_mentioned")
    error_recovery = judgment.get("error_recovery", "not_mentioned")
    modification = judgment.get("order_modification", "not_mentioned")

    accuracy_score = ACCURACY_SCORES.get(order_accuracy, 0)
    request_score = REQUEST_SCORES.get(special_request, 0)
    timing_score = TIMING_SCORES.get(timing, 0)
    communication_score = COMMUNICATION_SCORES.get(communication, 0)
    recovery_score = RECOVERY_SCORES.get(error_recovery, 0)
    modification_score = MODIFICATION_SCORES.get(modification, 0)

    l1_execution_score = (
        accuracy_score + request_score + timing_score +
        communication_score + recovery_score + modification_score
    )

    # Interaction effects
    complete_failure_penalty = -3.0 if (order_accuracy == "completely_wrong" and error_recovery in ("poor", "none")) else 0.0
    flawless_execution_bonus = 2.0 if (order_accuracy == "perfect" and timing == "perfectly_timed" and special_request in ("exceeded", "honored")) else 0.0

    l1_total_score = l1_execution_score + complete_failure_penalty + flawless_execution_bonus

    return {
        "accuracy_score": accuracy_score,
        "request_score": request_score,
        "timing_score": timing_score,
        "communication_score": communication_score,
        "recovery_score": recovery_score,
        "modification_score": modification_score,
        "l1_execution_score": round(l1_execution_score, 2),
        "complete_failure_penalty": complete_failure_penalty,
        "flawless_execution_bonus": flawless_execution_bonus,
        "l1_total_score": round(l1_total_score, 2),
    }


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

    sum_l1_score = 0.0
    n_major_error = 0
    n_flawless = 0
    n_poor_recovery = 0

    for j in execution_judgments:
        l1 = compute_l1_complex(j)
        sum_l1_score += l1["l1_total_score"]

        if j.get("order_accuracy") in ("major_error", "completely_wrong"):
            n_major_error += 1
        if l1["flawless_execution_bonus"] > 0:
            n_flawless += 1
        if j.get("error_recovery") in ("poor", "none"):
            n_poor_recovery += 1

    mean_l1_score = sum_l1_score / max(n_execution_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score
    final_score = max(0.0, min(10.0, raw_score))

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

    if n_major_error >= 2 and verdict in ("Excellent Execution", "Good Execution"):
        override_applied = "major_error_max_average"
        verdict = "Average Execution"
    elif n_flawless >= 2 and n_major_error == 0 and verdict in ("Average Execution", "Poor Execution"):
        override_applied = "flawless_min_good"
        verdict = "Good Execution"
    elif n_poor_recovery >= 2:
        recovery_warning = "Poor error recovery reported multiple times"
    elif mean_l1_score < -3:
        override_applied = "low_mean_score"
        verdict = "Poor Execution"

    result = {
        "N_EXECUTION_REVIEWS": n_execution_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_MAJOR_ERROR": n_major_error,
        "N_FLAWLESS": n_flawless,
        "N_POOR_RECOVERY": n_poor_recovery,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_execution_reviews": n_execution_reviews,
    }

    if recovery_warning:
        result["recovery_warning"] = recovery_warning

    return result
