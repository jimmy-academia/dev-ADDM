"""
Ground Truth computation for G5h (Order Execution - Complex + L1.5).

Implements the formula from data/tasks/yelp/G5h_prompt.txt.
Complex formula with weighted execution factors + L1.5 error-type grouping.

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

EXECUTION_PATTERN_BONUS = {
    "execution_excellence": 2.0,
    "recovers_well": 1.5,
    "timing_challenged": -0.5,
    "communication_gaps": -0.5,
    "systemic_issues": -1.5,
    "isolated_weakness": -0.5,
    "needs_improvement": -1.0,
}


# =============================================================================
# Helpers
# =============================================================================


def get_error_bucket(judgment: Dict[str, Any]) -> str:
    if judgment.get("order_accuracy", "not_mentioned") in ("minor_error", "major_error", "completely_wrong"):
        return "order_errors"
    elif judgment.get("timing_coordination", "not_mentioned") in ("poorly_timed", "chaotic"):
        return "timing_errors"
    elif (judgment.get("communication_clarity", "not_mentioned") in ("unclear", "poor") or
          judgment.get("special_request_handling", "not_mentioned") in ("ignored", "refused")):
        return "communication_errors"
    return "other"


def is_error_resolved(judgment: Dict[str, Any]) -> bool:
    return judgment.get("error_recovery", "not_mentioned") in ("excellent", "good")


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

    complete_failure_penalty = -3.0 if (order_accuracy == "completely_wrong" and error_recovery in ("poor", "none")) else 0.0
    flawless_execution_bonus = 2.0 if (order_accuracy == "perfect" and timing == "perfectly_timed" and special_request in ("exceeded", "honored")) else 0.0

    l1_total_score = l1_execution_score + complete_failure_penalty + flawless_execution_bonus

    return {
        "l1_execution_score": round(l1_execution_score, 2),
        "complete_failure_penalty": complete_failure_penalty,
        "flawless_execution_bonus": flawless_execution_bonus,
        "l1_total_score": round(l1_total_score, 2),
    }


def compute_l15_buckets(execution_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "order_errors": {"n_reviews": 0, "n_resolved": 0, "n_unresolved": 0},
        "timing_errors": {"n_reviews": 0, "n_resolved": 0, "n_unresolved": 0},
        "communication_errors": {"n_reviews": 0, "n_resolved": 0, "n_unresolved": 0},
    }

    total_errors = 0
    for j in execution_judgments:
        bucket = get_error_bucket(j)
        if bucket not in buckets:
            continue

        if bucket != "other":
            buckets[bucket]["n_reviews"] += 1
            total_errors += 1

            if is_error_resolved(j):
                buckets[bucket]["n_resolved"] += 1
            else:
                buckets[bucket]["n_unresolved"] += 1

    for key, bucket in buckets.items():
        bucket["resolution_rate"] = bucket["n_resolved"] / max(bucket["n_reviews"], 1)

    buckets_with_errors = [(k, v) for k, v in buckets.items() if v["n_unresolved"] > 0]
    if buckets_with_errors:
        worst_error_type, _ = max(buckets_with_errors, key=lambda x: x[1]["n_unresolved"])
    else:
        worst_error_type = None

    buckets_with_resolved = [(k, v) for k, v in buckets.items() if v["n_resolved"] > 0]
    if buckets_with_resolved:
        best_resolution, _ = max(buckets_with_resolved, key=lambda x: x[1]["resolution_rate"])
    else:
        best_resolution = None

    n_error_types_problematic = sum(1 for k, v in buckets.items() if v["resolution_rate"] < 0.5 and v["n_reviews"] > 0)

    if n_error_types_problematic == 0 and total_errors < 2:
        execution_pattern = "execution_excellence"
    elif buckets["order_errors"]["resolution_rate"] >= 0.8 and buckets["order_errors"]["n_reviews"] > 0:
        execution_pattern = "recovers_well"
    elif buckets["timing_errors"]["n_reviews"] > 3:
        execution_pattern = "timing_challenged"
    elif buckets["communication_errors"]["n_reviews"] > 3:
        execution_pattern = "communication_gaps"
    elif n_error_types_problematic >= 2:
        execution_pattern = "systemic_issues"
    elif n_error_types_problematic == 1:
        execution_pattern = "isolated_weakness"
    else:
        execution_pattern = "needs_improvement"

    return {
        "order_errors": buckets["order_errors"],
        "timing_errors": buckets["timing_errors"],
        "communication_errors": buckets["communication_errors"],
        "worst_error_type": worst_error_type,
        "best_resolution": best_resolution,
        "n_error_types_problematic": n_error_types_problematic,
        "execution_pattern": execution_pattern,
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

    for j in execution_judgments:
        l1 = compute_l1_complex(j)
        sum_l1_score += l1["l1_total_score"]

        if j.get("order_accuracy") in ("major_error", "completely_wrong"):
            n_major_error += 1
        if l1["flawless_execution_bonus"] > 0:
            n_flawless += 1

    # L1.5 Error Buckets
    l15_buckets = compute_l15_buckets(execution_judgments)
    execution_pattern = l15_buckets["execution_pattern"]
    execution_pattern_bonus = EXECUTION_PATTERN_BONUS.get(execution_pattern, 0.0)

    mean_l1_score = sum_l1_score / max(n_execution_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score + execution_pattern_bonus
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

    if execution_pattern == "execution_excellence" and verdict in ("Average Execution", "Poor Execution"):
        override_applied = "excellence_min_good"
        verdict = "Good Execution"
    elif execution_pattern == "systemic_issues" and n_major_error >= 2:
        override_applied = "systemic_with_errors"
        verdict = "Poor Execution"
    elif mean_l1_score < -3:
        override_applied = "low_mean_score"
        verdict = "Poor Execution"

    result = {
        "L1_5_error_buckets": l15_buckets,
        "N_EXECUTION_REVIEWS": n_execution_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_MAJOR_ERROR": n_major_error,
        "N_FLAWLESS": n_flawless,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "EXECUTION_PATTERN_BONUS": execution_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_execution_reviews": n_execution_reviews,
    }

    return result
