"""
Ground Truth computation for G5f (Order Execution - Simple + L1.5).

Implements the formula from data/tasks/yelp/G5f_prompt.txt.
Simple formula with L1.5 error-type grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

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


def compute_l1_positive_execution(judgment: Dict[str, Any]) -> bool:
    order_accuracy = judgment.get("order_accuracy", "not_mentioned")
    special_request = judgment.get("special_request_handling", "not_mentioned")
    timing = judgment.get("timing_coordination", "not_mentioned")

    return (order_accuracy in ("perfect", "correct") or
            special_request in ("exceeded", "honored") or
            timing in ("perfectly_timed", "well_coordinated"))


def compute_l1_negative_execution(judgment: Dict[str, Any]) -> bool:
    order_accuracy = judgment.get("order_accuracy", "not_mentioned")
    special_request = judgment.get("special_request_handling", "not_mentioned")
    timing = judgment.get("timing_coordination", "not_mentioned")

    return (order_accuracy in ("major_error", "completely_wrong") or
            special_request in ("ignored", "refused") or
            timing in ("poorly_timed", "chaotic"))


def is_error_resolved(judgment: Dict[str, Any]) -> bool:
    return judgment.get("error_recovery", "not_mentioned") in ("excellent", "good")


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

    # Find worst error type (most unresolved)
    buckets_with_errors = [(k, v) for k, v in buckets.items() if v["n_unresolved"] > 0]
    if buckets_with_errors:
        worst_error_type, _ = max(buckets_with_errors, key=lambda x: x[1]["n_unresolved"])
    else:
        worst_error_type = None

    # Find best resolution
    buckets_with_resolved = [(k, v) for k, v in buckets.items() if v["n_resolved"] > 0]
    if buckets_with_resolved:
        best_resolution, _ = max(buckets_with_resolved, key=lambda x: x[1]["resolution_rate"])
    else:
        best_resolution = None

    n_error_types_problematic = sum(1 for k, v in buckets.items() if v["resolution_rate"] < 0.5 and v["n_reviews"] > 0)

    # Determine execution pattern
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

    n_positive = 0
    n_negative = 0

    for j in execution_judgments:
        if compute_l1_positive_execution(j):
            n_positive += 1
        if compute_l1_negative_execution(j):
            n_negative += 1

    # L1.5 Error Buckets
    l15_buckets = compute_l15_buckets(execution_judgments)
    execution_pattern = l15_buckets["execution_pattern"]
    execution_pattern_bonus = EXECUTION_PATTERN_BONUS.get(execution_pattern, 0.0)

    # Formulas
    satisfaction_rate = n_positive / max(n_positive + n_negative, 1)
    positive_score = n_positive * 1.5
    negative_score = n_negative * 1.5

    raw_score = BASE_SCORE + positive_score - negative_score + execution_pattern_bonus
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
    strength_note = None
    improvement_note = None

    if execution_pattern == "execution_excellence" and verdict in ("Average Execution", "Poor Execution"):
        override_applied = "excellence_min_good"
        verdict = "Good Execution"
    elif execution_pattern == "systemic_issues" and verdict in ("Excellent Execution", "Good Execution"):
        override_applied = "systemic_max_average"
        verdict = "Average Execution"

    if execution_pattern == "recovers_well":
        strength_note = "Recovers well from errors"
    elif execution_pattern == "execution_excellence":
        strength_note = "Excellent order execution"

    if l15_buckets["worst_error_type"]:
        improvement_note = f"Could improve on {l15_buckets['worst_error_type'].replace('_', ' ')}"

    result = {
        "L1_5_error_buckets": l15_buckets,
        "N_EXECUTION_REVIEWS": n_execution_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "SATISFACTION_RATE": round(satisfaction_rate, 3),
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "EXECUTION_PATTERN_BONUS": execution_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_execution_reviews": n_execution_reviews,
    }

    if strength_note:
        result["strength_note"] = strength_note
    if improvement_note:
        result["improvement_note"] = improvement_note

    return result
