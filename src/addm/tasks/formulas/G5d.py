"""
Ground Truth computation for G5d (Capacity Management - Complex + L1.5).

Implements the formula from data/tasks/yelp/G5d_prompt.txt.
Complex formula with weighted capacity factors + L1.5 time-period grouping.

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

CAPACITY_PATTERN_BONUS = {
    "consistently_managed": 2.0,
    "peak_specialist": 1.5,
    "struggles_at_peak": -0.5,
    "event_challenged": -1.0,
    "inconsistent_handling": -0.5,
    "capacity_issues": -1.0,
}


# =============================================================================
# Helpers
# =============================================================================


def get_time_bucket(judgment: Dict[str, Any]) -> str:
    visit_timing = judgment.get("visit_timing", "not_mentioned")
    if visit_timing in ("peak_dinner", "peak_lunch"):
        return "peak_hours"
    elif visit_timing == "off_peak":
        return "off_peak"
    elif visit_timing == "special_event":
        return "special_events"
    return "other"


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

    chaos_penalty = -2.0 if (crowd_level in ("packed", "overwhelming") and staff_coverage in ("stretched", "insufficient")) else 0.0
    smooth_operation_bonus = 2.0 if (wait_time in ("none", "short") and staff_coverage == "excellent" and peak_handling in ("excellent", "good")) else 0.0

    l1_total_score = l1_capacity_score + chaos_penalty + smooth_operation_bonus

    return {
        "l1_capacity_score": round(l1_capacity_score, 2),
        "chaos_penalty": chaos_penalty,
        "smooth_operation_bonus": smooth_operation_bonus,
        "l1_total_score": round(l1_total_score, 2),
    }


def compute_l1_positive_capacity(judgment: Dict[str, Any]) -> bool:
    crowd_level = judgment.get("crowd_level", "not_mentioned")
    wait_time = judgment.get("wait_time", "not_mentioned")
    staff_coverage = judgment.get("staff_coverage", "not_mentioned")
    reservation_system = judgment.get("reservation_system", "not_mentioned")

    return (crowd_level in ("empty", "comfortable") or
            wait_time in ("none", "short") or
            staff_coverage in ("excellent", "adequate") or
            reservation_system in ("excellent", "good"))


def compute_l1_negative_capacity(judgment: Dict[str, Any]) -> bool:
    crowd_level = judgment.get("crowd_level", "not_mentioned")
    wait_time = judgment.get("wait_time", "not_mentioned")
    staff_coverage = judgment.get("staff_coverage", "not_mentioned")
    reservation_system = judgment.get("reservation_system", "not_mentioned")

    return (crowd_level in ("packed", "overwhelming") or
            wait_time in ("long", "excessive") or
            staff_coverage in ("stretched", "insufficient") or
            reservation_system in ("problematic", "failed"))


def compute_l15_buckets(capacity_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "peak_hours": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
        "off_peak": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
        "special_events": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
    }

    for j in capacity_judgments:
        bucket = get_time_bucket(j)
        if bucket not in buckets:
            continue

        buckets[bucket]["n_reviews"] += 1

        if compute_l1_positive_capacity(j):
            buckets[bucket]["n_positive"] += 1
        if compute_l1_negative_capacity(j):
            buckets[bucket]["n_negative"] += 1

    for key, bucket in buckets.items():
        n_positive = bucket["n_positive"]
        n_negative = bucket["n_negative"]
        bucket["success_rate"] = n_positive / max(n_positive + n_negative, 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_positive"] + v["n_negative"] > 0]

    if buckets_with_data:
        best_time, best_bucket = max(buckets_with_data, key=lambda x: x[1]["success_rate"])
        best_time_rate = best_bucket["success_rate"]
        worst_time, _ = min(buckets_with_data, key=lambda x: x[1]["success_rate"])
    else:
        best_time = None
        best_time_rate = 0.0
        worst_time = None

    peak_rate = buckets["peak_hours"]["success_rate"] if buckets["peak_hours"]["n_reviews"] > 0 else 0
    offpeak_rate = buckets["off_peak"]["success_rate"] if buckets["off_peak"]["n_reviews"] > 0 else 0
    peak_vs_offpeak_gap = peak_rate - offpeak_rate

    if peak_rate >= 0.7 and offpeak_rate >= 0.7:
        capacity_pattern = "consistently_managed"
    elif peak_rate >= 0.7:
        capacity_pattern = "peak_specialist"
    elif offpeak_rate >= 0.7 and peak_rate < 0.5:
        capacity_pattern = "struggles_at_peak"
    elif buckets["special_events"]["n_negative"] > buckets["special_events"]["n_positive"] and buckets["special_events"]["n_reviews"] > 0:
        capacity_pattern = "event_challenged"
    elif abs(peak_vs_offpeak_gap) > 0.3:
        capacity_pattern = "inconsistent_handling"
    else:
        capacity_pattern = "capacity_issues"

    return {
        "peak_hours": buckets["peak_hours"],
        "off_peak": buckets["off_peak"],
        "special_events": buckets["special_events"],
        "best_time": best_time,
        "best_time_rate": round(best_time_rate, 3),
        "worst_time": worst_time,
        "peak_vs_offpeak_gap": round(peak_vs_offpeak_gap, 3),
        "capacity_pattern": capacity_pattern,
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
    n_reservation_failed = 0

    for j in capacity_judgments:
        l1 = compute_l1_complex(j)
        sum_l1_score += l1["l1_total_score"]

        if j.get("wait_time") == "excessive":
            n_excessive_wait += 1
        if j.get("reservation_system") == "failed":
            n_reservation_failed += 1

    # L1.5 Time Buckets
    l15_buckets = compute_l15_buckets(capacity_judgments)
    capacity_pattern = l15_buckets["capacity_pattern"]
    capacity_pattern_bonus = CAPACITY_PATTERN_BONUS.get(capacity_pattern, 0.0)

    mean_l1_score = sum_l1_score / max(n_capacity_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score + capacity_pattern_bonus
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
    peak_warning = None

    if capacity_pattern == "consistently_managed" and verdict in ("Average Capacity Management", "Poor Capacity Management"):
        override_applied = "consistent_min_good"
        verdict = "Good Capacity Management"
    elif capacity_pattern == "struggles_at_peak" and n_excessive_wait >= 2:
        override_applied = "peak_struggles_with_waits"
        verdict = "Average Capacity Management"
        peak_warning = "Struggles during peak hours"
    elif mean_l1_score < -3:
        override_applied = "low_mean_score"
        verdict = "Poor Capacity Management"

    result = {
        "L1_5_time_buckets": l15_buckets,
        "N_CAPACITY_REVIEWS": n_capacity_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_EXCESSIVE_WAIT": n_excessive_wait,
        "N_RESERVATION_FAILED": n_reservation_failed,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "CAPACITY_PATTERN_BONUS": capacity_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_capacity_reviews": n_capacity_reviews,
    }

    if peak_warning:
        result["peak_warning"] = peak_warning

    return result
