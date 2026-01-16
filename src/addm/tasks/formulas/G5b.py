"""
Ground Truth computation for G5b (Capacity Management - Simple + L1.5).

Implements the formula from data/tasks/yelp/G5b_prompt.txt.
Simple formula with L1.5 time-period grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

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

    # Determine capacity pattern
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

    n_positive = 0
    n_negative = 0

    for j in capacity_judgments:
        if compute_l1_positive_capacity(j):
            n_positive += 1
        if compute_l1_negative_capacity(j):
            n_negative += 1

    # L1.5 Time Buckets
    l15_buckets = compute_l15_buckets(capacity_judgments)
    capacity_pattern = l15_buckets["capacity_pattern"]
    capacity_pattern_bonus = CAPACITY_PATTERN_BONUS.get(capacity_pattern, 0.0)

    # Formulas
    satisfaction_rate = n_positive / max(n_positive + n_negative, 1)
    positive_score = n_positive * 1.5
    negative_score = n_negative * 1.5

    raw_score = BASE_SCORE + positive_score - negative_score + capacity_pattern_bonus
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
    strength_note = None
    timing_recommendation = None
    peak_warning = None

    if capacity_pattern == "consistently_managed" and verdict in ("Average Capacity Management", "Poor Capacity Management"):
        override_applied = "consistent_min_good"
        verdict = "Good Capacity Management"

    if capacity_pattern == "peak_specialist":
        strength_note = "Handles peak hours well"
    elif capacity_pattern == "struggles_at_peak":
        peak_warning = "May struggle during peak hours"
        timing_recommendation = "Consider visiting during off-peak hours"

    if l15_buckets["best_time"]:
        timing_recommendation = f"Best experience during {l15_buckets['best_time'].replace('_', ' ')}"

    result = {
        "L1_5_time_buckets": l15_buckets,
        "N_CAPACITY_REVIEWS": n_capacity_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "SATISFACTION_RATE": round(satisfaction_rate, 3),
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "CAPACITY_PATTERN_BONUS": capacity_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_capacity_reviews": n_capacity_reviews,
    }

    if strength_note:
        result["strength_note"] = strength_note
    if timing_recommendation:
        result["timing_recommendation"] = timing_recommendation
    if peak_warning:
        result["peak_warning"] = peak_warning

    return result
