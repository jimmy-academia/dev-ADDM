"""
Ground Truth computation for G3j (Time-Value - Simple + L1.5).

Implements the formula from data/tasks/yelp/G3j_prompt.txt.
Simple formula with L1.5 time aspect grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

# L1.5 Time Pattern Bonus values
TIME_PATTERN_BONUS = {
    "consistently_efficient": 2.0,
    "quick_seating": 1.0,
    "good_food_pacing": 1.0,
    "reservation_reliable": 1.5,
    "time_dependent": 0.0,
    "unreliable_timing": -1.5,
}


# =============================================================================
# Helpers
# =============================================================================


def get_time_bucket(judgment: Dict[str, Any]) -> str:
    """Determine which time aspect is primary for this review."""
    if judgment.get("wait_for_table", "not_mentioned") != "not_mentioned":
        return "seating"
    elif judgment.get("food_delivery_speed", "not_mentioned") != "not_mentioned":
        return "food_delivery"
    elif judgment.get("reservation_accuracy", "not_applicable") not in ("not_applicable", "not_mentioned"):
        return "reservation"
    return "other"


def compute_l1_good_time(judgment: Dict[str, Any]) -> bool:
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


def compute_l15_buckets(time_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "seating": {"n_reviews": 0, "n_good": 0, "n_bad": 0},
        "food_delivery": {"n_reviews": 0, "n_good": 0, "n_bad": 0},
        "reservation": {"n_reviews": 0, "n_good": 0, "n_bad": 0},
    }

    for j in time_judgments:
        bucket = get_time_bucket(j)
        if bucket not in buckets:
            continue

        buckets[bucket]["n_reviews"] += 1

        if compute_l1_good_time(j):
            buckets[bucket]["n_good"] += 1
        if compute_l1_bad_time(j):
            buckets[bucket]["n_bad"] += 1

    for key, bucket in buckets.items():
        n_good = bucket["n_good"]
        n_bad = bucket["n_bad"]
        bucket["efficiency_rate"] = n_good / max(n_good + n_bad, 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_good"] + v["n_bad"] > 0]

    if buckets_with_data:
        best_aspect, best_bucket = max(buckets_with_data, key=lambda x: x[1]["efficiency_rate"])
        best_efficiency_rate = best_bucket["efficiency_rate"]
        worst_aspect, worst_bucket = min(buckets_with_data, key=lambda x: x[1]["efficiency_rate"])
    else:
        best_aspect = None
        best_efficiency_rate = 0.0
        worst_aspect = None

    n_aspects_efficient = sum(1 for k, v in buckets.items() if v["efficiency_rate"] >= 0.7 and v["n_reviews"] > 0)

    # Determine time pattern
    if n_aspects_efficient >= 3:
        time_pattern = "consistently_efficient"
    elif buckets["seating"]["efficiency_rate"] >= 0.8 and buckets["seating"]["n_reviews"] > 0:
        time_pattern = "quick_seating"
    elif buckets["food_delivery"]["efficiency_rate"] >= 0.8 and buckets["food_delivery"]["n_reviews"] > 0:
        time_pattern = "good_food_pacing"
    elif buckets["reservation"]["efficiency_rate"] >= 0.8 and buckets["reservation"]["n_reviews"] > 0:
        time_pattern = "reservation_reliable"
    elif n_aspects_efficient >= 1:
        time_pattern = "time_dependent"
    else:
        time_pattern = "unreliable_timing"

    return {
        "seating": buckets["seating"],
        "food_delivery": buckets["food_delivery"],
        "reservation": buckets["reservation"],
        "best_aspect": best_aspect,
        "best_efficiency_rate": round(best_efficiency_rate, 3),
        "worst_aspect": worst_aspect,
        "n_aspects_efficient": n_aspects_efficient,
        "time_pattern": time_pattern,
    }


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
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

    for j in time_judgments:
        if compute_l1_good_time(j):
            n_good += 1
        if compute_l1_bad_time(j):
            n_bad += 1

    # L1.5 Time Buckets
    l15_buckets = compute_l15_buckets(time_judgments)
    time_pattern = l15_buckets["time_pattern"]
    time_pattern_bonus = TIME_PATTERN_BONUS.get(time_pattern, 0.0)

    # Formulas
    efficiency_rate = n_good / max(n_good + n_bad, 1)
    positive_score = n_good * 1.5
    negative_score = n_bad * 1.5

    raw_score = BASE_SCORE + positive_score - negative_score + time_pattern_bonus
    final_score = max(0.0, min(10.0, raw_score))

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
    timing_recommendation = None

    if time_pattern == "consistently_efficient" and verdict in ("Variable Timing", "Poor Timing"):
        override_applied = "consistent_min_good"
        verdict = "Good Timing"
    elif time_pattern == "unreliable_timing" and n_bad >= 3:
        override_applied = "unreliable_with_bad"
        verdict = "Poor Timing"
    elif time_pattern == "quick_seating":
        timing_recommendation = "Walk-ins handled efficiently"
    elif time_pattern == "reservation_reliable":
        timing_recommendation = "Reservations honored reliably"

    result = {
        "L1_5_time_buckets": l15_buckets,
        "N_TIME_REVIEWS": n_time_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_GOOD": n_good,
        "N_BAD": n_bad,
        "EFFICIENCY_RATE": round(efficiency_rate, 3),
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "TIME_PATTERN_BONUS": time_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_time_reviews": n_time_reviews,
    }

    if timing_recommendation:
        result["timing_recommendation"] = timing_recommendation

    return result
