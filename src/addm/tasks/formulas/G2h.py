"""
Ground Truth computation for G2h (Business - Complex + L1.5).

Implements the formula from data/tasks/yelp/G2h_prompt.txt.
Complex formula with meeting stakes weighting + L1.5 meeting type grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

# Meeting stakes weights
MEETING_STAKES_WEIGHTS = {
    "client_meeting": 2.0,
    "interview": 1.8,
    "team_lunch": 1.0,
    "working_session": 1.2,
    "business_general": 1.0,
    "not_business": 0,
}

# Noise scores
NOISE_SCORES = {
    "quiet": 2.5,
    "moderate": 1,
    "loud": -3,
    "not_mentioned": 0,
}

# WiFi scores
WIFI_SCORES = {
    "fast": 2,
    "available": 1,
    "slow": -0.5,
    "unavailable": -1.5,
    "not_mentioned": 0,
}

# Atmosphere scores
ATMOSPHERE_SCORES = {
    "formal": 2.5,
    "business_casual": 1.5,
    "casual": 0.5,
    "inappropriate": -2,
    "not_mentioned": 0,
}

# Timing scores
TIMING_SCORES = {
    "efficient": 2,
    "appropriate": 0.5,
    "slow": -2,
    "rushed": -1.5,
    "not_mentioned": 0,
}

# Seating scores
SEATING_SCORES = {
    "excellent": 2,
    "adequate": 0.5,
    "poor": -1.5,
    "not_mentioned": 0,
}

# Outcome modifiers
OUTCOME_MODIFIERS = {
    "impressed": 3,
    "satisfied": 1,
    "embarrassed": -4,
    "not_mentioned": 0,
}

# L1.5 Business Versatility Bonus values
BUSINESS_VERSATILITY_BONUS = {
    "all_purpose_business": 2.0,
    "client_meeting_specialist": 1.5,
    "great_for_remote_work": 1.0,
    "team_lunch_favorite": 1.0,
    "situational": 0.0,
    "not_recommended": -1.0,
}


# =============================================================================
# Helpers
# =============================================================================


def get_meeting_bucket(meeting_type: str) -> str:
    """Map meeting type to L1.5 bucket."""
    if meeting_type in ("client_meeting", "interview"):
        return "high_stakes"
    elif meeting_type in ("team_lunch", "business_general"):
        return "team_casual"
    elif meeting_type == "working_session":
        return "working"
    else:
        return "other"


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    """Compute L1 composites using complex formula."""
    meeting_type = judgment.get("meeting_type", "not_business")
    noise_level = judgment.get("noise_level", "not_mentioned")
    wifi_availability = judgment.get("wifi_availability", "not_mentioned")
    professional_atmosphere = judgment.get("professional_atmosphere", "not_mentioned")
    service_timing = judgment.get("service_timing", "not_mentioned")
    seating_suitability = judgment.get("seating_suitability", "not_mentioned")
    outcome_impression = judgment.get("outcome_impression", "not_mentioned")

    meeting_stakes_weight = MEETING_STAKES_WEIGHTS.get(meeting_type, 0)
    noise_score = NOISE_SCORES.get(noise_level, 0)
    wifi_score = WIFI_SCORES.get(wifi_availability, 0)
    atmosphere_score = ATMOSPHERE_SCORES.get(professional_atmosphere, 0)
    timing_score = TIMING_SCORES.get(service_timing, 0)
    seating_score = SEATING_SCORES.get(seating_suitability, 0)
    outcome_modifier = OUTCOME_MODIFIERS.get(outcome_impression, 0)

    l1_review_score = (noise_score + atmosphere_score + timing_score + seating_score + outcome_modifier) * meeting_stakes_weight

    if meeting_type == "working_session" and wifi_availability == "fast":
        wifi_bonus = 2.0
    elif meeting_type == "working_session" and wifi_availability == "unavailable":
        wifi_bonus = -3.0
    else:
        wifi_bonus = wifi_score * 0.5

    embarrassed_client = -5.0 if (meeting_type == "client_meeting" and outcome_impression == "embarrassed") else 0.0

    l1_total_score = l1_review_score + wifi_bonus + embarrassed_client

    return {
        "meeting_stakes_weight": meeting_stakes_weight,
        "noise_score": noise_score,
        "wifi_score": wifi_score,
        "atmosphere_score": atmosphere_score,
        "timing_score": timing_score,
        "seating_score": seating_score,
        "outcome_modifier": outcome_modifier,
        "l1_review_score": round(l1_review_score, 2),
        "wifi_bonus": round(wifi_bonus, 2),
        "embarrassed_client": embarrassed_client,
        "l1_total_score": round(l1_total_score, 2),
    }


def compute_l1_success(judgment: Dict[str, Any]) -> bool:
    """Simple success check for L1.5 bucket tracking."""
    meeting_type = judgment.get("meeting_type", "not_business")
    noise_level = judgment.get("noise_level", "not_mentioned")
    professional_atmosphere = judgment.get("professional_atmosphere", "not_mentioned")

    if meeting_type == "not_business":
        return False

    if noise_level not in ("quiet", "moderate"):
        return False

    if professional_atmosphere not in ("formal", "business_casual", "casual"):
        return False

    return True


def compute_l1_failed(judgment: Dict[str, Any]) -> bool:
    """Simple failure check for L1.5 bucket tracking."""
    meeting_type = judgment.get("meeting_type", "not_business")
    noise_level = judgment.get("noise_level", "not_mentioned")
    professional_atmosphere = judgment.get("professional_atmosphere", "not_mentioned")
    service_timing = judgment.get("service_timing", "not_mentioned")

    if meeting_type == "not_business":
        return False

    return noise_level == "loud" or professional_atmosphere == "inappropriate" or service_timing == "slow"


def compute_l15_buckets(business_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute L1.5 meeting type buckets."""
    buckets = {
        "high_stakes": {"n_reviews": 0, "n_success": 0, "n_failed": 0},
        "team_casual": {"n_reviews": 0, "n_success": 0, "n_failed": 0},
        "working": {"n_reviews": 0, "n_success": 0, "n_failed": 0},
    }

    for j in business_judgments:
        meeting_type = j.get("meeting_type", "not_business")
        bucket = get_meeting_bucket(meeting_type)
        if bucket not in buckets:
            continue

        buckets[bucket]["n_reviews"] += 1

        if compute_l1_success(j):
            buckets[bucket]["n_success"] += 1
        if compute_l1_failed(j):
            buckets[bucket]["n_failed"] += 1

    for key, bucket in buckets.items():
        n_success = bucket["n_success"]
        n_failed = bucket["n_failed"]
        bucket["success_rate"] = n_success / max(n_success + n_failed, 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_success"] + v["n_failed"] > 0]

    if buckets_with_data:
        best_meeting_type, best_bucket = max(buckets_with_data, key=lambda x: x[1]["success_rate"])
        best_meeting_success_rate = best_bucket["success_rate"]
        worst_meeting_type, _ = min(buckets_with_data, key=lambda x: x[1]["success_rate"])
    else:
        best_meeting_type = None
        best_meeting_success_rate = 0.0
        worst_meeting_type = None

    n_meeting_types_served_well = sum(1 for k, v in buckets.items() if v["success_rate"] >= 0.7 and v["n_reviews"] > 0)

    if n_meeting_types_served_well >= 3:
        business_versatility = "all_purpose_business"
    elif buckets["high_stakes"]["success_rate"] >= 0.8 and buckets["high_stakes"]["n_reviews"] > 0:
        business_versatility = "client_meeting_specialist"
    elif buckets["working"]["success_rate"] >= 0.8 and buckets["working"]["n_reviews"] > 0:
        business_versatility = "great_for_remote_work"
    elif buckets["team_casual"]["success_rate"] >= 0.8 and buckets["team_casual"]["n_reviews"] > 0:
        business_versatility = "team_lunch_favorite"
    elif any(v["success_rate"] >= 0.6 and v["n_reviews"] > 0 for v in buckets.values()):
        business_versatility = "situational"
    else:
        business_versatility = "not_recommended"

    return {
        "high_stakes": buckets["high_stakes"],
        "team_casual": buckets["team_casual"],
        "working": buckets["working"],
        "best_meeting_type": best_meeting_type,
        "best_meeting_success_rate": round(best_meeting_success_rate, 3),
        "worst_meeting_type": worst_meeting_type,
        "n_meeting_types_served_well": n_meeting_types_served_well,
        "business_versatility": business_versatility,
    }


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
    business_judgments = [j for j in judgments if j.get("is_business_related", False)]
    n_business_reviews = len(business_judgments)

    if n_business_reviews == 0:
        confidence_level = "none"
    elif n_business_reviews <= 2:
        confidence_level = "low"
    elif n_business_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    sum_l1_score = 0.0
    sum_stakes_weight = 0.0
    n_high_stakes = 0
    n_high_stakes_success = 0
    n_embarrassed = 0
    n_working = 0
    n_working_with_wifi = 0

    for j in business_judgments:
        l1 = compute_l1_complex(j)

        sum_l1_score += l1["l1_total_score"]
        sum_stakes_weight += l1["meeting_stakes_weight"]

        meeting_type = j.get("meeting_type", "not_business")
        outcome_impression = j.get("outcome_impression", "not_mentioned")
        wifi_availability = j.get("wifi_availability", "not_mentioned")

        if meeting_type in ("client_meeting", "interview"):
            n_high_stakes += 1
            if outcome_impression == "impressed":
                n_high_stakes_success += 1

        if outcome_impression == "embarrassed":
            n_embarrassed += 1

        if meeting_type == "working_session":
            n_working += 1
            if wifi_availability in ("available", "fast"):
                n_working_with_wifi += 1

    # L1.5 Meeting Buckets
    l15_buckets = compute_l15_buckets(business_judgments)
    business_versatility = l15_buckets["business_versatility"]
    business_versatility_bonus = BUSINESS_VERSATILITY_BONUS.get(business_versatility, 0.0)

    # Weighted aggregation
    weighted_mean_score = sum_l1_score / max(sum_stakes_weight, 1)

    # Formulas (G2h: complex + L1.5)
    adjusted_score = weighted_mean_score

    # G2h: adds BUSINESS_VERSATILITY_BONUS to complex formula
    raw_score = BASE_SCORE + adjusted_score + business_versatility_bonus
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
    if final_score >= 7.5:
        base_verdict_by_score = "Excellent for Business"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good for Business"
    elif final_score >= 3.5:
        base_verdict_by_score = "Acceptable"
    else:
        base_verdict_by_score = "Not Suitable"

    override_applied = "none"
    verdict = base_verdict_by_score
    wifi_note = None

    if n_embarrassed >= 1 and verdict in ("Excellent for Business", "Good for Business"):
        override_applied = "embarrassed_max_acceptable"
        verdict = "Acceptable"
    elif n_high_stakes_success >= 2 and n_embarrassed == 0 and verdict in ("Acceptable", "Not Suitable"):
        override_applied = "high_stakes_success_min_good"
        verdict = "Good for Business"
    elif n_working >= 2 and n_working_with_wifi == 0:
        wifi_note = "Note: WiFi availability may be limited for working sessions"
    elif weighted_mean_score < -2:
        override_applied = "low_weighted_mean"
        verdict = "Not Suitable"

    result = {
        # L1.5 Meeting Buckets
        "L1_5_meeting_buckets": l15_buckets,
        # L2 Aggregates
        "N_BUSINESS_REVIEWS": n_business_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_HIGH_STAKES": n_high_stakes,
        "N_HIGH_STAKES_SUCCESS": n_high_stakes_success,
        "N_EMBARRASSED": n_embarrassed,
        "N_WORKING": n_working,
        "N_WORKING_WITH_WIFI": n_working_with_wifi,
        # Weighted aggregation
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "SUM_STAKES_WEIGHT": round(sum_stakes_weight, 2),
        "WEIGHTED_MEAN_SCORE": round(weighted_mean_score, 3),
        # Formula results
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "BUSINESS_VERSATILITY_BONUS": business_versatility_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_business_reviews": n_business_reviews,
    }

    if wifi_note:
        result["wifi_note"] = wifi_note

    return result
