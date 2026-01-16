"""
Ground Truth computation for G2g (Business - Complex).

Implements the formula from data/tasks/yelp/G2g_prompt.txt.
Complex formula with meeting stakes weighting and interaction effects.

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


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute L1 composites using complex formula.

    Returns dict with all L1 computed values.
    """
    meeting_type = judgment.get("meeting_type", "not_business")
    noise_level = judgment.get("noise_level", "not_mentioned")
    wifi_availability = judgment.get("wifi_availability", "not_mentioned")
    professional_atmosphere = judgment.get("professional_atmosphere", "not_mentioned")
    service_timing = judgment.get("service_timing", "not_mentioned")
    seating_suitability = judgment.get("seating_suitability", "not_mentioned")
    outcome_impression = judgment.get("outcome_impression", "not_mentioned")

    # Get scores
    meeting_stakes_weight = MEETING_STAKES_WEIGHTS.get(meeting_type, 0)
    noise_score = NOISE_SCORES.get(noise_level, 0)
    wifi_score = WIFI_SCORES.get(wifi_availability, 0)
    atmosphere_score = ATMOSPHERE_SCORES.get(professional_atmosphere, 0)
    timing_score = TIMING_SCORES.get(service_timing, 0)
    seating_score = SEATING_SCORES.get(seating_suitability, 0)
    outcome_modifier = OUTCOME_MODIFIERS.get(outcome_impression, 0)

    # L1_REVIEW_SCORE
    l1_review_score = (noise_score + atmosphere_score + timing_score + seating_score + outcome_modifier) * meeting_stakes_weight

    # WIFI_BONUS (interaction for working sessions)
    if meeting_type == "working_session" and wifi_availability == "fast":
        wifi_bonus = 2.0
    elif meeting_type == "working_session" and wifi_availability == "unavailable":
        wifi_bonus = -3.0
    else:
        wifi_bonus = wifi_score * 0.5

    # EMBARRASSED_CLIENT (interaction effect)
    embarrassed_client = -5.0 if (meeting_type == "client_meeting" and outcome_impression == "embarrassed") else 0.0

    # L1_TOTAL_SCORE
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
    # Filter to business-related judgments only
    business_judgments = [j for j in judgments if j.get("is_business_related", False)]
    n_business_reviews = len(business_judgments)

    # CONFIDENCE_LEVEL
    if n_business_reviews == 0:
        confidence_level = "none"
    elif n_business_reviews <= 2:
        confidence_level = "low"
    elif n_business_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    # Compute L1 for each judgment and aggregate
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

    # Weighted aggregation
    weighted_mean_score = sum_l1_score / max(sum_stakes_weight, 1)

    # Formulas
    adjusted_score = weighted_mean_score
    raw_score = BASE_SCORE + adjusted_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy: Base verdict by score
    if final_score >= 7.5:
        base_verdict_by_score = "Excellent for Business"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good for Business"
    elif final_score >= 3.5:
        base_verdict_by_score = "Acceptable"
    else:
        base_verdict_by_score = "Not Suitable"

    # Decision Policy: Overrides
    override_applied = "none"
    verdict = base_verdict_by_score
    wifi_note = None

    # Override 1: If N_EMBARRASSED >= 1 => max Acceptable
    if n_embarrassed >= 1 and verdict in ("Excellent for Business", "Good for Business"):
        override_applied = "embarrassed_max_acceptable"
        verdict = "Acceptable"
    # Override 2: If N_HIGH_STAKES_SUCCESS >= 2 AND N_EMBARRASSED == 0 => min Good for Business
    elif n_high_stakes_success >= 2 and n_embarrassed == 0 and verdict in ("Acceptable", "Not Suitable"):
        override_applied = "high_stakes_success_min_good"
        verdict = "Good for Business"
    # Override 3: If N_WORKING >= 2 AND N_WORKING_WITH_WIFI == 0 => note WiFi limitation
    elif n_working >= 2 and n_working_with_wifi == 0:
        wifi_note = "Note: WiFi availability may be limited for working sessions"
    # Override 4: If WEIGHTED_MEAN_SCORE < -2 => Not Suitable
    elif weighted_mean_score < -2:
        override_applied = "low_weighted_mean"
        verdict = "Not Suitable"

    result = {
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
