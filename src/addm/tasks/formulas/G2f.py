"""
Ground Truth computation for G2f (Business - Simple + L1.5).

Implements the formula from data/tasks/yelp/G2f_prompt.txt.
Simple formula with L1.5 meeting type grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

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


def compute_l1_success(judgment: Dict[str, Any]) -> bool:
    """
    SUCCESSFUL_BUSINESS_MEAL = true iff ALL:
      - MEETING_TYPE != not_business
      - NOISE_LEVEL in {quiet, moderate}
      - PROFESSIONAL_ATMOSPHERE in {formal, business_casual, casual}
    """
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
    """
    FAILED_BUSINESS_MEAL = true iff ALL:
      - MEETING_TYPE != not_business
      - Any of: NOISE_LEVEL = loud, PROFESSIONAL_ATMOSPHERE = inappropriate, SERVICE_TIMING = slow
    """
    meeting_type = judgment.get("meeting_type", "not_business")
    noise_level = judgment.get("noise_level", "not_mentioned")
    professional_atmosphere = judgment.get("professional_atmosphere", "not_mentioned")
    service_timing = judgment.get("service_timing", "not_mentioned")

    if meeting_type == "not_business":
        return False

    has_failure = (
        noise_level == "loud"
        or professional_atmosphere == "inappropriate"
        or service_timing == "slow"
    )

    return has_failure


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

    # Compute success rates
    for key, bucket in buckets.items():
        n_success = bucket["n_success"]
        n_failed = bucket["n_failed"]
        bucket["success_rate"] = n_success / max(n_success + n_failed, 1)

    # Find best and worst meeting types
    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_success"] + v["n_failed"] > 0]

    if buckets_with_data:
        best_meeting_type, best_bucket = max(buckets_with_data, key=lambda x: x[1]["success_rate"])
        best_meeting_success_rate = best_bucket["success_rate"]
        worst_meeting_type, _ = min(buckets_with_data, key=lambda x: x[1]["success_rate"])
    else:
        best_meeting_type = None
        best_meeting_success_rate = 0.0
        worst_meeting_type = None

    # Count meeting types served well
    n_meeting_types_served_well = sum(1 for k, v in buckets.items() if v["success_rate"] >= 0.7 and v["n_reviews"] > 0)

    # Determine business versatility
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

    # Compute L1 for each judgment
    n_success = 0
    n_failed = 0

    for j in business_judgments:
        if compute_l1_success(j):
            n_success += 1
        if compute_l1_failed(j):
            n_failed += 1

    # L1.5 Meeting Buckets
    l15_buckets = compute_l15_buckets(business_judgments)
    business_versatility = l15_buckets["business_versatility"]
    business_versatility_bonus = BUSINESS_VERSATILITY_BONUS.get(business_versatility, 0.0)

    # Formulas
    success_rate = n_success / max(n_success + n_failed, 1)
    positive_score = n_success * 2
    negative_score = n_failed * 2

    # G2f: adds BUSINESS_VERSATILITY_BONUS to simple formula
    raw_score = BASE_SCORE + positive_score - negative_score + business_versatility_bonus
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

    # Override: If HIGH_STAKES bucket N_FAILED >= 2 => max Acceptable
    if l15_buckets["high_stakes"]["n_failed"] >= 2 and verdict in ("Excellent for Business", "Good for Business"):
        override_applied = "high_stakes_failures"
        verdict = "Acceptable"

    return {
        # L1.5 Meeting Buckets
        "L1_5_meeting_buckets": l15_buckets,
        # L2 Aggregates
        "N_BUSINESS_REVIEWS": n_business_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_SUCCESS": n_success,
        "N_FAILED": n_failed,
        "SUCCESS_RATE": round(success_rate, 3),
        # Formula results
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
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
