"""
Ground Truth computation for G2e (Business - Simple).

Implements the formula from data/tasks/yelp/G2e_prompt.txt.
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
    n_quiet = 0
    n_loud = 0
    n_wifi = 0
    n_formal = 0

    for j in business_judgments:
        if compute_l1_success(j):
            n_success += 1
        if compute_l1_failed(j):
            n_failed += 1

        if j.get("noise_level") == "quiet":
            n_quiet += 1
        if j.get("noise_level") == "loud":
            n_loud += 1
        if j.get("wifi_availability") in ("available", "fast"):
            n_wifi += 1
        if j.get("professional_atmosphere") == "formal":
            n_formal += 1

    # Formulas
    positive_score = (n_success * 2) + (n_quiet * 1) + (n_wifi * 0.5) + (n_formal * 1)
    negative_score = (n_failed * 2) + (n_loud * 1.5)

    raw_score = BASE_SCORE + positive_score - negative_score
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

    # Override 1: If N_LOUD >= 2 => max Acceptable
    if n_loud >= 2 and verdict in ("Excellent for Business", "Good for Business"):
        override_applied = "loud_max_acceptable"
        verdict = "Acceptable"
    # Override 2: If N_SUCCESS >= 3 AND N_FAILED == 0 => min Good for Business
    elif n_success >= 3 and n_failed == 0 and verdict in ("Acceptable", "Not Suitable"):
        override_applied = "success_min_good"
        verdict = "Good for Business"
    # Override 3: If N_FORMAL >= 2 AND N_QUIET >= 2 => min Good for Business
    elif n_formal >= 2 and n_quiet >= 2 and verdict in ("Acceptable", "Not Suitable"):
        override_applied = "formal_quiet_min_good"
        verdict = "Good for Business"

    return {
        # L2 Aggregates
        "N_BUSINESS_REVIEWS": n_business_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_SUCCESS": n_success,
        "N_FAILED": n_failed,
        "N_QUIET": n_quiet,
        "N_LOUD": n_loud,
        "N_WIFI": n_wifi,
        "N_FORMAL": n_formal,
        # Formula results
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_business_reviews": n_business_reviews,
    }
