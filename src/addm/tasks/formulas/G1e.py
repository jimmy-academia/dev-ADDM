"""
Ground Truth computation for G1e (Dietary Accommodation - Simple).

Implements the formula from data/tasks/yelp/G1e_prompt.txt.
Basic scoring without weighting.

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


def derive_l1(judgment: Dict[str, Any]) -> Dict[str, bool]:
    """
    Derive L1 composites from L0 primitives.

    L1 composites (from prompt):
    - FIRSTHAND_FAILURE = true iff ALL:
        ACCOUNT_TYPE = firsthand
        ACCOMMODATION_OUTCOME = failure

    - POSITIVE_EXPERIENCE = true iff ALL:
        ACCOUNT_TYPE = firsthand
        ACCOMMODATION_OUTCOME = success
        STAFF_KNOWLEDGE in {knowledgeable, uncertain}
    """
    account_type = judgment.get("account_type", "hypothetical")
    accommodation_outcome = judgment.get("accommodation_outcome", "not_attempted")
    staff_knowledge = judgment.get("staff_knowledge", "none")

    is_firsthand = account_type == "firsthand"

    # FIRSTHAND_FAILURE
    firsthand_failure = is_firsthand and accommodation_outcome == "failure"

    # POSITIVE_EXPERIENCE
    positive_experience = (
        is_firsthand
        and accommodation_outcome == "success"
        and staff_knowledge in ("knowledgeable", "uncertain")
    )

    return {
        "FIRSTHAND_FAILURE": firsthand_failure,
        "POSITIVE_EXPERIENCE": positive_experience,
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

    Args:
        judgments: List of L0 judgments (one per review)
        restaurant_meta: Restaurant metadata

    Returns:
        Dict with all computed values matching OUTPUT SCHEMA in prompt
    """
    # Filter to dietary-related judgments only
    dietary_judgments = [j for j in judgments if j.get("is_dietary_related", False)]
    n_dietary_reviews = len(dietary_judgments)

    # CONFIDENCE_LEVEL
    if n_dietary_reviews == 0:
        confidence_level = "none"
    elif n_dietary_reviews <= 2:
        confidence_level = "low"
    elif n_dietary_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    # Compute L2 aggregates
    n_success = 0
    n_failure = 0
    n_partial = 0
    n_knowledgeable = 0
    n_uninformed = 0

    for j in dietary_judgments:
        account_type = j.get("account_type", "hypothetical")
        accommodation_outcome = j.get("accommodation_outcome", "not_attempted")
        staff_knowledge = j.get("staff_knowledge", "none")

        is_firsthand = account_type == "firsthand"

        # Count outcomes (firsthand only for success/failure/partial)
        if is_firsthand:
            if accommodation_outcome == "success":
                n_success += 1
            elif accommodation_outcome == "failure":
                n_failure += 1
            elif accommodation_outcome == "partial":
                n_partial += 1

        # Count staff knowledge
        if staff_knowledge == "knowledgeable":
            n_knowledgeable += 1
        elif staff_knowledge == "uninformed":
            n_uninformed += 1

    # Formulas
    total_outcomes = n_success + n_failure + n_partial
    success_rate = n_success / max(total_outcomes, 1)
    failure_rate = n_failure / max(total_outcomes, 1)

    positive_score = (n_success * 2) + (n_knowledgeable * 1)
    negative_score = (n_failure * 3) + (n_partial * 1) + (n_uninformed * 1)

    raw_score = BASE_SCORE + positive_score - negative_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy: Base verdict by score
    if final_score >= 7.0:
        base_verdict_by_score = "Excellent"
    elif final_score >= 5.0:
        base_verdict_by_score = "Adequate"
    elif final_score >= 3.0:
        base_verdict_by_score = "Poor"
    else:
        base_verdict_by_score = "Very Poor"

    # Decision Policy: Overrides
    override_applied = "none"
    verdict = base_verdict_by_score

    # Override 1: N_FAILURE >= 3 => Very Poor
    if n_failure >= 3:
        override_applied = "pattern_of_failures"
        verdict = "Very Poor"
    # Override 2: SUCCESS_RATE >= 0.8 AND N_SUCCESS >= 3 => min Adequate
    elif success_rate >= 0.8 and n_success >= 3 and verdict == "Poor":
        override_applied = "high_success_rate_min_adequate"
        verdict = "Adequate"
    elif success_rate >= 0.8 and n_success >= 3 and verdict == "Very Poor":
        override_applied = "high_success_rate_min_adequate"
        verdict = "Adequate"
    # Override 3: FAILURE_RATE >= 0.5 => max Poor
    elif failure_rate >= 0.5 and verdict in ("Excellent", "Adequate"):
        override_applied = "high_failure_rate_max_poor"
        verdict = "Poor"

    return {
        # L2 Aggregates
        "N_DIETARY_REVIEWS": n_dietary_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_SUCCESS": n_success,
        "N_FAILURE": n_failure,
        "N_PARTIAL": n_partial,
        "N_KNOWLEDGEABLE": n_knowledgeable,
        "N_UNINFORMED": n_uninformed,
        # Formula results
        "SUCCESS_RATE": round(success_rate, 3),
        "FAILURE_RATE": round(failure_rate, 3),
        "POSITIVE_SCORE": positive_score,
        "NEGATIVE_SCORE": negative_score,
        "BASE_SCORE": BASE_SCORE,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_dietary_reviews": n_dietary_reviews,
    }
