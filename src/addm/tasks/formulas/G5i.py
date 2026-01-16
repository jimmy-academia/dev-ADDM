"""
Ground Truth computation for G5i (Consistency - Simple).

Implements the formula from data/tasks/yelp/G5i_prompt.txt.
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


def compute_l1_positive_consistency(judgment: Dict[str, Any]) -> bool:
    food_consistency = judgment.get("food_consistency", "not_mentioned")
    service_consistency = judgment.get("service_consistency", "not_mentioned")
    overall_change = judgment.get("overall_change", "not_mentioned")
    expectation_match = judgment.get("expectation_match", "not_mentioned")

    if food_consistency in ("always_excellent", "consistent", "improving"):
        return True
    if service_consistency in ("always_excellent", "consistent", "improving"):
        return True
    if overall_change == "better_than_before":
        return True
    if expectation_match == "exceeded":
        return True
    return False


def compute_l1_negative_consistency(judgment: Dict[str, Any]) -> bool:
    food_consistency = judgment.get("food_consistency", "not_mentioned")
    service_consistency = judgment.get("service_consistency", "not_mentioned")
    overall_change = judgment.get("overall_change", "not_mentioned")
    recommendation_change = judgment.get("recommendation_change", "not_mentioned")

    if food_consistency in ("variable", "declining"):
        return True
    if service_consistency in ("variable", "declining"):
        return True
    if overall_change == "worse_than_before":
        return True
    if recommendation_change in ("wont_return", "lost_customer"):
        return True
    return False


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    consistency_judgments = [j for j in judgments if j.get("is_consistency_related", False)]
    n_consistency_reviews = len(consistency_judgments)

    if n_consistency_reviews == 0:
        confidence_level = "none"
    elif n_consistency_reviews <= 2:
        confidence_level = "low"
    elif n_consistency_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    n_positive = 0
    n_negative = 0
    n_repeat_visitors = 0
    n_declining = 0
    n_lost_customers = 0

    for j in consistency_judgments:
        if compute_l1_positive_consistency(j):
            n_positive += 1
        if compute_l1_negative_consistency(j):
            n_negative += 1

        if j.get("visit_type") in ("repeat_visit", "regular", "returning_after_gap"):
            n_repeat_visitors += 1
        if j.get("food_consistency") == "declining" or j.get("service_consistency") == "declining":
            n_declining += 1
        if j.get("recommendation_change") == "lost_customer":
            n_lost_customers += 1

    # Formulas
    positive_score = n_positive * 1.5
    negative_score = (n_negative * 1.5) + (n_declining * 1.0) + (n_lost_customers * 2.0)

    raw_score = BASE_SCORE + positive_score - negative_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
    if final_score >= 7.5:
        base_verdict_by_score = "Excellent Consistency"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good Consistency"
    elif final_score >= 3.5:
        base_verdict_by_score = "Variable Consistency"
    else:
        base_verdict_by_score = "Poor Consistency"

    override_applied = "none"
    verdict = base_verdict_by_score
    decline_warning = None
    confidence_note = None

    if n_lost_customers >= 2 and verdict in ("Excellent Consistency", "Good Consistency"):
        override_applied = "lost_customers_max_variable"
        verdict = "Variable Consistency"

    if n_declining >= 2:
        decline_warning = "Multiple reports of declining quality"
    if n_repeat_visitors < 2:
        confidence_note = "Limited repeat visit data"

    result = {
        "N_CONSISTENCY_REVIEWS": n_consistency_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "N_REPEAT_VISITORS": n_repeat_visitors,
        "N_DECLINING": n_declining,
        "N_LOST_CUSTOMERS": n_lost_customers,
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_consistency_reviews": n_consistency_reviews,
    }

    if decline_warning:
        result["decline_warning"] = decline_warning
    if confidence_note:
        result["confidence_note"] = confidence_note

    return result
