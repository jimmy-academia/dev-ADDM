"""
Ground Truth computation for G5k (Consistency - Complex).

Implements the formula from data/tasks/yelp/G5k_prompt.txt.
Complex formula with weighted consistency factors.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

FOOD_CONSISTENCY_SCORES = {"always_excellent": 3.0, "consistent": 2.0, "improving": 1.5, "variable": -1.5, "declining": -3.0, "not_mentioned": 0}
SERVICE_CONSISTENCY_SCORES = {"always_excellent": 3.0, "consistent": 2.0, "improving": 1.5, "variable": -1.5, "declining": -3.0, "not_mentioned": 0}
TIMING_CONSISTENCY_SCORES = {"always_prompt": 2.0, "consistent": 1.5, "improving": 1.0, "variable": -1.0, "declining": -2.0, "not_mentioned": 0}
CHANGE_SCORES = {"better_than_before": 2.5, "same_as_before": 1.0, "worse_than_before": -2.5, "not_applicable": 0, "not_mentioned": 0}
EXPECTATION_SCORES = {"exceeded": 2.0, "met": 0.5, "disappointed": -2.5, "not_applicable": 0, "not_mentioned": 0}
LOYALTY_SCORES = {"would_return": 2.0, "might_return": 0, "wont_return": -2.5, "lost_customer": -4.0, "not_mentioned": 0}


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    food_consistency = judgment.get("food_consistency", "not_mentioned")
    service_consistency = judgment.get("service_consistency", "not_mentioned")
    timing_consistency = judgment.get("timing_consistency", "not_mentioned")
    overall_change = judgment.get("overall_change", "not_mentioned")
    expectation_match = judgment.get("expectation_match", "not_mentioned")
    recommendation_change = judgment.get("recommendation_change", "not_mentioned")
    visit_type = judgment.get("visit_type", "not_mentioned")

    food_score = FOOD_CONSISTENCY_SCORES.get(food_consistency, 0)
    service_score = SERVICE_CONSISTENCY_SCORES.get(service_consistency, 0)
    timing_score = TIMING_CONSISTENCY_SCORES.get(timing_consistency, 0)
    change_score = CHANGE_SCORES.get(overall_change, 0)
    expectation_score = EXPECTATION_SCORES.get(expectation_match, 0)
    loyalty_score = LOYALTY_SCORES.get(recommendation_change, 0)

    l1_consistency_score = (
        food_score + service_score + timing_score +
        change_score + expectation_score + loyalty_score
    )

    # Repeat visitor weight
    if visit_type == "regular":
        repeat_visitor_weight = 1.5
    elif visit_type in ("repeat_visit", "returning_after_gap"):
        repeat_visitor_weight = 1.2
    else:
        repeat_visitor_weight = 1.0

    l1_weighted_score = l1_consistency_score * repeat_visitor_weight

    # Lost regular penalty
    lost_regular_penalty = -3.0 if (visit_type == "regular" and recommendation_change == "lost_customer") else 0.0

    l1_total_score = l1_weighted_score + lost_regular_penalty

    return {
        "food_consistency_score": food_score,
        "service_consistency_score": service_score,
        "timing_consistency_score": timing_score,
        "change_score": change_score,
        "expectation_score": expectation_score,
        "loyalty_score": loyalty_score,
        "l1_consistency_score": round(l1_consistency_score, 2),
        "repeat_visitor_weight": repeat_visitor_weight,
        "l1_weighted_score": round(l1_weighted_score, 2),
        "lost_regular_penalty": lost_regular_penalty,
        "l1_total_score": round(l1_total_score, 2),
    }


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

    sum_l1_score = 0.0
    n_repeat_visitors = 0
    n_regulars = 0
    n_declining = 0
    n_lost_customers = 0

    for j in consistency_judgments:
        l1 = compute_l1_complex(j)
        sum_l1_score += l1["l1_total_score"]

        if j.get("visit_type") in ("repeat_visit", "regular", "returning_after_gap"):
            n_repeat_visitors += 1
        if j.get("visit_type") == "regular":
            n_regulars += 1
        if j.get("food_consistency") == "declining" or j.get("service_consistency") == "declining":
            n_declining += 1
        if j.get("recommendation_change") == "lost_customer":
            n_lost_customers += 1

    mean_l1_score = sum_l1_score / max(n_consistency_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score
    final_score = max(0.0, min(10.0, raw_score))

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
    loyalty_warning = None
    decline_warning = None

    if n_lost_customers >= 2 and verdict in ("Excellent Consistency", "Good Consistency"):
        override_applied = "lost_customers_max_variable"
        verdict = "Variable Consistency"
        loyalty_warning = "Multiple customers report they won't return"
    elif n_regulars >= 3 and mean_l1_score >= 3 and verdict in ("Variable Consistency", "Poor Consistency"):
        override_applied = "regulars_min_good"
        verdict = "Good Consistency"
    elif n_declining >= 2:
        decline_warning = "Multiple reports of declining quality"
    elif mean_l1_score < -3:
        override_applied = "low_mean_score"
        verdict = "Poor Consistency"

    result = {
        "N_CONSISTENCY_REVIEWS": n_consistency_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_REPEAT_VISITORS": n_repeat_visitors,
        "N_REGULARS": n_regulars,
        "N_DECLINING": n_declining,
        "N_LOST_CUSTOMERS": n_lost_customers,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_consistency_reviews": n_consistency_reviews,
    }

    if loyalty_warning:
        result["loyalty_warning"] = loyalty_warning
    if decline_warning:
        result["decline_warning"] = decline_warning

    return result
