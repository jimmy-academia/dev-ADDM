"""
Ground Truth computation for G6g (Comparison - Complex).

Implements the formula from data/tasks/yelp/G6g_prompt.txt.
Complex formula with weighted comparison factors.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

COMPETITOR_SCORES = {"favorable": 3.0, "different": 1.0, "similar": 0, "unfavorable": -3.0, "not_mentioned": 0}
RANKING_SCORES = {"best_in_category": 4.0, "top_tier": 2.5, "competitive": 1.0, "middle_pack": -0.5, "bottom_tier": -3.0, "not_mentioned": 0}
PRICE_SCORES = {"best_value": 3.0, "underpriced": 2.0, "fair_value": 0.5, "overpriced": -2.5, "not_mentioned": 0}
QUALITY_SCORES = {"superior": 3.5, "comparable": 0.5, "inferior": -3.0, "not_mentioned": 0}
CONVENIENCE_SCORES = {"more_convenient": 2.0, "equally_convenient": 0.5, "less_convenient": -1.5, "not_mentioned": 0}
RECOMMENDATION_SCORES = {"over_competitor": 3.0, "for_specific_need": 2.0, "general_recommendation": 1.5, "conditional": 0, "not_recommended": -2.5, "not_mentioned": 0}
SWITCH_SCORES = {"switched_to": 3.0, "would_switch": 2.0, "loyal_to_other": -1.0, "switched_from": -3.0, "not_mentioned": 0}


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    competitor_mention = judgment.get("competitor_mention", "not_mentioned")
    category_ranking = judgment.get("category_ranking", "not_mentioned")
    price_comparison = judgment.get("price_comparison", "not_mentioned")
    quality_comparison = judgment.get("quality_comparison", "not_mentioned")
    convenience_comparison = judgment.get("convenience_comparison", "not_mentioned")
    recommendation_context = judgment.get("recommendation_context", "not_mentioned")
    switch_likelihood = judgment.get("switch_likelihood", "not_mentioned")

    competitor_score = COMPETITOR_SCORES.get(competitor_mention, 0)
    ranking_score = RANKING_SCORES.get(category_ranking, 0)
    price_score = PRICE_SCORES.get(price_comparison, 0)
    quality_score = QUALITY_SCORES.get(quality_comparison, 0)
    convenience_score = CONVENIENCE_SCORES.get(convenience_comparison, 0)
    recommendation_score = RECOMMENDATION_SCORES.get(recommendation_context, 0)
    switch_score = SWITCH_SCORES.get(switch_likelihood, 0)

    l1_comparison_score = (
        competitor_score + ranking_score + price_score +
        quality_score + convenience_score + recommendation_score + switch_score
    )

    # Best in class bonus
    best_in_class_bonus = 2.0 if (category_ranking == "best_in_category" and quality_comparison == "superior") else 0.0

    l1_total_score = l1_comparison_score + best_in_class_bonus

    return {
        "competitor_score": competitor_score,
        "ranking_score": ranking_score,
        "price_score": price_score,
        "quality_score": quality_score,
        "convenience_score": convenience_score,
        "recommendation_score": recommendation_score,
        "switch_score": switch_score,
        "l1_comparison_score": round(l1_comparison_score, 2),
        "best_in_class_bonus": best_in_class_bonus,
        "l1_total_score": round(l1_total_score, 2),
    }


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    comparison_judgments = [j for j in judgments if j.get("is_comparison_related", False)]
    n_comparison_reviews = len(comparison_judgments)

    if n_comparison_reviews == 0:
        confidence_level = "none"
    elif n_comparison_reviews <= 2:
        confidence_level = "low"
    elif n_comparison_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    sum_l1_score = 0.0
    n_best_in_category = 0
    n_bottom_tier = 0
    n_switched_to = 0
    n_switched_from = 0

    for j in comparison_judgments:
        l1 = compute_l1_complex(j)
        sum_l1_score += l1["l1_total_score"]

        if j.get("category_ranking") == "best_in_category":
            n_best_in_category += 1
        if j.get("category_ranking") == "bottom_tier":
            n_bottom_tier += 1
        if j.get("switch_likelihood") == "switched_to":
            n_switched_to += 1
        if j.get("switch_likelihood") == "switched_from":
            n_switched_from += 1

    mean_l1_score = sum_l1_score / max(n_comparison_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score
    final_score = max(0.0, min(10.0, raw_score))

    if final_score >= 7.5:
        base_verdict_by_score = "Category Leader"
    elif final_score >= 5.5:
        base_verdict_by_score = "Strong Competitor"
    elif final_score >= 3.5:
        base_verdict_by_score = "Average Competitor"
    else:
        base_verdict_by_score = "Weak Competitor"

    override_applied = "none"
    verdict = base_verdict_by_score
    churn_warning = None

    if n_best_in_category >= 2 and verdict in ("Average Competitor", "Weak Competitor"):
        override_applied = "best_in_category_min_strong"
        verdict = "Strong Competitor"
    elif n_switched_from >= 2:
        churn_warning = "Customers switching to competitors"
    elif n_bottom_tier >= 2 and verdict in ("Category Leader", "Strong Competitor"):
        override_applied = "bottom_tier_max_average"
        verdict = "Average Competitor"
    elif mean_l1_score < -3:
        override_applied = "low_mean_score"
        verdict = "Weak Competitor"

    result = {
        "N_COMPARISON_REVIEWS": n_comparison_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_BEST_IN_CATEGORY": n_best_in_category,
        "N_BOTTOM_TIER": n_bottom_tier,
        "N_SWITCHED_TO": n_switched_to,
        "N_SWITCHED_FROM": n_switched_from,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_comparison_reviews": n_comparison_reviews,
    }

    if churn_warning:
        result["churn_warning"] = churn_warning

    return result
