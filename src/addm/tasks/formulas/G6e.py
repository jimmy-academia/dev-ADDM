"""
Ground Truth computation for G6e (Comparison - Simple).

Implements the formula from data/tasks/yelp/G6e_prompt.txt.
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


def compute_l1_positive_comparison(judgment: Dict[str, Any]) -> bool:
    competitor_mention = judgment.get("competitor_mention", "not_mentioned")
    category_ranking = judgment.get("category_ranking", "not_mentioned")
    price_comparison = judgment.get("price_comparison", "not_mentioned")
    quality_comparison = judgment.get("quality_comparison", "not_mentioned")

    if competitor_mention == "favorable":
        return True
    if category_ranking in ("best_in_category", "top_tier"):
        return True
    if price_comparison in ("best_value", "underpriced"):
        return True
    if quality_comparison == "superior":
        return True
    return False


def compute_l1_negative_comparison(judgment: Dict[str, Any]) -> bool:
    competitor_mention = judgment.get("competitor_mention", "not_mentioned")
    category_ranking = judgment.get("category_ranking", "not_mentioned")
    price_comparison = judgment.get("price_comparison", "not_mentioned")
    quality_comparison = judgment.get("quality_comparison", "not_mentioned")

    if competitor_mention == "unfavorable":
        return True
    if category_ranking == "bottom_tier":
        return True
    if price_comparison == "overpriced":
        return True
    if quality_comparison == "inferior":
        return True
    return False


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

    n_positive = 0
    n_negative = 0
    n_best_in_category = 0
    n_bottom_tier = 0
    n_overpriced = 0

    for j in comparison_judgments:
        if compute_l1_positive_comparison(j):
            n_positive += 1
        if compute_l1_negative_comparison(j):
            n_negative += 1

        if j.get("category_ranking") == "best_in_category":
            n_best_in_category += 1
        if j.get("category_ranking") == "bottom_tier":
            n_bottom_tier += 1
        if j.get("price_comparison") == "overpriced":
            n_overpriced += 1

    # Formulas
    positive_score = (n_positive * 1.5) + (n_best_in_category * 1.5)
    negative_score = (n_negative * 1.5) + (n_bottom_tier * 1.0) + (n_overpriced * 0.5)

    raw_score = BASE_SCORE + positive_score - negative_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
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
    value_warning = None

    if n_best_in_category >= 2 and verdict in ("Average Competitor", "Weak Competitor"):
        override_applied = "best_in_category_min_strong"
        verdict = "Strong Competitor"
    elif n_bottom_tier >= 2 and verdict in ("Category Leader", "Strong Competitor"):
        override_applied = "bottom_tier_max_average"
        verdict = "Average Competitor"

    if n_overpriced >= 3:
        value_warning = "Multiple price concerns noted"

    result = {
        "N_COMPARISON_REVIEWS": n_comparison_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "N_BEST_IN_CATEGORY": n_best_in_category,
        "N_BOTTOM_TIER": n_bottom_tier,
        "N_OVERPRICED": n_overpriced,
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_comparison_reviews": n_comparison_reviews,
    }

    if value_warning:
        result["value_warning"] = value_warning

    return result
