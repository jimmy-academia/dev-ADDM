"""
Ground Truth computation for G6i (Loyalty - Simple).

Implements the formula from data/tasks/yelp/G6i_prompt.txt.
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


def compute_l1_positive_loyalty(judgment: Dict[str, Any]) -> bool:
    return_intention = judgment.get("return_intention", "not_mentioned")
    recommendation_likelihood = judgment.get("recommendation_likelihood", "not_mentioned")
    visit_frequency = judgment.get("visit_frequency", "not_mentioned")
    relationship_depth = judgment.get("relationship_depth", "not_mentioned")
    advocacy_behavior = judgment.get("advocacy_behavior", "not_mentioned")

    if return_intention in ("definitely_returning", "likely_returning"):
        return True
    if recommendation_likelihood in ("highly_recommend", "recommend"):
        return True
    if visit_frequency == "regular":
        return True
    if relationship_depth in ("personal_connection", "recognized"):
        return True
    if advocacy_behavior in ("brings_others", "shares_socially", "defends"):
        return True
    return False


def compute_l1_negative_loyalty(judgment: Dict[str, Any]) -> bool:
    return_intention = judgment.get("return_intention", "not_mentioned")
    recommendation_likelihood = judgment.get("recommendation_likelihood", "not_mentioned")
    visit_frequency = judgment.get("visit_frequency", "not_mentioned")
    relationship_depth = judgment.get("relationship_depth", "not_mentioned")

    if return_intention in ("unlikely_returning", "never_returning"):
        return True
    if recommendation_likelihood in ("discourage", "strongly_discourage"):
        return True
    if relationship_depth == "anonymous" and visit_frequency in ("regular", "occasional"):
        return True
    return False


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    loyalty_judgments = [j for j in judgments if j.get("is_loyalty_related", False)]
    n_loyalty_reviews = len(loyalty_judgments)

    if n_loyalty_reviews == 0:
        confidence_level = "none"
    elif n_loyalty_reviews <= 2:
        confidence_level = "low"
    elif n_loyalty_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    n_positive = 0
    n_negative = 0
    n_regulars = 0
    n_never_returning = 0
    n_advocates = 0

    for j in loyalty_judgments:
        if compute_l1_positive_loyalty(j):
            n_positive += 1
        if compute_l1_negative_loyalty(j):
            n_negative += 1

        if j.get("visit_frequency") == "regular":
            n_regulars += 1
        if j.get("return_intention") == "never_returning":
            n_never_returning += 1
        if j.get("advocacy_behavior") in ("brings_others", "shares_socially", "defends"):
            n_advocates += 1

    # Formulas
    positive_score = (n_positive * 1.5) + (n_regulars * 1.0) + (n_advocates * 1.0)
    negative_score = (n_negative * 1.5) + (n_never_returning * 2.0)

    raw_score = BASE_SCORE + positive_score - negative_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
    if final_score >= 7.5:
        base_verdict_by_score = "High Loyalty"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good Loyalty"
    elif final_score >= 3.5:
        base_verdict_by_score = "Moderate Loyalty"
    else:
        base_verdict_by_score = "Low Loyalty"

    override_applied = "none"
    verdict = base_verdict_by_score
    advocacy_highlight = None

    if n_regulars >= 3 and verdict in ("Moderate Loyalty", "Low Loyalty"):
        override_applied = "regulars_min_good"
        verdict = "Good Loyalty"
    elif n_never_returning >= 2 and verdict in ("High Loyalty", "Good Loyalty"):
        override_applied = "never_returning_max_moderate"
        verdict = "Moderate Loyalty"

    if n_advocates >= 2:
        advocacy_highlight = "Active customer advocates noted"

    result = {
        "N_LOYALTY_REVIEWS": n_loyalty_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "N_REGULARS": n_regulars,
        "N_NEVER_RETURNING": n_never_returning,
        "N_ADVOCATES": n_advocates,
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_loyalty_reviews": n_loyalty_reviews,
    }

    if advocacy_highlight:
        result["advocacy_highlight"] = advocacy_highlight

    return result
