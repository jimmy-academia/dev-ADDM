"""
Ground Truth computation for G6k (Loyalty - Complex).

Implements the formula from data/tasks/yelp/G6k_prompt.txt.
Complex formula with weighted loyalty factors.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

RETURN_SCORES = {"definitely_returning": 4.0, "likely_returning": 2.5, "uncertain": 0, "unlikely_returning": -2.0, "never_returning": -4.0, "not_mentioned": 0}
RECOMMENDATION_SCORES = {"highly_recommend": 4.0, "recommend": 2.5, "conditional_recommend": 1.0, "neutral": 0, "discourage": -2.5, "strongly_discourage": -4.0, "not_mentioned": 0}
FREQUENCY_SCORES = {"regular": 3.0, "occasional": 1.0, "rare": 0, "first_time": 0, "not_mentioned": 0}
RELATIONSHIP_SCORES = {"personal_connection": 3.5, "recognized": 2.0, "familiar": 1.0, "anonymous": -0.5, "not_mentioned": 0}
ADVOCACY_SCORES = {"brings_others": 3.5, "shares_socially": 2.5, "defends": 3.0, "passive": 0, "not_mentioned": 0}
TRIGGER_SCORES = {"multiple": 2.0, "food": 1.5, "service": 1.5, "atmosphere": 1.0, "value": 1.0, "convenience": 0.5, "habit": 0.5, "not_mentioned": 0}
TENURE_SCORES = {"longtime": 3.0, "established": 2.0, "new_regular": 1.5, "trying_out": 0, "not_mentioned": 0}


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    return_intention = judgment.get("return_intention", "not_mentioned")
    recommendation_likelihood = judgment.get("recommendation_likelihood", "not_mentioned")
    visit_frequency = judgment.get("visit_frequency", "not_mentioned")
    relationship_depth = judgment.get("relationship_depth", "not_mentioned")
    advocacy_behavior = judgment.get("advocacy_behavior", "not_mentioned")
    loyalty_trigger = judgment.get("loyalty_trigger", "not_mentioned")
    tenure = judgment.get("tenure", "not_mentioned")

    return_score = RETURN_SCORES.get(return_intention, 0)
    recommendation_score = RECOMMENDATION_SCORES.get(recommendation_likelihood, 0)
    frequency_score = FREQUENCY_SCORES.get(visit_frequency, 0)
    relationship_score = RELATIONSHIP_SCORES.get(relationship_depth, 0)
    advocacy_score = ADVOCACY_SCORES.get(advocacy_behavior, 0)
    trigger_score = TRIGGER_SCORES.get(loyalty_trigger, 0)
    tenure_score = TENURE_SCORES.get(tenure, 0)

    l1_loyalty_score = (
        return_score + recommendation_score + frequency_score +
        relationship_score + advocacy_score + trigger_score + tenure_score
    )

    # Loyal advocate bonus
    loyal_advocate_bonus = 2.0 if (visit_frequency == "regular" and
                                    advocacy_behavior in ("brings_others", "shares_socially", "defends")) else 0.0

    # Lost regular penalty
    lost_regular_penalty = -3.0 if (visit_frequency == "regular" and
                                     return_intention in ("unlikely_returning", "never_returning")) else 0.0

    l1_total_score = l1_loyalty_score + loyal_advocate_bonus + lost_regular_penalty

    return {
        "return_score": return_score,
        "recommendation_score": recommendation_score,
        "frequency_score": frequency_score,
        "relationship_score": relationship_score,
        "advocacy_score": advocacy_score,
        "trigger_score": trigger_score,
        "tenure_score": tenure_score,
        "l1_loyalty_score": round(l1_loyalty_score, 2),
        "loyal_advocate_bonus": loyal_advocate_bonus,
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

    sum_l1_score = 0.0
    n_regulars = 0
    n_longtime = 0
    n_advocates = 0
    n_never_returning = 0
    n_lost_regulars = 0

    for j in loyalty_judgments:
        l1 = compute_l1_complex(j)
        sum_l1_score += l1["l1_total_score"]

        if j.get("visit_frequency") == "regular":
            n_regulars += 1
        if j.get("tenure") == "longtime":
            n_longtime += 1
        if j.get("advocacy_behavior") in ("brings_others", "shares_socially", "defends"):
            n_advocates += 1
        if j.get("return_intention") == "never_returning":
            n_never_returning += 1
        if l1["lost_regular_penalty"] < 0:
            n_lost_regulars += 1

    mean_l1_score = sum_l1_score / max(n_loyalty_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score
    final_score = max(0.0, min(10.0, raw_score))

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
    tenure_highlight = None
    churn_warning = None

    if n_longtime >= 2 and verdict in ("Moderate Loyalty", "Low Loyalty"):
        override_applied = "longtime_min_good"
        verdict = "Good Loyalty"
        tenure_highlight = "Long-term loyal customers"
    elif n_lost_regulars >= 1:
        churn_warning = "Regular customers at risk of leaving"
    elif n_advocates >= 2 and n_never_returning == 0 and verdict in ("Moderate Loyalty", "Low Loyalty"):
        override_applied = "advocates_min_good"
        verdict = "Good Loyalty"
    elif mean_l1_score < -3:
        override_applied = "low_mean_score"
        verdict = "Low Loyalty"

    result = {
        "N_LOYALTY_REVIEWS": n_loyalty_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_REGULARS": n_regulars,
        "N_LONGTIME": n_longtime,
        "N_ADVOCATES": n_advocates,
        "N_NEVER_RETURNING": n_never_returning,
        "N_LOST_REGULARS": n_lost_regulars,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_loyalty_reviews": n_loyalty_reviews,
    }

    if tenure_highlight:
        result["tenure_highlight"] = tenure_highlight
    if churn_warning:
        result["churn_warning"] = churn_warning

    return result
