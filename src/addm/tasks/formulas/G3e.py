"""
Ground Truth computation for G3e (Hidden Costs - Simple).

Implements the formula from data/tasks/yelp/G3e_prompt.txt.
Simple formula without credibility weighting.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 7.0  # Assume transparent until proven otherwise


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_hidden_issue(judgment: Dict[str, Any]) -> bool:
    """
    HIDDEN_COST_ISSUE = true iff ALL:
      - COST_TYPE != none
      - DISCLOSURE_QUALITY in {buried, not_disclosed}
    """
    cost_type = judgment.get("cost_type", "none")
    disclosure_quality = judgment.get("disclosure_quality", "not_mentioned")

    if cost_type == "none":
        return False

    return disclosure_quality in ("buried", "not_disclosed")


def compute_l1_transparent(judgment: Dict[str, Any]) -> bool:
    """
    TRANSPARENT_PRICING = true iff ANY:
      - Explicitly states no hidden fees
      - DISCLOSURE_QUALITY = clear AND COST_TYPE != none
      - Review praises pricing transparency
    """
    cost_type = judgment.get("cost_type", "none")
    disclosure_quality = judgment.get("disclosure_quality", "not_mentioned")
    praises_transparency = judgment.get("praises_transparency", False)

    if praises_transparency:
        return True

    if disclosure_quality == "clear" and cost_type != "none":
        return True

    return False


def compute_l1_deceptive(judgment: Dict[str, Any]) -> bool:
    """
    DECEPTIVE_PRACTICE = true iff ALL:
      - DISCLOSURE_QUALITY = not_disclosed
      - SURPRISE_LEVEL in {shocked, outraged}
    """
    disclosure_quality = judgment.get("disclosure_quality", "not_mentioned")
    surprise_level = judgment.get("surprise_level", "not_mentioned")

    return disclosure_quality == "not_disclosed" and surprise_level in ("shocked", "outraged")


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
    cost_judgments = [j for j in judgments if j.get("is_cost_related", False)]
    n_cost_reviews = len(cost_judgments)

    if n_cost_reviews == 0:
        confidence_level = "none"
    elif n_cost_reviews <= 2:
        confidence_level = "low"
    elif n_cost_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    n_hidden_issues = 0
    n_transparent = 0
    n_deceptive = 0
    n_shocked = 0
    n_clear_disclosure = 0

    for j in cost_judgments:
        if compute_l1_hidden_issue(j):
            n_hidden_issues += 1
        if compute_l1_transparent(j):
            n_transparent += 1
        if compute_l1_deceptive(j):
            n_deceptive += 1
        if j.get("surprise_level") in ("shocked", "outraged"):
            n_shocked += 1
        if j.get("disclosure_quality") == "clear":
            n_clear_disclosure += 1

    # Formulas
    positive_score = (n_transparent * 1) + (n_clear_disclosure * 0.5)
    negative_score = (n_hidden_issues * 1.5) + (n_deceptive * 3) + (n_shocked * 1)

    raw_score = BASE_SCORE + positive_score - negative_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
    if final_score >= 8.0:
        base_verdict_by_score = "Transparent"
    elif final_score >= 6.0:
        base_verdict_by_score = "Mostly Transparent"
    elif final_score >= 4.0:
        base_verdict_by_score = "Some Hidden Costs"
    else:
        base_verdict_by_score = "Problematic"

    override_applied = "none"
    verdict = base_verdict_by_score

    if n_deceptive >= 2:
        override_applied = "deceptive_pattern"
        verdict = "Problematic"
    elif n_deceptive >= 1 and verdict in ("Transparent", "Mostly Transparent"):
        override_applied = "deceptive_max_some_hidden"
        verdict = "Some Hidden Costs"
    elif n_hidden_issues == 0 and n_cost_reviews >= 3 and verdict in ("Some Hidden Costs", "Problematic"):
        override_applied = "no_issues_min_mostly"
        verdict = "Mostly Transparent"

    return {
        "N_COST_REVIEWS": n_cost_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_HIDDEN_ISSUES": n_hidden_issues,
        "N_TRANSPARENT": n_transparent,
        "N_DECEPTIVE": n_deceptive,
        "N_SHOCKED": n_shocked,
        "N_CLEAR_DISCLOSURE": n_clear_disclosure,
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_cost_reviews": n_cost_reviews,
    }
