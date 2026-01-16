"""
Ground Truth computation for G1g (Dietary Accommodation - Complex).

Implements the formula from data/tasks/yelp/G1g_prompt.txt.
Complex formula with credibility weighting and interaction effects.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

# Credibility weights
CREDIBILITY_WEIGHTS = {
    "firsthand": 1.0,
    "secondhand": 0.5,
    "hypothetical": 0.1,
}

# Outcome scores
OUTCOME_SCORES = {
    "success": 3,
    "partial": 1,
    "failure": -3,
    "not_attempted": 0,
}

# Staff modifiers
STAFF_MODIFIERS = {
    "knowledgeable": 1.5,
    "uncertain": 0.0,
    "uninformed": -1.5,
    "none": 0.0,
}

# Menu modifiers
MENU_MODIFIERS = {
    "clear": 1.0,
    "partial": 0.0,
    "unclear": -0.5,
    "unknown": 0.0,
}

# Consequence penalties
CONSEQUENCE_PENALTIES = {
    "none": 0.0,
    "inconvenience": -1.0,
    "health_impact": -4.0,
    "serious": -8.0,
}


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute L1 composites using complex formula.

    Returns dict with all L1 computed values.
    """
    account_type = judgment.get("account_type", "hypothetical")
    accommodation_outcome = judgment.get("accommodation_outcome", "not_attempted")
    staff_knowledge = judgment.get("staff_knowledge", "none")
    menu_clarity = judgment.get("menu_clarity", "unknown")
    consequence_severity = judgment.get("consequence_severity", "none")

    # Basic weights and scores
    credibility_weight = CREDIBILITY_WEIGHTS.get(account_type, 0.1)
    outcome_score = OUTCOME_SCORES.get(accommodation_outcome, 0)
    staff_modifier = STAFF_MODIFIERS.get(staff_knowledge, 0.0)
    menu_modifier = MENU_MODIFIERS.get(menu_clarity, 0.0)
    consequence_penalty = CONSEQUENCE_PENALTIES.get(consequence_severity, 0.0)

    # L1_REVIEW_SCORE
    l1_review_score = (outcome_score * credibility_weight) + staff_modifier + menu_modifier + consequence_penalty

    # UNINFORMED_FAILURE (interaction effect)
    uninformed_failure = -2.0 if (staff_knowledge == "uninformed" and accommodation_outcome == "failure") else 0.0

    # HEALTH_INCIDENT
    is_firsthand = account_type == "firsthand"
    health_incident = is_firsthand and consequence_severity in ("health_impact", "serious")

    # L1_TOTAL_SCORE
    l1_total_score = l1_review_score + uninformed_failure

    return {
        "credibility_weight": credibility_weight,
        "outcome_score": outcome_score,
        "staff_modifier": staff_modifier,
        "menu_modifier": menu_modifier,
        "consequence_penalty": consequence_penalty,
        "l1_review_score": round(l1_review_score, 2),
        "uninformed_failure": uninformed_failure,
        "health_incident": health_incident,
        "l1_total_score": round(l1_total_score, 2),
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

    # Compute L1 for each judgment and aggregate
    sum_l1_score = 0.0
    sum_credibility = 0.0
    n_success = 0
    n_failure = 0
    n_health_incidents = 0
    n_uninformed_failures = 0

    for j in dietary_judgments:
        l1 = compute_l1_complex(j)

        sum_l1_score += l1["l1_total_score"]
        sum_credibility += l1["credibility_weight"]

        account_type = j.get("account_type", "hypothetical")
        accommodation_outcome = j.get("accommodation_outcome", "not_attempted")

        if account_type == "firsthand":
            if accommodation_outcome == "success":
                n_success += 1
            elif accommodation_outcome == "failure":
                n_failure += 1

        if l1["health_incident"]:
            n_health_incidents += 1

        if l1["uninformed_failure"] < 0:
            n_uninformed_failures += 1

    # Weighted aggregation
    weighted_mean_score = sum_l1_score / max(sum_credibility, 1)

    # Formulas
    adjusted_score = weighted_mean_score
    raw_score = BASE_SCORE + adjusted_score
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

    # Override 1: N_HEALTH_INCIDENTS >= 1 => Very Poor
    if n_health_incidents >= 1:
        override_applied = "health_risk"
        verdict = "Very Poor"
    # Override 2: N_UNINFORMED_FAILURES >= 2 => max Poor
    elif n_uninformed_failures >= 2 and verdict in ("Excellent", "Adequate"):
        override_applied = "systemic_staff_issue"
        verdict = "Poor"
    # Override 3: N_FAILURE >= 3 => Very Poor
    elif n_failure >= 3:
        override_applied = "pattern_of_failures"
        verdict = "Very Poor"
    # Override 4: WEIGHTED_MEAN_SCORE >= 3.0 AND N_SUCCESS >= 5 => min Adequate
    elif weighted_mean_score >= 3.0 and n_success >= 5 and verdict in ("Poor", "Very Poor"):
        override_applied = "high_success_min_adequate"
        verdict = "Adequate"

    return {
        # L2 Aggregates
        "N_DIETARY_REVIEWS": n_dietary_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_SUCCESS": n_success,
        "N_FAILURE": n_failure,
        "N_HEALTH_INCIDENTS": n_health_incidents,
        "N_UNINFORMED_FAILURES": n_uninformed_failures,
        # Weighted aggregation
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "SUM_CREDIBILITY": round(sum_credibility, 2),
        "WEIGHTED_MEAN_SCORE": round(weighted_mean_score, 3),
        # Formula results
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_dietary_reviews": n_dietary_reviews,
    }
