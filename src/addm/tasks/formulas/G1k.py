"""
Ground Truth computation for G1k (Hygiene - Complex).

Implements the formula from data/tasks/yelp/G1k_prompt.txt.
Complex formula with credibility weighting and resolution effects.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_RISK = 2.0

# Credibility weights
CREDIBILITY_WEIGHTS = {
    "firsthand": 1.0,
    "secondhand": 0.5,
    "hypothetical": 0.1,
}

# Severity scores
SEVERITY_SCORES = {
    "none": 0,
    "minor": 1,
    "moderate": 3,
    "severe": 8,
}

# Issue type multipliers
ISSUE_TYPE_MULTIPLIERS = {
    "food_handling": 1.5,
    "cleanliness": 1.0,
    "pest": 2.0,
    "illness": 2.5,
    "none": 0,
}

# Location modifiers
LOCATION_MODIFIERS = {
    "kitchen": 1.0,
    "food": 0.5,
    "dining_area": 0.0,
    "restroom": 0.5,
    "exterior": -0.5,
    "unknown": 0.0,
}

# Resolution modifiers
RESOLUTION_MODIFIERS = {
    "resolved": -2.0,
    "partial": -1.0,
    "unresolved": 1.0,
    "not_reported": 0.0,
    "unknown": 0.0,
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
    issue_type = judgment.get("issue_type", "none")
    issue_severity = judgment.get("issue_severity", "none")
    location = judgment.get("location", "unknown")
    resolution = judgment.get("resolution", "unknown")

    # Basic weights and scores
    credibility_weight = CREDIBILITY_WEIGHTS.get(account_type, 0.1)
    severity_score = SEVERITY_SCORES.get(issue_severity, 0)
    issue_type_mult = ISSUE_TYPE_MULTIPLIERS.get(issue_type, 0)
    location_modifier = LOCATION_MODIFIERS.get(location, 0.0)
    resolution_modifier = RESOLUTION_MODIFIERS.get(resolution, 0.0)

    # L1_REVIEW_RISK
    l1_review_risk = (severity_score * issue_type_mult * credibility_weight) + location_modifier + resolution_modifier

    # UNRESOLVED_SEVERE (interaction effect)
    unresolved_severe = 3.0 if (resolution == "unresolved" and issue_severity == "severe") else 0.0

    # L1_TOTAL_RISK
    l1_total_risk = l1_review_risk + unresolved_severe

    return {
        "credibility_weight": credibility_weight,
        "severity_score": severity_score,
        "issue_type_mult": issue_type_mult,
        "location_modifier": location_modifier,
        "resolution_modifier": resolution_modifier,
        "l1_review_risk": round(l1_review_risk, 2),
        "unresolved_severe": unresolved_severe,
        "l1_total_risk": round(l1_total_risk, 2),
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
    # Filter to hygiene-related judgments only
    hygiene_judgments = [j for j in judgments if j.get("is_hygiene_related", False)]
    n_hygiene_reviews = len(hygiene_judgments)

    # CONFIDENCE_LEVEL
    if n_hygiene_reviews == 0:
        confidence_level = "none"
    elif n_hygiene_reviews <= 2:
        confidence_level = "low"
    elif n_hygiene_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    # Compute L1 for each judgment and aggregate
    sum_l1_risk = 0.0
    sum_credibility = 0.0
    n_severe = 0
    n_unresolved_issues = 0
    n_illness_reports = 0
    n_pest_reports = 0

    for j in hygiene_judgments:
        l1 = compute_l1_complex(j)

        sum_l1_risk += l1["l1_total_risk"]
        sum_credibility += l1["credibility_weight"]

        account_type = j.get("account_type", "hypothetical")
        issue_type = j.get("issue_type", "none")
        issue_severity = j.get("issue_severity", "none")
        resolution = j.get("resolution", "unknown")

        is_firsthand = account_type == "firsthand"

        # Count severe incidents (firsthand only)
        if is_firsthand and issue_severity == "severe":
            n_severe += 1

        # Count unresolved issues
        if resolution == "unresolved" and issue_severity != "none":
            n_unresolved_issues += 1

        # Count illness reports (firsthand only)
        if is_firsthand and issue_type == "illness":
            n_illness_reports += 1

        # Count pest reports (firsthand only)
        if is_firsthand and issue_type == "pest":
            n_pest_reports += 1

    # Weighted aggregation
    weighted_mean_risk = sum_l1_risk / max(sum_credibility, 1)

    # Formulas
    adjusted_risk = weighted_mean_risk
    raw_risk = BASE_RISK + adjusted_risk
    final_risk_score = max(0.0, min(20.0, raw_risk))

    # Decision Policy: Base verdict by score
    if final_risk_score < 4.0:
        base_verdict_by_score = "Low Risk"
    elif final_risk_score < 8.0:
        base_verdict_by_score = "Moderate Risk"
    elif final_risk_score < 12.0:
        base_verdict_by_score = "High Risk"
    else:
        base_verdict_by_score = "Critical Risk"

    # Decision Policy: Overrides
    override_applied = "none"
    verdict = base_verdict_by_score

    # Override 1: N_ILLNESS_REPORTS >= 1 => Critical Risk
    if n_illness_reports >= 1:
        override_applied = "illness_report"
        verdict = "Critical Risk"
    # Override 2: N_PEST_REPORTS >= 1 => min High Risk
    elif n_pest_reports >= 1 and verdict in ("Low Risk", "Moderate Risk"):
        override_applied = "pest_report_min_high"
        verdict = "High Risk"
    # Override 3: N_UNRESOLVED_ISSUES >= 2 => min High Risk
    elif n_unresolved_issues >= 2 and verdict in ("Low Risk", "Moderate Risk"):
        override_applied = "ongoing_problems"
        verdict = "High Risk"
    # Override 4: N_SEVERE >= 2 => Critical Risk
    elif n_severe >= 2:
        override_applied = "multiple_severe"
        verdict = "Critical Risk"

    return {
        # L2 Aggregates
        "N_HYGIENE_REVIEWS": n_hygiene_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_SEVERE": n_severe,
        "N_UNRESOLVED_ISSUES": n_unresolved_issues,
        "N_ILLNESS_REPORTS": n_illness_reports,
        "N_PEST_REPORTS": n_pest_reports,
        # Weighted aggregation
        "SUM_L1_RISK": round(sum_l1_risk, 2),
        "SUM_CREDIBILITY": round(sum_credibility, 2),
        "WEIGHTED_MEAN_RISK": round(weighted_mean_risk, 3),
        # Formula results
        "BASE_RISK": BASE_RISK,
        "ADJUSTED_RISK": round(adjusted_risk, 3),
        "RAW_RISK": round(raw_risk, 2),
        "FINAL_RISK_SCORE": round(final_risk_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_hygiene_reviews": n_hygiene_reviews,
    }
