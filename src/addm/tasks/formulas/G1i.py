"""
Ground Truth computation for G1i (Hygiene - Simple).

Implements the formula from data/tasks/yelp/G1i_prompt.txt.
Basic scoring without weighting.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_RISK = 2.0


# =============================================================================
# Helpers
# =============================================================================


def derive_l1(judgment: Dict[str, Any]) -> Dict[str, bool]:
    """
    Derive L1 composites from L0 primitives.

    L1 composites (from prompt):
    - FIRSTHAND_ISSUE = true iff ALL:
        ACCOUNT_TYPE = firsthand
        ISSUE_TYPE != none

    - HEALTH_RISK = true iff ALL:
        ACCOUNT_TYPE = firsthand
        ISSUE_SEVERITY = severe

    - PEST_SIGHTING = true iff ALL:
        ISSUE_TYPE = pest
        ACCOUNT_TYPE = firsthand
    """
    account_type = judgment.get("account_type", "hypothetical")
    issue_type = judgment.get("issue_type", "none")
    issue_severity = judgment.get("issue_severity", "none")

    is_firsthand = account_type == "firsthand"

    # FIRSTHAND_ISSUE
    firsthand_issue = is_firsthand and issue_type != "none"

    # HEALTH_RISK
    health_risk = is_firsthand and issue_severity == "severe"

    # PEST_SIGHTING
    pest_sighting = issue_type == "pest" and is_firsthand

    return {
        "FIRSTHAND_ISSUE": firsthand_issue,
        "HEALTH_RISK": health_risk,
        "PEST_SIGHTING": pest_sighting,
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

    # Compute L2 aggregates
    n_minor = 0
    n_moderate = 0
    n_severe = 0
    n_pest_sightings = 0
    n_illness_reports = 0

    for j in hygiene_judgments:
        account_type = j.get("account_type", "hypothetical")
        issue_type = j.get("issue_type", "none")
        issue_severity = j.get("issue_severity", "none")

        is_firsthand = account_type == "firsthand"

        # Count by severity (firsthand only)
        if is_firsthand:
            if issue_severity == "minor":
                n_minor += 1
            elif issue_severity == "moderate":
                n_moderate += 1
            elif issue_severity == "severe":
                n_severe += 1

            # Count pest sightings
            if issue_type == "pest":
                n_pest_sightings += 1

            # Count illness reports
            if issue_type == "illness":
                n_illness_reports += 1

    # Formulas
    issue_score = (n_minor * 1) + (n_moderate * 3) + (n_severe * 8) + (n_pest_sightings * 5) + (n_illness_reports * 10)
    raw_risk = BASE_RISK + issue_score
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
    # Override 2: N_PEST_SIGHTINGS >= 1 => min High Risk
    elif n_pest_sightings >= 1 and verdict in ("Low Risk", "Moderate Risk"):
        override_applied = "pest_sighting_min_high"
        verdict = "High Risk"
    # Override 3: N_SEVERE >= 2 => Critical Risk
    elif n_severe >= 2:
        override_applied = "multiple_severe"
        verdict = "Critical Risk"

    return {
        # L2 Aggregates
        "N_HYGIENE_REVIEWS": n_hygiene_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_MINOR": n_minor,
        "N_MODERATE": n_moderate,
        "N_SEVERE": n_severe,
        "N_PEST_SIGHTINGS": n_pest_sightings,
        "N_ILLNESS_REPORTS": n_illness_reports,
        # Formula results
        "ISSUE_SCORE": issue_score,
        "BASE_RISK": BASE_RISK,
        "RAW_RISK": round(raw_risk, 2),
        "FINAL_RISK_SCORE": round(final_risk_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_hygiene_reviews": n_hygiene_reviews,
    }
