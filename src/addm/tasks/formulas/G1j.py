"""
Ground Truth computation for G1j (Hygiene + L1.5 Issue Type Grouping).

Implements the formula from data/tasks/yelp/G1j_prompt.txt.
Extends G1i with L1.5 issue type grouping and hygiene pattern bonus.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_RISK = 2.0

# L1.5 Hygiene Pattern Bonus values
HYGIENE_PATTERN_BONUS = {
    "illness_concern": 5.0,
    "pest_issue": 4.0,
    "systemic": 3.0,
    "isolated_issue": 1.0,
    "no_pattern": 0.0,
}


# =============================================================================
# Helpers
# =============================================================================


def get_issue_type(judgment: Dict[str, Any]) -> str:
    """
    Extract issue type from judgment.
    Returns: food_handling, cleanliness, pest, illness, or none
    """
    issue_type = judgment.get("issue_type", "none")
    return issue_type.lower() if issue_type else "none"


def derive_l1(judgment: Dict[str, Any]) -> Dict[str, bool]:
    """
    Derive L1 composites from L0 primitives.
    """
    account_type = judgment.get("account_type", "hypothetical")
    issue_type = judgment.get("issue_type", "none")
    issue_severity = judgment.get("issue_severity", "none")

    is_firsthand = account_type == "firsthand"

    firsthand_issue = is_firsthand and issue_type != "none"
    health_risk = is_firsthand and issue_severity == "severe"
    pest_sighting = issue_type == "pest" and is_firsthand

    return {
        "FIRSTHAND_ISSUE": firsthand_issue,
        "HEALTH_RISK": health_risk,
        "PEST_SIGHTING": pest_sighting,
    }


def compute_l15_buckets(hygiene_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute L1.5 issue type buckets.

    Groups reviews by issue type and computes:
    - n_reviews, n_severe, n_moderate, max_severity for each bucket
    - worst_issue_type, n_issue_types_reported
    - hygiene_pattern
    """
    buckets = {
        "food_handling": {"n_reviews": 0, "n_severe": 0, "n_moderate": 0, "severities": []},
        "cleanliness": {"n_reviews": 0, "n_severe": 0, "n_moderate": 0, "severities": []},
        "pest": {"n_reviews": 0, "n_severe": 0, "n_moderate": 0, "severities": []},
        "illness": {"n_reviews": 0, "n_severe": 0, "n_moderate": 0, "severities": []},
    }

    severity_order = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}

    for j in hygiene_judgments:
        issue_type = get_issue_type(j)
        if issue_type == "none" or issue_type not in buckets:
            continue

        buckets[issue_type]["n_reviews"] += 1

        account_type = j.get("account_type", "hypothetical")
        issue_severity = j.get("issue_severity", "none")

        if account_type == "firsthand":
            if issue_severity == "severe":
                buckets[issue_type]["n_severe"] += 1
            elif issue_severity == "moderate":
                buckets[issue_type]["n_moderate"] += 1

            buckets[issue_type]["severities"].append(issue_severity)

    # Compute max severity for each bucket
    for key, bucket in buckets.items():
        if bucket["severities"]:
            bucket["max_severity"] = max(bucket["severities"], key=lambda s: severity_order.get(s, 0))
        else:
            bucket["max_severity"] = "none"
        del bucket["severities"]

    # Find worst issue type (highest severity incident)
    buckets_with_issues = [(k, v) for k, v in buckets.items() if v["n_reviews"] > 0]

    if buckets_with_issues:
        worst_issue_type, _ = max(
            buckets_with_issues,
            key=lambda x: severity_order.get(x[1]["max_severity"], 0)
        )
    else:
        worst_issue_type = None

    n_issue_types_reported = len(buckets_with_issues)

    # Determine hygiene pattern
    has_illness = buckets["illness"]["n_reviews"] > 0
    has_pest = buckets["pest"]["n_reviews"] > 0

    if n_issue_types_reported >= 3:
        hygiene_pattern = "systemic"
    elif has_pest:
        hygiene_pattern = "pest_issue"
    elif has_illness:
        hygiene_pattern = "illness_concern"
    elif n_issue_types_reported >= 1:
        hygiene_pattern = "isolated_issue"
    else:
        hygiene_pattern = "no_pattern"

    return {
        "food_handling": buckets["food_handling"],
        "cleanliness": buckets["cleanliness"],
        "pest": buckets["pest"],
        "illness": buckets["illness"],
        "worst_issue_type": worst_issue_type,
        "n_issue_types_reported": n_issue_types_reported,
        "hygiene_pattern": hygiene_pattern,
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
    n_total_issues = 0

    for j in hygiene_judgments:
        account_type = j.get("account_type", "hypothetical")
        issue_severity = j.get("issue_severity", "none")

        is_firsthand = account_type == "firsthand"

        if is_firsthand:
            if issue_severity == "minor":
                n_minor += 1
            elif issue_severity == "moderate":
                n_moderate += 1
            elif issue_severity == "severe":
                n_severe += 1

    n_total_issues = n_minor + n_moderate + n_severe

    # L1.5 Issue Buckets
    l15_buckets = compute_l15_buckets(hygiene_judgments)
    hygiene_pattern = l15_buckets["hygiene_pattern"]
    hygiene_pattern_bonus = HYGIENE_PATTERN_BONUS.get(hygiene_pattern, 0.0)

    # Formulas (G1j: includes HYGIENE_PATTERN_BONUS)
    issue_score = (n_minor * 1) + (n_moderate * 3) + (n_severe * 8)

    # G1j formula: adds HYGIENE_PATTERN_BONUS
    raw_risk = BASE_RISK + issue_score + hygiene_pattern_bonus
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

    if n_severe >= 2:
        override_applied = "multiple_severe"
        verdict = "Critical Risk"

    return {
        # L1.5 Issue Buckets
        "L1_5_issue_buckets": l15_buckets,
        # L2 Aggregates
        "N_HYGIENE_REVIEWS": n_hygiene_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_MINOR": n_minor,
        "N_MODERATE": n_moderate,
        "N_SEVERE": n_severe,
        "N_TOTAL_ISSUES": n_total_issues,
        # Formula results
        "ISSUE_SCORE": issue_score,
        "BASE_RISK": BASE_RISK,
        "HYGIENE_PATTERN_BONUS": hygiene_pattern_bonus,
        "RAW_RISK": round(raw_risk, 2),
        "FINAL_RISK_SCORE": round(final_risk_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_hygiene_reviews": n_hygiene_reviews,
    }
