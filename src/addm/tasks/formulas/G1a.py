"""
Ground Truth computation for G1a (Allergy Safety).

Implements the formula from data/tasks/yelp/G1a_prompt.txt.
Takes extracted L0 judgments and computes L1 → L2 → Final verdict.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

CURRENT_YEAR = 2022
DEFAULT_INCIDENT_YEAR = 2020
BASE_RISK = 2.5

# Cuisine risk modifiers (match: highest)
CUISINE_MODIFIERS = {
    "Thai": 2.0,
    "Vietnamese": 1.8,
    "Chinese": 1.5,
    "Asian": 1.5,
    "Asian Fusion": 1.5,
    "Indian": 1.3,
    "Japanese": 1.2,
    "Korean": 1.2,
    "Mexican": 1.0,
    "Italian": 0.5,
    "American": 0.5,
    "American (Traditional)": 0.5,
    "American (New)": 0.5,
    "Pizza": 0.5,
}
DEFAULT_CUISINE_MODIFIER = 1.0


# =============================================================================
# Helpers
# =============================================================================


def get_cuisine_modifier(categories: str) -> float:
    """Get highest matching cuisine modifier from category string."""
    if not categories:
        return DEFAULT_CUISINE_MODIFIER

    cats = [c.strip() for c in categories.split(",")]
    max_modifier = DEFAULT_CUISINE_MODIFIER

    for cat in cats:
        # Direct match
        if cat in CUISINE_MODIFIERS:
            max_modifier = max(max_modifier, CUISINE_MODIFIERS[cat])
        # Partial match (e.g., "Thai Restaurant" contains "Thai")
        else:
            for cuisine, modifier in CUISINE_MODIFIERS.items():
                if cuisine.lower() in cat.lower():
                    max_modifier = max(max_modifier, modifier)

    return max_modifier


def derive_l1(judgment: Dict[str, Any]) -> Dict[str, bool]:
    """
    Derive L1 composites from L0 primitives.

    L1 composites (from prompt):
    - FALSE_ASSURANCE = true iff ALL:
        ACCOUNT_TYPE = firsthand
        INCIDENT_SEVERITY != none
        ASSURANCE_CLAIM = true

    - FIRSTHAND_INCIDENT = true iff ALL:
        ACCOUNT_TYPE = firsthand
        INCIDENT_SEVERITY in {mild, moderate, severe}
    """
    account_type = judgment.get("account_type", "hypothetical")
    incident_severity = judgment.get("incident_severity", "none")
    assurance_claim = judgment.get("assurance_claim", "false")

    # FALSE_ASSURANCE
    is_firsthand = account_type == "firsthand"
    has_incident = incident_severity in ("mild", "moderate", "severe")
    has_assurance = assurance_claim == "true"
    false_assurance = is_firsthand and has_incident and has_assurance

    # FIRSTHAND_INCIDENT
    firsthand_incident = is_firsthand and has_incident

    return {
        "FALSE_ASSURANCE": false_assurance,
        "FIRSTHAND_INCIDENT": firsthand_incident,
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
        restaurant_meta: Restaurant metadata with 'categories' field

    Returns:
        Dict with all computed values matching OUTPUT SCHEMA in prompt
    """
    categories = restaurant_meta.get("categories", "")

    # Filter to allergy-related judgments only
    allergy_judgments = [j for j in judgments if j.get("is_allergy_related", False)]
    n_allergy_reviews = len(allergy_judgments)

    # CONFIDENCE_LEVEL (from L2)
    if n_allergy_reviews == 0:
        confidence_level = "none"
    elif n_allergy_reviews <= 2:
        confidence_level = "low"
    elif n_allergy_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    # Derive L1 for each judgment and compute L2 aggregates
    n_mild = n_moderate = n_severe = 0
    n_false_assurance = 0
    incident_years: List[int] = []

    for j in allergy_judgments:
        l1 = derive_l1(j)

        # Count by severity (firsthand only)
        if l1["FIRSTHAND_INCIDENT"]:
            severity = j.get("incident_severity", "none")
            if severity == "mild":
                n_mild += 1
            elif severity == "moderate":
                n_moderate += 1
            elif severity == "severe":
                n_severe += 1

            # Track incident years for recency
            date_str = j.get("date", "")
            if date_str and len(date_str) >= 4:
                try:
                    year = int(date_str[:4])
                    incident_years.append(year)
                except ValueError:
                    pass

        # Count false assurance
        if l1["FALSE_ASSURANCE"]:
            n_false_assurance += 1

    n_total_incidents = n_mild + n_moderate + n_severe

    # MOST_RECENT_INCIDENT_YEAR
    if incident_years:
        most_recent_incident_year = max(incident_years)
    else:
        most_recent_incident_year = DEFAULT_INCIDENT_YEAR

    # Formulas (from prompt)
    incident_age = CURRENT_YEAR - most_recent_incident_year
    recency_decay = max(0.3, 1.0 - (incident_age * 0.15))
    cuisine_modifier = get_cuisine_modifier(categories)
    cuisine_impact = cuisine_modifier * 0.5
    incident_score = (n_mild * 2) + (n_moderate * 5) + (n_severe * 15)
    raw_risk = BASE_RISK + (incident_score * recency_decay) + cuisine_impact
    final_risk_score = max(0.0, min(20.0, raw_risk))  # clamp(0, 20)

    # Decision Policy: Base verdict by score
    if final_risk_score < 4.0:
        base_verdict_by_score = "Low Risk"
    elif final_risk_score < 8.0:
        base_verdict_by_score = "High Risk"
    else:
        base_verdict_by_score = "Critical Risk"

    # Decision Policy: Overrides
    override_applied = "none"
    verdict = base_verdict_by_score

    # Override 1: N_SEVERE >= 1 => Critical Risk
    if n_severe >= 1:
        override_applied = "severe_incident"
        verdict = "Critical Risk"
    # Override 2: N_FALSE_ASSURANCE >= 1 => min High Risk
    elif n_false_assurance >= 1 and verdict == "Low Risk":
        override_applied = "false_assurance_min_high"
        verdict = "High Risk"

    return {
        # L2 Aggregates
        "CURRENT_YEAR": CURRENT_YEAR,
        "N_ALLERGY_REVIEWS": n_allergy_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_MILD": n_mild,
        "N_MODERATE": n_moderate,
        "N_SEVERE": n_severe,
        "N_TOTAL_INCIDENTS": n_total_incidents,
        "N_FALSE_ASSURANCE": n_false_assurance,
        "MOST_RECENT_INCIDENT_YEAR": most_recent_incident_year,
        # Formula results
        "INCIDENT_AGE": incident_age,
        "RECENCY_DECAY": round(recency_decay, 3),
        "CUISINE_MODIFIER": cuisine_modifier,
        "INCIDENT_SCORE": incident_score,
        "BASE_RISK": BASE_RISK,
        "CUISINE_IMPACT": round(cuisine_impact, 3),
        "RAW_RISK": round(raw_risk, 3),
        "FINAL_RISK_SCORE": round(final_risk_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_allergy_reviews": n_allergy_reviews,  # lowercase alias for CLI
    }
