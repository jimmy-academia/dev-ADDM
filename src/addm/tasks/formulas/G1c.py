"""
Ground Truth computation for G1c (Allergy Safety - Complex).

Implements the formula from data/tasks/yelp/G1c_prompt.txt.
Complex formula with credibility weighting and interaction effects.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

CURRENT_YEAR = 2022
DEFAULT_INCIDENT_YEAR = 2020
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
    "mild": 2,
    "moderate": 5,
    "severe": 15,
}

# Staff modifiers
STAFF_MODIFIERS = {
    "accommodated": -1.0,
    "refused": 2.0,
    "dismissive": 3.0,
    "none": 0.0,
    "unknown": 0.0,
}

# Cuisine risk modifiers
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
        if cat in CUISINE_MODIFIERS:
            max_modifier = max(max_modifier, CUISINE_MODIFIERS[cat])
        else:
            for cuisine, modifier in CUISINE_MODIFIERS.items():
                if cuisine.lower() in cat.lower():
                    max_modifier = max(max_modifier, modifier)

    return max_modifier


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute L1 composites using complex formula.

    Returns dict with:
    - credibility_weight
    - severity_score
    - staff_modifier
    - l1_review_risk
    - false_assurance
    - broken_promise_penalty
    - dismissive_incident
    - l1_total_risk
    """
    account_type = judgment.get("account_type", "hypothetical")
    incident_severity = judgment.get("incident_severity", "none")
    assurance_claim = judgment.get("assurance_claim", "false")
    staff_response = judgment.get("staff_response", "none")

    # Basic weights and scores
    credibility_weight = CREDIBILITY_WEIGHTS.get(account_type, 0.1)
    severity_score = SEVERITY_SCORES.get(incident_severity, 0)
    staff_modifier = STAFF_MODIFIERS.get(staff_response, 0.0)

    # L1_REVIEW_RISK
    l1_review_risk = (severity_score * credibility_weight) + staff_modifier

    # FALSE_ASSURANCE
    is_firsthand = account_type == "firsthand"
    has_incident = incident_severity in ("mild", "moderate", "severe")
    has_assurance = assurance_claim == "true"
    false_assurance = is_firsthand and has_incident and has_assurance

    # BROKEN_PROMISE_PENALTY
    broken_promise_penalty = 5.0 if false_assurance else 0.0

    # DISMISSIVE_INCIDENT (interaction effect)
    dismissive_incident = 3.0 if (staff_response == "dismissive" and has_incident) else 0.0

    # L1_TOTAL_RISK
    l1_total_risk = l1_review_risk + broken_promise_penalty + dismissive_incident

    return {
        "credibility_weight": credibility_weight,
        "severity_score": severity_score,
        "staff_modifier": staff_modifier,
        "l1_review_risk": round(l1_review_risk, 2),
        "false_assurance": false_assurance,
        "broken_promise_penalty": broken_promise_penalty,
        "dismissive_incident": dismissive_incident,
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

    # CONFIDENCE_LEVEL
    if n_allergy_reviews == 0:
        confidence_level = "none"
    elif n_allergy_reviews <= 2:
        confidence_level = "low"
    elif n_allergy_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    # Compute L1 for each judgment and aggregate
    sum_l1_risk = 0.0
    sum_credibility = 0.0
    n_severe = 0
    n_false_assurance = 0
    n_dismissive_incidents = 0
    incident_years: List[int] = []

    for j in allergy_judgments:
        l1 = compute_l1_complex(j)

        sum_l1_risk += l1["l1_total_risk"]
        sum_credibility += l1["credibility_weight"]

        # Count severe incidents (firsthand only)
        if j.get("account_type") == "firsthand" and j.get("incident_severity") == "severe":
            n_severe += 1

        # Count false assurance
        if l1["false_assurance"]:
            n_false_assurance += 1

        # Count dismissive incidents
        if l1["dismissive_incident"] > 0:
            n_dismissive_incidents += 1

        # Track incident years for recency
        if j.get("incident_severity") != "none" and j.get("account_type") == "firsthand":
            date_str = j.get("date", "")
            if date_str and len(date_str) >= 4:
                try:
                    year = int(date_str[:4])
                    incident_years.append(year)
                except ValueError:
                    pass

    # MOST_RECENT_INCIDENT_YEAR
    if incident_years:
        most_recent_incident_year = max(incident_years)
    else:
        most_recent_incident_year = DEFAULT_INCIDENT_YEAR

    # Weighted aggregation
    weighted_mean_risk = sum_l1_risk / max(sum_credibility, 1)

    # Formulas
    incident_age = CURRENT_YEAR - most_recent_incident_year
    recency_decay = max(0.3, 1.0 - (incident_age * 0.15))
    cuisine_modifier = get_cuisine_modifier(categories)
    cuisine_impact = cuisine_modifier * 0.5
    adjusted_risk = weighted_mean_risk * recency_decay
    raw_risk = BASE_RISK + adjusted_risk + cuisine_impact
    final_risk_score = max(0.0, min(20.0, raw_risk))

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
    # Override 2: N_FALSE_ASSURANCE >= 2 => Critical Risk
    elif n_false_assurance >= 2:
        override_applied = "pattern_broken_promises"
        verdict = "Critical Risk"
    # Override 3: N_FALSE_ASSURANCE >= 1 => min High Risk
    elif n_false_assurance >= 1 and verdict == "Low Risk":
        override_applied = "false_assurance_min_high"
        verdict = "High Risk"
    # Override 4: N_DISMISSIVE_INCIDENTS >= 2 => min High Risk
    elif n_dismissive_incidents >= 2 and verdict == "Low Risk":
        override_applied = "systemic_staff_issue"
        verdict = "High Risk"

    return {
        # L2 Aggregates
        "CURRENT_YEAR": CURRENT_YEAR,
        "N_ALLERGY_REVIEWS": n_allergy_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_SEVERE": n_severe,
        "N_FALSE_ASSURANCE": n_false_assurance,
        "N_DISMISSIVE_INCIDENTS": n_dismissive_incidents,
        "MOST_RECENT_INCIDENT_YEAR": most_recent_incident_year,
        # Weighted aggregation
        "SUM_L1_RISK": round(sum_l1_risk, 2),
        "SUM_CREDIBILITY": round(sum_credibility, 2),
        "WEIGHTED_MEAN_RISK": round(weighted_mean_risk, 3),
        # Formula results
        "INCIDENT_AGE": incident_age,
        "RECENCY_DECAY": round(recency_decay, 3),
        "CUISINE_MODIFIER": cuisine_modifier,
        "CUISINE_IMPACT": round(cuisine_impact, 3),
        "ADJUSTED_RISK": round(adjusted_risk, 3),
        "BASE_RISK": BASE_RISK,
        "RAW_RISK": round(raw_risk, 3),
        "FINAL_RISK_SCORE": round(final_risk_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_allergy_reviews": n_allergy_reviews,
    }
