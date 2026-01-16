"""
Ground Truth computation for G1b (Allergy Safety + L1.5 Allergen Pattern).

Implements the formula from data/tasks/yelp/G1b_prompt.txt.
Extends G1a with L1.5 allergen type grouping and pattern bonus.

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

# L1.5 Allergen Pattern Bonus values
ALLERGEN_PATTERN_BONUS = {
    "systemic": 3.0,
    "specific_high_risk": 2.0,
    "specific_moderate_risk": 1.0,
    "no_pattern": 0.0,
}


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


def get_allergen_type(judgment: Dict[str, Any]) -> str:
    """
    Extract allergen type from judgment.
    Returns: peanut, tree_nut, shellfish, other, or unknown
    """
    # Check if allergen_type was extracted
    allergen_type = judgment.get("allergen_type", "unknown")
    if allergen_type and allergen_type != "unknown":
        return allergen_type.lower()

    # Fallback: try to infer from review text if available
    text = judgment.get("review_text", "").lower()
    if any(w in text for w in ["peanut", "peanuts"]):
        return "peanut"
    elif any(w in text for w in ["tree nut", "almond", "walnut", "cashew", "pistachio", "hazelnut"]):
        return "tree_nut"
    elif any(w in text for w in ["shellfish", "shrimp", "crab", "lobster", "oyster"]):
        return "shellfish"
    elif judgment.get("is_allergy_related", False):
        return "other"

    return "unknown"


def derive_l1(judgment: Dict[str, Any]) -> Dict[str, bool]:
    """
    Derive L1 composites from L0 primitives.
    """
    account_type = judgment.get("account_type", "hypothetical")
    incident_severity = judgment.get("incident_severity", "none")
    assurance_claim = judgment.get("assurance_claim", "false")

    is_firsthand = account_type == "firsthand"
    has_incident = incident_severity in ("mild", "moderate", "severe")
    has_assurance = assurance_claim == "true"

    false_assurance = is_firsthand and has_incident and has_assurance
    firsthand_incident = is_firsthand and has_incident

    return {
        "FALSE_ASSURANCE": false_assurance,
        "FIRSTHAND_INCIDENT": firsthand_incident,
    }


def compute_l15_buckets(allergy_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute L1.5 allergen type buckets.

    Groups reviews by allergen type and computes:
    - n_reviews, n_incidents, incident_rate, max_severity for each bucket
    - worst_allergen, worst_allergen_rate, n_allergens_with_incidents
    - allergen_pattern
    """
    buckets = {
        "peanut": {"n_reviews": 0, "n_incidents": 0, "severities": []},
        "tree_nut": {"n_reviews": 0, "n_incidents": 0, "severities": []},
        "shellfish": {"n_reviews": 0, "n_incidents": 0, "severities": []},
        "other": {"n_reviews": 0, "n_incidents": 0, "severities": []},
    }

    severity_order = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}

    for j in allergy_judgments:
        allergen = get_allergen_type(j)
        if allergen not in buckets:
            allergen = "other"

        buckets[allergen]["n_reviews"] += 1

        l1 = derive_l1(j)
        if l1["FIRSTHAND_INCIDENT"]:
            buckets[allergen]["n_incidents"] += 1
            severity = j.get("incident_severity", "none")
            buckets[allergen]["severities"].append(severity)

    # Compute rates and max severity for each bucket
    for key, bucket in buckets.items():
        n_reviews = bucket["n_reviews"]
        n_incidents = bucket["n_incidents"]
        bucket["incident_rate"] = n_incidents / max(n_reviews, 1)

        if bucket["severities"]:
            bucket["max_severity"] = max(bucket["severities"], key=lambda s: severity_order.get(s, 0))
        else:
            bucket["max_severity"] = "none"

        del bucket["severities"]  # Clean up temp data

    # Find worst allergen (highest incident rate among those with incidents)
    buckets_with_incidents = [(k, v) for k, v in buckets.items() if v["n_incidents"] > 0]

    if buckets_with_incidents:
        worst_allergen, worst_bucket = max(buckets_with_incidents, key=lambda x: x[1]["incident_rate"])
        worst_allergen_rate = worst_bucket["incident_rate"]
    else:
        worst_allergen = None
        worst_allergen_rate = 0.0

    n_allergens_with_incidents = len(buckets_with_incidents)

    # Determine allergen pattern
    if n_allergens_with_incidents >= 3:
        allergen_pattern = "systemic"
    elif n_allergens_with_incidents >= 1 and worst_allergen_rate > 0.5:
        allergen_pattern = "specific_high_risk"
    elif n_allergens_with_incidents >= 1:
        allergen_pattern = "specific_moderate_risk"
    else:
        allergen_pattern = "no_pattern"

    return {
        "peanut": buckets["peanut"],
        "tree_nut": buckets["tree_nut"],
        "shellfish": buckets["shellfish"],
        "other": buckets["other"],
        "worst_allergen": worst_allergen,
        "worst_allergen_rate": round(worst_allergen_rate, 3),
        "n_allergens_with_incidents": n_allergens_with_incidents,
        "allergen_pattern": allergen_pattern,
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

    # Derive L1 for each judgment and compute L2 aggregates
    n_mild = n_moderate = n_severe = 0
    n_false_assurance = 0
    incident_years: List[int] = []

    for j in allergy_judgments:
        l1 = derive_l1(j)

        if l1["FIRSTHAND_INCIDENT"]:
            severity = j.get("incident_severity", "none")
            if severity == "mild":
                n_mild += 1
            elif severity == "moderate":
                n_moderate += 1
            elif severity == "severe":
                n_severe += 1

            date_str = j.get("date", "")
            if date_str and len(date_str) >= 4:
                try:
                    year = int(date_str[:4])
                    incident_years.append(year)
                except ValueError:
                    pass

        if l1["FALSE_ASSURANCE"]:
            n_false_assurance += 1

    n_total_incidents = n_mild + n_moderate + n_severe

    # MOST_RECENT_INCIDENT_YEAR
    if incident_years:
        most_recent_incident_year = max(incident_years)
    else:
        most_recent_incident_year = DEFAULT_INCIDENT_YEAR

    # L1.5 Allergen Buckets
    l15_buckets = compute_l15_buckets(allergy_judgments)
    allergen_pattern = l15_buckets["allergen_pattern"]
    allergen_pattern_bonus = ALLERGEN_PATTERN_BONUS.get(allergen_pattern, 0.0)

    # Formulas (G1b: includes ALLERGEN_PATTERN_BONUS)
    incident_age = CURRENT_YEAR - most_recent_incident_year
    recency_decay = max(0.3, 1.0 - (incident_age * 0.15))
    cuisine_modifier = get_cuisine_modifier(categories)
    cuisine_impact = cuisine_modifier * 0.5
    incident_score = (n_mild * 2) + (n_moderate * 5) + (n_severe * 15)

    # G1b formula: adds ALLERGEN_PATTERN_BONUS
    raw_risk = BASE_RISK + (incident_score * recency_decay) + cuisine_impact + allergen_pattern_bonus
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

    if n_severe >= 1:
        override_applied = "severe_incident"
        verdict = "Critical Risk"
    elif n_false_assurance >= 1 and verdict == "Low Risk":
        override_applied = "false_assurance_min_high"
        verdict = "High Risk"

    return {
        # L1.5 Allergen Buckets
        "L1_5_allergen_buckets": l15_buckets,
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
        "ALLERGEN_PATTERN_BONUS": allergen_pattern_bonus,
        "RAW_RISK": round(raw_risk, 3),
        "FINAL_RISK_SCORE": round(final_risk_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_allergy_reviews": n_allergy_reviews,
    }
