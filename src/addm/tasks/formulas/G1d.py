"""
Ground Truth computation for G1d (Allergy Safety - Complex + L1.5).

Implements the formula from data/tasks/yelp/G1d_prompt.txt.
Complex formula with credibility weighting + L1.5 allergen pattern bonus.

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
    """Extract allergen type from judgment."""
    allergen_type = judgment.get("allergen_type", "unknown")
    if allergen_type and allergen_type != "unknown":
        return allergen_type.lower()

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


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    """Compute L1 composites using complex formula."""
    account_type = judgment.get("account_type", "hypothetical")
    incident_severity = judgment.get("incident_severity", "none")
    assurance_claim = judgment.get("assurance_claim", "false")
    staff_response = judgment.get("staff_response", "none")

    credibility_weight = CREDIBILITY_WEIGHTS.get(account_type, 0.1)
    severity_score = SEVERITY_SCORES.get(incident_severity, 0)
    staff_modifier = STAFF_MODIFIERS.get(staff_response, 0.0)

    l1_review_risk = (severity_score * credibility_weight) + staff_modifier

    is_firsthand = account_type == "firsthand"
    has_incident = incident_severity in ("mild", "moderate", "severe")
    has_assurance = assurance_claim == "true"
    false_assurance = is_firsthand and has_incident and has_assurance

    broken_promise_penalty = 5.0 if false_assurance else 0.0
    dismissive_incident = 3.0 if (staff_response == "dismissive" and has_incident) else 0.0

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


def compute_l15_buckets(allergy_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute L1.5 allergen type buckets."""
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

        if j.get("account_type") == "firsthand" and j.get("incident_severity") in ("mild", "moderate", "severe"):
            buckets[allergen]["n_incidents"] += 1
            severity = j.get("incident_severity", "none")
            buckets[allergen]["severities"].append(severity)

    for key, bucket in buckets.items():
        n_reviews = bucket["n_reviews"]
        n_incidents = bucket["n_incidents"]
        bucket["incident_rate"] = n_incidents / max(n_reviews, 1)

        if bucket["severities"]:
            bucket["max_severity"] = max(bucket["severities"], key=lambda s: severity_order.get(s, 0))
        else:
            bucket["max_severity"] = "none"

        del bucket["severities"]

    buckets_with_incidents = [(k, v) for k, v in buckets.items() if v["n_incidents"] > 0]

    if buckets_with_incidents:
        worst_allergen, worst_bucket = max(buckets_with_incidents, key=lambda x: x[1]["incident_rate"])
        worst_allergen_rate = worst_bucket["incident_rate"]
    else:
        worst_allergen = None
        worst_allergen_rate = 0.0

    n_allergens_with_incidents = len(buckets_with_incidents)

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
    """
    categories = restaurant_meta.get("categories", "")

    allergy_judgments = [j for j in judgments if j.get("is_allergy_related", False)]
    n_allergy_reviews = len(allergy_judgments)

    if n_allergy_reviews == 0:
        confidence_level = "none"
    elif n_allergy_reviews <= 2:
        confidence_level = "low"
    elif n_allergy_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

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

        if j.get("account_type") == "firsthand" and j.get("incident_severity") == "severe":
            n_severe += 1

        if l1["false_assurance"]:
            n_false_assurance += 1

        if l1["dismissive_incident"] > 0:
            n_dismissive_incidents += 1

        if j.get("incident_severity") != "none" and j.get("account_type") == "firsthand":
            date_str = j.get("date", "")
            if date_str and len(date_str) >= 4:
                try:
                    year = int(date_str[:4])
                    incident_years.append(year)
                except ValueError:
                    pass

    if incident_years:
        most_recent_incident_year = max(incident_years)
    else:
        most_recent_incident_year = DEFAULT_INCIDENT_YEAR

    # L1.5 Allergen Buckets
    l15_buckets = compute_l15_buckets(allergy_judgments)
    allergen_pattern = l15_buckets["allergen_pattern"]
    allergen_pattern_bonus = ALLERGEN_PATTERN_BONUS.get(allergen_pattern, 0.0)

    # Weighted aggregation
    weighted_mean_risk = sum_l1_risk / max(sum_credibility, 1)

    # Formulas (G1d: complex + L1.5)
    incident_age = CURRENT_YEAR - most_recent_incident_year
    recency_decay = max(0.3, 1.0 - (incident_age * 0.15))
    cuisine_modifier = get_cuisine_modifier(categories)
    cuisine_impact = cuisine_modifier * 0.5
    adjusted_risk = weighted_mean_risk * recency_decay

    # G1d: adds ALLERGEN_PATTERN_BONUS to complex formula
    raw_risk = BASE_RISK + adjusted_risk + cuisine_impact + allergen_pattern_bonus
    final_risk_score = max(0.0, min(20.0, raw_risk))

    # Decision Policy
    if final_risk_score < 4.0:
        base_verdict_by_score = "Low Risk"
    elif final_risk_score < 8.0:
        base_verdict_by_score = "High Risk"
    else:
        base_verdict_by_score = "Critical Risk"

    override_applied = "none"
    verdict = base_verdict_by_score

    if n_severe >= 1:
        override_applied = "severe_incident"
        verdict = "Critical Risk"
    elif n_false_assurance >= 2:
        override_applied = "pattern_broken_promises"
        verdict = "Critical Risk"
    elif n_false_assurance >= 1 and verdict == "Low Risk":
        override_applied = "false_assurance_min_high"
        verdict = "High Risk"
    elif n_dismissive_incidents >= 2 and verdict == "Low Risk":
        override_applied = "systemic_staff_issue"
        verdict = "High Risk"

    return {
        # L1.5 Allergen Buckets
        "L1_5_allergen_buckets": l15_buckets,
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
