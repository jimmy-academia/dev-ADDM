"""
Ground Truth computation for G1h (Dietary Accommodation - Complex + L1.5).

Implements the formula from data/tasks/yelp/G1h_prompt.txt.
Complex formula with credibility weighting + L1.5 diet type grouping.

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

# L1.5 Accommodation Pattern Bonus values
ACCOMMODATION_PATTERN_BONUS = {
    "comprehensive": 2.0,
    "selective": 1.0,
    "inconsistent": -1.0,
    "unknown": 0.0,
}


# =============================================================================
# Helpers
# =============================================================================


def get_diet_type(judgment: Dict[str, Any]) -> str:
    """Extract diet type from judgment."""
    diet_type = judgment.get("diet_type", "unknown")
    if diet_type and diet_type != "unknown":
        return diet_type.lower()

    text = judgment.get("review_text", "").lower()
    if any(w in text for w in ["vegetarian", "veggie"]):
        return "vegetarian"
    elif any(w in text for w in ["vegan", "plant-based", "plant based"]):
        return "vegan"
    elif any(w in text for w in ["gluten-free", "gluten free", "celiac"]):
        return "gluten_free"
    elif judgment.get("is_dietary_related", False):
        return "other"

    return "unknown"


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    """Compute L1 composites using complex formula."""
    account_type = judgment.get("account_type", "hypothetical")
    accommodation_outcome = judgment.get("accommodation_outcome", "not_attempted")
    staff_knowledge = judgment.get("staff_knowledge", "none")
    menu_clarity = judgment.get("menu_clarity", "unknown")
    consequence_severity = judgment.get("consequence_severity", "none")

    credibility_weight = CREDIBILITY_WEIGHTS.get(account_type, 0.1)
    outcome_score = OUTCOME_SCORES.get(accommodation_outcome, 0)
    staff_modifier = STAFF_MODIFIERS.get(staff_knowledge, 0.0)
    menu_modifier = MENU_MODIFIERS.get(menu_clarity, 0.0)
    consequence_penalty = CONSEQUENCE_PENALTIES.get(consequence_severity, 0.0)

    l1_review_score = (outcome_score * credibility_weight) + staff_modifier + menu_modifier + consequence_penalty

    uninformed_failure = -2.0 if (staff_knowledge == "uninformed" and accommodation_outcome == "failure") else 0.0

    is_firsthand = account_type == "firsthand"
    health_incident = is_firsthand and consequence_severity in ("health_impact", "serious")

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


def compute_l15_buckets(dietary_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute L1.5 diet type buckets."""
    buckets = {
        "vegetarian": {"n_reviews": 0, "n_success": 0, "n_failure": 0},
        "vegan": {"n_reviews": 0, "n_success": 0, "n_failure": 0},
        "gluten_free": {"n_reviews": 0, "n_success": 0, "n_failure": 0},
        "other": {"n_reviews": 0, "n_success": 0, "n_failure": 0},
    }

    for j in dietary_judgments:
        diet = get_diet_type(j)
        if diet not in buckets:
            diet = "other"

        buckets[diet]["n_reviews"] += 1

        account_type = j.get("account_type", "hypothetical")
        accommodation_outcome = j.get("accommodation_outcome", "not_attempted")

        if account_type == "firsthand":
            if accommodation_outcome == "success":
                buckets[diet]["n_success"] += 1
            elif accommodation_outcome == "failure":
                buckets[diet]["n_failure"] += 1

    for key, bucket in buckets.items():
        n_success = bucket["n_success"]
        n_failure = bucket["n_failure"]
        bucket["success_rate"] = n_success / max(n_success + n_failure, 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_success"] + v["n_failure"] > 0]

    if buckets_with_data:
        best_diet, best_bucket = max(buckets_with_data, key=lambda x: x[1]["success_rate"])
        best_diet_success_rate = best_bucket["success_rate"]
        worst_diet, worst_bucket = min(buckets_with_data, key=lambda x: x[1]["success_rate"])
        worst_diet_success_rate = worst_bucket["success_rate"]
    else:
        best_diet = None
        best_diet_success_rate = 0.0
        worst_diet = None
        worst_diet_success_rate = 0.0

    n_diets_well_served = sum(1 for k, v in buckets.items() if v["success_rate"] >= 0.7 and v["n_reviews"] > 0)

    any_failure = any(v["n_failure"] > 0 for v in buckets.values())

    if n_diets_well_served >= 3:
        accommodation_pattern = "comprehensive"
    elif n_diets_well_served >= 1:
        accommodation_pattern = "selective"
    elif any_failure:
        accommodation_pattern = "inconsistent"
    else:
        accommodation_pattern = "unknown"

    return {
        "vegetarian": buckets["vegetarian"],
        "vegan": buckets["vegan"],
        "gluten_free": buckets["gluten_free"],
        "other": buckets["other"],
        "best_diet": best_diet,
        "best_diet_success_rate": round(best_diet_success_rate, 3),
        "worst_diet": worst_diet,
        "worst_diet_success_rate": round(worst_diet_success_rate, 3),
        "n_diets_well_served": n_diets_well_served,
        "accommodation_pattern": accommodation_pattern,
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
    dietary_judgments = [j for j in judgments if j.get("is_dietary_related", False)]
    n_dietary_reviews = len(dietary_judgments)

    if n_dietary_reviews == 0:
        confidence_level = "none"
    elif n_dietary_reviews <= 2:
        confidence_level = "low"
    elif n_dietary_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

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

    # L1.5 Diet Buckets
    l15_buckets = compute_l15_buckets(dietary_judgments)
    accommodation_pattern = l15_buckets["accommodation_pattern"]
    accommodation_pattern_bonus = ACCOMMODATION_PATTERN_BONUS.get(accommodation_pattern, 0.0)

    # Weighted aggregation
    weighted_mean_score = sum_l1_score / max(sum_credibility, 1)

    # Formulas (G1h: complex + L1.5)
    adjusted_score = weighted_mean_score

    # G1h: adds ACCOMMODATION_PATTERN_BONUS to complex formula
    raw_score = BASE_SCORE + adjusted_score + accommodation_pattern_bonus
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
    if final_score >= 7.0:
        base_verdict_by_score = "Excellent"
    elif final_score >= 5.0:
        base_verdict_by_score = "Adequate"
    elif final_score >= 3.0:
        base_verdict_by_score = "Poor"
    else:
        base_verdict_by_score = "Very Poor"

    override_applied = "none"
    verdict = base_verdict_by_score

    if n_health_incidents >= 1:
        override_applied = "health_risk"
        verdict = "Very Poor"
    elif n_uninformed_failures >= 2 and verdict in ("Excellent", "Adequate"):
        override_applied = "systemic_staff_issue"
        verdict = "Poor"
    elif n_failure >= 3:
        override_applied = "pattern_of_failures"
        verdict = "Very Poor"
    elif weighted_mean_score >= 3.0 and n_success >= 5 and verdict in ("Poor", "Very Poor"):
        override_applied = "high_success_min_adequate"
        verdict = "Adequate"

    return {
        # L1.5 Diet Buckets
        "L1_5_diet_buckets": l15_buckets,
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
        "ACCOMMODATION_PATTERN_BONUS": accommodation_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_dietary_reviews": n_dietary_reviews,
    }
