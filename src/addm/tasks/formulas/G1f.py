"""
Ground Truth computation for G1f (Dietary Accommodation + L1.5 Diet Type Grouping).

Implements the formula from data/tasks/yelp/G1f_prompt.txt.
Extends G1e with L1.5 diet type grouping and accommodation pattern bonus.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

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
    """
    Extract diet type from judgment.
    Returns: vegetarian, vegan, gluten_free, other, or unknown
    """
    diet_type = judgment.get("diet_type", "unknown")
    if diet_type and diet_type != "unknown":
        return diet_type.lower()

    # Fallback: try to infer from review text if available
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


def derive_l1(judgment: Dict[str, Any]) -> Dict[str, bool]:
    """
    Derive L1 composites from L0 primitives.
    """
    account_type = judgment.get("account_type", "hypothetical")
    accommodation_outcome = judgment.get("accommodation_outcome", "not_attempted")
    staff_knowledge = judgment.get("staff_knowledge", "none")

    is_firsthand = account_type == "firsthand"

    firsthand_failure = is_firsthand and accommodation_outcome == "failure"

    positive_experience = (
        is_firsthand
        and accommodation_outcome == "success"
        and staff_knowledge in ("knowledgeable", "uncertain")
    )

    return {
        "FIRSTHAND_FAILURE": firsthand_failure,
        "POSITIVE_EXPERIENCE": positive_experience,
    }


def compute_l15_buckets(dietary_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute L1.5 diet type buckets.

    Groups reviews by diet type and computes:
    - n_reviews, n_success, n_failure, success_rate for each bucket
    - best_diet, best_diet_success_rate, worst_diet, worst_diet_success_rate
    - n_diets_well_served, accommodation_pattern
    """
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

    # Compute success rates for each bucket
    for key, bucket in buckets.items():
        n_success = bucket["n_success"]
        n_failure = bucket["n_failure"]
        bucket["success_rate"] = n_success / max(n_success + n_failure, 1)

    # Find best and worst diets
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

    # Count diets well served (success_rate >= 0.7)
    n_diets_well_served = sum(1 for k, v in buckets.items() if v["success_rate"] >= 0.7 and v["n_reviews"] > 0)

    # Determine accommodation pattern
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

    Args:
        judgments: List of L0 judgments (one per review)
        restaurant_meta: Restaurant metadata

    Returns:
        Dict with all computed values matching OUTPUT SCHEMA in prompt
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

    # Compute L2 aggregates
    n_success = 0
    n_failure = 0
    n_partial = 0
    n_knowledgeable = 0
    n_uninformed = 0

    for j in dietary_judgments:
        account_type = j.get("account_type", "hypothetical")
        accommodation_outcome = j.get("accommodation_outcome", "not_attempted")
        staff_knowledge = j.get("staff_knowledge", "none")

        is_firsthand = account_type == "firsthand"

        if is_firsthand:
            if accommodation_outcome == "success":
                n_success += 1
            elif accommodation_outcome == "failure":
                n_failure += 1
            elif accommodation_outcome == "partial":
                n_partial += 1

        if staff_knowledge == "knowledgeable":
            n_knowledgeable += 1
        elif staff_knowledge == "uninformed":
            n_uninformed += 1

    # L1.5 Diet Buckets
    l15_buckets = compute_l15_buckets(dietary_judgments)
    accommodation_pattern = l15_buckets["accommodation_pattern"]
    accommodation_pattern_bonus = ACCOMMODATION_PATTERN_BONUS.get(accommodation_pattern, 0.0)

    # Formulas
    total_outcomes = n_success + n_failure + n_partial
    success_rate = n_success / max(total_outcomes, 1)
    failure_rate = n_failure / max(total_outcomes, 1)

    positive_score = (n_success * 2)
    negative_score = (n_failure * 3) + (n_partial * 1)

    # G1f formula: adds ACCOMMODATION_PATTERN_BONUS
    raw_score = BASE_SCORE + positive_score - negative_score + accommodation_pattern_bonus
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

    if n_failure >= 3:
        override_applied = "pattern_of_failures"
        verdict = "Very Poor"
    elif failure_rate >= 0.5 and verdict in ("Excellent", "Adequate"):
        override_applied = "high_failure_rate_max_poor"
        verdict = "Poor"

    return {
        # L1.5 Diet Buckets
        "L1_5_diet_buckets": l15_buckets,
        # L2 Aggregates
        "N_DIETARY_REVIEWS": n_dietary_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_SUCCESS": n_success,
        "N_FAILURE": n_failure,
        "N_PARTIAL": n_partial,
        # Formula results
        "SUCCESS_RATE": round(success_rate, 3),
        "FAILURE_RATE": round(failure_rate, 3),
        "POSITIVE_SCORE": positive_score,
        "NEGATIVE_SCORE": negative_score,
        "BASE_SCORE": BASE_SCORE,
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
