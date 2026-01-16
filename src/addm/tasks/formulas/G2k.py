"""
Ground Truth computation for G2k (Group Dining - Complex).

Implements the formula from data/tasks/yelp/G2k_prompt.txt.
Complex formula with group size weighting and interaction effects.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

# Group size weights (larger = harder to accommodate)
GROUP_SIZE_WEIGHTS = {
    "small": 1.0,
    "medium": 1.5,
    "large": 2.0,
    "very_large": 2.5,
    "not_mentioned": 1.0,
}

# Seating scores
SEATING_SCORES = {
    "excellent": 3,
    "adequate": 1,
    "difficult": -2,
    "refused": -5,
    "not_mentioned": 0,
}

# Service scores
SERVICE_SCORES = {
    "smooth": 3,
    "adequate": 1,
    "struggled": -2,
    "overwhelmed": -4,
    "not_mentioned": 0,
}

# Noise scores
NOISE_SCORES = {
    "welcomed": 1.5,
    "tolerated": 0.5,
    "frowned_upon": -2,
    "not_mentioned": 0,
}

# Check scores
CHECK_SCORES = {
    "easy": 1.5,
    "available": 0.5,
    "refused": -1.5,
    "not_mentioned": 0,
}

# Kids scores (for family groups)
KIDS_SCORES = {
    "very_friendly": 2,
    "accommodating": 1,
    "not_suitable": -2,
    "not_mentioned": 0,
}

# Celebration scores (for celebrations)
CELEBRATION_SCORES = {
    "excellent": 2.5,
    "basic": 1,
    "none": -1,
    "not_mentioned": 0,
}


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute L1 composites using complex formula.

    Returns dict with all L1 computed values.
    """
    group_type = judgment.get("group_type", "not_group")
    group_size = judgment.get("group_size", "not_mentioned")
    seating_accommodation = judgment.get("seating_accommodation", "not_mentioned")
    service_handling = judgment.get("service_handling", "not_mentioned")
    noise_tolerance = judgment.get("noise_tolerance", "not_mentioned")
    check_splitting = judgment.get("check_splitting", "not_mentioned")
    kids_friendly = judgment.get("kids_friendly", "not_mentioned")
    celebration_support = judgment.get("celebration_support", "not_mentioned")

    # Get scores
    group_size_weight = GROUP_SIZE_WEIGHTS.get(group_size, 1.0)
    seating_score = SEATING_SCORES.get(seating_accommodation, 0)
    service_score = SERVICE_SCORES.get(service_handling, 0)
    noise_score = NOISE_SCORES.get(noise_tolerance, 0)
    check_score = CHECK_SCORES.get(check_splitting, 0)
    kids_score = KIDS_SCORES.get(kids_friendly, 0) if group_type == "family" else 0
    celebration_score = CELEBRATION_SCORES.get(celebration_support, 0) if group_type == "celebration" else 0

    # L1_REVIEW_SCORE
    l1_review_score = (seating_score + service_score + noise_score + check_score) * group_size_weight

    # FAMILY_BONUS (interaction)
    if group_type == "family" and kids_friendly == "very_friendly":
        family_bonus = 2.0
    elif group_type == "family" and kids_friendly == "not_suitable":
        family_bonus = -3.0
    else:
        family_bonus = 0.0

    # CELEBRATION_BONUS (interaction)
    celebration_bonus = 2.0 if (group_type == "celebration" and celebration_support == "excellent") else 0.0

    # REFUSED_LARGE_GROUP (interaction)
    refused_large_group = -5.0 if (group_size in ("large", "very_large") and seating_accommodation == "refused") else 0.0

    # L1_TOTAL_SCORE
    l1_total_score = l1_review_score + family_bonus + celebration_bonus + refused_large_group

    return {
        "group_size_weight": group_size_weight,
        "seating_score": seating_score,
        "service_score": service_score,
        "noise_score": noise_score,
        "check_score": check_score,
        "kids_score": kids_score,
        "celebration_score": celebration_score,
        "l1_review_score": round(l1_review_score, 2),
        "family_bonus": family_bonus,
        "celebration_bonus": celebration_bonus,
        "refused_large_group": refused_large_group,
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
    # Filter to group-related judgments only
    group_judgments = [j for j in judgments if j.get("is_group_related", False)]
    n_group_reviews = len(group_judgments)

    # CONFIDENCE_LEVEL
    if n_group_reviews == 0:
        confidence_level = "none"
    elif n_group_reviews <= 2:
        confidence_level = "low"
    elif n_group_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    # Compute L1 for each judgment and aggregate
    sum_l1_score = 0.0
    sum_size_weight = 0.0
    n_large_groups = 0
    n_large_success = 0
    n_refused = 0
    n_family = 0
    n_family_friendly = 0
    n_celebrations = 0
    n_celebration_success = 0

    for j in group_judgments:
        l1 = compute_l1_complex(j)

        sum_l1_score += l1["l1_total_score"]
        sum_size_weight += l1["group_size_weight"]

        group_type = j.get("group_type", "not_group")
        group_size = j.get("group_size", "not_mentioned")
        seating_accommodation = j.get("seating_accommodation", "not_mentioned")
        kids_friendly = j.get("kids_friendly", "not_mentioned")
        celebration_support = j.get("celebration_support", "not_mentioned")

        # Track large groups
        if group_size in ("large", "very_large"):
            n_large_groups += 1
            if seating_accommodation in ("excellent", "adequate"):
                n_large_success += 1

        if seating_accommodation == "refused":
            n_refused += 1

        # Track family
        if group_type == "family":
            n_family += 1
            if kids_friendly == "very_friendly":
                n_family_friendly += 1

        # Track celebrations
        if group_type == "celebration":
            n_celebrations += 1
            if celebration_support == "excellent":
                n_celebration_success += 1

    # Weighted aggregation
    weighted_mean_score = sum_l1_score / max(sum_size_weight, 1)

    # Formulas
    adjusted_score = weighted_mean_score
    raw_score = BASE_SCORE + adjusted_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy: Base verdict by score
    if final_score >= 7.5:
        base_verdict_by_score = "Excellent for Groups"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good for Groups"
    elif final_score >= 3.5:
        base_verdict_by_score = "Limited"
    else:
        base_verdict_by_score = "Not Suitable"

    # Decision Policy: Overrides
    override_applied = "none"
    verdict = base_verdict_by_score
    family_note = None
    celebration_note = None

    # Override 1: If N_REFUSED >= 1 => max Limited
    if n_refused >= 1 and verdict in ("Excellent for Groups", "Good for Groups"):
        override_applied = "refused_max_limited"
        verdict = "Limited"
    # Override 2: If N_LARGE_SUCCESS >= 2 => min Good for Groups
    elif n_large_success >= 2 and verdict in ("Limited", "Not Suitable"):
        override_applied = "large_success_min_good"
        verdict = "Good for Groups"
    # Override 3: If N_FAMILY_FRIENDLY >= 2 => note family-friendly
    elif n_family_friendly >= 2:
        family_note = "Especially family-friendly"
    # Override 4: If N_CELEBRATION_SUCCESS >= 2 => note celebration-friendly
    elif n_celebration_success >= 2:
        celebration_note = "Great for celebrations"
    # Override 5: If WEIGHTED_MEAN_SCORE < -2 => Not Suitable
    elif weighted_mean_score < -2:
        override_applied = "low_weighted_mean"
        verdict = "Not Suitable"

    result = {
        # L2 Aggregates
        "N_GROUP_REVIEWS": n_group_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_LARGE_GROUPS": n_large_groups,
        "N_LARGE_SUCCESS": n_large_success,
        "N_REFUSED": n_refused,
        "N_FAMILY": n_family,
        "N_FAMILY_FRIENDLY": n_family_friendly,
        "N_CELEBRATIONS": n_celebrations,
        "N_CELEBRATION_SUCCESS": n_celebration_success,
        # Weighted aggregation
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "SUM_SIZE_WEIGHT": round(sum_size_weight, 2),
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
        "n_group_reviews": n_group_reviews,
    }

    if family_note:
        result["family_note"] = family_note
    if celebration_note:
        result["celebration_note"] = celebration_note

    return result
