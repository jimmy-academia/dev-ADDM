"""
Ground Truth computation for G2l (Group Dining - Complex + L1.5).

Implements the formula from data/tasks/yelp/G2l_prompt.txt.
Complex formula with group size weighting + L1.5 group type grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

# Group size weights
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

# Kids scores
KIDS_SCORES = {
    "very_friendly": 2,
    "accommodating": 1,
    "not_suitable": -2,
    "not_mentioned": 0,
}

# Celebration scores
CELEBRATION_SCORES = {
    "excellent": 2.5,
    "basic": 1,
    "none": -1,
    "not_mentioned": 0,
}

# L1.5 Group Versatility Bonus values
GROUP_VERSATILITY_BONUS = {
    "all_groups_welcome": 2.0,
    "family_friendly": 1.5,
    "great_for_celebrations": 1.0,
    "handles_work_events": 1.0,
    "situational": 0.0,
    "not_recommended": -1.0,
}


# =============================================================================
# Helpers
# =============================================================================


def get_group_bucket(group_type: str) -> str:
    """Map group type to L1.5 bucket."""
    if group_type == "family":
        return "family"
    elif group_type in ("friends", "celebration"):
        return "social"
    elif group_type in ("work_team", "large_party"):
        return "professional"
    else:
        return "other"


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    """Compute L1 composites using complex formula."""
    group_type = judgment.get("group_type", "not_group")
    group_size = judgment.get("group_size", "not_mentioned")
    seating_accommodation = judgment.get("seating_accommodation", "not_mentioned")
    service_handling = judgment.get("service_handling", "not_mentioned")
    noise_tolerance = judgment.get("noise_tolerance", "not_mentioned")
    check_splitting = judgment.get("check_splitting", "not_mentioned")
    kids_friendly = judgment.get("kids_friendly", "not_mentioned")
    celebration_support = judgment.get("celebration_support", "not_mentioned")

    group_size_weight = GROUP_SIZE_WEIGHTS.get(group_size, 1.0)
    seating_score = SEATING_SCORES.get(seating_accommodation, 0)
    service_score = SERVICE_SCORES.get(service_handling, 0)
    noise_score = NOISE_SCORES.get(noise_tolerance, 0)
    check_score = CHECK_SCORES.get(check_splitting, 0)
    kids_score = KIDS_SCORES.get(kids_friendly, 0) if group_type == "family" else 0
    celebration_score = CELEBRATION_SCORES.get(celebration_support, 0) if group_type == "celebration" else 0

    l1_review_score = (seating_score + service_score + noise_score + check_score) * group_size_weight

    if group_type == "family" and kids_friendly == "very_friendly":
        family_bonus = 2.0
    elif group_type == "family" and kids_friendly == "not_suitable":
        family_bonus = -3.0
    else:
        family_bonus = 0.0

    celebration_bonus = 2.0 if (group_type == "celebration" and celebration_support == "excellent") else 0.0

    refused_large_group = -5.0 if (group_size in ("large", "very_large") and seating_accommodation == "refused") else 0.0

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


def compute_l1_success(judgment: Dict[str, Any]) -> bool:
    """Simple success check for L1.5 bucket tracking."""
    group_type = judgment.get("group_type", "not_group")
    seating_accommodation = judgment.get("seating_accommodation", "not_mentioned")
    service_handling = judgment.get("service_handling", "not_mentioned")

    if group_type == "not_group":
        return False

    return seating_accommodation in ("excellent", "adequate") and service_handling in ("smooth", "adequate")


def compute_l1_failed(judgment: Dict[str, Any]) -> bool:
    """Simple failure check for L1.5 bucket tracking."""
    group_type = judgment.get("group_type", "not_group")
    seating_accommodation = judgment.get("seating_accommodation", "not_mentioned")
    service_handling = judgment.get("service_handling", "not_mentioned")

    if group_type == "not_group":
        return False

    return seating_accommodation in ("difficult", "refused") or service_handling in ("struggled", "overwhelmed")


def compute_l15_buckets(group_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute L1.5 group type buckets."""
    buckets = {
        "family": {"n_reviews": 0, "n_success": 0, "n_failed": 0},
        "social": {"n_reviews": 0, "n_success": 0, "n_failed": 0},
        "professional": {"n_reviews": 0, "n_success": 0, "n_failed": 0},
    }

    for j in group_judgments:
        group_type = j.get("group_type", "not_group")
        bucket = get_group_bucket(group_type)
        if bucket not in buckets:
            continue

        buckets[bucket]["n_reviews"] += 1

        if compute_l1_success(j):
            buckets[bucket]["n_success"] += 1
        if compute_l1_failed(j):
            buckets[bucket]["n_failed"] += 1

    for key, bucket in buckets.items():
        n_success = bucket["n_success"]
        n_failed = bucket["n_failed"]
        bucket["success_rate"] = n_success / max(n_success + n_failed, 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_success"] + v["n_failed"] > 0]

    if buckets_with_data:
        best_group_type, best_bucket = max(buckets_with_data, key=lambda x: x[1]["success_rate"])
        best_group_success_rate = best_bucket["success_rate"]
        worst_group_type, _ = min(buckets_with_data, key=lambda x: x[1]["success_rate"])
    else:
        best_group_type = None
        best_group_success_rate = 0.0
        worst_group_type = None

    n_group_types_served_well = sum(1 for k, v in buckets.items() if v["success_rate"] >= 0.7 and v["n_reviews"] > 0)

    if n_group_types_served_well >= 3:
        group_versatility = "all_groups_welcome"
    elif buckets["family"]["success_rate"] >= 0.8 and buckets["family"]["n_reviews"] > 0:
        group_versatility = "family_friendly"
    elif buckets["social"]["success_rate"] >= 0.8 and buckets["social"]["n_reviews"] > 0:
        group_versatility = "great_for_celebrations"
    elif buckets["professional"]["success_rate"] >= 0.8 and buckets["professional"]["n_reviews"] > 0:
        group_versatility = "handles_work_events"
    elif any(v["success_rate"] >= 0.6 and v["n_reviews"] > 0 for v in buckets.values()):
        group_versatility = "situational"
    else:
        group_versatility = "not_recommended"

    return {
        "family": buckets["family"],
        "social": buckets["social"],
        "professional": buckets["professional"],
        "best_group_type": best_group_type,
        "best_group_success_rate": round(best_group_success_rate, 3),
        "worst_group_type": worst_group_type,
        "n_group_types_served_well": n_group_types_served_well,
        "group_versatility": group_versatility,
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
    group_judgments = [j for j in judgments if j.get("is_group_related", False)]
    n_group_reviews = len(group_judgments)

    if n_group_reviews == 0:
        confidence_level = "none"
    elif n_group_reviews <= 2:
        confidence_level = "low"
    elif n_group_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

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

        if group_size in ("large", "very_large"):
            n_large_groups += 1
            if seating_accommodation in ("excellent", "adequate"):
                n_large_success += 1

        if seating_accommodation == "refused":
            n_refused += 1

        if group_type == "family":
            n_family += 1
            if kids_friendly == "very_friendly":
                n_family_friendly += 1

        if group_type == "celebration":
            n_celebrations += 1
            if celebration_support == "excellent":
                n_celebration_success += 1

    # L1.5 Group Buckets
    l15_buckets = compute_l15_buckets(group_judgments)
    group_versatility = l15_buckets["group_versatility"]
    group_versatility_bonus = GROUP_VERSATILITY_BONUS.get(group_versatility, 0.0)

    # Weighted aggregation
    weighted_mean_score = sum_l1_score / max(sum_size_weight, 1)

    # Formulas (G2l: complex + L1.5)
    adjusted_score = weighted_mean_score

    # G2l: adds GROUP_VERSATILITY_BONUS to complex formula
    raw_score = BASE_SCORE + adjusted_score + group_versatility_bonus
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
    if final_score >= 7.5:
        base_verdict_by_score = "Excellent for Groups"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good for Groups"
    elif final_score >= 3.5:
        base_verdict_by_score = "Limited"
    else:
        base_verdict_by_score = "Not Suitable"

    override_applied = "none"
    verdict = base_verdict_by_score
    family_note = None
    celebration_note = None

    if n_refused >= 1 and verdict in ("Excellent for Groups", "Good for Groups"):
        override_applied = "refused_max_limited"
        verdict = "Limited"
    elif n_large_success >= 2 and verdict in ("Limited", "Not Suitable"):
        override_applied = "large_success_min_good"
        verdict = "Good for Groups"
    elif n_family_friendly >= 2:
        family_note = "Especially family-friendly"
    elif n_celebration_success >= 2:
        celebration_note = "Great for celebrations"
    elif weighted_mean_score < -2:
        override_applied = "low_weighted_mean"
        verdict = "Not Suitable"

    result = {
        # L1.5 Group Buckets
        "L1_5_group_buckets": l15_buckets,
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
        "GROUP_VERSATILITY_BONUS": group_versatility_bonus,
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
