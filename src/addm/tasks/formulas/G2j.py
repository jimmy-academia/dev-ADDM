"""
Ground Truth computation for G2j (Group Dining - Simple + L1.5).

Implements the formula from data/tasks/yelp/G2j_prompt.txt.
Simple formula with L1.5 group type grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

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


def compute_l1_success(judgment: Dict[str, Any]) -> bool:
    """
    SUCCESSFUL_GROUP_EXPERIENCE = true iff ALL:
      - GROUP_TYPE != not_group
      - SEATING_ACCOMMODATION in {excellent, adequate}
      - SERVICE_HANDLING in {smooth, adequate}
    """
    group_type = judgment.get("group_type", "not_group")
    seating_accommodation = judgment.get("seating_accommodation", "not_mentioned")
    service_handling = judgment.get("service_handling", "not_mentioned")

    if group_type == "not_group":
        return False

    if seating_accommodation not in ("excellent", "adequate"):
        return False

    if service_handling not in ("smooth", "adequate"):
        return False

    return True


def compute_l1_failed(judgment: Dict[str, Any]) -> bool:
    """
    FAILED_GROUP_EXPERIENCE = true iff ALL:
      - GROUP_TYPE != not_group
      - Any of: SEATING_ACCOMMODATION in {difficult, refused}, SERVICE_HANDLING in {struggled, overwhelmed}
    """
    group_type = judgment.get("group_type", "not_group")
    seating_accommodation = judgment.get("seating_accommodation", "not_mentioned")
    service_handling = judgment.get("service_handling", "not_mentioned")

    if group_type == "not_group":
        return False

    has_failure = (
        seating_accommodation in ("difficult", "refused")
        or service_handling in ("struggled", "overwhelmed")
    )

    return has_failure


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

    # Compute success rates
    for key, bucket in buckets.items():
        n_success = bucket["n_success"]
        n_failed = bucket["n_failed"]
        bucket["success_rate"] = n_success / max(n_success + n_failed, 1)

    # Find best and worst group types
    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_success"] + v["n_failed"] > 0]

    if buckets_with_data:
        best_group_type, best_bucket = max(buckets_with_data, key=lambda x: x[1]["success_rate"])
        best_group_success_rate = best_bucket["success_rate"]
        worst_group_type, _ = min(buckets_with_data, key=lambda x: x[1]["success_rate"])
    else:
        best_group_type = None
        best_group_success_rate = 0.0
        worst_group_type = None

    # Count group types served well
    n_group_types_served_well = sum(1 for k, v in buckets.items() if v["success_rate"] >= 0.7 and v["n_reviews"] > 0)

    # Determine group versatility
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

    # Compute L1 for each judgment
    n_success = 0
    n_failed = 0

    for j in group_judgments:
        if compute_l1_success(j):
            n_success += 1
        if compute_l1_failed(j):
            n_failed += 1

    # L1.5 Group Buckets
    l15_buckets = compute_l15_buckets(group_judgments)
    group_versatility = l15_buckets["group_versatility"]
    group_versatility_bonus = GROUP_VERSATILITY_BONUS.get(group_versatility, 0.0)

    # Formulas
    success_rate = n_success / max(n_success + n_failed, 1)
    positive_score = n_success * 2
    negative_score = n_failed * 2.5

    # G2j: adds GROUP_VERSATILITY_BONUS to simple formula
    raw_score = BASE_SCORE + positive_score - negative_score + group_versatility_bonus
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
    group_recommendation = None

    # Override: If GROUP_VERSATILITY = "all_groups_welcome" => min Good for Groups
    if group_versatility == "all_groups_welcome" and verdict in ("Limited", "Not Suitable"):
        override_applied = "versatile_min_good"
        verdict = "Good for Groups"
    # Override: If GROUP_VERSATILITY = "family_friendly" AND FAMILY bucket N_SUCCESS >= 2 => min Limited + family recommendation
    elif group_versatility == "family_friendly" and l15_buckets["family"]["n_success"] >= 2 and verdict == "Not Suitable":
        override_applied = "family_friendly_min_limited"
        verdict = "Limited"
        group_recommendation = "Recommended for families"
    # Override: If any bucket has N_FAILED >= 2 => max Limited for that group type
    elif any(v["n_failed"] >= 2 for v in [l15_buckets["family"], l15_buckets["social"], l15_buckets["professional"]]) and verdict in ("Excellent for Groups", "Good for Groups"):
        override_applied = "bucket_failures_max_limited"
        verdict = "Limited"
    # Override: If GROUP_VERSATILITY = "not_recommended" AND N_FAILED >= 2 => Not Suitable
    elif group_versatility == "not_recommended" and n_failed >= 2:
        override_applied = "not_recommended_with_failures"
        verdict = "Not Suitable"

    result = {
        # L1.5 Group Buckets
        "L1_5_group_buckets": l15_buckets,
        # L2 Aggregates
        "N_GROUP_REVIEWS": n_group_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_SUCCESS": n_success,
        "N_FAILED": n_failed,
        "SUCCESS_RATE": round(success_rate, 3),
        # Formula results
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
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

    if group_recommendation:
        result["group_recommendation"] = group_recommendation

    return result
