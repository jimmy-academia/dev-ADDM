"""
Ground Truth computation for G2i (Group Dining - Simple).

Implements the formula from data/tasks/yelp/G2i_prompt.txt.
Simple formula without credibility weighting.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0


# =============================================================================
# Helpers
# =============================================================================


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
    n_excellent_seating = 0
    n_large_groups = 0
    n_large_success = 0
    n_easy_split = 0
    n_kids_friendly = 0
    has_refused = False

    for j in group_judgments:
        is_success = compute_l1_success(j)
        if is_success:
            n_success += 1
        if compute_l1_failed(j):
            n_failed += 1

        if j.get("seating_accommodation") == "excellent":
            n_excellent_seating += 1
        if j.get("seating_accommodation") == "refused":
            has_refused = True

        group_size = j.get("group_size", "not_mentioned")
        is_large = group_size in ("large", "very_large")
        if is_large:
            n_large_groups += 1
            if is_success:
                n_large_success += 1

        if j.get("check_splitting") == "easy":
            n_easy_split += 1

        if j.get("kids_friendly") == "very_friendly":
            n_kids_friendly += 1

    # Formulas
    positive_score = (n_success * 2) + (n_excellent_seating * 1) + (n_large_success * 1) + (n_easy_split * 0.5)
    negative_score = n_failed * 2.5

    raw_score = BASE_SCORE + positive_score - negative_score
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

    # Override 1: If N_LARGE_SUCCESS >= 2 => min Good for Groups
    if n_large_success >= 2 and verdict in ("Limited", "Not Suitable"):
        override_applied = "large_success_min_good"
        verdict = "Good for Groups"
    # Override 2: If SEATING_ACCOMMODATION = refused in any review => max Limited
    elif has_refused and verdict in ("Excellent for Groups", "Good for Groups"):
        override_applied = "refused_max_limited"
        verdict = "Limited"
    # Override 3: If N_FAILED >= 2 => max Limited
    elif n_failed >= 2 and verdict in ("Excellent for Groups", "Good for Groups"):
        override_applied = "failed_max_limited"
        verdict = "Limited"

    return {
        # L2 Aggregates
        "N_GROUP_REVIEWS": n_group_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_SUCCESS": n_success,
        "N_FAILED": n_failed,
        "N_EXCELLENT_SEATING": n_excellent_seating,
        "N_LARGE_GROUPS": n_large_groups,
        "N_LARGE_SUCCESS": n_large_success,
        "N_EASY_SPLIT": n_easy_split,
        "N_KIDS_FRIENDLY": n_kids_friendly,
        # Formula results
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_group_reviews": n_group_reviews,
    }
