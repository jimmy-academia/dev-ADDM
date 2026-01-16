"""
Ground Truth computation for G3f (Hidden Costs - Simple + L1.5).

Implements the formula from data/tasks/yelp/G3f_prompt.txt.
Simple formula with L1.5 cost type grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 7.0

# L1.5 Cost Pattern Bonus values
COST_PATTERN_BONUS = {
    "fully_transparent": 1.5,
    "service_charge_only": 0.5,
    "dining_fees": 0.0,
    "multiple_hidden": -1.5,
    "deceptive": -3.0,
}


# =============================================================================
# Helpers
# =============================================================================


def get_cost_bucket(cost_type: str) -> str:
    """Map cost type to L1.5 bucket."""
    if cost_type in ("service_charge", "corkage_fee", "minimum_spend"):
        return "service_fees"
    elif cost_type in ("split_plate_fee", "bread_charge"):
        return "dining_fees"
    elif cost_type in ("parking_fee", "reservation_fee"):
        return "access_fees"
    elif cost_type in ("price_increase", "other_fee"):
        return "other"
    return "none"


def compute_l1_hidden_issue(judgment: Dict[str, Any]) -> bool:
    cost_type = judgment.get("cost_type", "none")
    disclosure_quality = judgment.get("disclosure_quality", "not_mentioned")
    return cost_type != "none" and disclosure_quality in ("buried", "not_disclosed")


def compute_l1_transparent(judgment: Dict[str, Any]) -> bool:
    cost_type = judgment.get("cost_type", "none")
    disclosure_quality = judgment.get("disclosure_quality", "not_mentioned")
    praises_transparency = judgment.get("praises_transparency", False)
    if praises_transparency:
        return True
    if disclosure_quality == "clear" and cost_type != "none":
        return True
    return False


def compute_l1_deceptive(judgment: Dict[str, Any]) -> bool:
    disclosure_quality = judgment.get("disclosure_quality", "not_mentioned")
    surprise_level = judgment.get("surprise_level", "not_mentioned")
    return disclosure_quality == "not_disclosed" and surprise_level in ("shocked", "outraged")


def compute_l15_buckets(cost_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "service_fees": {"n_reviews": 0, "n_hidden": 0, "n_transparent": 0},
        "dining_fees": {"n_reviews": 0, "n_hidden": 0, "n_transparent": 0},
        "access_fees": {"n_reviews": 0, "n_hidden": 0, "n_transparent": 0},
        "other": {"n_reviews": 0, "n_hidden": 0, "n_transparent": 0},
    }

    for j in cost_judgments:
        cost_type = j.get("cost_type", "none")
        bucket = get_cost_bucket(cost_type)
        if bucket == "none" or bucket not in buckets:
            continue

        buckets[bucket]["n_reviews"] += 1

        if compute_l1_hidden_issue(j):
            buckets[bucket]["n_hidden"] += 1
        if compute_l1_transparent(j):
            buckets[bucket]["n_transparent"] += 1

    for key, bucket in buckets.items():
        n_hidden = bucket["n_hidden"]
        n_transparent = bucket["n_transparent"]
        bucket["transparency_rate"] = n_transparent / max(n_hidden + n_transparent, 1)

    buckets_with_issues = [k for k, v in buckets.items() if v["n_hidden"] > 0]
    n_fee_types_with_issues = len(buckets_with_issues)

    any_deceptive = any(compute_l1_deceptive(j) for j in cost_judgments)

    if n_fee_types_with_issues == 0:
        cost_pattern = "fully_transparent"
    elif any_deceptive:
        cost_pattern = "deceptive"
    elif n_fee_types_with_issues >= 2:
        cost_pattern = "multiple_hidden"
    elif buckets["service_fees"]["n_hidden"] > 0 and n_fee_types_with_issues == 1:
        cost_pattern = "service_charge_only"
    else:
        cost_pattern = "dining_fees"

    return {
        "service_fees": buckets["service_fees"],
        "dining_fees": buckets["dining_fees"],
        "access_fees": buckets["access_fees"],
        "other": buckets["other"],
        "n_fee_types_with_issues": n_fee_types_with_issues,
        "cost_pattern": cost_pattern,
    }


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    cost_judgments = [j for j in judgments if j.get("is_cost_related", False)]
    n_cost_reviews = len(cost_judgments)

    if n_cost_reviews == 0:
        confidence_level = "none"
    elif n_cost_reviews <= 2:
        confidence_level = "low"
    elif n_cost_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    n_hidden_issues = 0
    n_transparent = 0
    n_deceptive = 0

    for j in cost_judgments:
        if compute_l1_hidden_issue(j):
            n_hidden_issues += 1
        if compute_l1_transparent(j):
            n_transparent += 1
        if compute_l1_deceptive(j):
            n_deceptive += 1

    # L1.5 Cost Buckets
    l15_buckets = compute_l15_buckets(cost_judgments)
    cost_pattern = l15_buckets["cost_pattern"]
    cost_pattern_bonus = COST_PATTERN_BONUS.get(cost_pattern, 0.0)

    # Formulas
    transparency_rate = n_transparent / max(n_hidden_issues + n_transparent, 1)
    positive_score = n_transparent * 1
    negative_score = (n_hidden_issues * 1.5) + (n_deceptive * 3)

    raw_score = BASE_SCORE + positive_score - negative_score + cost_pattern_bonus
    final_score = max(0.0, min(10.0, raw_score))

    if final_score >= 8.0:
        base_verdict_by_score = "Transparent"
    elif final_score >= 6.0:
        base_verdict_by_score = "Mostly Transparent"
    elif final_score >= 4.0:
        base_verdict_by_score = "Some Hidden Costs"
    else:
        base_verdict_by_score = "Problematic"

    override_applied = "none"
    verdict = base_verdict_by_score

    if cost_pattern == "fully_transparent" and verdict in ("Some Hidden Costs", "Problematic"):
        override_applied = "transparent_pattern_min_mostly"
        verdict = "Mostly Transparent"
    elif cost_pattern == "deceptive":
        override_applied = "deceptive_pattern"
        verdict = "Problematic"
    elif cost_pattern == "multiple_hidden" and verdict in ("Transparent", "Mostly Transparent"):
        override_applied = "multiple_hidden_max_some"
        verdict = "Some Hidden Costs"

    return {
        "L1_5_cost_buckets": l15_buckets,
        "N_COST_REVIEWS": n_cost_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_HIDDEN_ISSUES": n_hidden_issues,
        "N_TRANSPARENT": n_transparent,
        "N_DECEPTIVE": n_deceptive,
        "TRANSPARENCY_RATE": round(transparency_rate, 3),
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "COST_PATTERN_BONUS": cost_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_cost_reviews": n_cost_reviews,
    }
