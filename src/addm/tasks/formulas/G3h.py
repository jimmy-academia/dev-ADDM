"""
Ground Truth computation for G3h (Hidden Costs - Complex + L1.5).

Implements the formula from data/tasks/yelp/G3h_prompt.txt.
Complex formula with cost severity weighting + L1.5 cost type grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 7.0

COST_TYPE_WEIGHTS = {
    "service_charge": 1.5, "corkage_fee": 1.0, "split_plate_fee": 1.2,
    "bread_charge": 1.3, "parking_fee": 0.8, "reservation_fee": 1.4,
    "minimum_spend": 1.2, "price_increase": 1.8, "other_fee": 1.0, "none": 0,
}
DISCLOSURE_SCORES = {"clear": 2, "buried": -1, "not_disclosed": -3, "not_mentioned": 0}
SURPRISE_MODIFIERS = {"expected": 1, "mild_surprise": 0, "shocked": -2, "outraged": -4, "not_mentioned": 0}
COST_AMOUNT_MODIFIERS = {"minor": 0.5, "moderate": 0, "significant": -1, "percentage": -0.5, "not_specified": 0}

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
    if cost_type in ("service_charge", "corkage_fee", "minimum_spend"):
        return "service_fees"
    elif cost_type in ("split_plate_fee", "bread_charge"):
        return "dining_fees"
    elif cost_type in ("parking_fee", "reservation_fee"):
        return "access_fees"
    elif cost_type in ("price_increase", "other_fee"):
        return "other"
    return "none"


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    cost_type = judgment.get("cost_type", "none")
    disclosure_quality = judgment.get("disclosure_quality", "not_mentioned")
    surprise_level = judgment.get("surprise_level", "not_mentioned")
    cost_amount = judgment.get("cost_amount", "not_specified")
    staff_explanation = judgment.get("staff_explanation", "not_mentioned")

    cost_type_weight = COST_TYPE_WEIGHTS.get(cost_type, 0)
    disclosure_score = DISCLOSURE_SCORES.get(disclosure_quality, 0)
    surprise_modifier = SURPRISE_MODIFIERS.get(surprise_level, 0)
    cost_amount_modifier = COST_AMOUNT_MODIFIERS.get(cost_amount, 0)

    if staff_explanation == "proactive":
        staff_bonus = 1.0
    elif staff_explanation == "upon_request":
        staff_bonus = 0.5
    elif staff_explanation == "defensive":
        staff_bonus = -1.0
    elif staff_explanation == "no_explanation":
        staff_bonus = -0.5
    else:
        staff_bonus = 0.0

    l1_review_score = (disclosure_score + surprise_modifier + cost_amount_modifier + staff_bonus) * (1 + cost_type_weight * 0.2)
    deceptive_penalty = -3.0 if (disclosure_quality == "not_disclosed" and surprise_level in ("shocked", "outraged")) else 0.0
    l1_total_score = l1_review_score + deceptive_penalty

    return {
        "cost_type_weight": cost_type_weight,
        "disclosure_score": disclosure_score,
        "l1_review_score": round(l1_review_score, 2),
        "deceptive_penalty": deceptive_penalty,
        "l1_total_score": round(l1_total_score, 2),
    }


def compute_l1_hidden_issue(judgment: Dict[str, Any]) -> bool:
    cost_type = judgment.get("cost_type", "none")
    disclosure_quality = judgment.get("disclosure_quality", "not_mentioned")
    return cost_type != "none" and disclosure_quality in ("buried", "not_disclosed")


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
        if j.get("disclosure_quality") == "clear":
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

    sum_l1_score = 0.0
    n_deceptive = 0
    n_proactive_staff = 0

    for j in cost_judgments:
        l1 = compute_l1_complex(j)
        sum_l1_score += l1["l1_total_score"]

        if l1["deceptive_penalty"] < 0:
            n_deceptive += 1
        if j.get("staff_explanation") == "proactive":
            n_proactive_staff += 1

    # L1.5 Cost Buckets
    l15_buckets = compute_l15_buckets(cost_judgments)
    cost_pattern = l15_buckets["cost_pattern"]
    cost_pattern_bonus = COST_PATTERN_BONUS.get(cost_pattern, 0.0)

    mean_l1_score = sum_l1_score / max(n_cost_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score + cost_pattern_bonus
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

    if cost_pattern == "deceptive":
        override_applied = "deceptive_pattern"
        verdict = "Problematic"
    elif n_deceptive >= 2:
        override_applied = "multiple_deceptive"
        verdict = "Problematic"
    elif n_proactive_staff >= 2 and n_deceptive == 0 and verdict in ("Some Hidden Costs", "Problematic"):
        override_applied = "proactive_staff_min_mostly"
        verdict = "Mostly Transparent"
    elif mean_l1_score < -2:
        override_applied = "low_mean_score"
        verdict = "Problematic"

    return {
        "L1_5_cost_buckets": l15_buckets,
        "N_COST_REVIEWS": n_cost_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_DECEPTIVE": n_deceptive,
        "N_PROACTIVE_STAFF": n_proactive_staff,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "COST_PATTERN_BONUS": cost_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_cost_reviews": n_cost_reviews,
    }
