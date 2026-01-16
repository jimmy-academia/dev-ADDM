"""
Ground Truth computation for G6j (Loyalty - Simple + L1.5).

Implements the formula from data/tasks/yelp/G6j_prompt.txt.
Simple formula with L1.5 loyalty-type grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

LOYALTY_PATTERN_BONUS = {
    "true_loyalty": 2.0,
    "habitual_loyalty": 1.5,
    "emotional_attachment": 1.5,
    "brand_ambassador": 1.5,
    "partial_loyalty": 0.5,
    "at_risk": -1.5,
    "transactional": -1.0,
}


# =============================================================================
# Helpers
# =============================================================================


def is_strong_behavioral(judgment: Dict[str, Any]) -> bool:
    return (judgment.get("visit_frequency") == "regular" and
            judgment.get("return_intention") == "definitely_returning")


def is_strong_emotional(judgment: Dict[str, Any]) -> bool:
    return (judgment.get("relationship_depth") == "personal_connection" and
            judgment.get("loyalty_trigger") == "multiple")


def is_strong_advocacy(judgment: Dict[str, Any]) -> bool:
    return (judgment.get("recommendation_likelihood") == "highly_recommend" and
            judgment.get("advocacy_behavior") in ("brings_others", "shares_socially"))


def compute_l1_positive_loyalty(judgment: Dict[str, Any]) -> bool:
    return_intention = judgment.get("return_intention", "not_mentioned")
    recommendation_likelihood = judgment.get("recommendation_likelihood", "not_mentioned")
    visit_frequency = judgment.get("visit_frequency", "not_mentioned")
    advocacy_behavior = judgment.get("advocacy_behavior", "not_mentioned")

    return (return_intention in ("definitely_returning", "likely_returning") or
            recommendation_likelihood in ("highly_recommend", "recommend") or
            visit_frequency == "regular" or
            advocacy_behavior in ("brings_others", "shares_socially", "defends"))


def compute_l1_negative_loyalty(judgment: Dict[str, Any]) -> bool:
    return_intention = judgment.get("return_intention", "not_mentioned")
    recommendation_likelihood = judgment.get("recommendation_likelihood", "not_mentioned")

    return (return_intention in ("unlikely_returning", "never_returning") or
            recommendation_likelihood in ("discourage", "strongly_discourage"))


def compute_l15_buckets(loyalty_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "behavioral": {"n_reviews": 0, "n_strong": 0, "n_weak": 0},
        "emotional": {"n_reviews": 0, "n_strong": 0, "n_weak": 0},
        "advocacy": {"n_reviews": 0, "n_strong": 0, "n_weak": 0},
    }

    for j in loyalty_judgments:
        visit_frequency = j.get("visit_frequency", "not_mentioned")
        return_intention = j.get("return_intention", "not_mentioned")
        relationship_depth = j.get("relationship_depth", "not_mentioned")
        loyalty_trigger = j.get("loyalty_trigger", "not_mentioned")
        recommendation_likelihood = j.get("recommendation_likelihood", "not_mentioned")
        advocacy_behavior = j.get("advocacy_behavior", "not_mentioned")

        # Behavioral
        if visit_frequency != "not_mentioned" or return_intention != "not_mentioned":
            buckets["behavioral"]["n_reviews"] += 1
            if is_strong_behavioral(j):
                buckets["behavioral"]["n_strong"] += 1
            if return_intention in ("unlikely_returning", "never_returning"):
                buckets["behavioral"]["n_weak"] += 1

        # Emotional
        if relationship_depth != "not_mentioned" or loyalty_trigger != "not_mentioned":
            buckets["emotional"]["n_reviews"] += 1
            if is_strong_emotional(j):
                buckets["emotional"]["n_strong"] += 1
            if relationship_depth == "anonymous":
                buckets["emotional"]["n_weak"] += 1

        # Advocacy
        if recommendation_likelihood != "not_mentioned" or advocacy_behavior != "not_mentioned":
            buckets["advocacy"]["n_reviews"] += 1
            if is_strong_advocacy(j):
                buckets["advocacy"]["n_strong"] += 1
            if recommendation_likelihood in ("discourage", "strongly_discourage"):
                buckets["advocacy"]["n_weak"] += 1

    for key, bucket in buckets.items():
        bucket["loyalty_rate"] = bucket["n_strong"] / max(bucket["n_reviews"], 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_reviews"] > 0]

    if buckets_with_data:
        strongest_type, strongest_bucket = max(buckets_with_data, key=lambda x: x[1]["loyalty_rate"])
        strongest_rate = strongest_bucket["loyalty_rate"]
        weakest_type, _ = min(buckets_with_data, key=lambda x: x[1]["loyalty_rate"])
    else:
        strongest_type = None
        strongest_rate = 0.0
        weakest_type = None

    n_types_strong = sum(1 for k, v in buckets.items() if v["loyalty_rate"] >= 0.6 and v["n_reviews"] > 0)
    all_at_risk = all(v["loyalty_rate"] < 0.3 for v in buckets.values() if v["n_reviews"] > 0)

    if n_types_strong >= 3:
        loyalty_pattern = "true_loyalty"
    elif buckets["behavioral"]["loyalty_rate"] >= 0.7 and buckets["behavioral"]["n_reviews"] > 0:
        loyalty_pattern = "habitual_loyalty"
    elif buckets["emotional"]["loyalty_rate"] >= 0.7 and buckets["emotional"]["n_reviews"] > 0:
        loyalty_pattern = "emotional_attachment"
    elif buckets["advocacy"]["loyalty_rate"] >= 0.7 and buckets["advocacy"]["n_reviews"] > 0:
        loyalty_pattern = "brand_ambassador"
    elif n_types_strong >= 1:
        loyalty_pattern = "partial_loyalty"
    elif all_at_risk and any(v["n_reviews"] > 0 for v in buckets.values()):
        loyalty_pattern = "at_risk"
    else:
        loyalty_pattern = "transactional"

    return {
        "behavioral": buckets["behavioral"],
        "emotional": buckets["emotional"],
        "advocacy": buckets["advocacy"],
        "strongest_type": strongest_type,
        "strongest_rate": round(strongest_rate, 3),
        "weakest_type": weakest_type,
        "n_types_strong": n_types_strong,
        "loyalty_pattern": loyalty_pattern,
    }


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    loyalty_judgments = [j for j in judgments if j.get("is_loyalty_related", False)]
    n_loyalty_reviews = len(loyalty_judgments)

    if n_loyalty_reviews == 0:
        confidence_level = "none"
    elif n_loyalty_reviews <= 2:
        confidence_level = "low"
    elif n_loyalty_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    n_positive = 0
    n_negative = 0

    for j in loyalty_judgments:
        if compute_l1_positive_loyalty(j):
            n_positive += 1
        if compute_l1_negative_loyalty(j):
            n_negative += 1

    # L1.5 Type Buckets
    l15_buckets = compute_l15_buckets(loyalty_judgments)
    loyalty_pattern = l15_buckets["loyalty_pattern"]
    loyalty_pattern_bonus = LOYALTY_PATTERN_BONUS.get(loyalty_pattern, 0.0)

    # Formulas
    loyalty_rate = n_positive / max(n_positive + n_negative, 1)
    positive_score = n_positive * 1.5
    negative_score = n_negative * 1.5

    raw_score = BASE_SCORE + positive_score - negative_score + loyalty_pattern_bonus
    final_score = max(0.0, min(10.0, raw_score))

    if final_score >= 7.5:
        base_verdict_by_score = "High Loyalty"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good Loyalty"
    elif final_score >= 3.5:
        base_verdict_by_score = "Moderate Loyalty"
    else:
        base_verdict_by_score = "Low Loyalty"

    override_applied = "none"
    verdict = base_verdict_by_score
    strength_note = None
    development_note = None

    if loyalty_pattern == "true_loyalty" and verdict in ("Moderate Loyalty", "Low Loyalty"):
        override_applied = "true_loyalty_min_good"
        verdict = "Good Loyalty"
    elif loyalty_pattern == "at_risk" and verdict in ("High Loyalty", "Good Loyalty"):
        override_applied = "at_risk_max_moderate"
        verdict = "Moderate Loyalty"

    if loyalty_pattern in ("true_loyalty", "habitual_loyalty", "emotional_attachment", "brand_ambassador"):
        strength_note = f"Pattern: {loyalty_pattern.replace('_', ' ').title()}"

    if l15_buckets["weakest_type"]:
        weakest_bucket = l15_buckets[l15_buckets["weakest_type"]]
        if weakest_bucket["loyalty_rate"] < 0.4:
            development_note = f"Could develop: {l15_buckets['weakest_type']} loyalty"

    result = {
        "L1_5_type_buckets": l15_buckets,
        "N_LOYALTY_REVIEWS": n_loyalty_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "LOYALTY_RATE": round(loyalty_rate, 3),
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "LOYALTY_PATTERN_BONUS": loyalty_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_loyalty_reviews": n_loyalty_reviews,
    }

    if strength_note:
        result["strength_note"] = strength_note
    if development_note:
        result["development_note"] = development_note

    return result
