"""
Ground Truth computation for G6l (Loyalty - Complex + L1.5).

Implements the formula from data/tasks/yelp/G6l_prompt.txt.
Complex formula with weighted loyalty factors + L1.5 loyalty-type grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

RETURN_SCORES = {"definitely_returning": 4.0, "likely_returning": 2.5, "uncertain": 0, "unlikely_returning": -2.0, "never_returning": -4.0, "not_mentioned": 0}
RECOMMENDATION_SCORES = {"highly_recommend": 4.0, "recommend": 2.5, "conditional_recommend": 1.0, "neutral": 0, "discourage": -2.5, "strongly_discourage": -4.0, "not_mentioned": 0}
FREQUENCY_SCORES = {"regular": 3.0, "occasional": 1.0, "rare": 0, "first_time": 0, "not_mentioned": 0}
RELATIONSHIP_SCORES = {"personal_connection": 3.5, "recognized": 2.0, "familiar": 1.0, "anonymous": -0.5, "not_mentioned": 0}
ADVOCACY_SCORES = {"brings_others": 3.5, "shares_socially": 2.5, "defends": 3.0, "passive": 0, "not_mentioned": 0}
TRIGGER_SCORES = {"multiple": 2.0, "food": 1.5, "service": 1.5, "atmosphere": 1.0, "value": 1.0, "convenience": 0.5, "not_mentioned": 0}


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    return_intention = judgment.get("return_intention", "not_mentioned")
    recommendation_likelihood = judgment.get("recommendation_likelihood", "not_mentioned")
    visit_frequency = judgment.get("visit_frequency", "not_mentioned")
    relationship_depth = judgment.get("relationship_depth", "not_mentioned")
    advocacy_behavior = judgment.get("advocacy_behavior", "not_mentioned")
    loyalty_trigger = judgment.get("loyalty_trigger", "not_mentioned")

    return_score = RETURN_SCORES.get(return_intention, 0)
    recommendation_score = RECOMMENDATION_SCORES.get(recommendation_likelihood, 0)
    frequency_score = FREQUENCY_SCORES.get(visit_frequency, 0)
    relationship_score = RELATIONSHIP_SCORES.get(relationship_depth, 0)
    advocacy_score = ADVOCACY_SCORES.get(advocacy_behavior, 0)
    trigger_score = TRIGGER_SCORES.get(loyalty_trigger, 0)

    l1_loyalty_score = (
        return_score + recommendation_score + frequency_score +
        relationship_score + advocacy_score + trigger_score
    )

    # Loyal advocate bonus
    loyal_advocate_bonus = 2.0 if (visit_frequency == "regular" and
                                    advocacy_behavior in ("brings_others", "shares_socially", "defends")) else 0.0

    l1_total_score = l1_loyalty_score + loyal_advocate_bonus

    return {
        "return_score": return_score,
        "recommendation_score": recommendation_score,
        "frequency_score": frequency_score,
        "relationship_score": relationship_score,
        "advocacy_score": advocacy_score,
        "trigger_score": trigger_score,
        "l1_loyalty_score": round(l1_loyalty_score, 2),
        "loyal_advocate_bonus": loyal_advocate_bonus,
        "l1_total_score": round(l1_total_score, 2),
    }


def compute_l15_buckets(loyalty_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "behavioral": {"n_reviews": 0, "sum_score": 0.0, "n_strong": 0, "n_weak": 0},
        "emotional": {"n_reviews": 0, "sum_score": 0.0, "n_strong": 0, "n_weak": 0},
        "advocacy": {"n_reviews": 0, "sum_score": 0.0, "n_strong": 0, "n_weak": 0},
    }

    for j in loyalty_judgments:
        visit_frequency = j.get("visit_frequency", "not_mentioned")
        return_intention = j.get("return_intention", "not_mentioned")
        relationship_depth = j.get("relationship_depth", "not_mentioned")
        loyalty_trigger = j.get("loyalty_trigger", "not_mentioned")
        recommendation_likelihood = j.get("recommendation_likelihood", "not_mentioned")
        advocacy_behavior = j.get("advocacy_behavior", "not_mentioned")

        frequency_score = FREQUENCY_SCORES.get(visit_frequency, 0)
        return_score = RETURN_SCORES.get(return_intention, 0)
        relationship_score = RELATIONSHIP_SCORES.get(relationship_depth, 0)
        trigger_score = TRIGGER_SCORES.get(loyalty_trigger, 0)
        recommendation_score = RECOMMENDATION_SCORES.get(recommendation_likelihood, 0)
        advocacy_score = ADVOCACY_SCORES.get(advocacy_behavior, 0)

        # Behavioral
        if visit_frequency != "not_mentioned" or return_intention != "not_mentioned":
            behavioral_score = frequency_score + return_score
            buckets["behavioral"]["n_reviews"] += 1
            buckets["behavioral"]["sum_score"] += behavioral_score
            if behavioral_score >= 4.0:
                buckets["behavioral"]["n_strong"] += 1
            if behavioral_score < 0:
                buckets["behavioral"]["n_weak"] += 1

        # Emotional
        if relationship_depth != "not_mentioned" or loyalty_trigger != "not_mentioned":
            emotional_score = relationship_score + trigger_score
            buckets["emotional"]["n_reviews"] += 1
            buckets["emotional"]["sum_score"] += emotional_score
            if emotional_score >= 4.0:
                buckets["emotional"]["n_strong"] += 1
            if emotional_score < 0:
                buckets["emotional"]["n_weak"] += 1

        # Advocacy
        if recommendation_likelihood != "not_mentioned" or advocacy_behavior != "not_mentioned":
            adv_score = recommendation_score + advocacy_score
            buckets["advocacy"]["n_reviews"] += 1
            buckets["advocacy"]["sum_score"] += adv_score
            if adv_score >= 4.0:
                buckets["advocacy"]["n_strong"] += 1
            if adv_score < 0:
                buckets["advocacy"]["n_weak"] += 1

    for key, bucket in buckets.items():
        bucket["mean_score"] = bucket["sum_score"] / max(bucket["n_reviews"], 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_reviews"] > 0]

    if buckets_with_data:
        strongest_type, strongest_bucket = max(buckets_with_data, key=lambda x: x[1]["mean_score"])
        strongest_score = strongest_bucket["mean_score"]
        weakest_type, weakest_bucket = min(buckets_with_data, key=lambda x: x[1]["mean_score"])
        weakest_score = weakest_bucket["mean_score"]
    else:
        strongest_type = None
        strongest_score = 0.0
        weakest_type = None
        weakest_score = 0.0

    n_types_strong = sum(1 for k, v in buckets.items() if v["mean_score"] >= 3.0 and v["n_reviews"] > 0)
    any_at_risk = any(v["mean_score"] < -1.0 for v in buckets.values() if v["n_reviews"] > 0)

    if n_types_strong >= 3:
        loyalty_pattern = "loyalty_fortress"
    elif buckets["behavioral"]["mean_score"] >= 4.0 and buckets["behavioral"]["n_reviews"] > 0:
        loyalty_pattern = "habit_driven"
    elif buckets["emotional"]["mean_score"] >= 4.0 and buckets["emotional"]["n_reviews"] > 0:
        loyalty_pattern = "emotionally_connected"
    elif buckets["advocacy"]["mean_score"] >= 4.0 and buckets["advocacy"]["n_reviews"] > 0:
        loyalty_pattern = "active_promoter"
    elif strongest_score - weakest_score > 4.0 and buckets_with_data:
        loyalty_pattern = "loyalty_gap"
    elif any_at_risk:
        loyalty_pattern = "loyalty_risk"
    elif n_types_strong >= 1:
        loyalty_pattern = "emerging_loyalty"
    else:
        loyalty_pattern = "loyalty_challenge"

    return {
        "behavioral": buckets["behavioral"],
        "emotional": buckets["emotional"],
        "advocacy": buckets["advocacy"],
        "strongest_type": strongest_type,
        "strongest_score": round(strongest_score, 3) if strongest_score else 0.0,
        "weakest_type": weakest_type,
        "weakest_score": round(weakest_score, 3) if weakest_score else 0.0,
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

    n_regulars = 0
    n_never_returning = 0

    for j in loyalty_judgments:
        if j.get("visit_frequency") == "regular":
            n_regulars += 1
        if j.get("return_intention") == "never_returning":
            n_never_returning += 1

    # L1.5 Type Buckets
    l15_buckets = compute_l15_buckets(loyalty_judgments)
    loyalty_pattern = l15_buckets["loyalty_pattern"]

    # Pattern multiplier
    if loyalty_pattern == "loyalty_fortress":
        pattern_mult = 1.25
    elif loyalty_pattern in ("habit_driven", "emotionally_connected", "active_promoter"):
        pattern_mult = 1.15
    elif loyalty_pattern == "loyalty_risk":
        pattern_mult = 0.85
    elif loyalty_pattern == "loyalty_challenge":
        pattern_mult = 0.75
    else:
        pattern_mult = 1.0

    # Aggregate L1.5 scores
    total_type_score = sum(
        v["mean_score"] for v in [l15_buckets["behavioral"], l15_buckets["emotional"], l15_buckets["advocacy"]]
        if v["n_reviews"] > 0
    )
    n_type_buckets_active = sum(
        1 for v in [l15_buckets["behavioral"], l15_buckets["emotional"], l15_buckets["advocacy"]]
        if v["n_reviews"] > 0
    )
    mean_type_score = total_type_score / max(n_type_buckets_active, 1)

    adjusted_score = mean_type_score * pattern_mult
    raw_score = BASE_SCORE + adjusted_score
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
    driver_note = None
    development_note = None
    risk_warning = None

    if loyalty_pattern == "loyalty_fortress" and verdict in ("Moderate Loyalty", "Low Loyalty"):
        override_applied = "loyalty_fortress_min_good"
        verdict = "Good Loyalty"
    elif loyalty_pattern == "loyalty_risk" and n_never_returning >= 1 and verdict in ("High Loyalty", "Good Loyalty"):
        override_applied = "loyalty_risk_max_moderate"
        verdict = "Moderate Loyalty"
        risk_warning = "Customer retention at risk"

    if l15_buckets["strongest_type"]:
        driver_note = f"Loyalty driver: {l15_buckets['strongest_type']}"

    if l15_buckets["weakest_type"] and l15_buckets["weakest_score"] < 1.0:
        development_note = f"Could develop: {l15_buckets['weakest_type']}"

    result = {
        "L1_5_type_buckets": l15_buckets,
        "N_LOYALTY_REVIEWS": n_loyalty_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_REGULARS": n_regulars,
        "N_NEVER_RETURNING": n_never_returning,
        "TOTAL_TYPE_SCORE": round(total_type_score, 3),
        "MEAN_TYPE_SCORE": round(mean_type_score, 3),
        "PATTERN_MULT": pattern_mult,
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_loyalty_reviews": n_loyalty_reviews,
    }

    if driver_note:
        result["driver_note"] = driver_note
    if development_note:
        result["development_note"] = development_note
    if risk_warning:
        result["risk_warning"] = risk_warning

    return result
