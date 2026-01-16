"""
Ground Truth computation for G2d (Romance - Complex + L1.5).

Implements the formula from data/tasks/yelp/G2d_prompt.txt.
Complex formula with occasion weighting + L1.5 occasion type grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

# Occasion weights (higher stakes = more weight)
OCCASION_WEIGHTS = {
    "proposal": 2.0,
    "anniversary": 1.8,
    "valentines": 1.5,
    "date_night": 1.0,
    "first_date": 1.2,
    "general_romantic": 1.0,
    "not_romantic": 0,
}

# Ambiance scores
AMBIANCE_SCORES = {
    "excellent": 3,
    "good": 1.5,
    "neutral": 0,
    "poor": -2,
    "not_mentioned": 0,
}

# Privacy scores
PRIVACY_SCORES = {
    "intimate": 2,
    "adequate": 0.5,
    "lacking": -1.5,
    "not_mentioned": 0,
}

# Noise scores
NOISE_SCORES = {
    "quiet": 2,
    "moderate": 0,
    "loud": -3,
    "not_mentioned": 0,
}

# Service scores
SERVICE_SCORES = {
    "attentive": 1.5,
    "inattentive": -1,
    "intrusive": -1.5,
    "not_mentioned": 0,
}

# Romantic elements scores
ROMANTIC_ELEMENTS_SCORES = {
    "present": 1.5,
    "absent": 0,
    "negative": -2,
    "not_mentioned": 0,
}

# Outcome modifiers
OUTCOME_MODIFIERS = {
    "delighted": 2,
    "satisfied": 1,
    "disappointed": -2,
    "ruined": -4,
    "not_mentioned": 0,
}

# L1.5 Romance Versatility Bonus values
ROMANCE_VERSATILITY_BONUS = {
    "versatile": 2.0,
    "special_occasion_specialist": 1.5,
    "date_night_favorite": 1.0,
    "first_date_friendly": 0.5,
    "situational": 0.0,
    "not_recommended": -1.0,
}


# =============================================================================
# Helpers
# =============================================================================


def get_occasion_bucket(occasion_type: str) -> str:
    """Map occasion type to L1.5 bucket."""
    if occasion_type in ("anniversary", "proposal", "valentines"):
        return "special_occasion"
    elif occasion_type in ("date_night", "general_romantic"):
        return "date_night"
    elif occasion_type == "first_date":
        return "first_date"
    else:
        return "other"


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    """Compute L1 composites using complex formula."""
    occasion_type = judgment.get("occasion_type", "not_romantic")
    ambiance_rating = judgment.get("ambiance_rating", "not_mentioned")
    privacy_level = judgment.get("privacy_level", "not_mentioned")
    noise_level = judgment.get("noise_level", "not_mentioned")
    service_attentiveness = judgment.get("service_attentiveness", "not_mentioned")
    romantic_elements = judgment.get("romantic_elements", "not_mentioned")
    outcome_sentiment = judgment.get("outcome_sentiment", "not_mentioned")

    occasion_weight = OCCASION_WEIGHTS.get(occasion_type, 0)
    ambiance_score = AMBIANCE_SCORES.get(ambiance_rating, 0)
    privacy_score = PRIVACY_SCORES.get(privacy_level, 0)
    noise_score = NOISE_SCORES.get(noise_level, 0)
    service_score = SERVICE_SCORES.get(service_attentiveness, 0)
    romantic_elements_score = ROMANTIC_ELEMENTS_SCORES.get(romantic_elements, 0)
    outcome_modifier = OUTCOME_MODIFIERS.get(outcome_sentiment, 0)

    l1_review_score = (ambiance_score + privacy_score + noise_score + service_score + romantic_elements_score + outcome_modifier) * occasion_weight

    ruined_special_occasion = -5.0 if (occasion_type in ("proposal", "anniversary") and outcome_sentiment == "ruined") else 0.0
    perfect_proposal = 3.0 if (occasion_type == "proposal" and outcome_sentiment == "delighted") else 0.0

    l1_total_score = l1_review_score + ruined_special_occasion + perfect_proposal

    return {
        "occasion_weight": occasion_weight,
        "ambiance_score": ambiance_score,
        "privacy_score": privacy_score,
        "noise_score": noise_score,
        "service_score": service_score,
        "romantic_elements_score": romantic_elements_score,
        "outcome_modifier": outcome_modifier,
        "l1_review_score": round(l1_review_score, 2),
        "ruined_special_occasion": ruined_special_occasion,
        "perfect_proposal": perfect_proposal,
        "l1_total_score": round(l1_total_score, 2),
    }


def compute_l1_positive(judgment: Dict[str, Any]) -> bool:
    """Simple positive check for L1.5 bucket tracking."""
    occasion_type = judgment.get("occasion_type", "not_romantic")
    ambiance_rating = judgment.get("ambiance_rating", "not_mentioned")
    privacy_level = judgment.get("privacy_level", "not_mentioned")
    noise_level = judgment.get("noise_level", "not_mentioned")
    romantic_elements = judgment.get("romantic_elements", "not_mentioned")

    if occasion_type == "not_romantic":
        return False

    if ambiance_rating not in ("excellent", "good"):
        return False

    return privacy_level == "intimate" or noise_level == "quiet" or romantic_elements == "present"


def compute_l1_negative(judgment: Dict[str, Any]) -> bool:
    """Simple negative check for L1.5 bucket tracking."""
    occasion_type = judgment.get("occasion_type", "not_romantic")
    ambiance_rating = judgment.get("ambiance_rating", "not_mentioned")
    privacy_level = judgment.get("privacy_level", "not_mentioned")
    noise_level = judgment.get("noise_level", "not_mentioned")

    if occasion_type == "not_romantic":
        return False

    return ambiance_rating == "poor" or noise_level == "loud" or privacy_level == "lacking"


def compute_l15_buckets(romantic_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute L1.5 occasion type buckets."""
    buckets = {
        "special_occasion": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
        "date_night": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
        "first_date": {"n_reviews": 0, "n_positive": 0, "n_negative": 0},
    }

    for j in romantic_judgments:
        occasion_type = j.get("occasion_type", "not_romantic")
        bucket = get_occasion_bucket(occasion_type)
        if bucket not in buckets:
            continue

        buckets[bucket]["n_reviews"] += 1

        if compute_l1_positive(j):
            buckets[bucket]["n_positive"] += 1
        if compute_l1_negative(j):
            buckets[bucket]["n_negative"] += 1

    for key, bucket in buckets.items():
        n_positive = bucket["n_positive"]
        n_negative = bucket["n_negative"]
        bucket["success_rate"] = n_positive / max(n_positive + n_negative, 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_positive"] + v["n_negative"] > 0]

    if buckets_with_data:
        best_occasion, best_bucket = max(buckets_with_data, key=lambda x: x[1]["success_rate"])
        best_occasion_success_rate = best_bucket["success_rate"]
        worst_occasion, _ = min(buckets_with_data, key=lambda x: x[1]["success_rate"])
    else:
        best_occasion = None
        best_occasion_success_rate = 0.0
        worst_occasion = None

    n_occasions_served_well = sum(1 for k, v in buckets.items() if v["success_rate"] >= 0.7 and v["n_reviews"] > 0)

    if n_occasions_served_well >= 3:
        romance_versatility = "versatile"
    elif buckets["special_occasion"]["success_rate"] >= 0.8 and buckets["special_occasion"]["n_reviews"] > 0:
        romance_versatility = "special_occasion_specialist"
    elif buckets["date_night"]["success_rate"] >= 0.8 and buckets["date_night"]["n_reviews"] > 0:
        romance_versatility = "date_night_favorite"
    elif buckets["first_date"]["success_rate"] >= 0.8 and buckets["first_date"]["n_reviews"] > 0:
        romance_versatility = "first_date_friendly"
    elif any(v["success_rate"] >= 0.6 and v["n_reviews"] > 0 for v in buckets.values()):
        romance_versatility = "situational"
    else:
        romance_versatility = "not_recommended"

    return {
        "special_occasion": buckets["special_occasion"],
        "date_night": buckets["date_night"],
        "first_date": buckets["first_date"],
        "best_occasion": best_occasion,
        "best_occasion_success_rate": round(best_occasion_success_rate, 3),
        "worst_occasion": worst_occasion,
        "n_occasions_served_well": n_occasions_served_well,
        "romance_versatility": romance_versatility,
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
    romantic_judgments = [j for j in judgments if j.get("is_romantic_related", False)]
    n_romantic_reviews = len(romantic_judgments)

    if n_romantic_reviews == 0:
        confidence_level = "none"
    elif n_romantic_reviews <= 2:
        confidence_level = "low"
    elif n_romantic_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    sum_l1_score = 0.0
    sum_occasion_weight = 0.0
    n_delighted = 0
    n_ruined = 0
    n_special_occasions = 0
    n_special_success = 0

    for j in romantic_judgments:
        l1 = compute_l1_complex(j)

        sum_l1_score += l1["l1_total_score"]
        sum_occasion_weight += l1["occasion_weight"]

        occasion_type = j.get("occasion_type", "not_romantic")
        outcome_sentiment = j.get("outcome_sentiment", "not_mentioned")

        if outcome_sentiment == "delighted":
            n_delighted += 1
        if outcome_sentiment == "ruined":
            n_ruined += 1

        if occasion_type in ("proposal", "anniversary", "valentines"):
            n_special_occasions += 1
            if outcome_sentiment in ("delighted", "satisfied"):
                n_special_success += 1

    # L1.5 Occasion Buckets
    l15_buckets = compute_l15_buckets(romantic_judgments)
    romance_versatility = l15_buckets["romance_versatility"]
    romance_versatility_bonus = ROMANCE_VERSATILITY_BONUS.get(romance_versatility, 0.0)

    # Weighted aggregation
    weighted_mean_score = sum_l1_score / max(sum_occasion_weight, 1)

    # Formulas (G2d: complex + L1.5)
    adjusted_score = weighted_mean_score

    # G2d: adds ROMANCE_VERSATILITY_BONUS to complex formula
    raw_score = BASE_SCORE + adjusted_score + romance_versatility_bonus
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy
    if final_score >= 7.5:
        base_verdict_by_score = "Highly Romantic"
    elif final_score >= 5.5:
        base_verdict_by_score = "Romantic"
    elif final_score >= 3.5:
        base_verdict_by_score = "Somewhat Romantic"
    else:
        base_verdict_by_score = "Not Romantic"

    override_applied = "none"
    verdict = base_verdict_by_score

    has_ruined_special = any(
        j.get("occasion_type") in ("proposal", "anniversary") and j.get("outcome_sentiment") == "ruined"
        for j in romantic_judgments
    )

    if has_ruined_special and verdict in ("Highly Romantic", "Romantic"):
        override_applied = "ruined_special_occasion"
        verdict = "Somewhat Romantic"
    elif n_delighted >= 3 and n_ruined == 0 and verdict in ("Somewhat Romantic", "Not Romantic"):
        override_applied = "many_delighted_min_romantic"
        verdict = "Romantic"
    elif n_special_success >= 2 and n_special_occasions >= 2 and verdict in ("Somewhat Romantic", "Not Romantic"):
        override_applied = "special_success_min_romantic"
        verdict = "Romantic"
    elif weighted_mean_score < -2:
        override_applied = "low_weighted_mean"
        verdict = "Not Romantic"

    return {
        # L1.5 Occasion Buckets
        "L1_5_occasion_buckets": l15_buckets,
        # L2 Aggregates
        "N_ROMANTIC_REVIEWS": n_romantic_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_DELIGHTED": n_delighted,
        "N_RUINED": n_ruined,
        "N_SPECIAL_OCCASIONS": n_special_occasions,
        "N_SPECIAL_SUCCESS": n_special_success,
        # Weighted aggregation
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "SUM_OCCASION_WEIGHT": round(sum_occasion_weight, 2),
        "WEIGHTED_MEAN_SCORE": round(weighted_mean_score, 3),
        # Formula results
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "ROMANCE_VERSATILITY_BONUS": romance_versatility_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_romantic_reviews": n_romantic_reviews,
    }
