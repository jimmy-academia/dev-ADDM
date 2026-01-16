"""
Ground Truth computation for G2b (Romance - Simple + L1.5).

Implements the formula from data/tasks/yelp/G2b_prompt.txt.
Simple formula with L1.5 occasion type grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

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


def compute_l1_positive(judgment: Dict[str, Any]) -> bool:
    """
    POSITIVE_ROMANTIC_EXPERIENCE = true iff ALL:
      - OCCASION_TYPE != not_romantic
      - AMBIANCE_RATING in {excellent, good}
      - At least one of: PRIVACY_LEVEL = intimate, NOISE_LEVEL = quiet, ROMANTIC_ELEMENTS = present
    """
    occasion_type = judgment.get("occasion_type", "not_romantic")
    ambiance_rating = judgment.get("ambiance_rating", "not_mentioned")
    privacy_level = judgment.get("privacy_level", "not_mentioned")
    noise_level = judgment.get("noise_level", "not_mentioned")
    romantic_elements = judgment.get("romantic_elements", "not_mentioned")

    if occasion_type == "not_romantic":
        return False

    if ambiance_rating not in ("excellent", "good"):
        return False

    has_positive_feature = (
        privacy_level == "intimate"
        or noise_level == "quiet"
        or romantic_elements == "present"
    )

    return has_positive_feature


def compute_l1_negative(judgment: Dict[str, Any]) -> bool:
    """
    NEGATIVE_ROMANTIC_EXPERIENCE = true iff ALL:
      - OCCASION_TYPE != not_romantic
      - Any of: AMBIANCE_RATING = poor, NOISE_LEVEL = loud, PRIVACY_LEVEL = lacking
    """
    occasion_type = judgment.get("occasion_type", "not_romantic")
    ambiance_rating = judgment.get("ambiance_rating", "not_mentioned")
    privacy_level = judgment.get("privacy_level", "not_mentioned")
    noise_level = judgment.get("noise_level", "not_mentioned")

    if occasion_type == "not_romantic":
        return False

    has_negative = (
        ambiance_rating == "poor"
        or noise_level == "loud"
        or privacy_level == "lacking"
    )

    return has_negative


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

    # Compute success rates
    for key, bucket in buckets.items():
        n_positive = bucket["n_positive"]
        n_negative = bucket["n_negative"]
        bucket["success_rate"] = n_positive / max(n_positive + n_negative, 1)

    # Find best and worst occasions
    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_positive"] + v["n_negative"] > 0]

    if buckets_with_data:
        best_occasion, best_bucket = max(buckets_with_data, key=lambda x: x[1]["success_rate"])
        best_occasion_success_rate = best_bucket["success_rate"]
        worst_occasion, worst_bucket = min(buckets_with_data, key=lambda x: x[1]["success_rate"])
    else:
        best_occasion = None
        best_occasion_success_rate = 0.0
        worst_occasion = None

    # Count occasions served well
    n_occasions_served_well = sum(1 for k, v in buckets.items() if v["success_rate"] >= 0.7 and v["n_reviews"] > 0)

    # Determine romance versatility
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
    # Filter to romantic-related judgments only
    romantic_judgments = [j for j in judgments if j.get("is_romantic_related", False)]
    n_romantic_reviews = len(romantic_judgments)

    # CONFIDENCE_LEVEL
    if n_romantic_reviews == 0:
        confidence_level = "none"
    elif n_romantic_reviews <= 2:
        confidence_level = "low"
    elif n_romantic_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    # Compute L1 for each judgment
    n_positive = 0
    n_negative = 0

    for j in romantic_judgments:
        if compute_l1_positive(j):
            n_positive += 1
        if compute_l1_negative(j):
            n_negative += 1

    # L1.5 Occasion Buckets
    l15_buckets = compute_l15_buckets(romantic_judgments)
    romance_versatility = l15_buckets["romance_versatility"]
    romance_versatility_bonus = ROMANCE_VERSATILITY_BONUS.get(romance_versatility, 0.0)

    # Formulas
    success_rate = n_positive / max(n_positive + n_negative, 1)
    positive_score = n_positive * 2
    negative_score = n_negative * 2

    # G2b: adds ROMANCE_VERSATILITY_BONUS to simple formula
    raw_score = BASE_SCORE + positive_score - negative_score + romance_versatility_bonus
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy: Base verdict by score
    if final_score >= 7.5:
        base_verdict_by_score = "Highly Romantic"
    elif final_score >= 5.5:
        base_verdict_by_score = "Romantic"
    elif final_score >= 3.5:
        base_verdict_by_score = "Somewhat Romantic"
    else:
        base_verdict_by_score = "Not Romantic"

    # Decision Policy: Overrides
    override_applied = "none"
    verdict = base_verdict_by_score

    # Override: If ROMANCE_VERSATILITY = "not_recommended" AND N_NEGATIVE >= 2 => Not Romantic
    if romance_versatility == "not_recommended" and n_negative >= 2:
        override_applied = "not_recommended_with_negatives"
        verdict = "Not Romantic"
    # Override: If SPECIAL_OCCASION bucket has N_NEGATIVE >= 2 => max Somewhat Romantic
    elif l15_buckets["special_occasion"]["n_negative"] >= 2 and verdict in ("Highly Romantic", "Romantic"):
        override_applied = "special_occasion_negatives"
        verdict = "Somewhat Romantic"

    return {
        # L1.5 Occasion Buckets
        "L1_5_occasion_buckets": l15_buckets,
        # L2 Aggregates
        "N_ROMANTIC_REVIEWS": n_romantic_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "SUCCESS_RATE": round(success_rate, 3),
        # Formula results
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
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
