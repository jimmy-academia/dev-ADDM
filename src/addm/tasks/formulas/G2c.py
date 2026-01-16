"""
Ground Truth computation for G2c (Romance - Complex).

Implements the formula from data/tasks/yelp/G2c_prompt.txt.
Complex formula with occasion weighting and interaction effects.

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


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute L1 composites using complex formula.

    Returns dict with all L1 computed values.
    """
    occasion_type = judgment.get("occasion_type", "not_romantic")
    ambiance_rating = judgment.get("ambiance_rating", "not_mentioned")
    privacy_level = judgment.get("privacy_level", "not_mentioned")
    noise_level = judgment.get("noise_level", "not_mentioned")
    service_attentiveness = judgment.get("service_attentiveness", "not_mentioned")
    romantic_elements = judgment.get("romantic_elements", "not_mentioned")
    outcome_sentiment = judgment.get("outcome_sentiment", "not_mentioned")

    # Get scores
    occasion_weight = OCCASION_WEIGHTS.get(occasion_type, 0)
    ambiance_score = AMBIANCE_SCORES.get(ambiance_rating, 0)
    privacy_score = PRIVACY_SCORES.get(privacy_level, 0)
    noise_score = NOISE_SCORES.get(noise_level, 0)
    service_score = SERVICE_SCORES.get(service_attentiveness, 0)
    romantic_elements_score = ROMANTIC_ELEMENTS_SCORES.get(romantic_elements, 0)
    outcome_modifier = OUTCOME_MODIFIERS.get(outcome_sentiment, 0)

    # L1_REVIEW_SCORE
    l1_review_score = (ambiance_score + privacy_score + noise_score + service_score + romantic_elements_score + outcome_modifier) * occasion_weight

    # RUINED_SPECIAL_OCCASION (interaction effect)
    ruined_special_occasion = -5.0 if (occasion_type in ("proposal", "anniversary") and outcome_sentiment == "ruined") else 0.0

    # PERFECT_PROPOSAL (interaction effect)
    perfect_proposal = 3.0 if (occasion_type == "proposal" and outcome_sentiment == "delighted") else 0.0

    # L1_TOTAL_SCORE
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

    # Compute L1 for each judgment and aggregate
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

    # Weighted aggregation
    weighted_mean_score = sum_l1_score / max(sum_occasion_weight, 1)

    # Formulas
    adjusted_score = weighted_mean_score
    raw_score = BASE_SCORE + adjusted_score
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

    # Track if any ruined was special occasion
    has_ruined_special = any(
        j.get("occasion_type") in ("proposal", "anniversary") and j.get("outcome_sentiment") == "ruined"
        for j in romantic_judgments
    )

    # Override 1: If N_RUINED >= 1 AND OCCASION_TYPE was special => max Somewhat Romantic
    if has_ruined_special and verdict in ("Highly Romantic", "Romantic"):
        override_applied = "ruined_special_occasion"
        verdict = "Somewhat Romantic"
    # Override 2: If N_DELIGHTED >= 3 AND N_RUINED == 0 => min Romantic
    elif n_delighted >= 3 and n_ruined == 0 and verdict in ("Somewhat Romantic", "Not Romantic"):
        override_applied = "many_delighted_min_romantic"
        verdict = "Romantic"
    # Override 3: If N_SPECIAL_SUCCESS >= 2 AND N_SPECIAL_OCCASIONS >= 2 => min Romantic
    elif n_special_success >= 2 and n_special_occasions >= 2 and verdict in ("Somewhat Romantic", "Not Romantic"):
        override_applied = "special_success_min_romantic"
        verdict = "Romantic"
    # Override 4: If WEIGHTED_MEAN_SCORE < -2 => Not Romantic
    elif weighted_mean_score < -2:
        override_applied = "low_weighted_mean"
        verdict = "Not Romantic"

    return {
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
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        # Decision
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        # Meta
        "n_romantic_reviews": n_romantic_reviews,
    }
