"""
Ground Truth computation for G3k (Time-Value - Complex).

Implements the formula from data/tasks/yelp/G3k_prompt.txt.
Complex formula with time aspect weighting.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

# Wait scores (shorter = better)
WAIT_SCORES = {
    "none": 3,
    "short": 2,
    "moderate": 0.5,
    "long": -2,
    "excessive": -4,
    "not_mentioned": 0,
}

# Reservation accuracy scores
RESERVATION_SCORES = {
    "honored": 2,
    "slight_delay": 0.5,
    "significant_delay": -2,
    "lost": -5,
    "not_applicable": 0,
    "not_mentioned": 0,
}

# Food delivery scores
FOOD_DELIVERY_SCORES = {
    "fast": 2,
    "reasonable": 1,
    "slow": -1.5,
    "very_slow": -3,
    "not_mentioned": 0,
}

# Service pacing scores
PACING_SCORES = {
    "perfect": 2.5,
    "slightly_rushed": -0.5,
    "very_rushed": -2,
    "too_slow": -1.5,
    "forgotten": -3,
    "not_mentioned": 0,
}

# Overall sentiment modifiers
SENTIMENT_MODIFIERS = {
    "efficient": 2,
    "acceptable": 1,
    "frustrating": -2,
    "unacceptable": -4,
    "not_mentioned": 0,
}


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    wait_for_table = judgment.get("wait_for_table", "not_mentioned")
    reservation_accuracy = judgment.get("reservation_accuracy", "not_mentioned")
    food_delivery = judgment.get("food_delivery_speed", "not_mentioned")
    service_pacing = judgment.get("service_pacing", "not_mentioned")
    overall_sentiment = judgment.get("overall_time_sentiment", "not_mentioned")

    wait_score = WAIT_SCORES.get(wait_for_table, 0)
    reservation_score = RESERVATION_SCORES.get(reservation_accuracy, 0)
    food_delivery_score = FOOD_DELIVERY_SCORES.get(food_delivery, 0)
    pacing_score = PACING_SCORES.get(service_pacing, 0)
    sentiment_modifier = SENTIMENT_MODIFIERS.get(overall_sentiment, 0)

    l1_review_score = wait_score + reservation_score + food_delivery_score + pacing_score + sentiment_modifier

    # Lost reservation interaction
    lost_reservation_penalty = -3.0 if reservation_accuracy == "lost" else 0.0

    # Forgotten penalty
    forgotten_penalty = -2.0 if service_pacing == "forgotten" else 0.0

    l1_total_score = l1_review_score + lost_reservation_penalty + forgotten_penalty

    return {
        "wait_score": wait_score,
        "reservation_score": reservation_score,
        "food_delivery_score": food_delivery_score,
        "pacing_score": pacing_score,
        "sentiment_modifier": sentiment_modifier,
        "l1_review_score": round(l1_review_score, 2),
        "lost_reservation_penalty": lost_reservation_penalty,
        "forgotten_penalty": forgotten_penalty,
        "l1_total_score": round(l1_total_score, 2),
    }


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    time_judgments = [j for j in judgments if j.get("is_time_related", False)]
    n_time_reviews = len(time_judgments)

    if n_time_reviews == 0:
        confidence_level = "none"
    elif n_time_reviews <= 2:
        confidence_level = "low"
    elif n_time_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    sum_l1_score = 0.0
    n_lost_reservation = 0
    n_forgotten = 0
    n_excessive_wait = 0
    n_efficient = 0

    for j in time_judgments:
        l1 = compute_l1_complex(j)
        sum_l1_score += l1["l1_total_score"]

        if j.get("reservation_accuracy") == "lost":
            n_lost_reservation += 1
        if j.get("service_pacing") == "forgotten":
            n_forgotten += 1
        if j.get("wait_for_table") == "excessive":
            n_excessive_wait += 1
        if j.get("overall_time_sentiment") == "efficient":
            n_efficient += 1

    mean_l1_score = sum_l1_score / max(n_time_reviews, 1)

    adjusted_score = mean_l1_score
    raw_score = BASE_SCORE + adjusted_score
    final_score = max(0.0, min(10.0, raw_score))

    if final_score >= 7.5:
        base_verdict_by_score = "Excellent Timing"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good Timing"
    elif final_score >= 3.5:
        base_verdict_by_score = "Variable Timing"
    else:
        base_verdict_by_score = "Poor Timing"

    override_applied = "none"
    verdict = base_verdict_by_score

    if n_lost_reservation >= 1 and verdict in ("Excellent Timing", "Good Timing"):
        override_applied = "lost_reservation_max_variable"
        verdict = "Variable Timing"
    elif n_excessive_wait >= 2:
        override_applied = "excessive_wait_pattern"
        verdict = "Poor Timing"
    elif n_forgotten >= 2:
        override_applied = "forgotten_pattern"
        verdict = "Poor Timing"
    elif n_efficient >= 3 and n_lost_reservation == 0 and n_forgotten == 0 and verdict in ("Variable Timing", "Poor Timing"):
        override_applied = "efficient_min_good"
        verdict = "Good Timing"
    elif mean_l1_score < -2:
        override_applied = "low_mean_score"
        verdict = "Poor Timing"

    return {
        "N_TIME_REVIEWS": n_time_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_LOST_RESERVATION": n_lost_reservation,
        "N_FORGOTTEN": n_forgotten,
        "N_EXCESSIVE_WAIT": n_excessive_wait,
        "N_EFFICIENT": n_efficient,
        "SUM_L1_SCORE": round(sum_l1_score, 2),
        "MEAN_L1_SCORE": round(mean_l1_score, 3),
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_time_reviews": n_time_reviews,
    }
