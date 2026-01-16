"""
Ground Truth computation for G3a (Price-Worth - Simple).

Implements the formula from data/tasks/yelp/G3a_prompt.txt.
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


def compute_l1_positive(judgment: Dict[str, Any]) -> bool:
    """
    POSITIVE_VALUE = true iff ANY:
      - VALUE_SENTIMENT in {excellent_value, good_value}
      - (PRICE_LEVEL in {cheap, moderate} AND QUALITY_PERCEPTION in {exceptional, good})
      - (PORTION_SIZE = generous AND PRICE_LEVEL != very_expensive)
    """
    value_sentiment = judgment.get("value_sentiment", "not_mentioned")
    price_level = judgment.get("price_level", "not_mentioned")
    quality_perception = judgment.get("quality_perception", "not_mentioned")
    portion_size = judgment.get("portion_size", "not_mentioned")

    if value_sentiment in ("excellent_value", "good_value"):
        return True

    if price_level in ("cheap", "moderate") and quality_perception in ("exceptional", "good"):
        return True

    if portion_size == "generous" and price_level != "very_expensive":
        return True

    return False


def compute_l1_negative(judgment: Dict[str, Any]) -> bool:
    """
    NEGATIVE_VALUE = true iff ANY:
      - VALUE_SENTIMENT in {poor_value, ripoff}
      - (PRICE_LEVEL in {expensive, very_expensive} AND QUALITY_PERCEPTION in {average, poor})
      - (PORTION_SIZE in {small, tiny} AND PRICE_LEVEL in {expensive, very_expensive})
    """
    value_sentiment = judgment.get("value_sentiment", "not_mentioned")
    price_level = judgment.get("price_level", "not_mentioned")
    quality_perception = judgment.get("quality_perception", "not_mentioned")
    portion_size = judgment.get("portion_size", "not_mentioned")

    if value_sentiment in ("poor_value", "ripoff"):
        return True

    if price_level in ("expensive", "very_expensive") and quality_perception in ("average", "poor"):
        return True

    if portion_size in ("small", "tiny") and price_level in ("expensive", "very_expensive"):
        return True

    return False


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
    # Filter to value-related judgments only
    value_judgments = [j for j in judgments if j.get("is_value_related", False)]
    n_value_reviews = len(value_judgments)

    # CONFIDENCE_LEVEL
    if n_value_reviews == 0:
        confidence_level = "none"
    elif n_value_reviews <= 2:
        confidence_level = "low"
    elif n_value_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    # Compute L1 for each judgment
    n_positive = 0
    n_negative = 0
    n_excellent = 0
    n_ripoff = 0
    n_generous = 0
    n_tiny = 0
    n_would_return = 0

    for j in value_judgments:
        if compute_l1_positive(j):
            n_positive += 1
        if compute_l1_negative(j):
            n_negative += 1

        if j.get("value_sentiment") == "excellent_value":
            n_excellent += 1
        if j.get("value_sentiment") == "ripoff":
            n_ripoff += 1
        if j.get("portion_size") == "generous":
            n_generous += 1
        if j.get("portion_size") == "tiny":
            n_tiny += 1
        if j.get("would_return_for_value") == "yes":
            n_would_return += 1

    # Formulas
    positive_score = (n_positive * 1.5) + (n_excellent * 1) + (n_generous * 0.5) + (n_would_return * 0.5)
    negative_score = (n_negative * 1.5) + (n_ripoff * 2) + (n_tiny * 0.5)

    raw_score = BASE_SCORE + positive_score - negative_score
    final_score = max(0.0, min(10.0, raw_score))

    # Decision Policy: Base verdict by score
    if final_score >= 7.5:
        base_verdict_by_score = "Excellent Value"
    elif final_score >= 5.5:
        base_verdict_by_score = "Good Value"
    elif final_score >= 3.5:
        base_verdict_by_score = "Mixed Value"
    else:
        base_verdict_by_score = "Poor Value"

    # Decision Policy: Overrides
    override_applied = "none"
    verdict = base_verdict_by_score

    # Override 1: If N_RIPOFF >= 2 => Poor Value
    if n_ripoff >= 2:
        override_applied = "ripoff_pattern"
        verdict = "Poor Value"
    # Override 2: If N_EXCELLENT >= 3 AND N_NEGATIVE == 0 => min Good Value
    elif n_excellent >= 3 and n_negative == 0 and verdict in ("Mixed Value", "Poor Value"):
        override_applied = "excellent_min_good"
        verdict = "Good Value"
    # Override 3: If N_POSITIVE >= 5 AND N_NEGATIVE <= 1 => min Good Value
    elif n_positive >= 5 and n_negative <= 1 and verdict in ("Mixed Value", "Poor Value"):
        override_applied = "positive_pattern_min_good"
        verdict = "Good Value"

    return {
        # L2 Aggregates
        "N_VALUE_REVIEWS": n_value_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "N_EXCELLENT": n_excellent,
        "N_RIPOFF": n_ripoff,
        "N_GENEROUS": n_generous,
        "N_TINY": n_tiny,
        "N_WOULD_RETURN": n_would_return,
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
        "n_value_reviews": n_value_reviews,
    }
