"""
Ground Truth computation for G6f (Comparison - Simple + L1.5).

Implements the formula from data/tasks/yelp/G6f_prompt.txt.
Simple formula with L1.5 comparison-dimension grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

COMPETITIVE_PATTERN_BONUS = {
    "market_leader": 2.0,
    "value_leader": 1.5,
    "quality_leader": 1.5,
    "convenience_leader": 1.0,
    "selective_advantage": 0.5,
    "struggling_competitor": -1.5,
    "undifferentiated": -1.0,
}


# =============================================================================
# Helpers
# =============================================================================


def is_favorable(value: str, dimension: str) -> bool:
    if dimension == "value":
        return value in ("best_value", "underpriced")
    elif dimension == "quality":
        return value in ("superior", "best_in_category", "top_tier")
    elif dimension == "convenience":
        return value == "more_convenient"
    return False


def is_unfavorable(value: str, dimension: str) -> bool:
    if dimension == "value":
        return value == "overpriced"
    elif dimension == "quality":
        return value in ("inferior", "bottom_tier")
    elif dimension == "convenience":
        return value == "less_convenient"
    return False


def compute_l1_positive_comparison(judgment: Dict[str, Any]) -> bool:
    competitor_mention = judgment.get("competitor_mention", "not_mentioned")
    category_ranking = judgment.get("category_ranking", "not_mentioned")
    price_comparison = judgment.get("price_comparison", "not_mentioned")
    quality_comparison = judgment.get("quality_comparison", "not_mentioned")

    return (competitor_mention == "favorable" or
            category_ranking in ("best_in_category", "top_tier") or
            price_comparison in ("best_value", "underpriced") or
            quality_comparison == "superior")


def compute_l1_negative_comparison(judgment: Dict[str, Any]) -> bool:
    competitor_mention = judgment.get("competitor_mention", "not_mentioned")
    category_ranking = judgment.get("category_ranking", "not_mentioned")
    price_comparison = judgment.get("price_comparison", "not_mentioned")
    quality_comparison = judgment.get("quality_comparison", "not_mentioned")

    return (competitor_mention == "unfavorable" or
            category_ranking == "bottom_tier" or
            price_comparison == "overpriced" or
            quality_comparison == "inferior")


def compute_l15_buckets(comparison_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "value": {"n_reviews": 0, "n_favorable": 0, "n_unfavorable": 0},
        "quality": {"n_reviews": 0, "n_favorable": 0, "n_unfavorable": 0},
        "convenience": {"n_reviews": 0, "n_favorable": 0, "n_unfavorable": 0},
    }

    for j in comparison_judgments:
        price_comparison = j.get("price_comparison", "not_mentioned")
        quality_comparison = j.get("quality_comparison", "not_mentioned")
        category_ranking = j.get("category_ranking", "not_mentioned")
        convenience_comparison = j.get("convenience_comparison", "not_mentioned")

        if price_comparison != "not_mentioned":
            buckets["value"]["n_reviews"] += 1
            if is_favorable(price_comparison, "value"):
                buckets["value"]["n_favorable"] += 1
            if is_unfavorable(price_comparison, "value"):
                buckets["value"]["n_unfavorable"] += 1

        if quality_comparison != "not_mentioned" or category_ranking != "not_mentioned":
            buckets["quality"]["n_reviews"] += 1
            if is_favorable(quality_comparison, "quality") or is_favorable(category_ranking, "quality"):
                buckets["quality"]["n_favorable"] += 1
            if is_unfavorable(quality_comparison, "quality") or is_unfavorable(category_ranking, "quality"):
                buckets["quality"]["n_unfavorable"] += 1

        if convenience_comparison != "not_mentioned":
            buckets["convenience"]["n_reviews"] += 1
            if is_favorable(convenience_comparison, "convenience"):
                buckets["convenience"]["n_favorable"] += 1
            if is_unfavorable(convenience_comparison, "convenience"):
                buckets["convenience"]["n_unfavorable"] += 1

    for key, bucket in buckets.items():
        total = bucket["n_favorable"] + bucket["n_unfavorable"]
        bucket["win_rate"] = bucket["n_favorable"] / max(total, 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_reviews"] > 0]

    if buckets_with_data:
        strongest_dimension, strongest_bucket = max(buckets_with_data, key=lambda x: x[1]["win_rate"])
        strongest_win_rate = strongest_bucket["win_rate"]
        weakest_dimension, _ = min(buckets_with_data, key=lambda x: x[1]["win_rate"])
    else:
        strongest_dimension = None
        strongest_win_rate = 0.0
        weakest_dimension = None

    n_dimensions_winning = sum(1 for k, v in buckets.items() if v["win_rate"] >= 0.6 and v["n_reviews"] > 0)
    all_struggling = all(v["win_rate"] < 0.4 for v in buckets.values() if v["n_reviews"] > 0)

    if n_dimensions_winning >= 3:
        competitive_pattern = "market_leader"
    elif buckets["value"]["win_rate"] >= 0.7 and buckets["value"]["n_reviews"] > 0:
        competitive_pattern = "value_leader"
    elif buckets["quality"]["win_rate"] >= 0.7 and buckets["quality"]["n_reviews"] > 0:
        competitive_pattern = "quality_leader"
    elif buckets["convenience"]["win_rate"] >= 0.7 and buckets["convenience"]["n_reviews"] > 0:
        competitive_pattern = "convenience_leader"
    elif n_dimensions_winning >= 1:
        competitive_pattern = "selective_advantage"
    elif all_struggling and any(v["n_reviews"] > 0 for v in buckets.values()):
        competitive_pattern = "struggling_competitor"
    else:
        competitive_pattern = "undifferentiated"

    return {
        "value": buckets["value"],
        "quality": buckets["quality"],
        "convenience": buckets["convenience"],
        "strongest_dimension": strongest_dimension,
        "strongest_win_rate": round(strongest_win_rate, 3),
        "weakest_dimension": weakest_dimension,
        "n_dimensions_winning": n_dimensions_winning,
        "competitive_pattern": competitive_pattern,
    }


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    comparison_judgments = [j for j in judgments if j.get("is_comparison_related", False)]
    n_comparison_reviews = len(comparison_judgments)

    if n_comparison_reviews == 0:
        confidence_level = "none"
    elif n_comparison_reviews <= 2:
        confidence_level = "low"
    elif n_comparison_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    n_positive = 0
    n_negative = 0

    for j in comparison_judgments:
        if compute_l1_positive_comparison(j):
            n_positive += 1
        if compute_l1_negative_comparison(j):
            n_negative += 1

    # L1.5 Dimension Buckets
    l15_buckets = compute_l15_buckets(comparison_judgments)
    competitive_pattern = l15_buckets["competitive_pattern"]
    competitive_pattern_bonus = COMPETITIVE_PATTERN_BONUS.get(competitive_pattern, 0.0)

    # Formulas
    win_rate = n_positive / max(n_positive + n_negative, 1)
    positive_score = n_positive * 1.5
    negative_score = n_negative * 1.5

    raw_score = BASE_SCORE + positive_score - negative_score + competitive_pattern_bonus
    final_score = max(0.0, min(10.0, raw_score))

    if final_score >= 7.5:
        base_verdict_by_score = "Category Leader"
    elif final_score >= 5.5:
        base_verdict_by_score = "Strong Competitor"
    elif final_score >= 3.5:
        base_verdict_by_score = "Average Competitor"
    else:
        base_verdict_by_score = "Weak Competitor"

    override_applied = "none"
    verdict = base_verdict_by_score
    strength_note = None
    weakness_note = None

    if competitive_pattern == "market_leader" and verdict in ("Average Competitor", "Weak Competitor"):
        override_applied = "market_leader_min_strong"
        verdict = "Strong Competitor"
    elif competitive_pattern == "struggling_competitor" and verdict in ("Category Leader", "Strong Competitor"):
        override_applied = "struggling_max_average"
        verdict = "Average Competitor"

    if l15_buckets["strongest_dimension"]:
        strength_note = f"Competitive strength: {l15_buckets['strongest_dimension']}"

    if l15_buckets["weakest_dimension"]:
        weakest_bucket = l15_buckets[l15_buckets["weakest_dimension"]]
        if weakest_bucket["win_rate"] < 0.4:
            weakness_note = f"Could improve: {l15_buckets['weakest_dimension']}"

    result = {
        "L1_5_dimension_buckets": l15_buckets,
        "N_COMPARISON_REVIEWS": n_comparison_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "WIN_RATE": round(win_rate, 3),
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "COMPETITIVE_PATTERN_BONUS": competitive_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_comparison_reviews": n_comparison_reviews,
    }

    if strength_note:
        result["strength_note"] = strength_note
    if weakness_note:
        result["weakness_note"] = weakness_note

    return result
