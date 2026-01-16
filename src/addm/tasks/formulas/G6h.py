"""
Ground Truth computation for G6h (Comparison - Complex + L1.5).

Implements the formula from data/tasks/yelp/G6h_prompt.txt.
Complex formula with weighted comparison factors + L1.5 comparison-dimension grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

COMPETITOR_SCORES = {"favorable": 3.0, "different": 1.0, "similar": 0, "unfavorable": -3.0, "not_mentioned": 0}
RANKING_SCORES = {"best_in_category": 4.0, "top_tier": 2.5, "competitive": 1.0, "middle_pack": -0.5, "bottom_tier": -3.0, "not_mentioned": 0}
PRICE_SCORES = {"best_value": 3.0, "underpriced": 2.0, "fair_value": 0.5, "overpriced": -2.5, "not_mentioned": 0}
QUALITY_SCORES = {"superior": 3.5, "comparable": 0.5, "inferior": -3.0, "not_mentioned": 0}
CONVENIENCE_SCORES = {"more_convenient": 2.0, "equally_convenient": 0.5, "less_convenient": -1.5, "not_mentioned": 0}
RECOMMENDATION_SCORES = {"over_competitor": 3.0, "for_specific_need": 2.0, "general_recommendation": 1.5, "conditional": 0, "not_recommended": -2.5, "not_mentioned": 0}


# =============================================================================
# Helpers
# =============================================================================


def compute_l1_complex(judgment: Dict[str, Any]) -> Dict[str, Any]:
    competitor_mention = judgment.get("competitor_mention", "not_mentioned")
    category_ranking = judgment.get("category_ranking", "not_mentioned")
    price_comparison = judgment.get("price_comparison", "not_mentioned")
    quality_comparison = judgment.get("quality_comparison", "not_mentioned")
    convenience_comparison = judgment.get("convenience_comparison", "not_mentioned")
    recommendation_context = judgment.get("recommendation_context", "not_mentioned")

    competitor_score = COMPETITOR_SCORES.get(competitor_mention, 0)
    ranking_score = RANKING_SCORES.get(category_ranking, 0)
    price_score = PRICE_SCORES.get(price_comparison, 0)
    quality_score = QUALITY_SCORES.get(quality_comparison, 0)
    convenience_score = CONVENIENCE_SCORES.get(convenience_comparison, 0)
    recommendation_score = RECOMMENDATION_SCORES.get(recommendation_context, 0)

    l1_comparison_score = (
        competitor_score + ranking_score + price_score +
        quality_score + convenience_score + recommendation_score
    )

    return {
        "competitor_score": competitor_score,
        "ranking_score": ranking_score,
        "price_score": price_score,
        "quality_score": quality_score,
        "convenience_score": convenience_score,
        "recommendation_score": recommendation_score,
        "l1_comparison_score": round(l1_comparison_score, 2),
    }


def compute_l15_buckets(comparison_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "value": {"n_reviews": 0, "sum_score": 0.0, "n_wins": 0, "n_losses": 0},
        "quality": {"n_reviews": 0, "sum_score": 0.0, "n_wins": 0, "n_losses": 0},
        "convenience": {"n_reviews": 0, "sum_score": 0.0, "n_wins": 0, "n_losses": 0},
    }

    for j in comparison_judgments:
        price_comparison = j.get("price_comparison", "not_mentioned")
        quality_comparison = j.get("quality_comparison", "not_mentioned")
        category_ranking = j.get("category_ranking", "not_mentioned")
        convenience_comparison = j.get("convenience_comparison", "not_mentioned")

        price_score = PRICE_SCORES.get(price_comparison, 0)
        quality_score = QUALITY_SCORES.get(quality_comparison, 0)
        ranking_score = RANKING_SCORES.get(category_ranking, 0)
        convenience_score = CONVENIENCE_SCORES.get(convenience_comparison, 0)

        if price_comparison != "not_mentioned":
            buckets["value"]["n_reviews"] += 1
            buckets["value"]["sum_score"] += price_score
            if price_score > 0:
                buckets["value"]["n_wins"] += 1
            if price_score < 0:
                buckets["value"]["n_losses"] += 1

        if quality_comparison != "not_mentioned" or category_ranking != "not_mentioned":
            buckets["quality"]["n_reviews"] += 1
            buckets["quality"]["sum_score"] += quality_score + ranking_score
            if quality_score > 0 or ranking_score > 0:
                buckets["quality"]["n_wins"] += 1
            if quality_score < 0 or ranking_score < 0:
                buckets["quality"]["n_losses"] += 1

        if convenience_comparison != "not_mentioned":
            buckets["convenience"]["n_reviews"] += 1
            buckets["convenience"]["sum_score"] += convenience_score
            if convenience_score > 0:
                buckets["convenience"]["n_wins"] += 1
            if convenience_score < 0:
                buckets["convenience"]["n_losses"] += 1

    for key, bucket in buckets.items():
        bucket["mean_score"] = bucket["sum_score"] / max(bucket["n_reviews"], 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_reviews"] > 0]

    if buckets_with_data:
        strongest_dimension, strongest_bucket = max(buckets_with_data, key=lambda x: x[1]["mean_score"])
        strongest_score = strongest_bucket["mean_score"]
        weakest_dimension, weakest_bucket = min(buckets_with_data, key=lambda x: x[1]["mean_score"])
        weakest_score = weakest_bucket["mean_score"]
    else:
        strongest_dimension = None
        strongest_score = 0.0
        weakest_dimension = None
        weakest_score = 0.0

    n_dimensions_dominant = sum(1 for k, v in buckets.items() if v["mean_score"] >= 2.0 and v["n_reviews"] > 0)

    if n_dimensions_dominant >= 3:
        competitive_pattern = "dominant_player"
    elif buckets["value"]["mean_score"] >= 2.5 and buckets["value"]["n_reviews"] > 0:
        competitive_pattern = "value_champion"
    elif buckets["quality"]["mean_score"] >= 3.0 and buckets["quality"]["n_reviews"] > 0:
        competitive_pattern = "quality_champion"
    elif buckets["convenience"]["mean_score"] >= 2.0 and buckets["convenience"]["n_reviews"] > 0:
        competitive_pattern = "accessibility_champion"
    elif strongest_score - weakest_score > 3.0 and buckets_with_data:
        competitive_pattern = "niche_player"
    elif n_dimensions_dominant >= 1:
        competitive_pattern = "competitive_in_segment"
    else:
        competitive_pattern = "losing_ground"

    return {
        "value": buckets["value"],
        "quality": buckets["quality"],
        "convenience": buckets["convenience"],
        "strongest_dimension": strongest_dimension,
        "strongest_score": round(strongest_score, 3) if strongest_score else 0.0,
        "weakest_dimension": weakest_dimension,
        "weakest_score": round(weakest_score, 3) if weakest_score else 0.0,
        "n_dimensions_dominant": n_dimensions_dominant,
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

    n_best_in_category = 0
    n_bottom_tier = 0

    for j in comparison_judgments:
        if j.get("category_ranking") == "best_in_category":
            n_best_in_category += 1
        if j.get("category_ranking") == "bottom_tier":
            n_bottom_tier += 1

    # L1.5 Dimension Buckets
    l15_buckets = compute_l15_buckets(comparison_judgments)
    competitive_pattern = l15_buckets["competitive_pattern"]

    # Pattern multiplier
    if competitive_pattern == "dominant_player":
        pattern_mult = 1.25
    elif competitive_pattern in ("value_champion", "quality_champion", "accessibility_champion"):
        pattern_mult = 1.15
    elif competitive_pattern == "niche_player":
        pattern_mult = 1.0
    elif competitive_pattern == "losing_ground":
        pattern_mult = 0.75
    else:
        pattern_mult = 1.0

    # Aggregate L1.5 scores
    total_dimension_score = sum(
        v["mean_score"] for v in [l15_buckets["value"], l15_buckets["quality"], l15_buckets["convenience"]]
        if v["n_reviews"] > 0
    )
    n_dimension_buckets_active = sum(
        1 for v in [l15_buckets["value"], l15_buckets["quality"], l15_buckets["convenience"]]
        if v["n_reviews"] > 0
    )
    mean_dimension_score = total_dimension_score / max(n_dimension_buckets_active, 1)

    adjusted_score = mean_dimension_score * pattern_mult
    raw_score = BASE_SCORE + adjusted_score
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
    positioning_note = None
    strength_note = None

    if n_best_in_category >= 2 and verdict in ("Average Competitor", "Weak Competitor"):
        override_applied = "best_in_category_min_strong"
        verdict = "Strong Competitor"
    elif competitive_pattern == "dominant_player" and verdict in ("Average Competitor", "Weak Competitor"):
        override_applied = "dominant_player_min_strong"
        verdict = "Strong Competitor"
    elif competitive_pattern == "losing_ground" and n_bottom_tier >= 1 and verdict in ("Category Leader", "Strong Competitor"):
        override_applied = "losing_ground_max_average"
        verdict = "Average Competitor"

    if competitive_pattern in ("dominant_player", "value_champion", "quality_champion", "accessibility_champion"):
        positioning_note = f"Positioning: {competitive_pattern.replace('_', ' ').title()}"

    if l15_buckets["strongest_dimension"]:
        strength_note = f"Strongest dimension: {l15_buckets['strongest_dimension']}"

    result = {
        "L1_5_dimension_buckets": l15_buckets,
        "N_COMPARISON_REVIEWS": n_comparison_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_BEST_IN_CATEGORY": n_best_in_category,
        "N_BOTTOM_TIER": n_bottom_tier,
        "TOTAL_DIMENSION_SCORE": round(total_dimension_score, 3),
        "MEAN_DIMENSION_SCORE": round(mean_dimension_score, 3),
        "PATTERN_MULT": pattern_mult,
        "BASE_SCORE": BASE_SCORE,
        "ADJUSTED_SCORE": round(adjusted_score, 3),
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_comparison_reviews": n_comparison_reviews,
    }

    if positioning_note:
        result["positioning_note"] = positioning_note
    if strength_note:
        result["strength_note"] = strength_note

    return result
