"""
Ground Truth computation for G6b (Uniqueness - Simple + L1.5).

Implements the formula from data/tasks/yelp/G6b_prompt.txt.
Simple formula with L1.5 uniqueness-type grouping.

IMPORTANT: All constants must EXACTLY match the prompt file.
"""

from typing import Any, Dict, List

# =============================================================================
# Constants (MUST match prompt exactly)
# =============================================================================

BASE_SCORE = 5.0

UNIQUENESS_PATTERN_BONUS = {
    "total_original": 2.0,
    "culinary_innovator": 1.5,
    "atmosphere_destination": 1.5,
    "experience_pioneer": 1.5,
    "selective_distinction": 0.5,
    "cookie_cutter": -1.5,
    "needs_identity": -1.0,
}


# =============================================================================
# Helpers
# =============================================================================


def is_exceptional(value: str, category: str) -> bool:
    if category == "culinary":
        return value == "signature_standout"
    elif category == "atmosphere":
        return value == "one_of_a_kind"
    elif category == "experience":
        return value in ("exceptional_approach", "groundbreaking")
    return False


def is_generic(value: str, category: str) -> bool:
    if category == "culinary":
        return value == "standard"
    elif category == "atmosphere":
        return value == "generic"
    elif category == "experience":
        return value in ("standard", "derivative")
    return False


def compute_l1_positive_uniqueness(judgment: Dict[str, Any]) -> bool:
    unique_dishes = judgment.get("unique_dishes", "not_mentioned")
    atmosphere_distinctiveness = judgment.get("atmosphere_distinctiveness", "not_mentioned")
    service_style = judgment.get("service_style", "not_mentioned")
    memorability = judgment.get("memorability", "not_mentioned")

    return (unique_dishes in ("signature_standout", "distinctive", "creative") or
            atmosphere_distinctiveness in ("one_of_a_kind", "memorable", "themed") or
            service_style in ("exceptional_approach", "personalized") or
            memorability in ("unforgettable", "memorable"))


def compute_l1_negative_uniqueness(judgment: Dict[str, Any]) -> bool:
    standout_feature = judgment.get("standout_feature", "not_mentioned")
    memorability = judgment.get("memorability", "not_mentioned")

    return (standout_feature == "none" or
            memorability == "forgettable")


def compute_l15_buckets(uniqueness_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = {
        "culinary": {"n_reviews": 0, "n_exceptional": 0, "n_generic": 0},
        "atmosphere": {"n_reviews": 0, "n_exceptional": 0, "n_generic": 0},
        "experience": {"n_reviews": 0, "n_exceptional": 0, "n_generic": 0},
    }

    for j in uniqueness_judgments:
        unique_dishes = j.get("unique_dishes", "not_mentioned")
        atmosphere_distinctiveness = j.get("atmosphere_distinctiveness", "not_mentioned")
        service_style = j.get("service_style", "not_mentioned")
        concept_innovation = j.get("concept_innovation", "not_mentioned")

        if unique_dishes != "not_mentioned":
            buckets["culinary"]["n_reviews"] += 1
            if is_exceptional(unique_dishes, "culinary"):
                buckets["culinary"]["n_exceptional"] += 1
            if is_generic(unique_dishes, "culinary"):
                buckets["culinary"]["n_generic"] += 1

        if atmosphere_distinctiveness != "not_mentioned":
            buckets["atmosphere"]["n_reviews"] += 1
            if is_exceptional(atmosphere_distinctiveness, "atmosphere"):
                buckets["atmosphere"]["n_exceptional"] += 1
            if is_generic(atmosphere_distinctiveness, "atmosphere"):
                buckets["atmosphere"]["n_generic"] += 1

        if service_style != "not_mentioned" or concept_innovation != "not_mentioned":
            buckets["experience"]["n_reviews"] += 1
            if is_exceptional(service_style, "experience") or is_exceptional(concept_innovation, "experience"):
                buckets["experience"]["n_exceptional"] += 1
            if is_generic(service_style, "experience") or is_generic(concept_innovation, "experience"):
                buckets["experience"]["n_generic"] += 1

    for key, bucket in buckets.items():
        bucket["uniqueness_rate"] = bucket["n_exceptional"] / max(bucket["n_reviews"], 1)

    buckets_with_data = [(k, v) for k, v in buckets.items() if v["n_reviews"] > 0]

    if buckets_with_data:
        strongest_category, strongest_bucket = max(buckets_with_data, key=lambda x: x[1]["uniqueness_rate"])
        strongest_rate = strongest_bucket["uniqueness_rate"]
        weakest_category, _ = min(buckets_with_data, key=lambda x: x[1]["uniqueness_rate"])
    else:
        strongest_category = None
        strongest_rate = 0.0
        weakest_category = None

    n_categories_unique = sum(1 for k, v in buckets.items() if v["uniqueness_rate"] >= 0.5 and v["n_reviews"] > 0)
    all_low = all(v["uniqueness_rate"] < 0.3 for v in buckets.values() if v["n_reviews"] > 0)

    if n_categories_unique >= 3:
        uniqueness_pattern = "total_original"
    elif buckets["culinary"]["uniqueness_rate"] >= 0.7 and buckets["culinary"]["n_reviews"] > 0:
        uniqueness_pattern = "culinary_innovator"
    elif buckets["atmosphere"]["uniqueness_rate"] >= 0.7 and buckets["atmosphere"]["n_reviews"] > 0:
        uniqueness_pattern = "atmosphere_destination"
    elif buckets["experience"]["uniqueness_rate"] >= 0.7 and buckets["experience"]["n_reviews"] > 0:
        uniqueness_pattern = "experience_pioneer"
    elif n_categories_unique >= 1:
        uniqueness_pattern = "selective_distinction"
    elif all_low and any(v["n_reviews"] > 0 for v in buckets.values()):
        uniqueness_pattern = "cookie_cutter"
    else:
        uniqueness_pattern = "needs_identity"

    return {
        "culinary": buckets["culinary"],
        "atmosphere": buckets["atmosphere"],
        "experience": buckets["experience"],
        "strongest_category": strongest_category,
        "strongest_rate": round(strongest_rate, 3),
        "weakest_category": weakest_category,
        "n_categories_unique": n_categories_unique,
        "uniqueness_pattern": uniqueness_pattern,
    }


# =============================================================================
# Main computation
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
) -> Dict[str, Any]:
    uniqueness_judgments = [j for j in judgments if j.get("is_uniqueness_related", False)]
    n_uniqueness_reviews = len(uniqueness_judgments)

    if n_uniqueness_reviews == 0:
        confidence_level = "none"
    elif n_uniqueness_reviews <= 2:
        confidence_level = "low"
    elif n_uniqueness_reviews <= 5:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    n_positive = 0
    n_negative = 0

    for j in uniqueness_judgments:
        if compute_l1_positive_uniqueness(j):
            n_positive += 1
        if compute_l1_negative_uniqueness(j):
            n_negative += 1

    # L1.5 Category Buckets
    l15_buckets = compute_l15_buckets(uniqueness_judgments)
    uniqueness_pattern = l15_buckets["uniqueness_pattern"]
    uniqueness_pattern_bonus = UNIQUENESS_PATTERN_BONUS.get(uniqueness_pattern, 0.0)

    # Formulas
    uniqueness_rate = n_positive / max(n_positive + n_negative, 1)
    positive_score = n_positive * 1.5
    negative_score = n_negative * 1.5

    raw_score = BASE_SCORE + positive_score - negative_score + uniqueness_pattern_bonus
    final_score = max(0.0, min(10.0, raw_score))

    if final_score >= 7.5:
        base_verdict_by_score = "Highly Unique"
    elif final_score >= 5.5:
        base_verdict_by_score = "Distinctive"
    elif final_score >= 3.5:
        base_verdict_by_score = "Average Uniqueness"
    else:
        base_verdict_by_score = "Generic"

    override_applied = "none"
    verdict = base_verdict_by_score
    strength_note = None
    identity_note = None

    if uniqueness_pattern == "total_original" and verdict in ("Average Uniqueness", "Generic"):
        override_applied = "total_original_min_distinctive"
        verdict = "Distinctive"
    elif uniqueness_pattern == "cookie_cutter" and verdict in ("Highly Unique", "Distinctive"):
        override_applied = "cookie_cutter_max_average"
        verdict = "Average Uniqueness"

    if uniqueness_pattern in ("total_original", "culinary_innovator", "atmosphere_destination", "experience_pioneer"):
        strength_note = f"Pattern: {uniqueness_pattern.replace('_', ' ').title()}"

    if l15_buckets["strongest_category"]:
        identity_note = f"Identity strength: {l15_buckets['strongest_category']}"

    result = {
        "L1_5_category_buckets": l15_buckets,
        "N_UNIQUENESS_REVIEWS": n_uniqueness_reviews,
        "CONFIDENCE_LEVEL": confidence_level,
        "N_POSITIVE": n_positive,
        "N_NEGATIVE": n_negative,
        "UNIQUENESS_RATE": round(uniqueness_rate, 3),
        "POSITIVE_SCORE": round(positive_score, 2),
        "NEGATIVE_SCORE": round(negative_score, 2),
        "BASE_SCORE": BASE_SCORE,
        "UNIQUENESS_PATTERN_BONUS": uniqueness_pattern_bonus,
        "RAW_SCORE": round(raw_score, 2),
        "FINAL_SCORE": round(final_score, 2),
        "base_verdict_by_score": base_verdict_by_score,
        "override_applied": override_applied,
        "verdict": verdict,
        "n_uniqueness_reviews": n_uniqueness_reviews,
    }

    if strength_note:
        result["strength_note"] = strength_note
    if identity_note:
        result["identity_note"] = identity_note

    return result
