"""Scoring logic for policy ground truth computation.

Handles both V0/V1 rule-based evaluation and V2/V3 point-based scoring.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Default severity point values (used if not specified in policy)
DEFAULT_SEVERITY_POINTS = {
    "mild": 2,
    "moderate": 5,
    "severe": 15,
}

# High-risk cuisines for allergy topics
HIGH_RISK_CUISINES = {"Thai", "Vietnamese", "Chinese", "Asian", "Asian Fusion"}

# Current year for recency calculations
CURRENT_YEAR = 2022


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of how a score was computed."""

    severity_counts: Dict[str, int]  # e.g., {"mild": 2, "moderate": 1}
    severity_points: float
    modifier_points: Dict[str, float]  # e.g., {"false_assurance": 5}
    recency_adjustments: float
    total_score: float


def get_severity_points(severity: str, scoring_points: List[Any]) -> float:
    """Get point value for a severity level.

    Args:
        severity: Severity level (none, mild, moderate, severe)
        scoring_points: List of ScoringPoint from policy

    Returns:
        Point value for this severity, 0 if not found
    """
    if severity == "none":
        return 0.0

    # Map scoring point labels to severity
    label_map = {
        "mild incident": "mild",
        "moderate incident": "moderate",
        "severe incident": "severe",
    }

    for sp in scoring_points:
        label_lower = sp.label.lower()
        if label_lower in label_map and label_map[label_lower] == severity:
            return float(sp.points)

    # Fallback to defaults
    return float(DEFAULT_SEVERITY_POINTS.get(severity, 0))


def get_modifier_points(modifier_label: str, modifiers: List[Any]) -> float:
    """Get point value for a modifier.

    Args:
        modifier_label: Label to search for (case-insensitive partial match)
        modifiers: List of ScoringPoint modifiers from policy

    Returns:
        Point value for this modifier, 0 if not found
    """
    label_lower = modifier_label.lower()
    for m in modifiers:
        if label_lower in m.label.lower():
            return float(m.points)
    return 0.0


def is_high_risk_cuisine(categories: str) -> bool:
    """Check if restaurant categories include high-risk cuisines.

    Args:
        categories: Comma-separated category string

    Returns:
        True if any high-risk cuisine is present
    """
    if not categories:
        return False

    cats = [c.strip() for c in categories.split(",")]
    for cat in cats:
        for cuisine in HIGH_RISK_CUISINES:
            if cuisine.lower() in cat.lower():
                return True
    return False


def parse_date_year(date_str: Optional[str]) -> int:
    """Extract year from a date string.

    Args:
        date_str: Date string (e.g., "2019-12-15 19:02:50")

    Returns:
        Year as int, or default year if unparseable
    """
    if not date_str or len(date_str) < 4:
        return CURRENT_YEAR - 2  # Default to 2 years ago

    try:
        return int(date_str[:4])
    except ValueError:
        return CURRENT_YEAR - 2


def get_recency_weight(date_str: Optional[str], recency_rules: List[Any]) -> float:
    """Compute recency weight for a judgment based on its date.

    Args:
        date_str: Date string from judgment
        recency_rules: List of RecencyRule from policy

    Returns:
        Weight multiplier (0.0 to 1.0)
    """
    if not recency_rules:
        return 1.0  # No recency rules = full weight

    year = parse_date_year(date_str)
    age = CURRENT_YEAR - year

    # Parse recency rules (simplified parsing)
    # Rules are like: {age: "within 2 years", weight: "full point value"}
    for rule in recency_rules:
        age_str = rule.age.lower()
        weight_str = rule.weight.lower()

        # Parse age condition
        if "within" in age_str:
            # "within 2 years" -> age <= 2
            try:
                years = int("".join(c for c in age_str if c.isdigit()))
                if age <= years:
                    return _parse_weight(weight_str)
            except ValueError:
                continue
        elif "older" in age_str or "more than" in age_str:
            # "older than 3 years" or "more than 3 years" -> age > 3
            try:
                years = int("".join(c for c in age_str if c.isdigit()))
                if age > years:
                    return _parse_weight(weight_str)
            except ValueError:
                continue
        elif "-" in age_str:
            # "2-3 years" -> 2 <= age <= 3
            try:
                parts = age_str.replace("years", "").strip().split("-")
                low, high = int(parts[0].strip()), int(parts[1].strip())
                if low <= age <= high:
                    return _parse_weight(weight_str)
            except (ValueError, IndexError):
                continue

    return 1.0  # Default to full weight


def _parse_weight(weight_str: str) -> float:
    """Parse weight description to float multiplier."""
    if "full" in weight_str:
        return 1.0
    if "discard" in weight_str or "ignore" in weight_str:
        return 0.0
    if "multiply by" in weight_str or "×" in weight_str:
        # "multiply by 0.5" or "× 0.5"
        try:
            return float("".join(c for c in weight_str if c.isdigit() or c == "."))
        except ValueError:
            pass
    if "0." in weight_str:
        # Direct decimal like "0.5" or "0.25"
        try:
            return float("".join(c for c in weight_str if c.isdigit() or c == "."))
        except ValueError:
            pass
    return 1.0


def compute_score(
    judgments: List[Dict[str, Any]],
    scoring: Any,  # ScoringSystem
    restaurant_meta: Dict[str, Any],
    topic_filter_field: str = "is_allergy_related",
) -> Tuple[float, ScoreBreakdown]:
    """Compute total score from judgments using policy scoring system.

    Args:
        judgments: List of L0 judgments for the restaurant
        scoring: ScoringSystem from PolicyIR
        restaurant_meta: Restaurant metadata (name, categories)
        topic_filter_field: Field to filter relevant judgments

    Returns:
        Tuple of (total_score, breakdown)
    """
    total = 0.0
    severity_counts: Dict[str, int] = {"mild": 0, "moderate": 0, "severe": 0}
    severity_points_total = 0.0
    modifier_points: Dict[str, float] = {}
    recency_adjustments = 0.0

    for j in judgments:
        # Filter to relevant judgments
        if not j.get(topic_filter_field, False):
            continue

        # Only firsthand accounts count as incidents
        if j.get("account_type") != "firsthand":
            continue

        severity = j.get("incident_severity", "none")
        if severity == "none":
            continue

        # Count severity
        if severity in severity_counts:
            severity_counts[severity] += 1

        # Get base points for severity
        base_points = get_severity_points(severity, scoring.severity_points)

        # Apply recency weight if rules exist
        if scoring.recency_rules:
            weight = get_recency_weight(j.get("date"), scoring.recency_rules)
            adjusted_points = base_points * weight
            recency_adjustments += base_points - adjusted_points
            base_points = adjusted_points

        severity_points_total += base_points
        total += base_points

        # Check for modifiers on this judgment
        # False assurance: incident after explicit safety guarantee
        if j.get("assurance_claim") == "true":
            pts = get_modifier_points("false assurance", scoring.modifiers)
            if pts > 0:
                modifier_points["false_assurance"] = modifier_points.get(
                    "false_assurance", 0
                ) + pts
                total += pts

        # Dismissive staff
        if j.get("staff_response") == "dismissive":
            pts = get_modifier_points("dismissive", scoring.modifiers)
            if pts > 0:
                modifier_points["dismissive_staff"] = modifier_points.get(
                    "dismissive_staff", 0
                ) + pts
                total += pts

    # Restaurant-level modifiers
    categories = restaurant_meta.get("categories", "")
    if is_high_risk_cuisine(categories):
        pts = get_modifier_points("high-risk cuisine", scoring.modifiers)
        if pts > 0:
            modifier_points["high_risk_cuisine"] = pts
            total += pts

    breakdown = ScoreBreakdown(
        severity_counts=severity_counts,
        severity_points=severity_points_total,
        modifier_points=modifier_points,
        recency_adjustments=recency_adjustments,
        total_score=total,
    )

    return total, breakdown


def apply_thresholds(
    score: float, thresholds: List[Any]  # List[ScoringThreshold]
) -> str:
    """Map score to verdict using thresholds.

    Thresholds are checked in descending order of min_score.

    Args:
        score: Computed score
        thresholds: List of ScoringThreshold from policy

    Returns:
        Verdict string
    """
    # Sort by min_score descending
    sorted_thresholds = sorted(thresholds, key=lambda t: t.min_score, reverse=True)

    for t in sorted_thresholds:
        if score >= t.min_score:
            return t.verdict

    return "Low Risk"  # Default if no threshold matched
