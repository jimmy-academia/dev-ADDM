"""Policy-based ground truth computation.

This module provides functions to compute ground truth verdicts from L0 judgments
using PolicyIR scoring rules (V3+) or qualitative decision rules (V1/V2).

Two-step flow:
1. extract.py: Extract L0 judgments from reviews (multi-model, aggregated)
2. compute_gt.py: Apply policy scoring to produce verdicts (this module)
"""

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from addm.query.libraries import TERMS_DIR
from addm.query.models.policy import PolicyIR, ScoringSystem
from addm.query.models.term import Term, TermLibrary

# =============================================================================
# Human judgment overrides
# =============================================================================

OVERRIDE_FILE = Path("data/answers/yelp/judgment_overrides.json")
OVERRIDE_KEYWORD_FILE = Path("data/answers/yelp/judgment_overrides_keyword.json")


def load_overrides(topic: str) -> Dict[str, Dict[str, Any]]:
    """Load human judgment overrides for a topic.

    Merges overrides from both manual override file and keyword-based override file.

    Args:
        topic: Topic name (e.g., "G1_allergy")

    Returns:
        Dict mapping review_id to override spec
    """
    result: Dict[str, Dict[str, Any]] = {}

    # Load manual overrides
    if OVERRIDE_FILE.exists():
        with open(OVERRIDE_FILE) as f:
            data = json.load(f)
        # Get overrides for this topic (list format)
        topic_overrides = data.get(topic, [])
        for o in topic_overrides:
            result[o["review_id"]] = o

    # Load keyword-based overrides
    if OVERRIDE_KEYWORD_FILE.exists():
        with open(OVERRIDE_KEYWORD_FILE) as f:
            data = json.load(f)
        # Keyword file has nested structure: topic -> overrides -> [list]
        if topic in data and isinstance(data[topic], dict):
            keyword_overrides = data[topic].get("overrides", [])
            for o in keyword_overrides:
                # Keyword overrides don't overwrite manual overrides
                if o["review_id"] not in result:
                    result[o["review_id"]] = o

    return result


def apply_overrides(
    judgment: Dict[str, Any],
    overrides: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Apply human overrides to an aggregated judgment.

    Args:
        judgment: Aggregated judgment dict
        overrides: Dict of review_id -> override spec

    Returns:
        Judgment with overrides applied (if any)
    """
    review_id = judgment.get("review_id")
    if review_id not in overrides:
        return judgment

    override = overrides[review_id]
    result = judgment.copy()

    # Apply corrected values
    for field, value in override.get("corrected", {}).items():
        result[field] = value

    # Mark as overridden for audit trail
    result["_override_applied"] = True
    result["_override_reason"] = override.get("reason", "")

    return result


# =============================================================================
# Multi-model aggregation constants (cost-optimized)
# See docs/future/high_quality_gt.md for more robust config
# =============================================================================

MODEL_WEIGHTS = {
    "gpt-5-mini": 2,
    "gpt-5-nano": 1,
}
REQUIRED_RUNS = {
    "gpt-5-mini": 1,
    "gpt-5-nano": 3,
}
TOTAL_WEIGHT = sum(MODEL_WEIGHTS[m] * REQUIRED_RUNS[m] for m in MODEL_WEIGHTS)  # 5

# =============================================================================
# Cuisine risk detection
# =============================================================================

HIGH_RISK_CUISINES = [
    "Thai",
    "Vietnamese",
    "Chinese",
    "Asian",
    "Asian Fusion",
]


def is_high_risk_cuisine(categories: str) -> bool:
    """Check if restaurant categories include high-risk cuisines."""
    if not categories:
        return False
    categories_lower = categories.lower()
    for cuisine in HIGH_RISK_CUISINES:
        if cuisine.lower() in categories_lower:
            return True
    return False


# =============================================================================
# Term library loading
# =============================================================================


def load_term_library() -> TermLibrary:
    """Load all term libraries from the standard location."""
    library = TermLibrary()

    # Load shared terms
    shared_path = TERMS_DIR / "_shared.yaml"
    if shared_path.exists():
        library.load_domain("shared", shared_path)

    # Load topic-specific terms (allergy, dietary, etc.)
    for yaml_file in TERMS_DIR.glob("*.yaml"):
        if yaml_file.name.startswith("_"):
            continue
        domain = yaml_file.stem  # e.g., "allergy" from "allergy.yaml"
        library.load_domain(domain, yaml_file)

    return library


def build_l0_schema_from_policy(
    policy: PolicyIR, library: TermLibrary
) -> Dict[str, Dict[str, str]]:
    """
    Build L0 extraction schema from policy's term references.

    Args:
        policy: PolicyIR with term refs in normative.terms
        library: Loaded TermLibrary

    Returns:
        Dict mapping field name (lowercase) to {value_id: description}
    """
    schema = {}
    for ref in policy.get_term_refs():
        term = library.resolve(ref)
        # Field name is term ID in lowercase (e.g., "ACCOUNT_TYPE" -> "account_type")
        field_name = term.id.lower()
        schema[field_name] = {v.id: v.description for v in term.values}
    return schema


def build_l0_schema_from_topic(topic: str, library: TermLibrary) -> Dict[str, Dict[str, str]]:
    """
    Build L0 extraction schema from a topic (union of all V* policy term refs).

    Topic format: "G1_allergy" (group_topic)

    Uses the union of terms across V1-V4 to ensure all fields needed by any
    policy version are extracted.

    Args:
        topic: Topic identifier (e.g., "G1_allergy")
        library: Loaded TermLibrary

    Returns:
        Dict mapping field name to {value_id: description}
    """
    # Find policy directory for this topic
    parts = topic.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid topic format: {topic} (expected G1_allergy)")

    group, topic_name = parts

    # Collect union of terms across all versions
    policies_dir = Path("src/addm/query/policies")
    topic_dir = policies_dir / group / topic_name

    if not topic_dir.exists():
        raise FileNotFoundError(f"Topic directory not found: {topic_dir}")

    schema: Dict[str, Dict[str, str]] = {}
    for version in ["V1", "V2", "V3", "V4"]:
        policy_path = topic_dir / f"{version}.yaml"
        if policy_path.exists():
            policy = PolicyIR.load(policy_path)
            version_schema = build_l0_schema_from_policy(policy, library)
            # Merge into union (later versions may add fields)
            for field, values in version_schema.items():
                if field not in schema:
                    schema[field] = values

    if not schema:
        raise ValueError(f"No valid policies found for topic: {topic}")

    return schema


# =============================================================================
# Term version tracking (hash-based)
# =============================================================================


def compute_term_hash(l0_schema: Dict[str, Dict[str, str]]) -> str:
    """
    Compute deterministic hash of term definitions for version tracking.

    Args:
        l0_schema: L0 schema dict

    Returns:
        16-char hex hash string
    """
    canonical = json.dumps(l0_schema, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# =============================================================================
# Multi-model aggregation
# =============================================================================


def aggregate_judgments(
    raw_judgments: List[Dict[str, Any]], l0_schema: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    """
    Aggregate multiple model extractions via weighted majority voting.

    Args:
        raw_judgments: List of raw judgments with "_model" field
        l0_schema: L0 schema for field enumeration

    Returns:
        Aggregated judgment with "_confidence" per field
    """
    if not raw_judgments:
        return {"_aggregated": True, "_confidence": {}}

    aggregated: Dict[str, Any] = {"_aggregated": True, "_confidence": {}}

    # Copy non-schema fields from first judgment (review_id, date, etc.)
    first = raw_judgments[0]
    for key in ["review_id", "date", "stars", "useful"]:
        if key in first:
            aggregated[key] = first[key]

    # Aggregate each L0 field via weighted voting
    for field in l0_schema.keys():
        votes: Dict[str, int] = defaultdict(int)

        for j in raw_judgments:
            model = j.get("_model", "gpt-5-nano")
            value = j.get(field, "none")
            weight = MODEL_WEIGHTS.get(model, 1)
            votes[value] += weight

        if votes:
            winner, weight = max(votes.items(), key=lambda x: x[1])
            aggregated[field] = winner
            total_votes = sum(votes.values())
            aggregated["_confidence"][field] = round(weight / total_votes, 3)
        else:
            # Default to first valid value
            aggregated[field] = list(l0_schema[field].keys())[0]
            aggregated["_confidence"][field] = 0.0

    # Handle is_relevant specially (not in l0_schema)
    # Support both is_relevant (new) and is_allergy_related (legacy cache)
    relevance_votes: Dict[str, int] = defaultdict(int)
    for j in raw_judgments:
        model = j.get("_model", "gpt-5-nano")
        # Check is_relevant first (new), fallback to is_allergy_related (legacy)
        is_rel = j.get("is_relevant", j.get("is_allergy_related", False))
        relevance_votes[str(is_rel).lower()] += MODEL_WEIGHTS.get(model, 1)

    if relevance_votes:
        winner, _ = max(relevance_votes.items(), key=lambda x: x[1])
        aggregated["is_relevant"] = winner == "true"

    return aggregated


# =============================================================================
# Scoring-based GT computation (V3+)
# =============================================================================


def get_severity_points(severity: str, scoring: ScoringSystem) -> int:
    """Get points for a severity level from scoring system."""
    # Map severity to label (case-insensitive partial match)
    for sp in scoring.severity_points:
        if severity.lower() in sp.label.lower():
            return sp.points
    return 0


def get_modifier_points(label: str, scoring: ScoringSystem) -> int:
    """Get points for a modifier by label (case-insensitive partial match)."""
    for mod in scoring.modifiers:
        if label.lower() in mod.label.lower():
            return mod.points
    return 0


def determine_verdict(
    score: int, scoring: ScoringSystem, default_verdict: str = "Unknown", k: int = 200
) -> str:
    """Determine verdict based on score and thresholds.

    Handles both min_score (score >= threshold) and max_score (score <= threshold)
    for bidirectional scoring systems (where positive is good, negative is bad).

    Args:
        score: Accumulated score
        scoring: ScoringSystem with thresholds
        default_verdict: Fallback verdict if no threshold matches
        k: Dataset K value (for K-specific thresholds)
    """
    # Check min_score thresholds first (sorted descending by K-specific threshold)
    min_thresholds = [t for t in scoring.thresholds if t.get_min_score(k) is not None]
    min_thresholds.sort(key=lambda t: t.get_min_score(k), reverse=True)

    for threshold in min_thresholds:
        threshold_value = threshold.get_min_score(k)
        if score >= threshold_value:
            return threshold.verdict

    # Check max_score thresholds (sorted ascending by K-specific threshold)
    max_thresholds = [t for t in scoring.thresholds if t.get_max_score(k) is not None]
    max_thresholds.sort(key=lambda t: t.get_max_score(k))

    for threshold in max_thresholds:
        threshold_value = threshold.get_max_score(k)
        if score <= threshold_value:
            return threshold.verdict

    # Return the default verdict if no threshold matches
    return default_verdict


def compute_recency_weight(review_date_str: Optional[str], scoring: ScoringSystem) -> float:
    """
    Compute recency weight for V4 scoring based on review age.

    Args:
        review_date_str: Review date string (e.g., "2019-12-15 19:02:50")
        scoring: ScoringSystem with recency_rules

    Returns:
        Weight multiplier (1.0, 0.5, or 0.25)
    """
    if not scoring.recency_rules or not review_date_str:
        return 1.0

    try:
        # Parse review date
        review_date = datetime.strptime(review_date_str[:10], "%Y-%m-%d")
        # Use a fixed reference date for reproducibility (latest data is ~2022)
        reference_date = datetime(2022, 1, 1)
        age_years = (reference_date - review_date).days / 365.25

        # Parse recency rules to determine weight
        # Rules are ordered by recency (most recent first)
        for rule in scoring.recency_rules:
            age_text = rule.age.lower()
            weight_text = rule.weight.lower()

            # Parse age condition
            if "within" in age_text:
                # "Within 1 year" or "Within 2 years"
                if "1 year" in age_text and age_years <= 1:
                    return _parse_weight(weight_text)
                elif "2 year" in age_text and age_years <= 2:
                    return _parse_weight(weight_text)
            elif "1-2" in age_text or "1 to 2" in age_text:
                if 1 < age_years <= 2:
                    return _parse_weight(weight_text)
            elif "2-3" in age_text or "2 to 3" in age_text:
                if 2 < age_years <= 3:
                    return _parse_weight(weight_text)
            elif "over" in age_text:
                # "Over 2 years" or "Over 3 years"
                if "2 year" in age_text and age_years > 2:
                    return _parse_weight(weight_text)
                elif "3 year" in age_text and age_years > 3:
                    return _parse_weight(weight_text)

        return 1.0  # Default: full weight
    except (ValueError, AttributeError):
        return 1.0  # If date parsing fails, use full weight


def _parse_weight(weight_text: str) -> float:
    """Parse weight from human-readable text."""
    if "quarter" in weight_text or "0.25" in weight_text:
        return 0.25
    elif "half" in weight_text or "0.5" in weight_text:
        return 0.5
    else:
        return 1.0  # "full" or default


def compute_gt_from_policy_scoring(
    judgments: List[Dict[str, Any]],
    policy: PolicyIR,
    restaurant_meta: Dict[str, Any],
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    k: int = 200,
) -> Dict[str, Any]:
    """
    Compute GT using policy scoring rules (V3+).

    Uses field_mapping from policy to determine which judgment fields to use
    for scoring. Falls back to G1-style hardcoded fields if no field_mapping.

    Args:
        judgments: List of aggregated L0 judgments (one per review)
        policy: PolicyIR with scoring system
        restaurant_meta: Restaurant metadata with 'categories' field
        overrides: Optional dict of review_id -> override spec (from load_overrides)
        k: Dataset K value (for K-specific score thresholds)

    Returns:
        Dict with verdict, score, and incident details
    """
    scoring = policy.normative.scoring
    if not scoring:
        raise ValueError(f"Policy {policy.policy_id} has no scoring system")

    # Apply overrides before processing
    overrides = overrides or {}
    judgments = [apply_overrides(j, overrides) for j in judgments]

    categories = restaurant_meta.get("categories", "")
    total_score = 0
    incidents: List[Dict[str, Any]] = []

    # Get field mapping (or use G1-style defaults for backward compatibility)
    field_mapping = scoring.field_mapping
    if field_mapping:
        severity_field = field_mapping.severity_field
        value_mappings = field_mapping.value_mappings
        modifier_mappings = field_mapping.modifier_mappings
    else:
        # Legacy G1-style hardcoded behavior
        severity_field = "incident_severity"
        value_mappings = {
            "mild": "Mild incident",
            "moderate": "Moderate incident",
            "severe": "Severe incident",
        }
        modifier_mappings = []  # Will use hardcoded logic below

    for j in judgments:
        # Filter non-relevant (support both new and legacy field names)
        is_relevant = j.get("is_relevant", j.get("is_allergy_related", False))
        if not is_relevant:
            continue

        # Filter non-firsthand (only for policies that use account_type)
        account_type = j.get("account_type")
        if account_type is not None and account_type != "firsthand":
            continue

        # Get severity from mapped field
        field_value = j.get(severity_field, "none")
        if field_value == "none" or field_value is None:
            continue

        # Map field value to severity label
        severity_label = value_mappings.get(field_value)
        if not severity_label:
            # No mapping for this value - skip (e.g., "adequate" might not score)
            continue

        # Base points for severity
        points = get_severity_points(severity_label, scoring)

        # Check modifiers
        applied_modifiers: List[str] = []

        if field_mapping and modifier_mappings:
            # Use policy-defined modifier mappings
            for mod_map in modifier_mappings:
                mod_field_value = j.get(mod_map.field)
                if mod_field_value == mod_map.value:
                    mod_pts = get_modifier_points(mod_map.label, scoring)
                    points += mod_pts
                    if mod_pts != 0:
                        applied_modifiers.append(mod_map.label)
        else:
            # Legacy G1-style hardcoded modifiers
            # False assurance modifier
            assurance = j.get("assurance_claim", "false")
            if assurance == "true":
                mod_pts = get_modifier_points("False assurance", scoring)
                points += mod_pts
                if mod_pts > 0:
                    applied_modifiers.append("False assurance")

            # Dismissive staff modifier
            staff_response = j.get("staff_response", "none")
            if staff_response == "dismissive":
                mod_pts = get_modifier_points("Dismissive staff", scoring)
                points += mod_pts
                if mod_pts > 0:
                    applied_modifiers.append("Dismissive staff")

        # Apply V4 recency weighting if policy has recency_rules
        recency_weight = compute_recency_weight(j.get("_review_date"), scoring)
        weighted_points = int(points * recency_weight)

        total_score += weighted_points
        incidents.append({
            "review_id": j.get("review_id", ""),
            "severity_field": severity_field,
            "severity_value": field_value,
            "severity_label": severity_label,
            "base_points": get_severity_points(severity_label, scoring),
            "modifiers": applied_modifiers,
            "total_points": points,
            "recency_weight": recency_weight,
            "weighted_points": weighted_points,
        })

    # Cuisine modifier (restaurant-level, not per-incident)
    # Only applies when there are actual incidents (per V3 policy)
    cuisine_modifier_applied = False
    if is_high_risk_cuisine(categories) and len(incidents) > 0:
        mod_pts = get_modifier_points("High-risk cuisine", scoring)
        total_score += mod_pts
        if mod_pts > 0:
            cuisine_modifier_applied = True

    # Determine verdict
    # Find the default verdict from decision rules
    default_verdict = "Unknown"
    for rule in policy.normative.decision.rules:
        if rule.default:
            default_verdict = rule.verdict
            break

    verdict = determine_verdict(total_score, scoring, default_verdict, k)

    return {
        "verdict": verdict,
        "score": total_score,
        "n_incidents": len(incidents),
        "incidents": incidents,
        "cuisine_modifier": cuisine_modifier_applied,
        "policy_id": policy.policy_id,
    }


# =============================================================================
# Qualitative GT computation (V1/V2)
# =============================================================================


def evaluate_structured_condition(
    condition: Dict[str, Any],
    judgments: List[Dict[str, Any]],
    k: int = 200,
) -> bool:
    """
    Evaluate a structured condition against judgments.

    Condition types:
    - count_threshold: count matching >= min_count
    - exists: at least one matching

    Args:
        condition: Structured condition from policy
        judgments: List of L0 judgments
        k: Dataset K value (used to lookup K-specific thresholds)
    """
    cond_type = condition.get("type", "count_threshold")
    filter_spec = condition.get("filter", {})

    # Check for K-specific threshold first, fall back to default min_count
    min_count_by_k = condition.get("min_count_by_k", {})
    if k in min_count_by_k:
        min_count = min_count_by_k[k]
    elif str(k) in min_count_by_k:
        min_count = min_count_by_k[str(k)]
    else:
        min_count = condition.get("min_count", 1)

    # Count matching judgments
    count = 0
    for j in judgments:
        # Filter non-relevant (support both new and legacy field names)
        is_relevant = j.get("is_relevant", j.get("is_allergy_related", False))
        if not is_relevant:
            continue

        # Check all filter conditions
        matches = True
        for field, expected in filter_spec.items():
            actual = j.get(field, "")
            if isinstance(expected, list):
                if actual not in expected:
                    matches = False
                    break
            elif actual != expected:
                matches = False
                break

        if matches:
            count += 1

    if cond_type == "exists":
        return count >= 1
    else:  # count_threshold
        return count >= min_count


def compute_gt_from_policy_qualitative(
    judgments: List[Dict[str, Any]],
    policy: PolicyIR,
    restaurant_meta: Dict[str, Any],
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    k: int = 200,
) -> Dict[str, Any]:
    """
    Compute GT using qualitative decision rules (V1/V2).

    Args:
        judgments: List of aggregated L0 judgments
        policy: PolicyIR with decision rules
        restaurant_meta: Restaurant metadata
        overrides: Optional dict of review_id -> override spec (from load_overrides)
        k: Dataset K value (for K-specific thresholds)

    Returns:
        Dict with verdict, rule that matched, and incidents list
    """
    # Apply overrides before processing
    overrides = overrides or {}
    judgments = [apply_overrides(j, overrides) for j in judgments]

    rules = policy.normative.decision.rules

    # Extract evidence from judgments (same as scoring, but without point calculation)
    # This enables evaluation metrics to work for V1/V2 policies
    from addm.eval.constants import get_evidence_config

    evidences: List[Dict[str, Any]] = []

    # Get evidence field config for this policy group
    evidence_config = get_evidence_config(policy.policy_id)
    evidence_field = evidence_config["field"]
    evidence_values = evidence_config["evidence_values"]

    for j in judgments:
        review_id = j.get("review_id", "")
        field_value = j.get(evidence_field, "").lower().strip()

        # Only include if it's actual evidence (not neutral/"none"/empty)
        if field_value in evidence_values:
            # Build modifiers list from assurance_claim and staff_response
            modifiers = []
            assurance = j.get("assurance_claim", "false").lower().strip()
            if assurance == "true":
                modifiers.append("False assurance")
            staff_response = j.get("staff_response", "none").lower().strip()
            if staff_response == "dismissive":
                modifiers.append("Dismissive staff")

            evidences.append({
                "review_id": review_id,
                "severity_field": evidence_field,  # Keep name for backward compat
                "severity_value": field_value,     # Keep name for backward compat
                # Include account_type for reference
                "account_type": j.get("account_type", "").lower().strip(),
                "modifiers": modifiers,
            })

    # Evaluate rules in order (precedence ladder)
    matched_verdict = None
    matched_rule = None

    for rule in rules:
        if rule.default:
            continue  # Skip default, evaluate it last

        # Get structured conditions
        structured = rule.get_structured_conditions()

        if not structured:
            # No structured conditions, skip (NL-only rules)
            continue

        # Evaluate based on logic (ANY or ALL)
        if rule.logic.value == "ANY":
            matched = any(
                evaluate_structured_condition(c, judgments, k) for c in structured
            )
        else:  # ALL
            matched = all(
                evaluate_structured_condition(c, judgments, k) for c in structured
            )

        if matched:
            matched_verdict = rule.verdict
            matched_rule = rule.label
            break

    # Find default rule if no match
    if matched_verdict is None:
        for rule in rules:
            if rule.default:
                matched_verdict = rule.verdict
                matched_rule = rule.label
                break

    if matched_verdict is None:
        matched_verdict = "Unknown"

    return {
        "verdict": matched_verdict,
        "matched_rule": matched_rule,
        "policy_id": policy.policy_id,
        "n_incidents": len(evidences),  # Keep field name for backward compat
        "incidents": evidences,          # Keep field name for backward compat
    }


# =============================================================================
# Main entry point
# =============================================================================


def compute_gt_from_policy(
    judgments: List[Dict[str, Any]],
    policy: PolicyIR,
    restaurant_meta: Dict[str, Any],
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    k: int = 200,
) -> Dict[str, Any]:
    """
    Compute ground truth using policy rules.

    Automatically selects scoring (V3+) or qualitative (V1/V2) based on policy.

    Args:
        judgments: List of aggregated L0 judgments (one per review)
        policy: PolicyIR loaded from YAML
        restaurant_meta: Restaurant metadata with 'categories' field
        overrides: Optional dict of review_id -> override spec (from load_overrides)
        k: Dataset K value (for K-specific thresholds in qualitative rules)

    Returns:
        Dict with verdict and supporting details
    """
    if policy.normative.scoring:
        return compute_gt_from_policy_scoring(judgments, policy, restaurant_meta, overrides, k)
    else:
        return compute_gt_from_policy_qualitative(judgments, policy, restaurant_meta, overrides, k)


def load_policy(policy_id: str) -> PolicyIR:
    """
    Load a policy by ID.

    Policy ID formats:
        T* format: "T1P1" -> policies/T1/P1.yaml
        G* format: "G1_allergy_V3" -> policies/G1/allergy/V3.yaml

    Args:
        policy_id: Policy identifier

    Returns:
        Loaded PolicyIR
    """
    policies_dir = Path("src/addm/query/policies")

    # T* policy format: T1P1, T2P3, etc.
    if len(policy_id) >= 4 and policy_id[0] == "T" and policy_id[1].isdigit() and "P" in policy_id:
        tier = policy_id[:2]  # T1
        variant = policy_id[2:]  # P1
        policy_path = policies_dir / tier / f"{variant}.yaml"

        if not policy_path.exists():
            raise FileNotFoundError(f"Policy not found: {policy_path}")

        return PolicyIR.load(policy_path)

    # G* policy format: G1_allergy_V3 -> policies/G1/allergy/V3.yaml
    parts = policy_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid policy_id format: {policy_id} (expected G1_allergy_V3 or T1P1)")

    # First part is group (G1, G2, ...), last part is version (V1, V2, ...)
    # Everything in between is the topic (may contain underscores)
    group = parts[0]
    version = parts[-1]
    topic = "_".join(parts[1:-1])

    policy_path = policies_dir / group / topic / f"{version}.yaml"

    if not policy_path.exists():
        raise FileNotFoundError(f"Policy not found: {policy_path}")

    return PolicyIR.load(policy_path)


# Re-export from constants for backward compatibility
from addm.tasks.constants import get_topic_from_policy_id
