"""Evaluation constants for ADDM benchmark.

Canonical scoring values for V3 policies. These are used for:
- Verdict consistency checks (recompute verdict from evidence)
- Judgment accuracy evaluation

Note: Future improvement is to load these from policy YAML files
to support different scoring schemes across policy groups (G1-G6).
"""

# Canonical severity base points (V3 policy scoring)
SEVERITY_BASE_POINTS = {
    "mild": 2,
    "moderate": 5,
    "severe": 15,
}

# Modifier point adjustments (V3 policy)
MODIFIER_POINTS = {
    "False assurance": 5,
    "Dismissive staff": 3,
}

# Verdict thresholds (V3 policy)
# Score >= threshold → that verdict
VERDICT_THRESHOLDS = {
    "Critical Risk": 8,
    "High Risk": 4,
    "Low Risk": 0,
}

# Ordinal mapping for AUPRC computation
VERDICT_TO_ORDINAL = {
    "Low Risk": 0,
    "High Risk": 1,
    "Critical Risk": 2,
}

# Class names in order (low to high)
CLASS_NAMES = ["Low Risk", "High Risk", "Critical Risk"]

# Evidence field definitions per policy topic
# AUTO-GENERATED from term libraries by scripts/generate_evidence_fields.py
# DO NOT EDIT MANUALLY - regenerate with: python scripts/generate_evidence_fields.py
#
# Each topic has a primary outcome field and values that constitute "evidence"
# (non-neutral values that indicate something noteworthy happened)
EVIDENCE_FIELDS = {
    "G1": {  # Fallback
        "field": "incident_severity",
        "evidence_values": {"mild", "moderate", "severe"},
    },
    # G1: Customer Safety
    "G1_allergy": {
        "field": "incident_severity",
        "evidence_values": {"mild", "moderate", "severe"},
    },
    "G1_dietary": {
        "field": "incident_severity",
        "evidence_values": {"mild", "moderate", "severe"},
    },
    "G1_hygiene": {
        "field": "issue_severity",
        "evidence_values": {"major", "minor", "severe"},
    },
    "G2": {  # Fallback
        "field": "date_outcome",
        "evidence_values": {"memorable", "negative", "positive"},
    },
    # G2: Customer Experience
    "G2_business": {
        "field": "meeting_outcome",
        "evidence_values": {"negative", "positive"},
    },
    "G2_group": {
        "field": "group_outcome",
        "evidence_values": {"disaster", "great", "mixed"},
    },
    "G2_romance": {
        "field": "date_outcome",
        "evidence_values": {"memorable", "negative", "positive"},
    },
    "G3": {  # Fallback
        "field": "price_perception",
        "evidence_values": {"good_value", "overpriced", "ripoff", "steal"},
    },
    # G3: Customer Value
    "G3_hidden_costs": {
        "field": "impact_severity",
        "evidence_values": {"minor", "significant"},
    },
    "G3_price_worth": {
        "field": "price_perception",
        "evidence_values": {"good_value", "overpriced", "ripoff", "steal"},
    },
    "G3_time_value": {
        "field": "wait_justification",
        "evidence_values": {"definitely_worth_it", "not_worth_it", "worth_it"},
    },
    "G4": {  # Fallback
        "field": "attentiveness",
        "evidence_values": {"absent", "excellent", "good", "poor"},
    },
    # G4: Owner Operations
    "G4_environment": {
        "field": "cleanliness",
        "evidence_values": {"dirty", "fair", "spotless"},
    },
    "G4_kitchen": {
        "field": "issue_severity",
        "evidence_values": {"inedible", "major", "minor"},
    },
    "G4_server": {
        "field": "attentiveness",
        "evidence_values": {"absent", "excellent", "good", "poor"},
    },
    "G5": {  # Fallback
        "field": "service_degradation",
        "evidence_values": {"complete", "minor", "significant"},
    },
    # G5: Owner Performance
    "G5_capacity": {
        "field": "service_degradation",
        "evidence_values": {"complete", "minor", "significant"},
    },
    "G5_consistency": {
        "field": "visit_comparison",
        "evidence_values": {"better", "much_better", "much_worse", "worse"},
    },
    "G5_execution": {
        "field": "issue_resolution",
        "evidence_values": {"excellent", "none", "poor"},
    },
    "G6": {  # Fallback
        "field": "memorability",
        "evidence_values": {"forgettable", "memorable", "unforgettable"},
    },
    # G6: Owner Strategy
    "G6_comparison": {
        "field": "comparison_outcome",
        "evidence_values": {"favorable", "unfavorable"},
    },
    "G6_loyalty": {
        "field": "loyalty_driver",
        "evidence_values": {"convenience", "love", "price", "quality", "service"},
    },
    "G6_uniqueness": {
        "field": "memorability",
        "evidence_values": {"forgettable", "memorable", "unforgettable"},
    },
}


def get_evidence_config(policy_id: str) -> dict:
    """Get evidence field configuration for a policy.

    Looks up by topic first (e.g., "G3_hidden_costs"), then falls back to group (e.g., "G3").

    Args:
        policy_id: Policy ID like "G1_allergy_V1" or "G3_hidden_costs_V3"

    Returns:
        Dict with 'field' and 'evidence_values' keys
    """
    # Parse policy_id: "G3_hidden_costs_V3" -> group="G3", topic="hidden_costs"
    # Format: {group}_{topic}_{variant} where topic can have underscores
    import re
    match = re.match(r"(G\d)_(.+)_V\d+$", policy_id)
    if match:
        group = match.group(1)
        topic_name = match.group(2)
        topic_key = f"{group}_{topic_name}"

        # Try topic-specific lookup first
        if topic_key in EVIDENCE_FIELDS:
            return EVIDENCE_FIELDS[topic_key]

        # Fall back to group-level config
        return EVIDENCE_FIELDS.get(group, EVIDENCE_FIELDS["G1"])

    # Fallback for non-standard formats
    parts = policy_id.split("_")
    group = parts[0] if parts else "G1"
    return EVIDENCE_FIELDS.get(group, EVIDENCE_FIELDS["G1"])


# Policy group verdict mappings (ordinal: lowest=0 to highest=n)
# Verified from src/addm/query/policies/G{1-6}/*/V1.yaml
POLICY_VERDICTS = {
    "G1": ["Low Risk", "High Risk", "Critical Risk"],
    "G2": ["Not Recommended", "Acceptable", "Recommended"],
    "G3": ["Poor Value", "Fair Value", "Good Value"],
    "G4": ["Needs Improvement", "Satisfactory", "Excellent"],
    "G5": ["Needs Improvement", "Satisfactory", "Excellent"],
    "G6": ["Generic", "Differentiated", "Highly Unique"],
}


def get_verdict_to_ordinal(policy_id: str) -> dict:
    """Get verdict-to-ordinal mapping for a policy.

    Args:
        policy_id: Policy ID like "G1_allergy_V1" or "G2_romance_V2"

    Returns:
        Dict mapping verdict strings to ordinal values (0, 1, 2, ...)
    """
    group = policy_id.split("_")[0] if "_" in policy_id else policy_id[:2]
    verdicts = POLICY_VERDICTS.get(group, POLICY_VERDICTS["G1"])
    return {v: i for i, v in enumerate(verdicts)}
