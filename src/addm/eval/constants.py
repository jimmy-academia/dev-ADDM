"""Evaluation constants for ADDM benchmark.

Canonical scoring values for V2 policies. These are used for:
- Verdict consistency checks (recompute verdict from evidence)
- Judgment accuracy evaluation

Note: Future improvement is to load these from policy YAML files
to support different scoring schemes across policy groups (G1-G6).
"""

# Canonical severity base points (V2 policy scoring)
SEVERITY_BASE_POINTS = {
    "mild": 2,
    "moderate": 5,
    "severe": 15,
}

# Modifier point adjustments (V2 policy)
MODIFIER_POINTS = {
    "False assurance": 5,
    "Dismissive staff": 3,
}

# Verdict thresholds (V2 policy)
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
# Each topic has a primary outcome field and values that constitute "evidence"
# (non-neutral values that indicate something noteworthy happened)
#
# Keyed by topic (e.g., "G1_allergy") with fallback to group (e.g., "G1")
# This allows topic-specific overrides within the same group.
EVIDENCE_FIELDS = {
    # G1: Customer Safety (all topics use incident_severity)
    "G1": {
        "field": "incident_severity",
        "evidence_values": {"mild", "moderate", "severe"},
    },
    # G2: Customer Experience (all topics use date_outcome)
    "G2": {
        "field": "date_outcome",
        "evidence_values": {"positive", "negative"},
    },
    # G3: Customer Value - TOPIC-SPECIFIC FIELDS
    "G3_price_worth": {
        "field": "price_perception",
        "evidence_values": {"ripoff", "overpriced", "good_value", "steal"},
    },
    "G3_hidden_costs": {
        "field": "impact_severity",
        "evidence_values": {"minor", "moderate", "significant"},
    },
    "G3_time_value": {
        "field": "wait_justification",
        "evidence_values": {"not_worth_it", "barely", "worth_it", "definitely_worth_it"},
    },
    "G3": {  # Fallback for unknown G3 topics
        "field": "quality_for_price",
        "evidence_values": {"excellent", "good", "poor", "terrible"},
    },
    # G4: Owner Operations - TOPIC-SPECIFIC FIELDS
    "G4_server": {
        "field": "attentiveness",
        "evidence_values": {"excellent", "good", "poor", "absent"},
    },
    "G4_kitchen": {
        "field": "issue_severity",
        "evidence_values": {"minor", "moderate", "significant"},
    },
    "G4_environment": {
        "field": "issue_severity",
        "evidence_values": {"minor", "moderate", "significant"},
    },
    "G4": {  # Fallback
        "field": "attentiveness",
        "evidence_values": {"excellent", "good", "poor", "terrible"},
    },
    # G5: Owner Performance
    "G5_capacity": {
        "field": "service_degradation",
        "evidence_values": {"minor", "moderate", "severe"},
    },
    "G5_execution": {
        "field": "execution_issue",
        "evidence_values": {"minor", "moderate", "significant"},
    },
    "G5_consistency": {
        "field": "consistency_issue",
        "evidence_values": {"minor", "moderate", "significant"},
    },
    "G5": {  # Fallback
        "field": "service_degradation",
        "evidence_values": {"minor", "moderate", "severe"},
    },
    # G6: Owner Strategy
    "G6_uniqueness": {
        "field": "memorability",
        "evidence_values": {"forgettable", "memorable", "unforgettable"},
    },
    "G6_comparison": {
        "field": "comparison_outcome",
        "evidence_values": {"worse", "similar", "better"},
    },
    "G6_loyalty": {
        "field": "loyalty_signal",
        "evidence_values": {"detractor", "neutral", "promoter"},
    },
    "G6": {  # Fallback
        "field": "memorability",
        "evidence_values": {"memorable", "remarkable", "forgettable"},
    },
}


def get_evidence_config(policy_id: str) -> dict:
    """Get evidence field configuration for a policy.

    Looks up by topic first (e.g., "G3_hidden_costs"), then falls back to group (e.g., "G3").

    Args:
        policy_id: Policy ID like "G1_allergy_V0" or "G3_hidden_costs_V2"

    Returns:
        Dict with 'field' and 'evidence_values' keys
    """
    # Parse policy_id: "G3_hidden_costs_V2" -> group="G3", topic="hidden_costs"
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
# Verified from src/addm/query/policies/G{1-6}/*/V0.yaml
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
        policy_id: Policy ID like "G1_allergy_V0" or "G2_romance_V1"

    Returns:
        Dict mapping verdict strings to ordinal values (0, 1, 2, ...)
    """
    group = policy_id.split("_")[0] if "_" in policy_id else policy_id[:2]
    verdicts = POLICY_VERDICTS.get(group, POLICY_VERDICTS["G1"])
    return {v: i for i, v in enumerate(verdicts)}
