"""Shared constants for tasks module.

T* System: 5 topics × 7 variants = 35 policies (T1P1, T1P2, etc.)

Variants:
- P1-P3: Rule variations (base, extended, ALL logic)
- P4-P7: Format variations (reorder v1, reorder v2, XML, prose)
"""

from typing import List, Optional

# =============================================================================
# G* topics (for extraction cache)
# =============================================================================

# Topics used in judgment extraction (maps to cache keys)
ALL_TOPICS = [
    "G1_allergy",
    "G3_price_worth",
    "G4_environment",
    "G5_execution",
    "G4_server",
]


def expand_topics(topic: Optional[str] = None) -> List[str]:
    """Expand topic specifier to list of topics.

    Args:
        topic: Single topic or None for all

    Returns:
        List of topic strings
    """
    if topic:
        return [topic]
    return ALL_TOPICS.copy()

# =============================================================================
# T* SYSTEM (35 policies: 5 topics × 7 variants)
# =============================================================================

# T1-T5 topic IDs (selected by Macro F1 performance)
TOPIC_IDS = ["T1", "T2", "T3", "T4", "T5"]

# Mapping from topic ID to G* topic (for GT file lookup)
# These G* files exist in data/answers/yelp/ from previous benchmark
TOPIC_ID_TO_GT_TOPIC = {
    "T1": "G1_allergy",      # F1: 0.64
    "T2": "G3_price_worth",  # F1: 0.56
    "T3": "G4_environment",  # F1: 0.56
    "T4": "G5_execution",    # F1: 0.56
    "T5": "G4_server",       # F1: 0.56
}

# P1-P7 variant descriptions
VARIANTS = {
    "P1": "base",           # Base rules (standard markdown)
    "P2": "extended",       # Extended rules (compound conditions)
    "P3": "tbd",            # TBD rule variation
    "P4": "reorder_v1",     # Verdicts before Definitions
    "P5": "reorder_v2",     # Interleaved structure
    "P6": "xml",            # XML format
    "P7": "prose",          # Prose format
}

# All 35 policies: 5 topics × 7 variants (T1P1, T1P2, ..., T5P7)
ALL_POLICIES = [f"{topic_id}P{v}" for topic_id in TOPIC_IDS for v in range(1, 8)]

# K values for ground truth / context sizes
K_VALUES = [25, 50, 100, 200]


def expand_policies(
    policy: Optional[str] = None,
    topic: Optional[str] = None,
) -> List[str]:
    """Expand policy specifier to list of policy IDs.

    Args:
        policy: Comma-separated policy IDs (e.g., "T1P1,T1P2")
        topic: Topic ID (e.g., "T1") or comma-separated topics (e.g., "T1,T2")

    Returns:
        List of policy IDs. Returns ALL_POLICIES if no argument specified.

    Raises:
        ValueError: If topic is invalid.
    """
    if policy:
        return [p.strip() for p in policy.split(",") if p.strip()]

    if topic:
        # Support comma-separated topics (e.g., "T1,T2")
        result = []
        for t in topic.split(","):
            t_upper = t.strip().upper()
            if t_upper not in TOPIC_IDS:
                raise ValueError(f"Unknown topic: {t}. Valid: {TOPIC_IDS}")
            result.extend([f"{t_upper}P{v}" for v in range(1, 8)])
        return result

    return ALL_POLICIES.copy()


def get_topic_id_from_policy_id(policy_id: str) -> str:
    """Extract topic ID from policy ID.

    "T1P1" -> "T1"
    "T3P5" -> "T3"
    """
    # Format: T{n}P{m}
    if len(policy_id) >= 2 and policy_id[0] == "T" and policy_id[1].isdigit():
        return policy_id[:2]
    raise ValueError(f"Invalid policy_id format: {policy_id}. Expected T{{n}}P{{m}}")


def get_variant_from_policy_id(policy_id: str) -> str:
    """Extract variant from policy ID.

    "T1P1" -> "P1"
    "T3P5" -> "P5"
    """
    # Format: T{n}P{m}
    if "P" in policy_id:
        idx = policy_id.index("P")
        return policy_id[idx:]
    raise ValueError(f"Invalid policy_id format: {policy_id}. Expected T{{n}}P{{m}}")


def get_gt_topic(topic_id: str) -> str:
    """Get the G* topic for ground truth lookup.

    "T1" -> "G1_allergy"
    "T2" -> "G3_price_worth"
    """
    return TOPIC_ID_TO_GT_TOPIC.get(topic_id, topic_id)


def is_valid_policy_id(policy_id: str) -> bool:
    """Check if policy ID is valid.

    "T1P1" -> True
    "T5P7" -> True
    "T1_P1" -> False (old format)
    "G1_allergy_V1" -> False (legacy)
    """
    return policy_id in ALL_POLICIES


def get_topic_from_policy_id(policy_id: str) -> str:
    """Get the topic (for judgment cache lookup) from a policy ID.

    T* policies: "T1P1" -> "G1_allergy" (uses TOPIC_ID_TO_GT_TOPIC mapping)
    G* policies: "G1_allergy_V1" -> "G1_allergy" (strips version suffix)

    Args:
        policy_id: Policy ID like "T1P1" or "G1_allergy_V1"

    Returns:
        Topic string for cache lookup (e.g., "G1_allergy")
    """
    # T* policy format: T1P1, T2P3, etc.
    if len(policy_id) >= 2 and policy_id[0] == "T" and policy_id[1].isdigit():
        topic_id = get_topic_id_from_policy_id(policy_id)
        return TOPIC_ID_TO_GT_TOPIC.get(topic_id, topic_id)

    # G* policy format: G1_allergy_V1 -> G1_allergy
    # Strip the _V{n} suffix
    parts = policy_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].startswith("V") and parts[1][1:].isdigit():
        return parts[0]

    return policy_id
