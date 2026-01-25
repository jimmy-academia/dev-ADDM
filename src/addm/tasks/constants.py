"""Shared constants for tasks module."""

from typing import List, Optional

# All 18 topics (6 groups × 3 topics)
ALL_TOPICS = [
    # G1: Customer Safety
    "G1_allergy",
    "G1_dietary",
    "G1_hygiene",
    # G2: Customer Experience
    "G2_romance",
    "G2_business",
    "G2_group",
    # G3: Customer Value
    "G3_price_worth",
    "G3_hidden_costs",
    "G3_time_value",
    # G4: Owner Operations
    "G4_server",
    "G4_kitchen",
    "G4_environment",
    # G5: Owner Performance
    "G5_capacity",
    "G5_execution",
    "G5_consistency",
    # G6: Owner Strategy
    "G6_uniqueness",
    "G6_comparison",
    "G6_loyalty",
]

# Group to topics mapping
GROUP_TOPICS = {
    "G1": ["G1_allergy", "G1_dietary", "G1_hygiene"],
    "G2": ["G2_romance", "G2_business", "G2_group"],
    "G3": ["G3_price_worth", "G3_hidden_costs", "G3_time_value"],
    "G4": ["G4_server", "G4_kitchen", "G4_environment"],
    "G5": ["G5_capacity", "G5_execution", "G5_consistency"],
    "G6": ["G6_uniqueness", "G6_comparison", "G6_loyalty"],
}

# All 72 policies: 18 topics × 4 variants (V1-V4)
ALL_POLICIES = [f"{topic}_{v}" for topic in ALL_TOPICS for v in ["V1", "V2", "V3", "V4"]]

# K values for ground truth generation
K_VALUES = [25, 50, 100, 200]


def expand_policies(
    policy: Optional[str] = None,
    topic: Optional[str] = None,
    group: Optional[str] = None,
) -> List[str]:
    """Expand policy specifier to list of policy IDs.

    Args:
        policy: Comma-separated policy IDs (e.g., "G1_allergy_V3,G1_allergy_V4")
        topic: Topic ID (e.g., "G1_allergy") - expands to V1-V4
        group: Group ID (e.g., "G1") - expands to all topics × V1-V4

    Returns:
        List of policy IDs. Returns ALL_POLICIES if no argument specified.

    Raises:
        ValueError: If topic or group is invalid.
    """
    if policy:
        return [p.strip() for p in policy.split(",") if p.strip()]
    if topic:
        if topic not in ALL_TOPICS:
            raise ValueError(f"Unknown topic: {topic}. Valid: {ALL_TOPICS}")
        return [f"{topic}_V{v}" for v in range(1, 5)]
    if group:
        group_upper = group.upper()
        if group_upper not in GROUP_TOPICS:
            raise ValueError(f"Unknown group: {group}. Valid: G1-G6")
        return [f"{t}_V{v}" for t in GROUP_TOPICS[group_upper] for v in range(1, 5)]
    return ALL_POLICIES.copy()


def expand_topics(
    topic: Optional[str] = None,
    group: Optional[str] = None,
) -> List[str]:
    """Expand topic specifier to list of topic IDs.

    Args:
        topic: Single topic ID (e.g., "G1_allergy")
        group: Group ID (e.g., "G1") - expands to all 3 topics in group

    Returns:
        List of topic IDs. Returns ALL_TOPICS if no argument specified.

    Raises:
        ValueError: If topic or group is invalid.
    """
    if topic:
        if topic not in ALL_TOPICS:
            raise ValueError(f"Unknown topic: {topic}. Valid: {ALL_TOPICS}")
        return [topic]
    if group:
        group_upper = group.upper()
        if group_upper not in GROUP_TOPICS:
            raise ValueError(f"Unknown group: {group}. Valid: G1-G6")
        return GROUP_TOPICS[group_upper].copy()
    return ALL_TOPICS.copy()


def get_topic_from_policy_id(policy_id: str) -> str:
    """Extract topic from policy ID.

    "G1_allergy_V2" -> "G1_allergy"
    "G3_price_worth_V2" -> "G3_price_worth"
    """
    parts = policy_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid policy_id: {policy_id}")
    # Variant is always last part matching V{digit}
    if parts[-1].startswith("V") and parts[-1][1:].isdigit():
        return "_".join(parts[:-1])
    return policy_id  # No variant suffix found


def get_group_from_policy_id(policy_id: str) -> str:
    """Extract group from policy ID.

    "G1_allergy_V2" -> "G1"
    "G3_price_worth_V2" -> "G3"
    """
    return policy_id.split("_")[0]
