"""Shared constants for tasks module."""

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

# All 72 policies: 18 topics × 4 variants (V0-V3)
ALL_POLICIES = [f"{topic}_{v}" for topic in ALL_TOPICS for v in ["V0", "V1", "V2", "V3"]]

# K values for ground truth generation
K_VALUES = [25, 50, 100, 200]
