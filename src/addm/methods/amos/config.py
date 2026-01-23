"""AMOS Configuration.

Dataclass for configuring AMOS execution modes and parameters.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Phase1Approach(Enum):
    """Approach for Phase 1 Formula Seed generation.

    PLAN_AND_ACT: Fixed 3-step pipeline (OBSERVE → PLAN → ACT)
                  Same cost as current approach, no self-correction.
    """

    PLAN_AND_ACT = "plan_and_act"


@dataclass
class AMOSConfig:
    """Configuration for AMOS method execution.

    Simplified single-pass extraction:
    - Process ALL reviews in parallel batches (no filtering/early exit)

    Attributes:
        max_concurrent: Max concurrent LLM calls for extraction.
        batch_size: Number of reviews per batch during extraction.
        max_reviews: Maximum number of reviews to process.
    """

    # Concurrency
    max_concurrent: int = 256

    # Extraction configuration
    batch_size: int = 10  # Reviews per LLM call
    max_reviews: int = 200  # Process all reviews (K=200 max)

    # Phase 1 (Formula Seed generation)
    phase1_approach: Phase1Approach = Phase1Approach.PLAN_AND_ACT

    def __post_init__(self):
        """Validate configuration."""
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_reviews < 1:
            raise ValueError("max_reviews must be >= 1")
