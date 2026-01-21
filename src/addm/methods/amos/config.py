"""AMOS Configuration.

Dataclass for configuring AMOS execution modes and parameters.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Phase1Approach(Enum):
    """Approach for Phase 1 Formula Seed generation.

    PLAN_AND_ACT: Fixed 3-step pipeline (OBSERVE → PLAN → ACT)
                  Same cost as current approach, no self-correction.

    REACT: Iterative loop with actions (Thought → Action → Observation)*
           Self-correcting, handles complex agendas. 5-10 LLM calls.

    REFLEXION: Initial generation + quality analysis + revision
               Highest quality, most expensive (7-15 LLM calls).
    """

    PLAN_AND_ACT = "plan_and_act"
    REACT = "react"
    REFLEXION = "reflexion"


@dataclass
class AMOSConfig:
    """Configuration for AMOS method execution.

    Attributes:
        adaptive: If True, use batch processing with early stopping.
                  If False (default), process all reviews in parallel.
        hybrid: If True, enable hybrid embedding retrieval when keyword
                filtering yields few results.
        batch_size: Number of reviews per batch in adaptive mode.
        max_concurrent: Max concurrent LLM calls for extraction.
    """

    # Execution mode
    adaptive: bool = False
    hybrid: bool = False

    # Batch processing (adaptive mode)
    batch_size: int = 8

    # Concurrency
    max_concurrent: int = 32

    # Phase 1
    phase1_approach: Phase1Approach = Phase1Approach.PLAN_AND_ACT
    react_max_iterations: int = 8
    reflexion_max_iterations: int = 2

    # Embedding configuration (hybrid mode)
    embedding_model: str = "text-embedding-3-large"
    embedding_cache_path: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
