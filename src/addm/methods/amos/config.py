"""AMOS Configuration.

Dataclass for configuring AMOS execution modes and parameters.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FilterMode(Enum):
    """Filter mode for Stage 1 quick scan.

    Controls which reviews are selected for initial extraction before
    the thorough sweep of remaining reviews.

    KEYWORD: Filter by keyword matching (fast, may miss semantic matches)
    EMBEDDING: Filter by embedding similarity (better recall, slower)
    HYBRID: Filter by keywords + embedding (best recall for Stage 1)
    """

    KEYWORD = "keyword"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"


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

    Two-stage retrieval:
    - Stage 1 (Quick Scan): Filter reviews using filter_mode, extract, check early exit
    - Stage 2 (Thorough Sweep): Process ALL remaining reviews (always on)

    Attributes:
        filter_mode: How to filter reviews for Stage 1 quick scan.
        max_concurrent: Max concurrent LLM calls for extraction.
        sweep_batch_size: Number of reviews per batch during thorough sweep.
        max_sweep_reviews: Maximum number of reviews to process in thorough sweep.
        sweep_early_exit: If True, stop sweep early when severe evidence found.
        embedding_model: Model for embedding-based filtering.
        embedding_cache_path: Path to cache embeddings.
    """

    # Stage 1: Quick scan filter strategy
    filter_mode: FilterMode = FilterMode.KEYWORD

    # Concurrency
    max_concurrent: int = 256

    # Stage 2: Thorough sweep configuration (always on)
    sweep_batch_size: int = 256  # Legacy - sweep is now fully parallel
    max_sweep_reviews: int = 200  # Process all reviews (K=200 max)
    sweep_early_exit: bool = False  # Disabled - sweep is fully parallel now

    # Phase 1 (Formula Seed generation)
    phase1_approach: Phase1Approach = Phase1Approach.PLAN_AND_ACT
    react_max_iterations: int = 8
    reflexion_max_iterations: int = 2

    # Embedding configuration (for EMBEDDING and HYBRID filter modes)
    embedding_model: str = "text-embedding-3-large"
    embedding_cache_path: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if self.sweep_batch_size < 1:
            raise ValueError("sweep_batch_size must be >= 1")
        if self.max_sweep_reviews < 1:
            raise ValueError("max_sweep_reviews must be >= 1")
        # Convert string to enum if needed
        if isinstance(self.filter_mode, str):
            self.filter_mode = FilterMode(self.filter_mode)
