"""AMOS Configuration.

Dataclass for configuring AMOS execution modes and parameters.
"""

from dataclasses import dataclass, field
from typing import Optional


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
        force_regenerate: Force regenerate Formula Seed even if cached.
    """

    # Execution mode
    adaptive: bool = False
    hybrid: bool = False

    # Batch processing (adaptive mode)
    batch_size: int = 8

    # Concurrency
    max_concurrent: int = 32

    # Phase 1
    force_regenerate: bool = False

    # Embedding configuration (hybrid mode)
    embedding_model: str = "text-embedding-3-large"
    embedding_cache_path: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
