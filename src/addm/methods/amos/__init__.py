"""AMOS - Agenda-Driven Mining with Observable Steps.

Two-phase method for evaluating restaurants based on LLM-generated specifications:
- Phase 1: LLM "compiles" agenda into Formula Seed
- Phase 2: Two-stage retrieval (quick scan + thorough sweep)

Stage 1 (Quick Scan): Filter using filter_mode, extract, check early exit
Stage 2 (Thorough Sweep): Process ALL remaining reviews (always on)

Filter modes for Stage 1:
- KEYWORD: Filter by keyword matching (fast, may miss semantic matches)
- EMBEDDING: Filter by embedding similarity (better recall, slower)
- HYBRID: Filter by keywords + embedding (best recall)
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from addm.data.types import Sample
from addm.llm import LLMService
from addm.methods.base import Method
from addm.methods.amos.config import AMOSConfig, FilterMode, Phase1Approach
from addm.methods.amos.phase1 import generate_formula_seed, generate_formula_seed_with_config
from addm.methods.amos.phase2 import FormulaSeedInterpreter


class AMOSMethod(Method):
    """AMOS - Agenda-Driven Mining with Observable Steps.

    Orchestrates Phase 1 (Formula Seed generation) and Phase 2 (two-stage retrieval).

    Two-stage retrieval (always on):
    - Stage 1: Quick scan using filter_mode (keyword/embedding/hybrid)
    - Stage 2: Thorough sweep of ALL remaining reviews
    """

    name = "amos"

    def __init__(
        self,
        policy_id: str = "G1_allergy_V2",
        max_concurrent: int = 32,
        config: Optional[AMOSConfig] = None,
        filter_mode: Union[str, FilterMode] = FilterMode.KEYWORD,
        system_prompt: Optional[str] = None,
    ):
        """Initialize AMOS method.

        Args:
            policy_id: Policy identifier for this run
            max_concurrent: Max concurrent LLM calls for extraction
            config: Full AMOS configuration (overrides filter_mode param)
            filter_mode: Stage 1 filter strategy: 'keyword', 'embedding', or 'hybrid'
            system_prompt: System prompt for output format (stored for reference)
        """
        self.policy_id = policy_id
        self.max_concurrent = max_concurrent
        self.system_prompt = system_prompt

        # Build config from parameters or use provided config
        if config:
            self.config = config
        else:
            # Convert string to enum if needed
            if isinstance(filter_mode, str):
                filter_mode = FilterMode(filter_mode)
            self.config = AMOSConfig(
                filter_mode=filter_mode,
                max_concurrent=max_concurrent,
            )

        # Formula Seed (generated once per run, reused for all samples)
        self._seed: Optional[Dict[str, Any]] = None
        self._phase1_usage: Dict[str, Any] = {}

    def save_formula_seed_to_run_dir(self, run_dir: Path) -> None:
        """Save a copy of the Formula Seed to the run directory.

        Creates formula_seed.json in the run directory for artifact tracking.

        Args:
            run_dir: Run directory (e.g., results/dev/20260120_G1_allergy_V2/)
        """
        if self._seed is None:
            return

        output_path = run_dir / "formula_seed.json"
        with open(output_path, "w") as f:
            json.dump(self._seed, f, indent=2)

    def get_formula_seed(self) -> Optional[Dict[str, Any]]:
        """Get the current Formula Seed (if loaded)."""
        return self._seed

    def set_formula_seed(self, seed: Dict[str, Any]) -> None:
        """Set Formula Seed directly (for Phase 2-only runs).

        Args:
            seed: Pre-generated Formula Seed
        """
        self._seed = seed
        self._phase1_usage = {}  # No Phase 1 usage when seed is injected

    async def _get_formula_seed(self, agenda: str, llm: LLMService) -> Dict[str, Any]:
        """Get or generate Formula Seed.

        Seeds are generated fresh each run and reused for all samples within the run.

        Args:
            agenda: Task agenda/query prompt
            llm: LLM service for Phase 1 generation

        Returns:
            Formula Seed specification
        """
        if self._seed is not None:
            return self._seed

        seed, usage = await generate_formula_seed(
            agenda=agenda,
            policy_id=self.policy_id,
            llm=llm,
        )

        self._seed = seed
        self._phase1_usage = usage

        return seed

    async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
        """Run AMOS evaluation on a sample.

        Args:
            sample: Input sample with query and context
            llm: LLM service for API calls

        Returns:
            Dict with sample_id, output (in standard format), verdict, and usage metrics
        """
        # Get or generate Formula Seed (Phase 1)
        seed = await self._get_formula_seed(sample.query, llm)

        # Parse restaurant data
        try:
            restaurant = json.loads(sample.context or "{}")
        except json.JSONDecodeError:
            restaurant = {}

        reviews = restaurant.get("reviews", [])
        business = restaurant.get("business", {})

        # Create interpreter with config and execute (Phase 2)
        interpreter = FormulaSeedInterpreter(
            seed=seed,
            llm=llm,
            max_concurrent=self.max_concurrent,
            config=self.config,
        )

        # Execute two-stage retrieval
        result = await interpreter.execute(
            reviews=reviews,
            business=business,
            query=sample.query,
            sample_id=sample.sample_id,
        )

        # Get standard output format (matching output_schema.txt)
        standard_output = interpreter.get_standard_output()

        # Extract verdict and risk score from computed output
        verdict = result.get("VERDICT")
        risk_score = result.get("FINAL_RISK_SCORE") or result.get("RISK_SCORE") or result.get("SCORE")

        # Get usage metrics from interpreter
        phase2_usage = interpreter.get_usage_metrics()

        # Combine Phase 1 + Phase 2 usage
        total_prompt_tokens = (
            self._phase1_usage.get("prompt_tokens", 0)
            + phase2_usage.get("prompt_tokens", 0)
        )
        total_completion_tokens = (
            self._phase1_usage.get("completion_tokens", 0)
            + phase2_usage.get("completion_tokens", 0)
        )
        total_cost = (
            self._phase1_usage.get("cost_usd", 0.0)
            + phase2_usage.get("cost_usd", 0.0)
        )
        total_latency = (
            self._phase1_usage.get("latency_ms", 0.0)
            + phase2_usage.get("latency_ms", 0.0)
        )
        total_llm_calls = (
            (1 if self._phase1_usage else 0)
            + phase2_usage.get("llm_calls", 0)
        )

        # Get strategy metrics
        strategy_metrics = phase2_usage.get("strategy_metrics", {})

        return {
            "sample_id": sample.sample_id,
            # Output in standard format (JSON string for compatibility)
            "output": json.dumps(standard_output),
            # Also include parsed standard output for direct access
            "parsed": standard_output,
            "verdict": verdict,
            "risk_score": risk_score,
            # Usage metrics
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "cost_usd": total_cost,
            "latency_ms": total_latency,
            "llm_calls": total_llm_calls,
            # AMOS-specific metrics
            "filter_stats": result.get("_filter_stats", {}),
            "extractions_count": len(result.get("_extractions", [])),
            "phase1_cached": not bool(self._phase1_usage),
            # Strategy metrics
            "filter_mode": strategy_metrics.get("filter_mode", "keyword"),
            "early_exit": strategy_metrics.get("early_exit", False),
            # Embedding metrics (for embedding/hybrid filter modes)
            "embedding_tokens": phase2_usage.get("embedding_tokens", 0),
            "embedding_cost_usd": phase2_usage.get("embedding_cost_usd", 0.0),
        }


__all__ = [
    "AMOSMethod",
    "AMOSConfig",
    "FilterMode",
    "Phase1Approach",
    "generate_formula_seed",
    "generate_formula_seed_with_config",
    "FormulaSeedInterpreter",
]
