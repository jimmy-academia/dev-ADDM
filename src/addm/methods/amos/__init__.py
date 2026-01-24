"""AMOS - Agenda-Driven Mining with Observable Steps.

Two-phase method for evaluating restaurants based on LLM-generated specifications:
- Phase 1: LLM "compiles" agenda into Formula Seed
- Phase 2: Process ALL reviews in parallel batches
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from addm.data.types import Sample
from addm.llm import LLMService
from addm.methods.base import Method
from addm.methods.amos.config import AMOSConfig
from addm.methods.amos.phase1 import generate_formula_seed_with_config
from addm.methods.amos.phase2 import FormulaSeedInterpreter
from addm.utils.output import output


class AMOSMethod(Method):
    """AMOS - Agenda-Driven Mining with Observable Steps.

    Orchestrates Phase 1 (Formula Seed generation) and Phase 2 (extraction).
    """

    name = "amos"

    # Type alias for progress callback
    ProgressCallback = Callable[[int, str, float, str], None]

    def __init__(
        self,
        policy_id: str = "G1_allergy_V2",
        max_concurrent: int = 256,
        config: Optional[AMOSConfig] = None,
        batch_size: int = 10,
        system_prompt: Optional[str] = None,
        progress_callback: Optional["AMOSMethod.ProgressCallback"] = None,
    ):
        """Initialize AMOS method.

        Args:
            policy_id: Policy identifier for this run
            max_concurrent: Max concurrent LLM calls for extraction
            config: Full AMOS configuration (overrides batch_size param)
            batch_size: Reviews per LLM call (default: 10)
            system_prompt: System prompt for output format (stored for reference)
            progress_callback: Optional callback for progress updates.
                Signature: callback(phase: int, step: str, progress: float, detail: str)
                - phase: 1 or 2
                - step: Current step name (e.g., "OBSERVE", "PLAN", "ACT")
                - progress: Percentage complete (0-100)
                - detail: Additional info (e.g., "25/50 reviews")
        """
        self.policy_id = policy_id
        self.max_concurrent = max_concurrent
        self.system_prompt = system_prompt
        self.progress_callback = progress_callback

        # Build config from parameters or use provided config
        if config:
            self.config = config
        else:
            self.config = AMOSConfig(
                max_concurrent=max_concurrent,
                batch_size=batch_size,
            )

        # Formula Seed (generated once per run, reused for all samples)
        self._seed: Optional[Dict[str, Any]] = None
        self._phase1_usage: Dict[str, Any] = {}

    def _report_progress(
        self, phase: int, step: str, progress: float, detail: str = ""
    ) -> None:
        """Report progress via callback if configured.

        Args:
            phase: 1 or 2
            step: Current step name (e.g., "OBSERVE", "PLAN", "ACT")
            progress: Percentage complete (0-100)
            detail: Additional info (e.g., "25/50 reviews")
        """
        if self.progress_callback:
            self.progress_callback(phase, step, progress, detail)

    def save_formula_seed_to_run_dir(self, run_dir: Path) -> None:
        """Save a copy of the Formula Seed to the run directory.

        Creates formula_seed.json in the run directory for artifact tracking.
        If hybrid approach was used, also saves policy_yaml.yaml.

        Args:
            run_dir: Run directory (e.g., results/dev/20260120_G1_allergy_V2/)
        """
        import yaml as yaml_module

        if self._seed is None:
            return

        # Save PolicyYAML if present (from hybrid approach)
        policy_yaml = self._seed.pop("_policy_yaml", None)
        if policy_yaml:
            yaml_path = run_dir / "policy_yaml.yaml"
            with open(yaml_path, "w") as f:
                yaml_module.dump(policy_yaml, f, default_flow_style=False, sort_keys=False)

        # Save Formula Seed (without _policy_yaml)
        output_path = run_dir / "formula_seed.json"
        with open(output_path, "w") as f:
            json.dump(self._seed, f, indent=2)

    def get_formula_seed(self) -> Optional[Dict[str, Any]]:
        """Get the current Formula Seed (if loaded)."""
        return self._seed

    def get_phase1_usage(self) -> Dict[str, Any]:
        """Get Phase 1 usage metrics (for aggregate tracking).

        Phase 1 runs once per policy, shared across all samples.
        This should be added ONCE to the aggregate, not per sample.
        """
        return self._phase1_usage

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

        seed, usage = await generate_formula_seed_with_config(
            agenda=agenda,
            policy_id=self.policy_id,
            llm=llm,
            config=self.config,
            progress_callback=self.progress_callback,
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
        sample_short = sample.sample_id[:12]

        # Get or generate Formula Seed (Phase 1)
        if self._seed is None:
            output.status(f"[AMOS] {sample_short}... | Phase 1: Generating Formula Seed")
            self._report_progress(1, "starting", 0, "generating seed")
        else:
            output.status(f"[AMOS] {sample_short}... | Phase 1: Using cached seed")
            self._report_progress(1, "cached", 50, "using cached seed")

        seed = await self._get_formula_seed(sample.query, llm)

        # Parse restaurant data
        try:
            restaurant = json.loads(sample.context or "{}")
        except json.JSONDecodeError:
            restaurant = {}

        reviews = restaurant.get("reviews", [])
        business = restaurant.get("business", {})
        biz_name = business.get("name", sample_short)
        output.status(f"[AMOS] {biz_name} | Phase 2: {len(reviews)} reviews")

        # Report Phase 2 start
        self._report_progress(2, "starting", 50, f"0/{len(reviews)} reviews")

        # Create interpreter with config and execute (Phase 2)
        interpreter = FormulaSeedInterpreter(
            seed=seed,
            llm=llm,
            max_concurrent=self.max_concurrent,
            config=self.config,
            progress_callback=self.progress_callback,
        )

        # Execute extraction on all reviews
        result = await interpreter.execute(
            reviews=reviews,
            business=business,
            query=sample.query,
            sample_id=sample.sample_id,
        )

        # Get standard output format (matching output_schema.txt)
        standard_output = interpreter.get_standard_output()

        # Get debug metadata separately (not included in serialized output)
        debug_metadata = interpreter.get_debug_metadata()

        # Extract verdict from standard_output (normalized to GT format like "High Risk")
        # NOT from result["VERDICT"] which has raw format like "HIGH_RISK"
        verdict = standard_output.get("verdict")
        risk_score = result.get("FINAL_RISK_SCORE") or result.get("RISK_SCORE") or result.get("SCORE")

        # Get usage metrics from interpreter
        phase2_usage = interpreter.get_usage_metrics()

        # NOTE: Phase 1 usage is tracked separately and added ONCE at aggregate level
        # Each sample only reports its own Phase 2 usage to avoid triple-counting
        total_prompt_tokens = phase2_usage.get("prompt_tokens", 0)
        total_completion_tokens = phase2_usage.get("completion_tokens", 0)
        total_cost = phase2_usage.get("cost_usd", 0.0)
        wall_clock_ms = phase2_usage.get("wall_clock_ms", 0.0)
        total_llm_calls = phase2_usage.get("llm_calls", 0)

        # Verbose output: Done
        output.status(f"[AMOS] {biz_name} | Done: {verdict} (${total_cost:.4f}, {wall_clock_ms:.0f}ms)")

        # Build aggregated usage dict
        usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "cost_usd": total_cost,
            "latency_ms": wall_clock_ms,  # Wall-clock time, not summed LLM latencies
        }

        return self._make_result(
            sample.sample_id,
            json.dumps(standard_output),
            usage,
            llm_calls=total_llm_calls,
            # Parsed standard output for direct access
            parsed=standard_output,
            verdict=verdict,
            risk_score=risk_score,
            # AMOS-specific metrics
            stats=result.get("_stats", {}),
            extractions_count=len(result.get("_extractions", [])),
            phase1_cached=not bool(self._phase1_usage),
            # Debug metadata (separate from output, for debugging only)
            _debug_metadata=debug_metadata,
        )


__all__ = [
    # Main classes
    "AMOSMethod",
    "AMOSConfig",
    # Phase 1 (Formula Seed generation)
    "generate_formula_seed_with_config",
    # Phase 2 (Formula Seed interpreter)
    "FormulaSeedInterpreter",
]
