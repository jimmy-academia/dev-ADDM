"""AMOS Method - Agenda-Driven Mining with Observable Steps.

Two-phase method where:
- Phase 1: LLM "compiles" agenda into Formula Seed (cached per policy)
- Phase 2: Interpreter executes Formula Seed against restaurant data
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from addm.data.types import Sample
from addm.llm import LLMService
from addm.methods.base import Method
from addm.methods.amos.phase1 import generate_formula_seed
from addm.methods.amos.phase2 import FormulaSeedInterpreter


class AMOSMethod(Method):
    """AMOS - Agenda-Driven Mining with Observable Steps."""

    name = "amos"

    def __init__(
        self,
        policy_id: str = "G1_allergy_V2",
        max_concurrent: int = 32,
        force_regenerate: bool = False,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize AMOS method.

        Args:
            policy_id: Policy identifier for Formula Seed caching
            max_concurrent: Max concurrent LLM calls for extraction
            force_regenerate: Force regenerate Formula Seed even if cached
            cache_dir: Override cache directory (default: data/formula_seeds/)
        """
        self.policy_id = policy_id
        self.max_concurrent = max_concurrent
        self.force_regenerate = force_regenerate
        self.cache_dir = cache_dir or Path("results/cache/formula_seeds")

        # Cached Formula Seed (loaded on first sample)
        self._seed: Optional[Dict[str, Any]] = None
        self._phase1_usage: Dict[str, Any] = {}

    async def _get_formula_seed(self, agenda: str, llm: LLMService) -> Dict[str, Any]:
        """Get or generate Formula Seed.

        Args:
            agenda: Task agenda/query prompt
            llm: LLM service for Phase 1 generation

        Returns:
            Formula Seed specification
        """
        if self._seed is not None and not self.force_regenerate:
            return self._seed

        seed, usage = await generate_formula_seed(
            agenda=agenda,
            policy_id=self.policy_id,
            llm=llm,
            cache_dir=self.cache_dir,
            force_regenerate=self.force_regenerate,
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
            Dict with sample_id, output, verdict, and usage metrics
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

        # Create interpreter and execute (Phase 2)
        interpreter = FormulaSeedInterpreter(seed, llm, self.max_concurrent)
        result = await interpreter.execute(reviews, business)

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
        # Phase 1 is 1 call (if not cached), Phase 2 is extraction calls
        total_llm_calls = (
            (1 if self._phase1_usage else 0)
            + phase2_usage.get("llm_calls", 0)
        )

        return {
            "sample_id": sample.sample_id,
            "output": json.dumps(result),
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
        }
