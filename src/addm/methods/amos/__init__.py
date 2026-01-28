"""AMOS method (Phase 1 compile + Phase 2 ATKD)."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from addm.data.types import Sample
from addm.llm import LLMService
from addm.methods.base import Method
from addm.utils.output import output

from .phase1 import generate_verdict_and_terms


class AMOSMethod(Method):
    """AMOS method (Phase 1 compile + Phase 2 ATKD).

    Phase 1: Extract verdict rules + term definitions from agenda.
    Phase 2: ATKD (Active Test-time Knowledge Discovery).
    """

    name = "amos"

    ProgressCallback = Callable[[int, str, float, str], None]

    def __init__(
        self,
        policy_id: str,
        max_concurrent: int = 256,
        batch_size: int = 10,
        system_prompt: Optional[str] = None,
        progress_callback: Optional["AMOSMethod.ProgressCallback"] = None,
    ) -> None:
        self.policy_id = policy_id
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.system_prompt = system_prompt
        self.progress_callback = progress_callback

        self._agenda_spec: Optional[Dict[str, Any]] = None
        self._phase1_usage: Dict[str, Any] = {}

    async def generate_phase1(
        self,
        agenda: str,
        llm: LLMService,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Run Phase 1 and cache the result."""
        spec, usage = await generate_verdict_and_terms(
            agenda=agenda,
            policy_id=self.policy_id,
            llm=llm,
            progress_callback=self.progress_callback,
            max_retries=max_retries,
        )
        self._agenda_spec = spec
        self._phase1_usage = usage
        return spec

    def get_phase1_usage(self) -> Dict[str, Any]:
        """Get Phase 1 usage metrics."""
        return self._phase1_usage

    def save_agenda_spec_to_run_dir(self, run_dir: Path) -> Optional[Path]:
        """Save agenda spec (Phase 1 output) to run directory."""
        if self._agenda_spec is None:
            return None
        output_path = run_dir / "agenda_spec.json"
        with open(output_path, "w") as f:
            json.dump(self._agenda_spec, f, indent=2)
        return output_path

    async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
        """AMOS Phase 2 is orchestrated at the policy level."""
        raise NotImplementedError("AMOS Phase 2 is orchestrated via run_amos_policy.")


__all__ = [
    "AMOSMethod",
]
