"""AMOS runner (policy-level orchestration)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from addm.data.types import Sample
from addm.llm import LLMService
from addm.utils.output import output

from addm.methods.amos import AMOSMethod


async def run_amos_policy(
    *,
    run_id: str,
    agenda: str,
    restaurants: List[Dict[str, Any]],
    output_dir: Path,
    llm: LLMService,
    system_prompt: Optional[str],
    batch_size: int,
    phase: Optional[str],
    progress_callback: Optional[Callable],
    phase1_retries: int,
) -> Dict[str, Any]:
    """Run AMOS for a single policy.

    Returns:
        Dict result compatible with run_experiment outputs.
    """
    amos_method = AMOSMethod(
        policy_id=run_id,
        max_concurrent=256,
        batch_size=batch_size,
        system_prompt=system_prompt,
        progress_callback=progress_callback,
    )

    # Prepare samples for future Phase 2 use
    _ = [
        Sample(
            sample_id=r["business"]["business_id"],
            query=agenda,
            context=json.dumps(r),
            metadata={"restaurant_name": r["business"]["name"]},
        )
        for r in restaurants
    ]

    if phase == "2":
        raise ValueError("AMOS Phase 2 is not wired yet. Use --phase 1.")

    # Phase 1: Extract agenda spec
    agenda_spec = await amos_method.generate_phase1(
        agenda,
        llm,
        max_retries=phase1_retries,
    )
    agenda_spec_path = amos_method.save_agenda_spec_to_run_dir(output_dir)

    if phase == "1":
        terms = agenda_spec.get("terms", [])
        verdict_spec = agenda_spec.get("verdict", {})
        verdicts = verdict_spec.get("verdicts", [])
        groups = verdict_spec.get("groups", [])
        clauses = sum(
            len(g.get("clauses", [])) for g in groups if not g.get("default")
        )

        output.success(f"Phase 1 complete: {run_id}")
        output.print(f"  Terms: {len(terms)}")
        output.print(f"  Verdicts: {len(verdicts)}")
        output.print(f"  Groups: {len(groups)}")
        output.print(f"  Clauses: {clauses}")
        if agenda_spec_path:
            output.print(f"  Saved to: {agenda_spec_path}")

        return {
            "phase": "1",
            "policy_id": run_id,
            "output_dir": str(output_dir),
            "agenda_spec": agenda_spec,
            "agenda_spec_summary": {
                "terms": len(terms),
                "verdicts": len(verdicts),
                "groups": len(groups),
                "clauses": clauses,
            },
        }

    raise ValueError("AMOS Phase 2 is not wired yet. Use --phase 1.")
