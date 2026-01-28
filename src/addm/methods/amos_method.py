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
        verdict_spec = agenda_spec.get("verdict_rules", {})
        labels = verdict_spec.get("labels", [])
        rules = verdict_spec.get("rules", [])
        clauses = sum(
            len(r.get("clauses", [])) for r in rules if not r.get("default")
        )

        output.success(f"Phase 1 complete: {run_id}")
        summary_parts = [
            f"Terms: {len(terms)}",
            f"Labels: {len(labels)}",
            f"Rules: {len(rules)}",
            f"Clauses: {clauses}",
        ]
        summary = "  " + " | ".join(summary_parts)
        if agenda_spec_path:
            summary += f" | Saved to: {agenda_spec_path}"
        output.print(summary)
        if agenda_spec_path:
            pass

        return {
            "phase": "1",
            "policy_id": run_id,
            "output_dir": str(output_dir),
            "agenda_spec": agenda_spec,
            "agenda_spec_summary": {
                "terms": len(terms),
                "labels": len(labels),
                "rules": len(rules),
                "clauses": clauses,
            },
        }

    raise ValueError("AMOS Phase 2 is not wired yet. Use --phase 1.")
