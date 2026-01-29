"""AMOS runner (policy-level orchestration)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from addm.data.types import Sample
from addm.llm import LLMService
from addm.utils.output import output

from addm.methods.amos import AMOSMethod
from addm.methods.amos.phase2_atkd import ATKDEngine, ATKDConfig


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
    agenda_spec_path: Optional[str],
    atkd_config: Optional[ATKDConfig],
    rng_seed: Optional[int],
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

    # Phase 2 may use pre-generated agenda spec
    agenda_spec = None

    if phase == "2" and agenda_spec_path:
        agenda_spec = _load_agenda_spec(Path(agenda_spec_path), run_id)

    # Phase 1: Extract agenda spec
    if phase in ("1", "1,2", None):
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

    # Phase 2: ATKD engine
    if phase in ("2", "1,2", None):
        if agenda_spec is None:
            raise ValueError("Agenda spec not found for Phase 2.")
        config = atkd_config or ATKDConfig()
        cache_dir = output_dir / "atkd_cache"
        engine = ATKDEngine(
            policy_id=run_id,
            agenda_spec=agenda_spec,
            agenda_text=agenda,
            restaurants=restaurants,
            llm=llm,
            config=config,
            cache_dir=cache_dir,
            rng_seed=rng_seed,
        )
        return await engine.run()

    raise ValueError("AMOS Phase 2 not executed. Check --phase.")


def _load_agenda_spec(path: Path, policy_id: str) -> Optional[Dict[str, Any]]:
    if path.is_dir():
        candidate = path / f"{policy_id}.json"
        if candidate.exists():
            return json.loads(candidate.read_text())
        candidate = path / f"{policy_id}_agenda_spec.json"
        if candidate.exists():
            return json.loads(candidate.read_text())
        # Fallback: first json file in directory
        for file in path.glob("*.json"):
            return json.loads(file.read_text())
        return None
    if path.exists():
        return json.loads(path.read_text())
    return None
