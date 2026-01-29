"""Phase 2 debug/output helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from addm.utils.debug_logger import get_debug_logger
from addm.utils.output import output


def log_event(policy_id: str, event_type: str, data: Dict[str, Any], sample_id: Optional[str] = None) -> None:
    logger = get_debug_logger()
    if not logger or not logger.enabled:
        return
    try:
        logger.log_event(sample_id or policy_id, event_type, data)
    except Exception:
        return


def pause(label: str, enabled: bool) -> None:
    output.status(f"ATKD step: {label}")
    if not enabled:
        return
    try:
        import sys

        if not sys.stdin.isatty():
            return
        input(f"[ATKD] {label} - press Enter to continue...")
    except EOFError:
        return


def print_gate_summary(
    primitives: List[Any],
    gate_library: Any,
    max_per: int = 3,
) -> None:
    def _short(text: Any, limit: int = 80) -> str:
        s = str(text)
        if len(s) <= limit:
            return s
        return s[: limit - 3] + "..."

    output.rule()
    output.print("Gate summary (sample)")
    for p in primitives:
        parts = []
        for modality, polarity, label in [
            ("bm25", "pos", "bm25+"),
            ("bm25", "neg", "bm25-"),
            ("emb", "pos", "emb+"),
            ("emb", "neg", "emb-"),
        ]:
            gates = gate_library.by_filter(p.primitive_id, modality, polarity)
            samples = ", ".join(_short(g.query) for g in gates[:max_per])
            parts.append(f"{label}[{len(gates)}]: {samples}")
        output.print(f"- {p.primitive_id}: " + " | ".join(parts))


def print_llm_gate_payload(label: str, data: Dict[str, Any]) -> None:
    output.rule()
    output.print(f"{label}: LLM gate output (full)")
    if not data:
        output.print("(empty)")
        return
    primitives = data.get("primitives")
    if not isinstance(primitives, list):
        output.print(json.dumps(data, indent=2, ensure_ascii=True))
        return

    def _flatten_bm25(items: Any) -> List[str]:
        flat: List[str] = []
        if not isinstance(items, list):
            return flat
        for g in items:
            if isinstance(g, list):
                flat.append(" ".join(str(x) for x in g if x is not None))
            else:
                flat.append(str(g))
        return flat

    lines: List[str] = []
    lines.append("{")
    lines.append('  "primitives": [')
    for idx, p in enumerate(primitives):
        if not isinstance(p, dict):
            continue
        pid = p.get("primitive_id")
        pos_bm25 = _flatten_bm25(p.get("pos_bm25_gates", []))
        neg_bm25 = _flatten_bm25(p.get("neg_bm25_gates", []))
        pos_emb = p.get("pos_emb_gates", [])
        neg_emb = p.get("neg_emb_gates", [])
        lines.append("    {")
        lines.append(f'      "primitive_id": {json.dumps(pid, ensure_ascii=True)},')
        lines.append(f'      "pos_bm25_gates": {json.dumps(pos_bm25, ensure_ascii=True)},')
        lines.append(f'      "neg_bm25_gates": {json.dumps(neg_bm25, ensure_ascii=True)},')
        lines.append(f'      "pos_emb_gates": {json.dumps(pos_emb, ensure_ascii=True)},')
        lines.append(f'      "neg_emb_gates": {json.dumps(neg_emb, ensure_ascii=True)}')
        lines.append("    }" + ("," if idx < len(primitives) - 1 else ""))
    lines.append("  ]")
    lines.append("}")
    output.print("\n".join(lines))


def dump_state(
    policy_id: str,
    cache_dir: Path,
    primitives: List[Any],
    gate_library: Any,
    tag_store: Any,
    score_store: Any,
    calibration: Any,
    label: str,
    *,
    save_arrays: bool = False,
    batch: Optional[List[int]] = None,
    iteration: Optional[int] = None,
    review_pool: Optional[List[Dict[str, Any]]] = None,
    last_batch_debug: Optional[Dict[str, Any]] = None,
) -> None:
    logger = get_debug_logger()
    if not logger or not logger.enabled:
        return

    def _to_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    dump_dir = cache_dir / "debug_state"
    dump_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace(" ", "_")
    name = f"{safe_label}_{stamp}"
    if iteration is not None:
        name = f"{safe_label}_iter{iteration}_{stamp}"
    payload: Dict[str, Any] = {
        "label": label,
        "policy_id": policy_id,
        "timestamp": stamp,
        "primitives": [p.__dict__ for p in primitives],
        "gate_counts": {
            p.primitive_id: {
                f"{modality}_{polarity}": len(
                    gate_library.by_filter(p.primitive_id, modality, polarity)
                )
                for modality in ("bm25", "emb")
                for polarity in ("pos", "neg")
            }
            for p in primitives
        },
        "tag_counts": {
            p.primitive_id: {
                "total": len(tag_store.records_for_primitive(p.primitive_id)),
                "pos": sum(
                    1
                    for r in tag_store.records_for_primitive(p.primitive_id)
                    if r.tags.get(p.primitive_id, 0) > 0
                ),
                "neg": sum(
                    1
                    for r in tag_store.records_for_primitive(p.primitive_id)
                    if r.tags.get(p.primitive_id, 0) == 0
                ),
            }
            for p in primitives
        },
    }
    if batch is not None and review_pool is not None:
        batch_rows = []
        for review_idx in batch:
            review = review_pool[review_idx]
            z_by_primitive = {}
            bin_by_primitive = {}
            if score_store:
                for p in primitives:
                    pid = p.primitive_id
                    z_by_primitive[pid] = float(score_store.z_scores[pid][review_idx])
                    bin_by_primitive[pid] = int(score_store.z_bins[pid][review_idx])
            batch_rows.append(
                {
                    "review_id": review.get("review_id"),
                    "business_id": review.get("business_id"),
                    "z_by_primitive": z_by_primitive,
                    "bin_by_primitive": bin_by_primitive,
                }
            )
        payload["batch"] = batch_rows
    if last_batch_debug:
        payload["voi_summary"] = last_batch_debug.get("summary", {})
        payload["voi_selected"] = last_batch_debug.get("selected", [])

    state_path = dump_dir / f"{name}.json"
    with open(state_path, "w") as f:
        json.dump(payload, f, indent=2)

    if score_store:
        payload["z_stats"] = {
            p.primitive_id: {
                "min": float(np.min(score_store.z_scores[p.primitive_id])),
                "max": float(np.max(score_store.z_scores[p.primitive_id])),
                "mean": float(np.mean(score_store.z_scores[p.primitive_id])),
            }
            for p in primitives
            if p.primitive_id in score_store.z_scores
        }
        payload["z_bin_edges"] = {
            p.primitive_id: score_store.z_bin_edges[p.primitive_id].tolist()
            for p in primitives
            if p.primitive_id in score_store.z_bin_edges
        }
        if save_arrays:
            arrays = {}
            for p in primitives:
                pid = p.primitive_id
                if pid not in score_store.z_scores:
                    continue
                z_path = dump_dir / f"{name}_{pid}_z.npy"
                b_path = dump_dir / f"{name}_{pid}_bins.npy"
                np.save(z_path, score_store.z_scores[pid])
                np.save(b_path, score_store.z_bins[pid])
                arrays[pid] = {
                    "z_scores": str(z_path),
                    "z_bins": str(b_path),
                }
            payload["z_arrays"] = arrays

    if calibration:
        payload["theta_hat"] = {
            p.primitive_id: _to_list(calibration.theta_hat.get(p.primitive_id))
            for p in primitives
        }
        payload["upper_bound"] = {
            p.primitive_id: _to_list(calibration.upper_bound.get(p.primitive_id))
            for p in primitives
        }

    gates_path = None
    if gate_library:
        gates_path = dump_dir / f"{name}_gates.json"
        with open(gates_path, "w") as f:
            json.dump(gate_library.to_dict(), f, indent=2)
        payload["gate_list_path"] = str(gates_path)

    log_event(
        policy_id,
        "phase2_state",
        {
            "label": label,
            "iteration": iteration,
            "state_path": str(state_path),
            "gate_list_path": str(gates_path) if gates_path else None,
        },
    )
