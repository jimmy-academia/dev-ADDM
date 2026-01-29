"""Run GateInit + GateScan only and print Z ranking diagnostics.

This script is intended for iterating on cheap-gate quality without paying for verifier calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from addm.llm import LLMService
from addm.methods.amos.phase2_atkd import ATKDEngine, ATKDConfig


def _load_dataset(domain: str, k: int) -> List[Dict[str, Any]]:
    path = Path(f"data/context/{domain}/dataset_K{k}.jsonl")
    restaurants = []
    with path.open() as f:
        for line in f:
            restaurants.append(json.loads(line))
    return restaurants


def _load_agenda(policy_id: str, domain: str, k: int) -> str:
    path = Path(f"data/query/{domain}/{policy_id}_K{k}_prompt.txt")
    return path.read_text()


def _load_agenda_spec(agenda_spec_dir: Path, policy_id: str) -> Dict[str, Any]:
    cand = agenda_spec_dir / f"{policy_id}_agenda_spec.json"
    if cand.exists():
        return json.loads(cand.read_text())
    cand = agenda_spec_dir / f"{policy_id}.json"
    if cand.exists():
        return json.loads(cand.read_text())
    raise FileNotFoundError(f"Agenda spec not found for {policy_id} in {agenda_spec_dir}")


def _load_gt(domain: str, policy_id: str, k: int) -> Dict[str, str]:
    gt_path = Path(f"data/answers/{domain}/{policy_id}_K{k}_groundtruth.json")
    data = json.loads(gt_path.read_text())
    verdicts = {}
    for biz_id, blob in data.get("restaurants", {}).items():
        verdicts[biz_id] = (blob.get("ground_truth") or {}).get("verdict", "Unknown")
    return verdicts


def _biz_to_restaurant_index(restaurants: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {}
    for i, r in enumerate(restaurants):
        out[r.get("business", {}).get("business_id", "")] = i
    return out


def _review_index_to_restaurant_index(restaurants: List[Dict[str, Any]]) -> np.ndarray:
    ridx = []
    for i, r in enumerate(restaurants):
        ridx.extend([i] * len(r.get("reviews", [])))
    return np.array(ridx, dtype=np.int32)


async def main_async(args: argparse.Namespace) -> int:
    restaurants = _load_dataset(args.domain, args.k)
    agenda = _load_agenda(args.policy, args.domain, args.k)
    agenda_spec = _load_agenda_spec(Path(args.agenda_spec_path), args.policy)

    llm = LLMService()
    llm.configure(model=args.model, max_concurrent=args.max_concurrent)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("results/dev") / f"{timestamp}_gate_scan_{args.domain}_{args.policy}_K{args.k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ATKDConfig(
        gate_init=True,
        gate_discover_every=0,
        batch_size=len(restaurants),
        verifier_batch_size=0,
        num_bins=args.num_bins,
        num_gates_suggest=args.num_gates_suggest,
        dataset_tag=args.domain,
        pause=False,
    )

    engine = ATKDEngine(
        policy_id=args.policy,
        agenda_spec=agenda_spec,
        agenda_text=agenda,
        restaurants=restaurants,
        llm=llm,
        config=cfg,
        cache_dir=out_dir / "atkd_cache",
        rng_seed=args.seed,
    )

    await engine.initialize_gates()
    # Persist the expanded GateLibrary for inspection (grouped by primitive/kind).
    gates_out: Dict[str, Any] = {
        "policy_id": args.policy,
        "domain": args.domain,
        "k": args.k,
        "topic_anchors": getattr(engine, "topic_anchors", []),
        "gates": {},
    }
    for p in engine.primitives:
        pid = p.primitive_id
        gates_out["gates"][pid] = {
            "bm25_pos": [g.query for g in engine.gate_library.by_filter(pid, "bm25", "pos")],
            "bm25_neg": [g.query for g in engine.gate_library.by_filter(pid, "bm25", "neg")],
            "emb_pos": [g.query for g in engine.gate_library.by_filter(pid, "emb", "pos")],
            "emb_neg": [g.query for g in engine.gate_library.by_filter(pid, "emb", "neg")],
        }
    (out_dir / "gates_expanded.json").write_text(json.dumps(gates_out, indent=2))
    (out_dir / "gate_library.json").write_text(json.dumps(engine.gate_library.to_dict(), indent=2))
    engine.scan_gates()

    if not engine.score_store:
        raise RuntimeError("score_store missing after gate scan")

    gt = _load_gt(args.domain, args.policy, args.k)
    biz_to_idx = _biz_to_restaurant_index(restaurants)
    review_to_rest = _review_index_to_restaurant_index(restaurants)

    positives = {biz: v for biz, v in gt.items() if v not in ("Low Risk", "Unknown")}
    print("GT positives:", positives)

    # Per-restaurant max Z rank for each primitive.
    for p in engine.primitives:
        pid = p.primitive_id
        z = engine.score_store.z_scores[pid]
        rest_max = np.full(len(restaurants), -1e9, dtype=np.float64)
        for ridx in range(len(restaurants)):
            mask = review_to_rest == ridx
            rest_max[ridx] = float(np.max(z[mask])) if np.any(mask) else -1e9
        print(f"\nPrimitive {pid}:")
        print("  z[min,p50,p90,max] =", float(np.min(z)), float(np.percentile(z, 50)), float(np.percentile(z, 90)), float(np.max(z)))
        for biz_id, verdict in positives.items():
            ridx = biz_to_idx.get(biz_id)
            if ridx is None:
                continue
            val = rest_max[ridx]
            rank = int(np.sum(rest_max > val) + 1)
            print(f"  GT {verdict} biz={biz_id} restaurant_rank={rank}/100 max_z={val:.4f}")

    # Top reviews by Z (per primitive) for quick sanity.
    top_n = 10
    for p in engine.primitives:
        pid = p.primitive_id
        z = engine.score_store.z_scores[pid]
        top = np.argsort(-z)[:top_n]
        print(f"\nTop {top_n} reviews by Z for {pid}:")
        for idx in top:
            r = engine.review_pool[int(idx)]
            snippet = (r.get("text") or "").replace("\n", " ")[:120]
            print(f"  idx={int(idx)} z={float(z[idx]):.4f} biz={r.get('business_id')} :: {snippet}")

    print(f"\nSaved cache to: {out_dir}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="T1P1")
    parser.add_argument("--domain", type=str, default="yelp")
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--agenda-spec-path", type=str, default="results/agenda_spec_temp")
    parser.add_argument("--model", type=str, default="gpt-5-nano")
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--num-gates-suggest", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
