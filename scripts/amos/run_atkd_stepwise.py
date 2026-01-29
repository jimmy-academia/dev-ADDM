"""Stepwise ATKD runner for manual inspection (Phase 2)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from addm.llm import LLMService
from addm.methods.amos.phase2_atkd import ATKDEngine, ATKDConfig, GateLibrary
from addm.utils.output import output


def load_agenda_spec(path: Path, policy_id: str) -> Dict[str, Any]:
    if path.is_dir():
        candidates = [
            path / f"{policy_id}.json",
            path / f"{policy_id}_agenda_spec.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return json.loads(candidate.read_text())
        for file in path.glob("*.json"):
            return json.loads(file.read_text())
        raise FileNotFoundError(f"No agenda spec found in {path}")
    if not path.exists():
        raise FileNotFoundError(f"Agenda spec path not found: {path}")
    return json.loads(path.read_text())


def load_dataset(domain: str, k: int) -> List[Dict[str, Any]]:
    dataset_path = Path(f"data/context/{domain}/dataset_K{k}.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    restaurants = []
    with open(dataset_path) as f:
        for line in f:
            restaurants.append(json.loads(line))
    return restaurants


def pause_if(enabled: bool) -> None:
    if enabled:
        input("Press Enter to continue...")


def summarize_primitives(engine: ATKDEngine) -> None:
    output.rule()
    output.print(f"Primitives: {len(engine.primitives)}")
    for p in engine.primitives:
        output.print(
            f"  - {p.primitive_id} | min_count={p.min_count} | logic={p.logic} | {p.clause_quote}"
        )


def summarize_gates(engine: ATKDEngine) -> None:
    output.rule()
    output.print(f"Gates total: {len(engine.gate_library.list())}")
    for p in engine.primitives:
        counts = {}
        for modality in ("bm25", "emb"):
            for polarity in ("pos", "neg"):
                key = f"{modality}_{polarity}"
                counts[key] = len(engine.gate_library.by_filter(p.primitive_id, modality, polarity))
        output.print(
            f"  - {p.primitive_id}: "
            f"bm25+={counts['bm25_pos']} bm25-={counts['bm25_neg']} "
            f"emb+={counts['emb_pos']} emb-={counts['emb_neg']}"
        )


def summarize_z(engine: ATKDEngine) -> None:
    output.rule()
    if not engine.score_store:
        output.print("No score store yet.")
        return
    for p in engine.primitives:
        z = engine.score_store.z_scores.get(p.primitive_id)
        if z is None:
            continue
        output.print(
            f"{p.primitive_id}: z(min={float(np.min(z)):.3f}, "
            f"max={float(np.max(z)):.3f}, mean={float(np.mean(z)):.3f})"
        )


def summarize_calibration(engine: ATKDEngine) -> None:
    output.rule()
    for p in engine.primitives:
        theta = engine.calibration.theta_hat.get(p.primitive_id)
        if theta is None:
            output.print(f"{p.primitive_id}: no calibration yet")
            continue
        output.print(f"{p.primitive_id}: theta_mean={float(np.mean(theta)):.3f}")


def dump_state(
    engine: ATKDEngine,
    label: str,
    out_dir: Path,
    batch: Optional[List[Any]] = None,
) -> None:
    def _to_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    out_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "label": label,
        "policy_id": engine.policy_id,
        "primitives": [p.__dict__ for p in engine.primitives],
        "gate_counts": {
            p.primitive_id: {
                f"{modality}_{polarity}": len(
                    engine.gate_library.by_filter(p.primitive_id, modality, polarity)
                )
                for modality in ("bm25", "emb")
                for polarity in ("pos", "neg")
            }
            for p in engine.primitives
        },
        "tag_counts": {
            p.primitive_id: {
                "total": len(engine.tag_store.records_for_primitive(p.primitive_id)),
                "pos": sum(
                    1
                    for r in engine.tag_store.records_for_primitive(p.primitive_id)
                    if r.tags.get(p.primitive_id, 0) > 0
                ),
                "neg": sum(
                    1
                    for r in engine.tag_store.records_for_primitive(p.primitive_id)
                    if r.tags.get(p.primitive_id, 0) == 0
                ),
            }
            for p in engine.primitives
        },
    }
    if engine.score_store:
        payload["z_stats"] = {
            p.primitive_id: {
                "min": float(np.min(engine.score_store.z_scores[p.primitive_id])),
                "max": float(np.max(engine.score_store.z_scores[p.primitive_id])),
                "mean": float(np.mean(engine.score_store.z_scores[p.primitive_id])),
            }
            for p in engine.primitives
            if p.primitive_id in engine.score_store.z_scores
        }
        payload["z_bin_edges"] = {
            p.primitive_id: engine.score_store.z_bin_edges[p.primitive_id].tolist()
            for p in engine.primitives
            if p.primitive_id in engine.score_store.z_bin_edges
        }
    if engine.calibration:
        payload["theta_hat"] = {
            p.primitive_id: _to_list(engine.calibration.theta_hat.get(p.primitive_id))
            for p in engine.primitives
        }
        payload["upper_bound"] = {
            p.primitive_id: _to_list(engine.calibration.upper_bound.get(p.primitive_id))
            for p in engine.primitives
        }
    if batch is not None:
        normalized = []
        for item in batch:
            if isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append({"review_index": item})
        payload["selected_batch"] = normalized
    (out_dir / f"{label}.json").write_text(json.dumps(payload, indent=2))


def summarize_batch(engine: ATKDEngine, batch: List[Any], max_rows: int = 10) -> None:
    output.rule()
    output.print(f"Batch preview (showing up to {max_rows})")
    if not engine.score_store:
        output.print("No score store yet.")
        return
    rows = []
    for item in batch[:max_rows]:
        review_idx = item.get("review_index") if isinstance(item, dict) else item
        if review_idx is None:
            continue
        best_pid = None
        best_z = -1e9
        for p in engine.primitives:
            pid = p.primitive_id
            z_val = float(engine.score_store.z_scores[pid][review_idx])
            if z_val > best_z:
                best_z = z_val
                best_pid = pid
        if best_pid is None:
            continue
        z = best_z
        bin_idx = int(engine.score_store.z_bins[best_pid][review_idx])
        review = engine.review_pool[review_idx]
        snippet = review["text"].strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        rows.append(
            [
                review.get("review_id"),
                best_pid,
                f"{z:.3f}",
                str(bin_idx),
                review.get("business_id"),
                snippet,
            ]
        )
    output.print_table(
        "Verifier Batch",
        ["review_id", "primitive_id", "z", "bin", "business_id", "snippet"],
        rows,
    )


async def run_stepwise(args: argparse.Namespace) -> None:
    agenda_spec = load_agenda_spec(Path(args.agenda_spec_path), args.policy)
    restaurants = load_dataset(args.domain, args.k)

    if args.sample_ids:
        allowed = {s.strip() for s in args.sample_ids.split(",") if s.strip()}
        restaurants = [
            r for r in restaurants if r.get("business", {}).get("business_id") in allowed
        ]
    if args.n > 0:
        restaurants = restaurants[args.skip : args.skip + args.n]
    else:
        restaurants = restaurants[args.skip :]

    llm = LLMService()
    llm.configure(model=args.model)

    config = ATKDConfig(
        epsilon=args.epsilon,
        delta=args.delta,
        gate_init=args.gate_init,
        gate_discover_period=args.gate_discover_period,
        gate_discover_every=args.gate_discover_every or args.gate_discover_period,
        explore_frac=args.explore_frac,
        batch_size=args.batch_size,
        verifier_batch_size=args.verifier_batch_size,
        num_bins=args.num_bins,
        gamma=args.gamma,
        lambda_H=args.lambda_H,
        lambda_L=args.lambda_L,
        lambda_G=args.lambda_G,
        stall_iters=args.stall_iters,
        min_bin_coverage=args.min_bin_coverage,
        prompt_version=args.prompt_version,
        dataset_tag=args.domain,
    )

    engine = ATKDEngine(
        policy_id=args.policy,
        agenda_spec=agenda_spec,
        restaurants=restaurants,
        llm=llm,
        config=config,
        cache_dir=Path(args.cache_dir),
        rng_seed=args.seed,
    )

    output.rule()
    output.print("Step 0: Compile primitives")
    summarize_primitives(engine)
    if args.dump_state_dir:
        dump_state(engine, "step0_compile", Path(args.dump_state_dir))
    pause_if(args.pause)

    output.rule()
    output.print("Step 1: Gate initialization")
    gate_cache_path = Path(args.gate_cache_path) if args.gate_cache_path else None
    if gate_cache_path and gate_cache_path.exists():
        data = json.loads(gate_cache_path.read_text())
        engine.gate_library = GateLibrary.from_dict(data)
        engine.config.gate_init = False
        engine._ensure_min_gates()
        output.print(f"Loaded gates from cache: {gate_cache_path}")
    else:
        await engine.initialize_gates()
        if gate_cache_path and args.save_gate_cache:
            gate_cache_path.parent.mkdir(parents=True, exist_ok=True)
            gate_cache_path.write_text(json.dumps(engine.gate_library.to_dict(), indent=2))
            output.print(f"Saved gate cache to: {gate_cache_path}")
    summarize_gates(engine)
    if args.dump_state_dir:
        dump_state(engine, "step1_gates", Path(args.dump_state_dir))
    pause_if(args.pause)

    output.rule()
    output.print("Step 2: Gate scan (BM25 + embeddings)")
    engine.scan_gates()
    summarize_z(engine)
    if args.dump_state_dir:
        dump_state(engine, "step2_scan", Path(args.dump_state_dir))
    pause_if(args.pause)

    output.rule()
    output.print("Step 3: Initial calibration")
    engine._recompute_calibration()
    summarize_calibration(engine)
    if args.dump_state_dir:
        dump_state(engine, "step3_calibration", Path(args.dump_state_dir))
    pause_if(args.pause)

    active_restaurants = list(range(len(restaurants)))
    per_restaurant_counts: Dict[int, Dict[str, int]] = {}
    restaurant_verdicts: Dict[int, Dict[str, Any]] = {}

    for iteration in range(1, args.max_iterations + 1):
        if not active_restaurants:
            break
        output.rule()
        output.print(f"Iteration {iteration}")

        restaurant_batch = active_restaurants[: config.batch_size]
        batch = engine._select_batch(
            restaurant_batch,
            per_restaurant_counts,
            config.explore_frac,
            config.verifier_batch_size,
        )
        output.print(f"Verifier batch size: {len(batch)}")
        summarize_batch(engine, batch, max_rows=args.preview_rows)
        if args.dump_state_dir:
            dump_state(
                engine,
                f"iter{iteration}_selected",
                Path(args.dump_state_dir),
                batch=batch,
            )
        pause_if(args.pause)

        if not args.skip_verify:
            await engine._verify_batch(batch)
            engine.tag_store.save()
            engine._recompute_calibration()
            if args.dump_state_dir:
                dump_state(engine, f"iter{iteration}_verified", Path(args.dump_state_dir))

        # Recompute counts (same logic as engine.run)
        per_restaurant_counts = engine._compute_counts_by_restaurant()

        # Check stopping for each restaurant in batch
        still_active = []
        for ridx in active_restaurants:
            counts = per_restaurant_counts.get(ridx, {})
            verdict = engine._evaluate_verdict(counts)
            if verdict:
                restaurant_verdicts[ridx] = {
                    "verdict": verdict,
                    "stop_reason": "rule_satisfied",
                }
                continue
            review_indices = engine._restaurant_review_indices(ridx)
            rho = engine._compute_default_bound(review_indices, counts)
            if rho <= config.epsilon:
                restaurant_verdicts[ridx] = {
                    "verdict": engine.default_label or "Default",
                    "stop_reason": "bound",
                    "rho": rho,
                }
                continue
            still_active.append(ridx)
        active_restaurants = still_active

        output.print(f"Active restaurants remaining: {len(active_restaurants)}")
        summarize_calibration(engine)
        if args.dump_state_dir:
            dump_state(engine, f"iter{iteration}_postcheck", Path(args.dump_state_dir))
        pause_if(args.pause)

        discover_every = config.gate_discover_every or config.gate_discover_period
        if discover_every > 0 and iteration % discover_every == 0:
            output.rule()
            output.print("GateDiscover")
            await engine._gate_discover()
            engine.scan_gates()
            engine._recompute_calibration()
            summarize_gates(engine)
            summarize_z(engine)
            pause_if(args.pause)

    output.rule()
    output.print("Stepwise run complete.")
    output.print(f"Finalized restaurants: {len(restaurant_verdicts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stepwise ATKD runner")
    parser.add_argument("--policy", type=str, default="T1P1")
    parser.add_argument("--domain", type=str, default="yelp")
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("-n", type=int, default=5)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt-5-nano")
    parser.add_argument("--agenda-spec-path", type=str, default="results/agenda_spec_temp")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--gate_init", action="store_true", default=True)
    parser.add_argument("--no-gate_init", dest="gate_init", action="store_false")
    parser.add_argument("--gate_discover_period", type=int, default=5)
    parser.add_argument("--gate_discover_every", type=int, default=None)
    parser.add_argument("--explore_frac", type=float, default=0.1)
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda_H", type=float, default=1.0)
    parser.add_argument("--lambda_L", type=float, default=1.0)
    parser.add_argument("--lambda_G", type=float, default=1.0)
    parser.add_argument("--stall_iters", type=int, default=3)
    parser.add_argument("--min_bin_coverage", type=int, default=5)
    parser.add_argument("--prompt_version", type=str, default="extract_evident_v1")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--verifier_batch_size", type=int, default=20)
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--sample-ids", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="results/dev/atkd_stepwise_cache")
    parser.add_argument("--gate-cache-path", type=str, default=None)
    parser.add_argument("--save-gate-cache", action="store_true", default=False)
    parser.add_argument("--pause", action="store_true", default=True)
    parser.add_argument("--no-pause", dest="pause", action="store_false")
    parser.add_argument("--dump-state-dir", type=str, default=None)
    parser.add_argument("--preview-rows", type=int, default=8)
    parser.add_argument("--skip-verify", action="store_true", default=False)

    args = parser.parse_args()
    asyncio.run(run_stepwise(args))


if __name__ == "__main__":
    main()
