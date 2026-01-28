"""Stepwise ATKD runner for manual inspection (Phase 2)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from addm.llm import LLMService
from addm.methods.amos.phase2_atkd import ATKDEngine, ATKDConfig
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
            f"{p.primitive_id}: z[min={float(np.min(z)):.3f}, "
            f"max={float(np.max(z)):.3f}, mean={float(np.mean(z)):.3f}]"
        )


def summarize_calibration(engine: ATKDEngine) -> None:
    output.rule()
    for p in engine.primitives:
        theta = engine.calibration.theta_hat.get(p.primitive_id)
        if theta is None:
            output.print(f"{p.primitive_id}: no calibration yet")
            continue
        output.print(f"{p.primitive_id}: theta_mean={float(np.mean(theta)):.3f}")


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
        explore_frac=args.explore_frac,
        batch_size=args.batch_size,
        verifier_batch_size=args.verifier_batch_size,
        num_bins=args.num_bins,
        gamma=args.gamma,
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
    pause_if(args.pause)

    output.rule()
    output.print("Step 1: Gate initialization")
    await engine.initialize_gates()
    summarize_gates(engine)
    pause_if(args.pause)

    output.rule()
    output.print("Step 2: Gate scan (BM25 + embeddings)")
    engine.scan_gates()
    summarize_z(engine)
    pause_if(args.pause)

    output.rule()
    output.print("Step 3: Initial calibration")
    engine._recompute_calibration()
    summarize_calibration(engine)
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
        pause_if(args.pause)

        await engine._verify_batch(batch)
        engine._recompute_calibration()

        # Recompute counts (same logic as engine.run)
        per_restaurant_counts = {}
        for record in engine.tag_store._records.values():
            ridx = engine._restaurant_index_for_review(record.review_id)
            if ridx is None:
                continue
            per_restaurant_counts.setdefault(ridx, {})
            per_restaurant_counts[ridx][record.primitive_id] = (
                per_restaurant_counts[ridx].get(record.primitive_id, 0) + record.y
            )

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
        pause_if(args.pause)

        if config.gate_discover_period > 0 and iteration % config.gate_discover_period == 0:
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
    parser.add_argument("--explore_frac", type=float, default=0.1)
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--verifier_batch_size", type=int, default=20)
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--sample-ids", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="results/dev/atkd_stepwise_cache")
    parser.add_argument("--pause", action="store_true", default=True)
    parser.add_argument("--no-pause", dest="pause", action="store_false")

    args = parser.parse_args()
    asyncio.run(run_stepwise(args))


if __name__ == "__main__":
    main()
