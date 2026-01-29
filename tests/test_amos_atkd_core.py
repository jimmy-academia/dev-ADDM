import json
from pathlib import Path

import numpy as np

from addm.methods.amos.phase2_atkd import (
    ATKDEngine,
    ATKDConfig,
    CalibrationEngine,
    Gate,
    GateLibrary,
    ScoreStore,
    EvidentRecord,
)
from addm.llm import LLMService


def test_gate_dedup_and_serialization(tmp_path: Path) -> None:
    lib = GateLibrary()
    g1 = Gate(
        gate_id="g1",
        primitive_ids=["p1"],
        modality="bm25",
        polarity="pos",
        query="peanut allergy",
        created_by="baseline",
        created_iter=0,
    )
    g2 = Gate(
        gate_id="g2",
        primitive_ids=["p1"],
        modality="bm25",
        polarity="pos",
        query="peanut allergy",
        created_by="baseline",
        created_iter=0,
    )
    assert lib.add_gate(g1) is True
    assert lib.add_gate(g2) is False

    data = lib.to_dict()
    path = tmp_path / "gates.json"
    path.write_text(json.dumps(data))

    lib2 = GateLibrary.from_dict(json.loads(path.read_text()))
    assert len(lib2.list()) == 1


def test_score_aggregation_max(tmp_path: Path) -> None:
    score_store = ScoreStore(3, tmp_path)
    g1 = Gate("g1", ["p1"], "bm25", "pos", "q1", "baseline", created_iter=0)
    g2 = Gate("g2", ["p1"], "bm25", "pos", "q2", "baseline", created_iter=0)
    lib = GateLibrary()
    lib.add_gate(g1)
    lib.add_gate(g2)

    score_store.add_gate_scores("g1", np.array([0.1, 0.2, 0.3]))
    score_store.add_gate_scores("g2", np.array([0.2, 0.1, 0.4]))
    score_store.recompute_for_primitives(["p1"], lib, num_bins=3, gamma=1.0)
    agg = score_store.aggregates["p1"]["pos_bm25"]
    assert np.allclose(agg, np.array([0.2, 0.2, 0.4]))


def test_calibration_recompute_deterministic() -> None:
    engine = CalibrationEngine(num_bins=3, delta=0.05)
    records = [
        EvidentRecord(
            review_id="r1",
            business_id="b1",
            review_text_hash="h1",
            prompt_version="v1",
            review_index=0,
            evident={"review_id": "r1", "evidents": []},
            tags={"p1": 1},
            bins={"p1": 0},
            evidence={},
            usage={},
            created_at="now",
        ),
        EvidentRecord(
            review_id="r2",
            business_id="b1",
            review_text_hash="h2",
            prompt_version="v1",
            review_index=1,
            evident={"review_id": "r2", "evidents": []},
            tags={"p1": 0},
            bins={"p1": 1},
            evidence={},
            usage={},
            created_at="now",
        ),
    ]
    engine.recompute("p1", records)
    first = engine.theta_hat["p1"].copy()
    engine.recompute("p1", records)
    second = engine.theta_hat["p1"].copy()
    assert np.allclose(first, second)


def test_incremental_gate_recompute_matches_full(tmp_path: Path) -> None:
    lib = GateLibrary()
    g1 = Gate("g1", ["p1"], "bm25", "pos", "q1", "baseline", created_iter=0)
    g2 = Gate("g2", ["p1"], "bm25", "pos", "q2", "baseline", created_iter=0)
    lib.add_gate(g1)
    lib.add_gate(g2)

    score_store = ScoreStore(3, tmp_path)
    score_store.add_gate_scores("g1", np.array([0.1, 0.2, 0.3]))
    score_store.add_gate_scores("g2", np.array([0.2, 0.1, 0.4]))
    score_store.recompute_for_primitives(["p1"], lib, num_bins=3, gamma=1.0)

    g3 = Gate("g3", ["p1"], "bm25", "pos", "q3", "baseline", created_iter=0)
    lib.add_gate(g3)
    score_store.add_gate_scores("g3", np.array([0.5, 0.0, 0.2]))
    score_store.recompute_for_primitives(["p1"], lib, num_bins=3, gamma=1.0)
    z_incremental = score_store.z_scores["p1"].copy()

    full = ScoreStore(3, tmp_path / "full")
    full.add_gate_scores("g1", np.array([0.1, 0.2, 0.3]))
    full.add_gate_scores("g2", np.array([0.2, 0.1, 0.4]))
    full.add_gate_scores("g3", np.array([0.5, 0.0, 0.2]))
    full.recompute_for_primitives(["p1"], lib, num_bins=3, gamma=1.0)
    z_full = full.z_scores["p1"]
    assert np.allclose(z_incremental, z_full)


def test_default_bound_monotonicity(tmp_path: Path) -> None:
    agenda_spec = {
        "terms": [],
        "verdict_rules": {
            "labels": ["High", "Low"],
            "default_label": "Low",
            "order": ["High", "Low"],
            "rules": [
                {
                    "label": "High",
                    "default": False,
                    "connective": "ANY",
                    "clauses": [
                        {"clause_quote": "x", "min_count": 1, "logic": "ALL", "conditions": []}
                    ],
                },
                {"label": "Low", "default": True},
            ],
        },
    }
    restaurants = [
        {
            "business": {"business_id": "b1", "name": "b1"},
            "reviews": [{"review_id": "r1", "text": "a"}, {"review_id": "r2", "text": "b"}],
        }
    ]
    engine = ATKDEngine(
        policy_id="T1P1",
        agenda_spec=agenda_spec,
        agenda_text="agenda",
        restaurants=restaurants,
        llm=LLMService(),
        config=ATKDConfig(),
        cache_dir=tmp_path,
        rng_seed=7,
    )
    engine.score_store = ScoreStore(2, tmp_path / "scores")
    pid = engine.primitives[0].primitive_id
    engine.score_store.z_bins[pid] = np.array([0, 1])
    engine.calibration.upper_bound[pid] = np.array([0.5, 0.5])

    counts = {pid: 0}
    review_indices = [0, 1]
    rho_before = engine._compute_default_bound(review_indices, counts)
    engine.tag_store.add(
        EvidentRecord(
            review_id="r1",
            business_id="b1",
            review_text_hash="h1",
            prompt_version="v1",
            review_index=0,
            evident={"review_id": "r1", "evidents": []},
            tags={pid: 0},
            bins={pid: 0},
            evidence={},
            usage={},
            created_at="now",
        )
    )
    rho_after = engine._compute_default_bound(review_indices, counts)
    assert rho_after <= rho_before
