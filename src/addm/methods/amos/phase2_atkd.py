"""Phase 2 ATKD (Active Test-time Knowledge Discovery) engine."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from addm.llm import LLMService
from addm.utils.async_utils import gather_with_concurrency

from .phase1_helpers import extract_json_from_response, parse_json_safely
from .phase2_prompts import (
    build_gate_discover_prompt,
    build_gate_init_prompt,
    build_verifier_prompt,
)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _text_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_query(query: Any) -> str:
    if isinstance(query, list):
        joined = " ".join(str(q) for q in query)
    else:
        joined = str(query or "")
    return re.sub(r"\\s+", " ", joined.strip().lower())


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    n = len(scores)
    if n == 0:
        return scores
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    denom = max(n - 1, 1)
    return ranks / denom


# =============================================================================
# Beta distribution utilities (regularized incomplete beta + inverse)
# =============================================================================


def _betacf(a: float, b: float, x: float) -> float:
    # Continued fraction for incomplete beta (Numerical Recipes).
    max_iter = 200
    eps = 3.0e-7
    fpmin = 1.0e-30

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        del_h = d * c
        h *= del_h
        if abs(del_h - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - ln_beta) / a
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x)
    return 1.0 - front * _betacf(b, a, 1.0 - x)


def beta_ppf(q: float, a: float, b: float) -> float:
    # Inverse regularized incomplete beta via bisection.
    q = min(max(q, 0.0), 1.0)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        cdf = _betainc(a, b, mid)
        if cdf < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# =============================================================================
# Data structures
# =============================================================================


@dataclass(frozen=True)
class Primitive:
    primitive_id: str
    clause_quote: str
    conditions: List[Dict[str, Any]]
    min_count: int
    logic: str


@dataclass(frozen=True)
class CompiledRule:
    label: str
    connective: str
    primitive_ids: List[str]


@dataclass
class Gate:
    gate_id: str
    primitive_ids: List[str]
    modality: str  # bm25 | emb
    polarity: str  # pos | neg
    query: str
    created_by: str


@dataclass
class TagRecord:
    review_id: str
    primitive_id: str
    review_text_hash: str
    y: int
    evidence_snippets: List[str]
    fields: Dict[str, Any]
    usage: Dict[str, Any]
    created_at: str


@dataclass
class ATKDConfig:
    epsilon: float = 0.01
    delta: float = 0.05
    gate_init: bool = True
    gate_discover_period: int = 5
    explore_frac: float = 0.1
    batch_size: int = 10  # restaurants per driver iteration
    verifier_batch_size: int = 32
    num_bins: int = 10
    gamma: float = 1.0
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    embedding_model: str = "text-embedding-3-large"
    max_concurrent: int = 32


class GateLibrary:
    def __init__(self) -> None:
        self._gates: Dict[str, Gate] = {}
        self._dedupe: Dict[Tuple[Tuple[str, ...], str, str, str], str] = {}

    def add_gate(self, gate: Gate) -> bool:
        key = (
            tuple(sorted(gate.primitive_ids)),
            gate.modality,
            gate.polarity,
            _normalize_query(gate.query),
        )
        if key in self._dedupe:
            return False
        self._gates[gate.gate_id] = gate
        self._dedupe[key] = gate.gate_id
        return True

    def add_many(self, gates: Iterable[Gate]) -> int:
        added = 0
        for g in gates:
            if self.add_gate(g):
                added += 1
        return added

    def list(self) -> List[Gate]:
        return list(self._gates.values())

    def by_primitive(self, primitive_id: str) -> List[Gate]:
        return [g for g in self._gates.values() if primitive_id in g.primitive_ids]

    def by_filter(self, primitive_id: str, modality: str, polarity: str) -> List[Gate]:
        return [
            g for g in self._gates.values()
            if primitive_id in g.primitive_ids and g.modality == modality and g.polarity == polarity
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {"gates": [g.__dict__ for g in self._gates.values()]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GateLibrary":
        lib = cls()
        for g in data.get("gates", []):
            lib.add_gate(Gate(**g))
        return lib


class TagStore:
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self._records: Dict[Tuple[str, str, str], TagRecord] = {}
        self._by_primitive: Dict[str, List[TagRecord]] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path) as f:
                for line in f:
                    data = json.loads(line)
                    record = TagRecord(**data)
                    key = (record.review_id, record.primitive_id, record.review_text_hash)
                    self._records[key] = record
                    self._by_primitive.setdefault(record.primitive_id, []).append(record)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            for record in self._records.values():
                f.write(json.dumps(record.__dict__) + "\n")

    def get(self, review_id: str, primitive_id: str, review_text_hash: str) -> Optional[TagRecord]:
        return self._records.get((review_id, primitive_id, review_text_hash))

    def add(self, record: TagRecord) -> None:
        key = (record.review_id, record.primitive_id, record.review_text_hash)
        if key in self._records:
            return
        self._records[key] = record
        self._by_primitive.setdefault(record.primitive_id, []).append(record)

    def records_for_primitive(self, primitive_id: str) -> List[TagRecord]:
        return self._by_primitive.get(primitive_id, [])


class BM25Index:
    def __init__(self, texts: List[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_tokens: List[List[str]] = [_tokenize(t) for t in texts]
        self._doc_len = np.array([len(t) for t in self._doc_tokens], dtype=np.float64)
        self._avg_len = float(np.mean(self._doc_len)) if len(self._doc_len) > 0 else 0.0
        self._tf: List[Dict[str, int]] = []
        self._df: Dict[str, int] = {}
        for tokens in self._doc_tokens:
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            for t in tf.keys():
                self._df[t] = self._df.get(t, 0) + 1
        self._idf: Dict[str, float] = {}
        n_docs = len(self._doc_tokens)
        for term, df in self._df.items():
            self._idf[term] = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> np.ndarray:
        tokens = _tokenize(query)
        if not tokens:
            return np.zeros(len(self._doc_tokens), dtype=np.float64)
        scores = np.zeros(len(self._doc_tokens), dtype=np.float64)
        for idx, tf in enumerate(self._tf):
            score = 0.0
            dl = self._doc_len[idx]
            for t in tokens:
                if t not in tf:
                    continue
                idf = self._idf.get(t, 0.0)
                freq = tf[t]
                denom = freq + self.k1 * (1 - self.b + self.b * (dl / (self._avg_len + 1e-9)))
                score += idf * (freq * (self.k1 + 1)) / (denom + 1e-9)
            scores[idx] = score
        return scores


class EmbeddingStore:
    def __init__(self, cache_dir: Path, model: str = "text-embedding-3-large") -> None:
        self.cache_dir = cache_dir
        self.model = model
        self._review_embeddings: Optional[np.ndarray] = None
        self._review_ids: List[str] = []
        self._review_hashes: List[str] = []
        self._query_cache: Dict[str, np.ndarray] = {}

    def _meta_path(self) -> Path:
        return self.cache_dir / "embedding_meta.json"

    def _emb_path(self) -> Path:
        return self.cache_dir / "review_embeddings.npy"

    def load_or_compute(self, reviews: List[Dict[str, Any]], llm: LLMService) -> np.ndarray:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        review_ids = [r["review_id"] for r in reviews]
        review_hashes = [r["review_text_hash"] for r in reviews]
        if self._emb_path().exists() and self._meta_path().exists():
            try:
                meta = json.loads(self._meta_path().read_text())
                if (
                    meta.get("review_ids") == review_ids
                    and meta.get("review_hashes") == review_hashes
                    and meta.get("model") == self.model
                ):
                    self._review_embeddings = np.load(self._emb_path())
                    self._review_ids = review_ids
                    self._review_hashes = review_hashes
                    return self._review_embeddings
            except (OSError, json.JSONDecodeError):
                pass

        embeddings = self._embed_texts([r["text"] for r in reviews], llm)
        self._review_embeddings = embeddings
        self._review_ids = review_ids
        self._review_hashes = review_hashes
        self._meta_path().write_text(
            json.dumps(
                {"review_ids": review_ids, "review_hashes": review_hashes, "model": self.model}
            )
        )
        np.save(self._emb_path(), embeddings)
        return embeddings

    def embed_query(self, query: str, llm: LLMService) -> np.ndarray:
        key = _text_hash(query)
        if key in self._query_cache:
            return self._query_cache[key]
        emb = self._embed_texts([query], llm)[0]
        self._query_cache[key] = emb
        return emb

    def _embed_texts(self, texts: List[str], llm: LLMService) -> np.ndarray:
        # Uses OpenAI embeddings API via LLMService config.
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for embeddings") from exc

        client = openai.OpenAI() if hasattr(openai, "OpenAI") else openai.Client()
        embeddings: List[List[float]] = []
        batch_size = 128
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(model=self.model, input=batch)
            embeddings.extend([item.embedding for item in response.data])
        return np.array(embeddings, dtype=np.float64)


class ScoreStore:
    def __init__(self, review_count: int, cache_dir: Path) -> None:
        self.review_count = review_count
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.gate_scores: Dict[str, np.ndarray] = {}
        self.aggregates: Dict[str, Dict[str, np.ndarray]] = {}
        self.z_scores: Dict[str, np.ndarray] = {}
        self.z_bins: Dict[str, np.ndarray] = {}
        self.z_bin_edges: Dict[str, np.ndarray] = {}

    def _gate_path(self, gate_id: str) -> Path:
        return self.cache_dir / f"gate_{gate_id}.npy"

    def add_gate_scores(self, gate_id: str, scores: np.ndarray) -> None:
        if gate_id in self.gate_scores:
            return
        self.gate_scores[gate_id] = scores
        np.save(self._gate_path(gate_id), scores)

    def load_gate_scores(self, gate_id: str) -> Optional[np.ndarray]:
        if gate_id in self.gate_scores:
            return self.gate_scores[gate_id]
        path = self._gate_path(gate_id)
        if path.exists():
            scores = np.load(path)
            self.gate_scores[gate_id] = scores
            return scores
        return None

    def recompute_for_primitives(
        self,
        primitive_ids: List[str],
        gate_library: GateLibrary,
        num_bins: int,
        gamma: float,
    ) -> None:
        for pid in primitive_ids:
            pos_bm25 = self._max_scores(gate_library.by_filter(pid, "bm25", "pos"))
            neg_bm25 = self._max_scores(gate_library.by_filter(pid, "bm25", "neg"))
            pos_emb = self._max_scores(gate_library.by_filter(pid, "emb", "pos"))
            neg_emb = self._max_scores(gate_library.by_filter(pid, "emb", "neg"))
            self.aggregates[pid] = {
                "pos_bm25": pos_bm25,
                "neg_bm25": neg_bm25,
                "pos_emb": pos_emb,
                "neg_emb": neg_emb,
            }
            pos_bm25_n = rank_normalize(pos_bm25)
            neg_bm25_n = rank_normalize(neg_bm25)
            pos_emb_n = rank_normalize(pos_emb)
            neg_emb_n = rank_normalize(neg_emb)
            z = np.maximum(pos_bm25_n, pos_emb_n) - gamma * np.maximum(neg_bm25_n, neg_emb_n)
            self.z_scores[pid] = z
            # Quantile bins
            if num_bins <= 1:
                edges = np.array([0.0, 1.0])
            else:
                qs = np.linspace(0, 1, num_bins + 1)
                edges = np.quantile(z, qs)
                edges[0] = -1e9
                edges[-1] = 1e9
            self.z_bin_edges[pid] = edges
            self.z_bins[pid] = np.digitize(z, edges[1:-1], right=True)

    def _max_scores(self, gates: List[Gate]) -> np.ndarray:
        if not gates:
            return np.zeros(self.review_count, dtype=np.float64)
        stacked = []
        for g in gates:
            scores = self.load_gate_scores(g.gate_id)
            if scores is None:
                continue
            stacked.append(scores)
        if not stacked:
            return np.zeros(self.review_count, dtype=np.float64)
        return np.max(np.vstack(stacked), axis=0)


class CalibrationEngine:
    def __init__(self, num_bins: int, delta: float) -> None:
        self.num_bins = num_bins
        self.delta = delta
        self.theta_hat: Dict[str, np.ndarray] = {}
        self.upper_bound: Dict[str, np.ndarray] = {}

    def recompute(self, primitive_id: str, z_bins: np.ndarray, tags: List[TagRecord]) -> None:
        alpha0, beta0 = 1.0, 1.0
        alpha = np.full(self.num_bins, alpha0, dtype=np.float64)
        beta = np.full(self.num_bins, beta0, dtype=np.float64)
        for record in tags:
            review_idx = record.fields.get("_review_index")
            if review_idx is None:
                continue
            bin_idx = int(z_bins[int(review_idx)])
            if bin_idx < 0 or bin_idx >= self.num_bins:
                continue
            if record.y == 1:
                alpha[bin_idx] += 1.0
            else:
                beta[bin_idx] += 1.0
        theta = alpha / (alpha + beta)
        u = np.zeros(self.num_bins, dtype=np.float64)
        for i in range(self.num_bins):
            u[i] = beta_ppf(1.0 - self.delta, alpha[i], beta[i])
        self.theta_hat[primitive_id] = theta
        self.upper_bound[primitive_id] = u


class ATKDEngine:
    def __init__(
        self,
        policy_id: str,
        agenda_spec: Dict[str, Any],
        restaurants: List[Dict[str, Any]],
        llm: LLMService,
        config: ATKDConfig,
        cache_dir: Path,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.policy_id = policy_id
        self.agenda_spec = agenda_spec
        self.restaurants = restaurants
        self.llm = llm
        self.config = config
        self.cache_dir = cache_dir
        self.rng = random.Random(rng_seed)

        (
            self.primitives,
            self.rules,
            self.default_label,
            self.label_order,
        ) = self._compile_primitives()
        self.term_defs = self._build_term_defs()
        self.gate_library = GateLibrary()
        self.tag_store = TagStore(cache_dir / "verifier_cache.jsonl")
        self.score_store: Optional[ScoreStore] = None
        self.calibration = CalibrationEngine(config.num_bins, config.delta)

        self._usage_records: List[Dict[str, Any]] = []

        # Build review pool
        self.review_pool = self._build_review_pool()
        self._review_to_restaurant = {
            r["review_id"]: r["restaurant_index"] for r in self.review_pool
        }
        self._restaurant_to_reviews: Dict[int, List[int]] = {}
        for idx, r in enumerate(self.review_pool):
            self._restaurant_to_reviews.setdefault(r["restaurant_index"], []).append(idx)

    # ---------------------------------------------------------------------
    # Compilation
    # ---------------------------------------------------------------------

    def _compile_primitives(self) -> Tuple[List[Primitive], List[CompiledRule], str, List[str]]:
        verdict_spec = self.agenda_spec.get("verdict_rules", {})
        rules = verdict_spec.get("rules", [])
        primitives: List[Primitive] = []
        compiled_rules: List[CompiledRule] = []
        default_label = verdict_spec.get("default_label", "")
        order = verdict_spec.get("order", [])

        for rule in rules:
            if rule.get("default"):
                continue
            label = rule.get("label")
            primitive_ids = []
            for idx, clause in enumerate(rule.get("clauses", [])):
                pid = f"{label}_clause_{idx + 1}"
                primitive_ids.append(pid)
                primitives.append(
                    Primitive(
                        primitive_id=pid,
                        clause_quote=clause.get("clause_quote", ""),
                        conditions=clause.get("conditions", []),
                        min_count=int(clause.get("min_count") or 1),
                        logic=clause.get("logic", "ALL"),
                    )
                )
            compiled_rules.append(
                CompiledRule(
                    label=label,
                    connective=rule.get("connective", "ANY"),
                    primitive_ids=primitive_ids,
                )
            )
        return primitives, compiled_rules, default_label, order

    def _build_term_defs(self) -> Dict[str, Any]:
        terms = self.agenda_spec.get("terms", [])
        term_defs = {}
        for t in terms:
            term_defs[t.get("field_id")] = t.get("values", [])
        return term_defs

    # ---------------------------------------------------------------------
    # Review pool
    # ---------------------------------------------------------------------

    def _build_review_pool(self) -> List[Dict[str, Any]]:
        pool = []
        for r_idx, restaurant in enumerate(self.restaurants):
            business = restaurant.get("business", {})
            biz_id = business.get("business_id", "")
            for review in restaurant.get("reviews", []):
                review_id = review.get("review_id") or f"{biz_id}::{len(pool)}"
                text = review.get("text", "")
                pool.append(
                    {
                        "review_id": review_id,
                        "business_id": biz_id,
                        "restaurant_index": r_idx,
                        "text": text,
                        "date": review.get("date"),
                        "review_text_hash": _text_hash(text),
                    }
                )
        return pool

    # ---------------------------------------------------------------------
    # Gate initialization
    # ---------------------------------------------------------------------

    async def initialize_gates(self) -> None:
        # Baseline gates from clause text.
        baseline_gates = self._baseline_gates()
        self.gate_library.add_many(baseline_gates)

        if not self.config.gate_init:
            self._ensure_min_gates()
            return

        primitives_payload = [p.__dict__ for p in self.primitives]
        prompt = build_gate_init_prompt(primitives_payload)
        response, usage = await self.llm.call_async_with_usage(
            [{"role": "user", "content": prompt}],
            context={"phase": "gate_init", "policy_id": self.policy_id},
        )
        self._usage_records.append(usage)
        try:
            data = parse_json_safely(extract_json_from_response(response))
        except Exception:
            data = {}
        for item in data.get("primitives", []):
            self._add_gates_from_payload(item, created_by="llm_init")
        self._ensure_min_gates()

    def _baseline_gates(self) -> List[Gate]:
        gates: List[Gate] = []
        all_value_tokens: List[str] = []
        for values in self.term_defs.values():
            for v in values:
                all_value_tokens.extend(_tokenize(v.get("value_id", "")))
        for p in self.primitives:
            tokens = _tokenize(p.clause_quote)
            if not tokens:
                tokens = all_value_tokens
            mid = max(1, len(tokens) // 2)
            groups = [tokens[:mid], tokens[mid:]]
            for idx, group in enumerate(groups):
                if not group:
                    continue
                query = " ".join(group)
                gates.append(self._make_gate([p.primitive_id], "bm25", "pos", query, "baseline"))
            gates.append(
                self._make_gate([p.primitive_id], "emb", "pos", p.clause_quote, "baseline")
            )
            gates.append(
                self._make_gate(
                    [p.primitive_id],
                    "emb",
                    "pos",
                    f"{p.clause_quote} example",
                    "baseline",
                )
            )
        return gates

    def _add_gates_from_payload(self, payload: Dict[str, Any], created_by: str) -> None:
        primitive_id = payload.get("primitive_id")
        if not primitive_id:
            return
        for gate_list, modality, polarity in [
            (payload.get("pos_bm25_gates", []), "bm25", "pos"),
            (payload.get("neg_bm25_gates", []), "bm25", "neg"),
            (payload.get("pos_emb_gates", []), "emb", "pos"),
            (payload.get("neg_emb_gates", []), "emb", "neg"),
        ]:
            for g in gate_list:
                query = " ".join(g) if isinstance(g, list) else str(g)
                self.gate_library.add_gate(
                    self._make_gate([primitive_id], modality, polarity, query, created_by)
                )

    def _ensure_min_gates(self) -> None:
        for p in self.primitives:
            for modality in ("bm25", "emb"):
                for polarity in ("pos", "neg"):
                    existing = self.gate_library.by_filter(p.primitive_id, modality, polarity)
                    if len(existing) >= 2:
                        continue
                    # Add a couple baseline-derived gates if missing.
                    query = p.clause_quote or " ".join(_tokenize(p.clause_quote))
                    self.gate_library.add_gate(
                        self._make_gate([p.primitive_id], modality, polarity, query, "baseline")
                    )
                    self.gate_library.add_gate(
                        self._make_gate(
                            [p.primitive_id],
                            modality,
                            polarity,
                            f"{query} context",
                            "baseline",
                        )
                    )

    def _make_gate(
        self,
        primitive_ids: List[str],
        modality: str,
        polarity: str,
        query: str,
        created_by: str,
    ) -> Gate:
        base = f"{modality}:{polarity}:{_normalize_query(query)}:{','.join(sorted(primitive_ids))}"
        gate_id = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
        return Gate(
            gate_id=gate_id,
            primitive_ids=primitive_ids,
            modality=modality,
            polarity=polarity,
            query=query,
            created_by=created_by,
        )

    # ---------------------------------------------------------------------
    # Gate scanning
    # ---------------------------------------------------------------------

    def scan_gates(self) -> None:
        review_count = len(self.review_pool)
        score_store = ScoreStore(review_count, self.cache_dir / "gate_scores")
        bm25 = BM25Index(
            [r["text"] for r in self.review_pool],
            self.config.bm25_k1,
            self.config.bm25_b,
        )
        emb_store = EmbeddingStore(self.cache_dir / "embeddings", model=self.config.embedding_model)
        review_embeddings = emb_store.load_or_compute(self.review_pool, self.llm)

        for gate in self.gate_library.list():
            if score_store.load_gate_scores(gate.gate_id) is not None:
                continue
            if gate.modality == "bm25":
                scores = bm25.score(gate.query)
            else:
                query_emb = emb_store.embed_query(gate.query, self.llm)
                q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
                r = review_embeddings / (
                    np.linalg.norm(review_embeddings, axis=1, keepdims=True) + 1e-9
                )
                scores = r.dot(q)
            score_store.add_gate_scores(gate.gate_id, scores)

        score_store.recompute_for_primitives(
            [p.primitive_id for p in self.primitives],
            self.gate_library,
            self.config.num_bins,
            self.config.gamma,
        )
        self.score_store = score_store

    # ---------------------------------------------------------------------
    # Calibration and scheduling
    # ---------------------------------------------------------------------

    def _recompute_calibration(self) -> None:
        if not self.score_store:
            return
        for p in self.primitives:
            bins = self.score_store.z_bins[p.primitive_id]
            tags = self.tag_store.records_for_primitive(p.primitive_id)
            self.calibration.recompute(p.primitive_id, bins, tags)

    def _rule_satisfied(self, counts: Dict[str, int], rule: CompiledRule) -> bool:
        if rule.connective == "ALL":
            return all(counts.get(pid, 0) >= self._min_count(pid) for pid in rule.primitive_ids)
        return any(counts.get(pid, 0) >= self._min_count(pid) for pid in rule.primitive_ids)

    def _min_count(self, primitive_id: str) -> int:
        for p in self.primitives:
            if p.primitive_id == primitive_id:
                return p.min_count
        return 1

    def _evaluate_verdict(self, counts: Dict[str, int]) -> Optional[str]:
        for label in self.label_order:
            if label == self.default_label:
                continue
            rule = next((r for r in self.rules if r.label == label), None)
            if rule and self._rule_satisfied(counts, rule):
                return label
        return None

    def _compute_default_bound(
        self,
        restaurant_review_indices: List[int],
        counts: Dict[str, int],
    ) -> float:
        if not self.score_store:
            return 1.0
        rho_total = 0.0
        for rule in self.rules:
            if rule.label == self.default_label:
                continue
            per_rule = []
            for pid in rule.primitive_ids:
                deficit = max(1, self._min_count(pid) - counts.get(pid, 0))
                bins = self.score_store.z_bins[pid]
                u = self.calibration.upper_bound.get(pid)
                if u is None:
                    u = np.ones(self.config.num_bins, dtype=np.float64)
                mu = 0.0
                for idx in restaurant_review_indices:
                    if idx in self._tagged_reviews(pid):
                        continue
                    mu += u[int(bins[idx])]
                per_rule.append(min(1.0, mu / float(deficit)))
            if rule.connective == "ALL":
                rho_rule = min(per_rule) if per_rule else 1.0
            else:
                rho_rule = min(1.0, sum(per_rule))
            rho_total += rho_rule
        return min(1.0, rho_total)

    def _tagged_reviews(self, primitive_id: str) -> set[int]:
        tagged = set()
        for record in self.tag_store.records_for_primitive(primitive_id):
            idx = record.fields.get("_review_index")
            if idx is not None:
                tagged.add(int(idx))
        return tagged

    def _select_batch(
        self,
        active_restaurants: List[int],
        per_restaurant_counts: Dict[int, Dict[str, int]],
        explore_frac: float,
        max_pairs: int,
    ) -> List[Tuple[int, str]]:
        if not self.score_store:
            return []
        explore_frac = min(max(explore_frac, 0.0), 1.0)
        candidates: List[Tuple[float, int, str]] = []
        for ridx in active_restaurants:
            review_indices = self._restaurant_review_indices(ridx)
            counts = per_restaurant_counts.get(ridx, {})
            for p in self.primitives:
                deficit = max(0, p.min_count - counts.get(p.primitive_id, 0))
                if deficit <= 0:
                    continue
                z = self.score_store.z_scores[p.primitive_id]
                for idx in review_indices:
                    if idx in self._tagged_reviews(p.primitive_id):
                        continue
                    priority = z[idx] * (1.0 + deficit)
                    candidates.append((priority, idx, p.primitive_id))
        if not candidates:
            return []
        candidates.sort(key=lambda x: x[0], reverse=True)
        n_explore = int(max_pairs * explore_frac)
        n_exploit = max(0, max_pairs - n_explore)
        selected = [(idx, pid) for _, idx, pid in candidates[:n_exploit]]
        if n_explore > 0:
            remaining = candidates[n_exploit:]
            if remaining:
                sample = self.rng.sample(remaining, k=min(n_explore, len(remaining)))
                selected.extend([(idx, pid) for _, idx, pid in sample])
        return selected[:max_pairs]

    def _restaurant_review_indices(self, restaurant_index: int) -> List[int]:
        return self._restaurant_to_reviews.get(restaurant_index, [])

    # ---------------------------------------------------------------------
    # Verifier + discovery
    # ---------------------------------------------------------------------

    async def _verify_batch(self, batch: List[Tuple[int, str]]) -> None:
        tasks = []
        for review_idx, primitive_id in batch:
            review = self.review_pool[review_idx]
            primitive = next(p for p in self.primitives if p.primitive_id == primitive_id)
            cached = self.tag_store.get(
                review["review_id"], primitive_id, review["review_text_hash"]
            )
            if cached:
                continue
            prompt = build_verifier_prompt(review["text"], primitive.__dict__, self.term_defs)
            tasks.append(self._call_verifier(prompt, review, primitive, review_idx))
        if not tasks:
            return
        results = await gather_with_concurrency(self.config.max_concurrent, tasks)
        for record in results:
            if record:
                self.tag_store.add(record)

    async def _call_verifier(
        self,
        prompt: str,
        review: Dict[str, Any],
        primitive: Primitive,
        review_idx: int,
    ) -> Optional[TagRecord]:
        response, usage = await self.llm.call_async_with_usage(
            [{"role": "user", "content": prompt}],
            context={"phase": "verifier", "policy_id": self.policy_id},
        )
        self._usage_records.append(usage)
        try:
            data = parse_json_safely(extract_json_from_response(response))
        except Exception:
            data = {}
        is_match = bool(data.get("is_match"))
        evidence_snippets = data.get("evidence_snippets") or []
        fields = data.get("fields") or {}
        fields["_review_index"] = review_idx
        return TagRecord(
            review_id=review["review_id"],
            primitive_id=primitive.primitive_id,
            review_text_hash=review["review_text_hash"],
            y=1 if is_match else 0,
            evidence_snippets=evidence_snippets,
            fields=fields,
            usage=usage,
            created_at=_now_iso(),
        )

    async def _gate_discover(self) -> None:
        if self.config.gate_discover_period <= 0:
            return
        for p in self.primitives:
            records = self.tag_store.records_for_primitive(p.primitive_id)
            positives = [
                r.evidence_snippets[0]
                for r in records
                if r.y == 1 and r.evidence_snippets
            ]
            negatives = []
            if self.score_store and p.primitive_id in self.score_store.z_scores:
                z = self.score_store.z_scores[p.primitive_id]
                thresh = float(np.quantile(z, 0.7))
                for r in records:
                    if r.y == 0 and r.evidence_snippets:
                        review_idx = r.fields.get("_review_index")
                        if review_idx is not None and z[int(review_idx)] >= thresh:
                            negatives.append(r.evidence_snippets[0])
            else:
                negatives = [
                    r.evidence_snippets[0]
                    for r in records
                    if r.y == 0 and r.evidence_snippets
                ]
            if len(positives) < 1 or len(negatives) < 1:
                continue
            prompt = build_gate_discover_prompt(p.__dict__, positives[:5], negatives[:5])
            response, usage = await self.llm.call_async_with_usage(
                [{"role": "user", "content": prompt}],
                context={"phase": "gate_discover", "policy_id": self.policy_id},
            )
            self._usage_records.append(usage)
            try:
                data = parse_json_safely(extract_json_from_response(response))
            except Exception:
                data = {}
            self._add_gates_from_payload(data, created_by="llm_discover")

    # ---------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------

    async def run(self) -> Dict[str, Any]:
        await self.initialize_gates()
        self.scan_gates()
        self._recompute_calibration()

        active_restaurants = list(range(len(self.restaurants)))
        per_restaurant_counts: Dict[int, Dict[str, int]] = {}
        restaurant_verdicts: Dict[int, Dict[str, Any]] = {}

        iteration = 0
        while active_restaurants:
            iteration += 1
            restaurant_batch = active_restaurants[: self.config.batch_size]
            batch = self._select_batch(
                restaurant_batch,
                per_restaurant_counts,
                self.config.explore_frac,
                self.config.verifier_batch_size,
            )
            if not batch:
                break
            await self._verify_batch(batch)
            self._recompute_calibration()

            # Recompute counts to avoid double counting
            per_restaurant_counts = {}
            for record in self.tag_store._records.values():
                ridx = self._restaurant_index_for_review(record.review_id)
                if ridx is None:
                    continue
                per_restaurant_counts.setdefault(ridx, {})
                per_restaurant_counts[ridx][record.primitive_id] = (
                    per_restaurant_counts[ridx].get(record.primitive_id, 0) + record.y
                )

            # Check stopping for each restaurant
            still_active = []
            for ridx in active_restaurants:
                counts = per_restaurant_counts.get(ridx, {})
                verdict = self._evaluate_verdict(counts)
                if verdict:
                    restaurant_verdicts[ridx] = {
                        "verdict": verdict,
                        "stop_reason": "rule_satisfied",
                    }
                    continue
                review_indices = self._restaurant_review_indices(ridx)
                rho = self._compute_default_bound(review_indices, counts)
                if rho <= self.config.epsilon:
                    restaurant_verdicts[ridx] = {
                        "verdict": self.default_label or "Default",
                        "stop_reason": "bound",
                        "rho": rho,
                    }
                    continue
                still_active.append(ridx)
            active_restaurants = still_active

            if (
                self.config.gate_discover_period > 0
                and iteration % self.config.gate_discover_period == 0
            ):
                await self._gate_discover()
                # Only scan new gates
                if self.score_store:
                    self.scan_gates()
                    self._recompute_calibration()

        # Any remaining restaurants default out
        for ridx in active_restaurants:
            restaurant_verdicts[ridx] = {
                "verdict": self.default_label or "Default",
                "stop_reason": "exhausted",
            }

        results = []
        for idx, restaurant in enumerate(self.restaurants):
            business = restaurant.get("business", {})
            biz_id = business.get("business_id", "")
            verdict = restaurant_verdicts.get(idx, {}).get("verdict", self.default_label)
            result = {
                "business_id": biz_id,
                "name": business.get("name", biz_id),
                "verdict": verdict,
                "risk_score": None,
                "response": json.dumps(
                    {
                        "verdict": verdict,
                        "stop_reason": restaurant_verdicts.get(idx, {}).get("stop_reason"),
                        "policy_id": self.policy_id,
                    }
                ),
                "parsed": {
                    "verdict": verdict,
                },
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "latency_ms": 0.0,
                "llm_calls": 0,
            }
            results.append(result)

        aggregated_usage = self._aggregate_usage()
        self.tag_store.save()

        return {
            "results": results,
            "aggregated_usage": aggregated_usage,
            "meta": {
                "policy_id": self.policy_id,
                "iterations": iteration,
                "gates": len(self.gate_library.list()),
                "primitives": len(self.primitives),
                "default_label": self.default_label,
            },
        }

    def _aggregate_usage(self) -> Dict[str, Any]:
        total_prompt = sum(u.get("prompt_tokens", 0) for u in self._usage_records)
        total_completion = sum(u.get("completion_tokens", 0) for u in self._usage_records)
        total_cost = sum(u.get("cost_usd", 0.0) for u in self._usage_records)
        total_latency = sum(u.get("latency_ms", 0.0) for u in self._usage_records)
        return {
            "total_llm_calls": len(self._usage_records),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_cost_usd": total_cost,
            "total_latency_ms": total_latency,
        }

    def _restaurant_index_for_review(self, review_id: str) -> Optional[int]:
        return self._review_to_restaurant.get(review_id)
