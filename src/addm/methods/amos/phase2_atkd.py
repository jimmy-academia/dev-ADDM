"""Phase 2 ATKD (Active Test-time Knowledge Discovery) engine."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from addm.llm import LLMService
from addm.utils.async_utils import gather_with_concurrency
from addm.utils.output import output

from .phase1_helpers import extract_json_from_response, parse_json_safely
from .phase2_prompts import (
    build_gate_discover_prompt,
    build_gate_init_prompt,
    build_extract_evident_prompt,
)
from .phase2_debug import (
    dump_state as debug_dump_state,
    log_event as debug_log_event,
    pause as debug_pause,
    print_llm_gate_payload,
)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "our",
    "the",
    "their",
    "they",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _extract_phrases(text: str, max_tokens: int = 6) -> List[str]:
    phrases: List[str] = []
    if not text:
        return phrases
    # Quoted phrases
    for quoted in re.findall(r"\"([^\"]+)\"", text):
        phrases.append(quoted.strip())
    for quoted in re.findall(r"'([^']+)'", text):
        phrases.append(quoted.strip())
    # Parenthetical examples
    for paren in re.findall(r"\(([^)]+)\)", text):
        parts = re.split(r",|;|/|\bor\b|\band\b", paren, flags=re.IGNORECASE)
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                phrases.append(cleaned)
    # e.g. / including snippets
    for m in re.findall(r"(?:e\.g\.|including)\s+([^.;]+)", text, flags=re.IGNORECASE):
        parts = re.split(r",|;|/|\bor\b|\band\b", m, flags=re.IGNORECASE)
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                phrases.append(cleaned)
    # Token-level fallbacks
    tokens = [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 2]
    for t in tokens[:max_tokens]:
        phrases.append(t)
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for p in phrases:
        key = p.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(p.strip())
    return uniq


def _text_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_query(query: Any) -> str:
    if isinstance(query, list):
        joined = " ".join(str(q) for q in query)
    else:
        joined = str(query or "")
    return re.sub(r"\s+", " ", joined.strip().lower())


def _contains_any_anchor(text: str, anchors: List[str]) -> bool:
    if not anchors:
        return True
    hay = _normalize_query(text)
    if not hay:
        return False
    for a in anchors:
        needle = _normalize_query(a)
        if needle and needle in hay:
            return True
    return False


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    n = len(scores)
    if n == 0:
        return scores
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]

    # Tie-aware ranks: equal scores get the same (average) rank.
    # Without this, large tie groups (e.g., many zeros) produce arbitrary ordering
    # and corrupt Z.
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        # Note: scores are floats, but ties here are typically exact (BM25 zeros).
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j)
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

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
    label: str
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
    query: Any
    created_by: str
    created_iter: int = 0


@dataclass
class EvidentRecord:
    review_id: str
    business_id: str
    review_text_hash: str
    prompt_version: str
    review_index: int
    evident: Dict[str, Any]
    tags: Dict[str, int]
    bins: Dict[str, int]
    evidence: Dict[str, Any]
    usage: Dict[str, Any]
    created_at: str


@dataclass
class ATKDConfig:
    epsilon: float = 0.01
    delta: float = 0.05
    gate_init: bool = True
    gate_discover_period: int = 5  # legacy alias
    gate_discover_every: int = 5
    num_gates_suggest: int = 5
    explore_frac: float = 0.1
    batch_size: int = 100  # restaurants per driver iteration
    verifier_batch_size: int = 32
    num_bins: int = 10
    gamma: float = 1.0
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    embedding_model: str = "text-embedding-3-large"
    max_concurrent: int = 32
    pause: bool = False
    lambda_H: float = 1.0
    lambda_L: float = 1.0
    lambda_G: float = 1.0
    stall_iters: int = 3
    min_bin_coverage: int = 5
    prompt_version: str = "extract_evident_v1"
    dataset_tag: str = "unknown"


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
        self._records: Dict[Tuple[str, str, str], EvidentRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path) as f:
                for line in f:
                    data = json.loads(line)
                    record = EvidentRecord(**data)
                    key = (record.review_id, record.review_text_hash, record.prompt_version)
                    self._records[key] = record
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            for record in self._records.values():
                f.write(json.dumps(record.__dict__) + "\n")

    def get(
        self,
        review_id: str,
        review_text_hash: str,
        prompt_version: str,
    ) -> Optional[EvidentRecord]:
        return self._records.get((review_id, review_text_hash, prompt_version))

    def add(self, record: EvidentRecord) -> None:
        key = (record.review_id, record.review_text_hash, record.prompt_version)
        if key in self._records:
            return
        self._records[key] = record

    def records(self) -> List[EvidentRecord]:
        return list(self._records.values())

    def records_for_primitive(self, primitive_id: str) -> List[EvidentRecord]:
        return [r for r in self._records.values() if primitive_id in r.tags]


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
            # IMPORTANT: Treat multi-token BM25 gates as an AND-group.
            # This prevents high false positives from generic single-token hits
            # (e.g., "emergency room" matching any review that contains only "room").
            if len(tokens) > 1 and any(t not in tf for t in tokens):
                continue
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
    def __init__(
        self,
        cache_root: Path,
        model: str = "text-embedding-3-large",
        dataset_tag: str = "unknown",
    ) -> None:
        self.cache_root = cache_root
        self.model = model
        self.dataset_tag = dataset_tag
        self._review_embeddings: Optional[np.ndarray] = None
        self._review_ids: List[str] = []
        self._review_hashes: List[str] = []
        self._query_cache: Dict[str, np.ndarray] = {}
        self._last_usage: Dict[str, Any] = {}
        self._active_cache_dir: Optional[Path] = None

    def _cache_dir_for(self, review_ids: List[str], review_hashes: List[str]) -> Path:
        payload = json.dumps(
            {"model": self.model, "review_ids": review_ids, "review_hashes": review_hashes},
            separators=(",", ":"),
        )
        signature = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        return self.cache_root / self.dataset_tag / self.model / signature

    def _meta_path(self, cache_dir: Path) -> Path:
        return cache_dir / "embedding_meta.json"

    def _emb_path(self, cache_dir: Path) -> Path:
        return cache_dir / "review_embeddings.npy"

    def load_or_compute(self, reviews: List[Dict[str, Any]], llm: LLMService) -> np.ndarray:
        review_ids = [r["review_id"] for r in reviews]
        review_hashes = [r["review_text_hash"] for r in reviews]
        cache_dir = self._cache_dir_for(review_ids, review_hashes)
        self._active_cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._meta_path(cache_dir)
        emb_path = self._emb_path(cache_dir)
        if emb_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if (
                    meta.get("review_ids") == review_ids
                    and meta.get("review_hashes") == review_hashes
                    and meta.get("model") == self.model
                ):
                    self._review_embeddings = np.load(emb_path)
                    self._review_ids = review_ids
                    self._review_hashes = review_hashes
                    return self._review_embeddings
            except (OSError, json.JSONDecodeError):
                pass

        texts = [r["text"] for r in reviews]
        embeddings = self._embed_texts(texts, llm, track_usage=True)
        self._review_embeddings = embeddings
        self._review_ids = review_ids
        self._review_hashes = review_hashes
        meta_payload = {
            "review_ids": review_ids,
            "review_hashes": review_hashes,
            "model": self.model,
            "dataset_tag": self.dataset_tag,
            "created_at": _now_iso(),
            "review_count": len(review_ids),
            "total_text_chars": int(sum(len(t) for t in texts)),
            "usage": self._last_usage,
        }
        meta_path.write_text(
            json.dumps(
                meta_payload
            )
        )
        np.save(emb_path, embeddings)
        return embeddings

    def embed_query(self, query: str, llm: LLMService) -> np.ndarray:
        key = _text_hash(query)
        if key in self._query_cache:
            return self._query_cache[key]
        emb = self._embed_texts([query], llm, show_progress=False, track_usage=False)[0]
        self._query_cache[key] = emb
        return emb

    def embed_queries(self, queries: List[str], llm: LLMService) -> Dict[str, np.ndarray]:
        if not queries:
            return {}
        # Uses OpenAI embeddings API via LLMService config.
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for embeddings") from exc

        client = openai.OpenAI() if hasattr(openai, "OpenAI") else openai.Client()
        batch_size = 128
        missing_queries: List[str] = []
        missing_keys: List[str] = []
        for q in queries:
            key = _text_hash(q)
            if key in self._query_cache:
                continue
            missing_queries.append(q)
            missing_keys.append(key)

        if missing_queries:
            total_batches = (len(missing_queries) + batch_size - 1) // batch_size
            embeddings: List[List[float]] = []
            with output.progress("Embedding gate-queries") as progress:
                task_id = progress.add_task("Embedding gate-queries", total=total_batches)
                for i in range(0, len(missing_queries), batch_size):
                    batch = missing_queries[i:i + batch_size]
                    response = client.embeddings.create(model=self.model, input=batch)
                    embeddings.extend([item.embedding for item in response.data])
                    progress.advance(task_id, 1)
            for key, emb in zip(missing_keys, embeddings):
                self._query_cache[key] = np.array(emb, dtype=np.float64)

        return {q: self._query_cache[_text_hash(q)] for q in queries}

    def _embed_texts(
        self,
        texts: List[str],
        llm: LLMService,
        *,
        show_progress: bool = True,
        track_usage: bool = False,
    ) -> np.ndarray:
        # Uses OpenAI embeddings API via LLMService config.
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for embeddings") from exc

        client = openai.OpenAI() if hasattr(openai, "OpenAI") else openai.Client()
        embeddings: List[List[float]] = []
        batch_size = 128
        total_batches = (len(texts) + batch_size - 1) // batch_size
        usage_total_tokens = 0
        usage_requests = 0
        if show_progress and total_batches > 1:
            with output.progress("Embedding batches") as progress:
                task_id = progress.add_task("Embedding reviews", total=total_batches)
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = client.embeddings.create(model=self.model, input=batch)
                    usage_requests += 1
                    usage = getattr(response, "usage", None)
                    if usage and getattr(usage, "total_tokens", None) is not None:
                        usage_total_tokens += int(usage.total_tokens)
                    embeddings.extend([item.embedding for item in response.data])
                    progress.advance(task_id, 1)
        else:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = client.embeddings.create(model=self.model, input=batch)
                usage_requests += 1
                usage = getattr(response, "usage", None)
                if usage and getattr(usage, "total_tokens", None) is not None:
                    usage_total_tokens += int(usage.total_tokens)
                embeddings.extend([item.embedding for item in response.data])
        if track_usage:
            self._last_usage = {
                "total_tokens": usage_total_tokens or None,
                "requests": usage_requests,
                "batch_size": batch_size,
                "total_batches": total_batches,
            }
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
        self.alpha: Dict[str, np.ndarray] = {}
        self.beta: Dict[str, np.ndarray] = {}

    def recompute(self, primitive_id: str, records: List[EvidentRecord]) -> None:
        alpha0, beta0 = 1.0, 1.0
        alpha = np.full(self.num_bins, alpha0, dtype=np.float64)
        beta = np.full(self.num_bins, beta0, dtype=np.float64)
        for record in records:
            bin_idx = record.bins.get(primitive_id)
            if bin_idx is None:
                continue
            if bin_idx < 0 or bin_idx >= self.num_bins:
                continue
            y = record.tags.get(primitive_id, 0)
            if y > 0:
                alpha[bin_idx] += 1.0
            else:
                beta[bin_idx] += 1.0
        theta = alpha / (alpha + beta)
        u = np.zeros(self.num_bins, dtype=np.float64)
        for i in range(self.num_bins):
            u[i] = beta_ppf(1.0 - self.delta, alpha[i], beta[i])
        self.theta_hat[primitive_id] = theta
        self.upper_bound[primitive_id] = u
        self.alpha[primitive_id] = alpha
        self.beta[primitive_id] = beta


class ATKDEngine:
    def __init__(
        self,
        policy_id: str,
        agenda_spec: Dict[str, Any],
        agenda_text: str,
        restaurants: List[Dict[str, Any]],
        llm: LLMService,
        config: ATKDConfig,
        cache_dir: Path,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.policy_id = policy_id
        self.agenda_spec = agenda_spec
        self.agenda_text = agenda_text
        self.restaurants = restaurants
        self.llm = llm
        self.config = config
        self.cache_dir = cache_dir
        self.rng = random.Random(rng_seed)
        self._step_prefix = ""
        self._current_iter = 0
        self._last_batch_debug: Dict[str, Any] = {}

        (
            self.primitives,
            self.rules,
            self.default_label,
            self.label_order,
        ) = self._compile_primitives()
        self.term_defs = self._build_term_defs()
        self.overview_text = (
            self.agenda_spec.get("overview", {}).get("text", "") if self.agenda_spec else ""
        )
        self.topic_anchors: List[str] = []
        self._topic_anchor_tokens: set[str] = set()
        self._topic_mask: Optional[np.ndarray] = None
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

    def _set_step_prefix(self, prefix: str) -> None:
        self._step_prefix = prefix

    def _status(self, message: str) -> None:
        if self._step_prefix:
            output.status(f"{self._step_prefix} - {message}")
        else:
            output.status(message)

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
                        label=label,
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
        if not self.config.gate_init:
            output.warn("GateInit disabled and baseline gates removed; no gates will be available.")
            return

        primitives_payload = [
            {
                "primitive_id": p.primitive_id,
                "conditions": p.conditions,
            }
            for p in self.primitives
        ]
        prompt = build_gate_init_prompt(
            primitives_payload,
            self.term_defs,
            agenda_text=self.agenda_text,
            num_gates_suggest=self.config.num_gates_suggest,
        )
        self._status("GateInit: calling LLM for gate proposals")
        response, usage = await self.llm.call_async_with_usage(
            [{"role": "user", "content": prompt}],
            context={
                "phase": "gate_init",
                "policy_id": self.policy_id,
                "sample_id": self.policy_id,
            },
        )
        self._usage_records.append(usage)
        try:
            data = parse_json_safely(extract_json_from_response(response))
        except Exception:
            data = {}
        anchors = data.get("topic_anchors") or []
        if isinstance(anchors, list):
            self.topic_anchors = [str(a) for a in anchors if str(a).strip()]
        else:
            self.topic_anchors = []
        added = 0
        for item in data.get("primitives", []):
            added += self._add_gates_from_payload(
                item,
                created_by="llm_init",
                topic_anchors=self.topic_anchors,
            )
        # Build a simple lexical topic mask to suppress embedding false positives on topic-irrelevant reviews.
        # Use both the (optional) topic_anchors and the BM25 positive-gate tokens to avoid over-restricting.
        self._topic_anchor_tokens = {
            t
            for a in self.topic_anchors
            for t in _tokenize(str(a))
            if t and t not in _STOPWORDS
        }
        for g in self.gate_library.list():
            if g.modality != "bm25" or g.polarity != "pos":
                continue
            for t in _tokenize(str(g.query)):
                if t and t not in _STOPWORDS:
                    self._topic_anchor_tokens.add(t)

        if self._topic_anchor_tokens:
            mask = np.zeros(len(self.review_pool), dtype=np.float64)
            for i, r in enumerate(self.review_pool):
                toks = set(_tokenize(r.get("text", "")))
                hits = len(toks.intersection(self._topic_anchor_tokens))
                # Soft mask: 0.0 (no hits), 0.5 (one hit), 1.0 (2+ hits).
                mask[i] = min(1.0, hits / 2.0)
            self._topic_mask = mask
        else:
            self._topic_mask = None
        self._status(
            f"GateInit: llm_added={added} total_gates={len(self.gate_library.list())}"
        )
        if self.config.pause:
            print_llm_gate_payload("GateInit", data)
        debug_log_event(
            self.policy_id,
            "phase2_gate_init",
            {
                "baseline_gates": 0,
                "llm_added": added,
                "total_gates": len(self.gate_library.list()),
            },
        )

    def _value_phrases(self, field_id: str, value_id: str) -> List[str]:
        phrases: List[str] = []
        if value_id:
            phrases.append(value_id.replace("_", " "))
        for v in self.term_defs.get(field_id, []):
            if v.get("value_id") == value_id:
                desc = v.get("description", "")
                phrases.extend(_extract_phrases(desc))
                break
        # Deduplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for p in phrases:
            key = p.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        return uniq

    def _condition_phrases(self, field_id: str, values: List[str]) -> List[str]:
        phrases: List[str] = []
        for value_id in values:
            phrases.extend(self._value_phrases(field_id, value_id))
        # Deduplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for p in phrases:
            key = p.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        return uniq

    def _build_phrase_groups(self, phrase_lists: List[List[str]], max_groups: int = 3) -> List[List[str]]:
        if not phrase_lists:
            return []
        max_len = max((len(lst) for lst in phrase_lists if lst), default=0)
        if max_len == 0:
            return []
        groups: List[List[str]] = []
        for i in range(min(max_groups, max_len)):
            group: List[str] = []
            for lst in phrase_lists:
                if not lst:
                    continue
                group.append(lst[i % len(lst)])
            if group:
                groups.append(group)
        return groups

    def _phrase_sentence(self, phrases: List[str]) -> str:
        if not phrases:
            return "Review mentions a relevant incident."
        if len(phrases) == 1:
            return f"Review mentions {phrases[0]}."
        return f"Review mentions {phrases[0]} and {phrases[1]}."

    def _baseline_gates(self) -> List[Gate]:
        gates: List[Gate] = []
        for p in self.primitives:
            cond_phrase_lists: List[List[str]] = []
            neg_phrase_lists: List[List[str]] = []
            for cond in p.conditions:
                field_id = cond.get("field_id")
                values = cond.get("values", []) or []
                if not field_id:
                    continue
                cond_phrase_lists.append(self._condition_phrases(field_id, values))
                # Negative phrases: other values for same field
                alt_values = [
                    v.get("value_id")
                    for v in self.term_defs.get(field_id, [])
                    if v.get("value_id") not in values
                ]
                neg_phrase_lists.append(self._condition_phrases(field_id, alt_values))

            pos_groups = self._build_phrase_groups(cond_phrase_lists, max_groups=3)
            neg_groups = self._build_phrase_groups(neg_phrase_lists, max_groups=2)

            for group in pos_groups:
                query = " ".join(group)
                gates.append(
                    self._make_gate(
                        [p.primitive_id], "bm25", "pos", query, "baseline", created_iter=0
                    )
                )
                gates.append(
                    self._make_gate(
                        [p.primitive_id],
                        "emb",
                        "pos",
                        self._phrase_sentence(group),
                        "baseline",
                        created_iter=0,
                    )
                )
            for group in neg_groups:
                query = " ".join(group)
                gates.append(
                    self._make_gate(
                        [p.primitive_id], "bm25", "neg", query, "baseline", created_iter=0
                    )
                )
                gates.append(
                    self._make_gate(
                        [p.primitive_id],
                        "emb",
                        "neg",
                        self._phrase_sentence(group),
                        "baseline",
                        created_iter=0,
                    )
                )
        return gates

    def _add_gates_from_payload(
        self,
        payload: Dict[str, Any],
        created_by: str,
        *,
        topic_anchors: Optional[List[str]] = None,
    ) -> int:
        primitive_id = payload.get("primitive_id")
        if not primitive_id:
            return 0
        added = 0
        anchors = topic_anchors or []
        for gate_list, modality, polarity in [
            (payload.get("pos_bm25_gates", []), "bm25", "pos"),
            (payload.get("neg_bm25_gates", []), "bm25", "neg"),
            (payload.get("pos_emb_gates", []), "emb", "pos"),
            (payload.get("neg_emb_gates", []), "emb", "neg"),
        ]:
            for g in gate_list:
                query = " ".join(g) if isinstance(g, list) else str(g)
                if self.gate_library.add_gate(
                    self._make_gate(
                        [primitive_id],
                        modality,
                        polarity,
                        query,
                        created_by,
                        created_iter=self._current_iter,
                    )
                ):
                    added += 1
        return added

    def _ensure_min_gates(self) -> None:
        for p in self.primitives:
            for modality in ("bm25", "emb"):
                for polarity in ("pos", "neg"):
                    existing = self.gate_library.by_filter(p.primitive_id, modality, polarity)
                    if len(existing) >= 2:
                        continue
                    # Add fallback gates derived from term definitions.
                    fallback = "review"
                    for cond in p.conditions:
                        field_id = cond.get("field_id")
                        values = cond.get("values", []) or []
                        phrases = self._condition_phrases(field_id, values) if field_id else []
                        if phrases:
                            fallback = phrases[0]
                            break
                    if modality == "emb":
                        query = self._phrase_sentence([fallback])
                    else:
                        query = fallback
                    self.gate_library.add_gate(
                        self._make_gate(
                            [p.primitive_id],
                            modality,
                            polarity,
                            query,
                            "baseline",
                            created_iter=0,
                        )
                    )
                    self.gate_library.add_gate(
                        self._make_gate(
                            [p.primitive_id],
                            modality,
                            polarity,
                            f"{query} review",
                            "baseline",
                            created_iter=0,
                        )
                    )

    def _make_gate(
        self,
        primitive_ids: List[str],
        modality: str,
        polarity: str,
        query: Any,
        created_by: str,
        created_iter: int = 0,
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
            created_iter=created_iter,
        )

    # ---------------------------------------------------------------------
    # Gate scanning
    # ---------------------------------------------------------------------

    def scan_gates(self) -> None:
        review_count = len(self.review_pool)
        score_store = ScoreStore(review_count, self.cache_dir / "gate_scores")
        self._status("GateScan: building BM25 index")
        bm25 = BM25Index(
            [r["text"] for r in self.review_pool],
            self.config.bm25_k1,
            self.config.bm25_b,
        )
        emb_store = EmbeddingStore(
            Path("results/shared/embeddings"),
            model=self.config.embedding_model,
            dataset_tag=self.config.dataset_tag,
        )
        self._status("GateScan: loading/creating embeddings")
        review_embeddings = emb_store.load_or_compute(self.review_pool, self.llm)

        self._status("GateScan: scoring gates")
        new_gates = [
            g for g in self.gate_library.list()
            if score_store.load_gate_scores(g.gate_id) is None
        ]
        scored_new = 0
        if new_gates:
            embed_gates = [g for g in new_gates if g.modality == "emb"]
            query_map: Dict[str, np.ndarray] = {}
            r_norm = None
            if embed_gates:
                query_map = emb_store.embed_queries([g.query for g in embed_gates], self.llm)
                r_norm = review_embeddings / (
                    np.linalg.norm(review_embeddings, axis=1, keepdims=True) + 1e-9
                )
            with output.progress("Gate scoring") as progress:
                task_id = progress.add_task("Scoring gates", total=len(new_gates))
                for gate in new_gates:
                    if gate.modality == "bm25":
                        scores = bm25.score(gate.query)
                    else:
                        query_emb = query_map.get(gate.query)
                        if query_emb is None:
                            query_emb = emb_store.embed_query(gate.query, self.llm)
                        q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
                        r = r_norm if r_norm is not None else review_embeddings / (
                            np.linalg.norm(review_embeddings, axis=1, keepdims=True) + 1e-9
                        )
                        scores = r.dot(q)
                    score_store.add_gate_scores(gate.gate_id, scores)
                    scored_new += 1
                    progress.advance(task_id, 1)
        else:
            self._status("GateScan: no new gates to score")

        score_store.recompute_for_primitives(
            [p.primitive_id for p in self.primitives],
            self.gate_library,
            self.config.num_bins,
            self.config.gamma,
        )
        self.score_store = score_store
        self._status(
            f"GateScan: completed (new_scored={scored_new}, total_gates={len(self.gate_library.list())})"
        )

    # ---------------------------------------------------------------------
    # Calibration and scheduling
    # ---------------------------------------------------------------------

    def _recompute_calibration(self) -> None:
        if not self.score_store:
            return
        # Refresh bins for all verified records using current z-bins.
        for record in self.tag_store.records():
            for p in self.primitives:
                pid = p.primitive_id
                record.bins[pid] = int(self.score_store.z_bins[pid][record.review_index])
        total_tags = sum(
            len(self.tag_store.records_for_primitive(p.primitive_id)) for p in self.primitives
        )
        self._status(f"Calibration: recompute (tags={total_tags})")
        for p in self.primitives:
            records = self.tag_store.records_for_primitive(p.primitive_id)
            self.calibration.recompute(p.primitive_id, records)
        self._status("Calibration: done")

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
        verified = self._verified_review_indices()
        rho_total = 0.0
        for rule in self.rules:
            if rule.label == self.default_label:
                continue
            per_rule = []
            for pid in rule.primitive_ids:
                deficit = max(0, self._min_count(pid) - counts.get(pid, 0))
                if deficit == 0:
                    continue
                bins = self.score_store.z_bins[pid]
                u = self.calibration.upper_bound.get(pid)
                if u is None:
                    u = np.ones(self.config.num_bins, dtype=np.float64)
                mu = 0.0
                for idx in restaurant_review_indices:
                    if idx in verified:
                        continue
                    mu += u[int(bins[idx])]
                per_rule.append(min(1.0, mu / float(deficit)))
            if rule.connective == "ALL":
                rho_rule = min(per_rule) if per_rule else 0.0
            else:
                rho_rule = min(1.0, sum(per_rule)) if per_rule else 0.0
            rho_total += rho_rule
        return min(1.0, rho_total)

    def _verified_review_indices(self) -> set[int]:
        return {record.review_index for record in self.tag_store.records()}

    def _compute_counts_by_restaurant(self) -> Dict[int, Dict[str, int]]:
        counts: Dict[int, Dict[str, int]] = {}
        for record in self.tag_store.records():
            ridx = self.review_pool[record.review_index]["restaurant_index"]
            counts.setdefault(ridx, {})
            for pid, y in record.tags.items():
                if y > 0:
                    counts[ridx][pid] = counts[ridx].get(pid, 0) + int(y)
        return counts

    def _compute_delta_var(self, primitive_id: str) -> np.ndarray:
        alpha = self.calibration.alpha.get(
            primitive_id, np.ones(self.config.num_bins, dtype=np.float64)
        )
        beta = self.calibration.beta.get(
            primitive_id, np.ones(self.config.num_bins, dtype=np.float64)
        )
        denom = (alpha + beta)
        var = (alpha * beta) / (denom * denom * (denom + 1.0))
        theta = alpha / denom
        var1 = (alpha + 1.0) * beta / (
            (denom + 1.0) * (denom + 1.0) * (denom + 2.0)
        )
        var0 = alpha * (beta + 1.0) / (
            (denom + 1.0) * (denom + 1.0) * (denom + 2.0)
        )
        e_var_next = theta * var1 + (1.0 - theta) * var0
        return np.maximum(0.0, var - e_var_next)

    def _select_batch(
        self,
        active_restaurants: List[int],
        per_restaurant_counts: Dict[int, Dict[str, int]],
        explore_frac: float,
        max_reviews: int,
        *,
        stall_mode: bool = False,
        rho_by_restaurant: Optional[Dict[int, float]] = None,
    ) -> List[int]:
        if not self.score_store:
            return []
        explore_frac = min(max(explore_frac, 0.0), 1.0)
        verified = self._verified_review_indices()

        # Build bin counts c[i,p,b] and global C[p,b]
        c: Dict[int, Dict[str, np.ndarray]] = {}
        C: Dict[str, np.ndarray] = {p.primitive_id: np.zeros(self.config.num_bins) for p in self.primitives}
        review_candidates: List[Tuple[int, int]] = []
        rest_unverified: Dict[int, int] = {}
        for ridx in active_restaurants:
            review_indices = self._restaurant_review_indices(ridx)
            c.setdefault(ridx, {p.primitive_id: np.zeros(self.config.num_bins) for p in self.primitives})
            for review_idx in review_indices:
                if review_idx in verified:
                    continue
                review_candidates.append((ridx, review_idx))
                rest_unverified[ridx] = rest_unverified.get(ridx, 0) + 1
                for p in self.primitives:
                    pid = p.primitive_id
                    b = int(self.score_store.z_bins[pid][review_idx])
                    c[ridx][pid][b] += 1
                    C[pid][b] += 1

        if not review_candidates:
            return []

        # Precompute delta variance per primitive/bin
        delta_var = {p.primitive_id: self._compute_delta_var(p.primitive_id) for p in self.primitives}

        # Severity weights by label rank
        label_rank = {label: idx for idx, label in enumerate(self.label_order)}
        min_rank = min(label_rank.get(p.label, 999) for p in self.primitives)
        top_labels = {p.label for p in self.primitives if label_rank.get(p.label, 999) == min_rank}
        w_p = {
            p.primitive_id: math.exp(-label_rank.get(p.label, 999))
            for p in self.primitives
        }

        # Candidate scores
        total_unverified = max(len(review_candidates), 1)
        scores: Dict[int, float] = {}
        components: Dict[int, Dict[str, Any]] = {}
        for ridx, review_idx in review_candidates:
            counts = per_restaurant_counts.get(ridx, {})
            v_hunt = 0.0
            v_local = 0.0
            v_global = 0.0
            rest_den = max(rest_unverified.get(ridx, 0), 1)
            for p in self.primitives:
                pid = p.primitive_id
                b = int(self.score_store.z_bins[pid][review_idx])
                if p.label in top_labels:
                    deficit = max(0, p.min_count - counts.get(pid, 0))
                    if deficit > 0:
                        # Hunt uses Z directly so it works before calibration has any supervision.
                        z_val = float(self.score_store.z_scores[pid][review_idx])
                        v_hunt = max(v_hunt, max(0.0, z_val) * w_p[pid])
                # Scale local/global VOI by proportions to keep score magnitudes stable across dataset sizes.
                v_local += (c[ridx][pid][b] / rest_den) * delta_var[pid][b]
                v_global += (C[pid][b] / total_unverified) * delta_var[pid][b]
            score = (
                self.config.lambda_H * v_hunt
                + self.config.lambda_L * v_local
                + self.config.lambda_G * v_global
            )
            if self._topic_mask is not None:
                topic_weight = 0.1 + 0.9 * float(self._topic_mask[review_idx])
                score *= topic_weight
            scores[review_idx] = score
            best_pid = None
            best_z = -1e9
            for p in self.primitives:
                pid = p.primitive_id
                z_val = float(self.score_store.z_scores[pid][review_idx])
                if z_val > best_z:
                    best_z = z_val
                    best_pid = pid
            components[review_idx] = {
                "restaurant_index": ridx,
                "v_hunt": float(v_hunt),
                "v_local": float(v_local),
                "v_global": float(v_global),
                "score": float(score),
                "topic_weight": float(topic_weight) if self._topic_mask is not None else 1.0,
                "top_pid": best_pid,
                "top_z": float(best_z),
                "top_bin": int(self.score_store.z_bins[best_pid][review_idx]) if best_pid else None,
            }

        # Stall mode: focus closeout on restaurant with smallest rho_i
        selected: List[int] = []
        selected_set: set[int] = set()
        stage: Dict[int, str] = {}
        if stall_mode and rho_by_restaurant:
            focus_ridx = min(rho_by_restaurant.items(), key=lambda x: x[1])[0]
            focus_reviews = [
                review_idx
                for ridx, review_idx in review_candidates
                if ridx == focus_ridx
            ]
            focus_reviews.sort(key=lambda r: scores.get(r, 0.0), reverse=True)
            closeout_n = max(1, max_reviews // 4)
            for review_idx in focus_reviews[:closeout_n]:
                selected.append(review_idx)
                selected_set.add(review_idx)
                stage.setdefault(review_idx, "stall_closeout")

        # Coverage: include top (p,b) by C[p,b]*DeltaVar
        coverage_added = 0
        if self.config.min_bin_coverage > 0:
            bin_scores: List[Tuple[float, str, int]] = []
            for p in self.primitives:
                pid = p.primitive_id
                for b in range(self.config.num_bins):
                    bin_scores.append((C[pid][b] * delta_var[pid][b], pid, b))
            bin_scores.sort(key=lambda x: x[0], reverse=True)
            for _, pid, b in bin_scores:
                if coverage_added >= self.config.min_bin_coverage:
                    break
                # Pick a review with bin b for primitive pid
                candidates = [
                    review_idx
                    for ridx, review_idx in review_candidates
                    if review_idx not in selected_set
                    and int(self.score_store.z_bins[pid][review_idx]) == b
                ]
                if not candidates:
                    continue
                candidates.sort(key=lambda r: scores.get(r, 0.0), reverse=True)
                selected.append(candidates[0])
                selected_set.add(candidates[0])
                coverage_added += 1
                stage.setdefault(candidates[0], "coverage")

        # Exploit: fill remaining with top scores
        n_explore = int(max_reviews * explore_frac)
        n_exploit = max(0, max_reviews - n_explore)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for review_idx, _ in ranked:
            if len(selected) >= n_exploit:
                break
            if review_idx in selected_set:
                continue
            selected.append(review_idx)
            selected_set.add(review_idx)
            stage.setdefault(review_idx, "exploit")

        # Explore: stratified by low-coverage bins
        if n_explore > 0 and len(selected) < max_reviews:
            bin_scores = []
            for p in self.primitives:
                pid = p.primitive_id
                for b in range(self.config.num_bins):
                    bin_scores.append((C[pid][b], pid, b))
            bin_scores.sort(key=lambda x: x[0])  # low coverage first
            for _, pid, b in bin_scores:
                if len(selected) >= max_reviews:
                    break
                candidates = [
                    review_idx
                    for ridx, review_idx in review_candidates
                    if review_idx not in selected_set
                    and int(self.score_store.z_bins[pid][review_idx]) == b
                ]
                if not candidates:
                    continue
                pick = self.rng.choice(candidates)
                selected.append(pick)
                selected_set.add(pick)
                stage.setdefault(pick, "explore")

        selected_scores = [scores.get(r, 0.0) for r in selected]
        top_pid_counts: Dict[str, int] = {}
        for r in selected:
            pid = components.get(r, {}).get("top_pid")
            if pid:
                top_pid_counts[pid] = top_pid_counts.get(pid, 0) + 1
        summary = {
            "candidates": len(review_candidates),
            "selected": len(selected),
            "coverage_added": coverage_added,
            "explore_target": n_explore,
            "exploit_target": n_exploit,
            "stall_mode": stall_mode,
            "stall_focus_ridx": min(rho_by_restaurant.items(), key=lambda x: x[1])[0]
            if stall_mode and rho_by_restaurant
            else None,
            "score_min": float(min(selected_scores)) if selected_scores else 0.0,
            "score_mean": float(sum(selected_scores) / len(selected_scores)) if selected_scores else 0.0,
            "score_max": float(max(selected_scores)) if selected_scores else 0.0,
            "top_pid_counts": top_pid_counts,
        }
        self._last_batch_debug = {
            "summary": summary,
            "selected": [
                {
                    "review_index": r,
                    "restaurant_index": components.get(r, {}).get("restaurant_index"),
                    "stage": stage.get(r),
                    "v_hunt": components.get(r, {}).get("v_hunt"),
                    "v_local": components.get(r, {}).get("v_local"),
                    "v_global": components.get(r, {}).get("v_global"),
                    "topic_weight": components.get(r, {}).get("topic_weight"),
                    "score": components.get(r, {}).get("score"),
                    "top_pid": components.get(r, {}).get("top_pid"),
                    "top_bin": components.get(r, {}).get("top_bin"),
                    "top_z": components.get(r, {}).get("top_z"),
                }
                for r in selected
            ],
        }

        return selected[:max_reviews]

    def _restaurant_review_indices(self, restaurant_index: int) -> List[int]:
        return self._restaurant_to_reviews.get(restaurant_index, [])

    # ---------------------------------------------------------------------
    # Verifier + discovery
    # ---------------------------------------------------------------------

    async def _verify_batch(self, batch: List[int]) -> None:
        self._status(f"Verifier: preparing batch (reviews={len(batch)})")
        tasks = []
        for review_idx in batch:
            review = self.review_pool[review_idx]
            cached = self.tag_store.get(
                review["review_id"],
                review["review_text_hash"],
                self.config.prompt_version,
            )
            if cached:
                continue
            prompt = build_extract_evident_prompt(
                review["text"],
                self.term_defs,
                review["review_id"],
            )
            tasks.append(self._call_evident(prompt, review, review_idx))
        if not tasks:
            self._status("Verifier: no new tasks (all cached)")
            return
        self._status(f"Verifier: calling LLM (tasks={len(tasks)})")
        results: List[Optional[EvidentRecord]] = [None] * len(tasks)
        sem = asyncio.Semaphore(self.config.max_concurrent)

        async def runner(idx: int, coro):
            async with sem:
                return idx, await coro

        with output.progress("Verifier progress") as progress:
            task_id = progress.add_task("Verifying reviews", total=len(tasks))
            pending = [asyncio.create_task(runner(i, t)) for i, t in enumerate(tasks)]
            for fut in asyncio.as_completed(pending):
                idx, res = await fut
                results[idx] = res
                progress.advance(task_id, 1)
        added_records: List[EvidentRecord] = []
        for record in results:
            if record:
                self.tag_store.add(record)
                added_records.append(record)
        self._status(f"Verifier: completed (added={len(added_records)})")
        if added_records:
            # Log per-item evidence for inspection
            for record in added_records:
                debug_log_event(
                    self.policy_id,
                    "phase2_verifier_result",
                    {
                        "review_id": record.review_id,
                        "tags": record.tags,
                        "evidence": record.evidence,
                    },
                    sample_id=record.business_id or self.policy_id,
                )
            debug_log_event(
                self.policy_id,
                "phase2_verifier_batch",
                {
                    "batch_size": len(batch),
                    "added": len(added_records),
                },
            )

    async def _call_evident(
        self,
        prompt: str,
        review: Dict[str, Any],
        review_idx: int,
    ) -> Optional[EvidentRecord]:
        response, usage = await self.llm.call_async_with_usage(
            [{"role": "user", "content": prompt}],
            context={
                "phase": "extract_evident",
                "policy_id": self.policy_id,
                "sample_id": review.get("business_id", self.policy_id),
                "review_id": review.get("review_id"),
            },
        )
        self._usage_records.append(usage)
        try:
            data = parse_json_safely(extract_json_from_response(response))
        except Exception:
            data = {}
        evidents = data.get("evidents", [])
        if not isinstance(evidents, list):
            evidents = []
        data["evidents"] = evidents
        tags, evidence = self._derive_tags(data)
        bins = {}
        if self.score_store:
            for p in self.primitives:
                pid = p.primitive_id
                bins[pid] = int(self.score_store.z_bins[pid][review_idx])
        return EvidentRecord(
            review_id=review["review_id"],
            business_id=review.get("business_id", ""),
            review_text_hash=review["review_text_hash"],
            prompt_version=self.config.prompt_version,
            review_index=review_idx,
            evident=data,
            tags=tags,
            bins=bins,
            evidence=evidence,
            usage=usage,
            created_at=_now_iso(),
        )

    def _derive_tags(self, evident: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, Any]]:
        evidents = evident.get("evidents") or []
        tags: Dict[str, int] = {}
        evidence: Dict[str, Any] = {}
        for p in self.primitives:
            matched_events: List[Dict[str, Any]] = []
            count = 0
            for event in evidents:
                fields = event.get("fields", {}) or {}
                if self._event_matches(p, fields):
                    count += 1
                    matched_events.append(
                        {
                            "event_id": event.get("event_id"),
                            "snippet": event.get("snippet"),
                            "fields": event.get("fields", {}),
                            "span": event.get("span"),
                        }
                    )
            tags[p.primitive_id] = count
            if matched_events:
                evidence[p.primitive_id] = matched_events
        return tags, evidence

    def _event_matches(self, primitive: Primitive, fields: Dict[str, Any]) -> bool:
        if not primitive.conditions:
            return False
        results = []
        for cond in primitive.conditions:
            field_id = cond.get("field_id")
            allowed = cond.get("values", [])
            if not field_id:
                continue
            value = fields.get(field_id)
            if value is None:
                results.append(False)
                continue
            if isinstance(value, list):
                match = any(v in allowed for v in value)
            else:
                match = value in allowed
            results.append(match)
        if not results:
            return False
        if primitive.logic == "ANY":
            return any(results)
        return all(results)

    async def _gate_discover(self) -> None:
        period = self.config.gate_discover_every or self.config.gate_discover_period
        if period <= 0:
            return
        self._status("GateDiscover: starting")
        total_added = 0
        gate_discover_payloads: List[Dict[str, Any]] = []
        for p in self.primitives:
            records = self.tag_store.records_for_primitive(p.primitive_id)
            positives = []
            for r in records:
                if r.tags.get(p.primitive_id, 0) > 0:
                    ev_list = r.evidence.get(p.primitive_id, [])
                    if isinstance(ev_list, dict):
                        ev_list = [ev_list]
                    for ev in ev_list:
                        if ev.get("snippet"):
                            positives.append(ev["snippet"])
            negatives = []
            if self.score_store and p.primitive_id in self.score_store.z_scores:
                z = self.score_store.z_scores[p.primitive_id]
                thresh = float(np.quantile(z, 0.7))
                for r in records:
                    if r.tags.get(p.primitive_id, 0) == 0:
                        review_idx = r.review_index
                        if z[int(review_idx)] >= thresh:
                            snippet = self.review_pool[review_idx]["text"][:200]
                            negatives.append(snippet)
            else:
                for r in records:
                    if r.tags.get(p.primitive_id, 0) == 0:
                        snippet = self.review_pool[r.review_index]["text"][:200]
                        negatives.append(snippet)
            if len(positives) < 1 or len(negatives) < 1:
                continue
            prompt = build_gate_discover_prompt(p.__dict__, positives[:5], negatives[:5])
            response, usage = await self.llm.call_async_with_usage(
                [{"role": "user", "content": prompt}],
                context={
                    "phase": "gate_discover",
                    "policy_id": self.policy_id,
                    "sample_id": self.policy_id,
                },
            )
            self._usage_records.append(usage)
            try:
                data = parse_json_safely(extract_json_from_response(response))
            except Exception:
                data = {}
            if data:
                gate_discover_payloads.append(data)
            total_added += self._add_gates_from_payload(data, created_by="llm_discover")
        self._status(
            f"GateDiscover: done (added={total_added}, total_gates={len(self.gate_library.list())})"
        )
        if total_added > 0 and self.config.pause:
            print_llm_gate_payload("GateDiscover", {"primitives": gate_discover_payloads})
        if total_added > 0:
            debug_log_event(
                self.policy_id,
                "phase2_gate_discover",
                {
                    "added": total_added,
                    "total_gates": len(self.gate_library.list()),
                },
            )

    # ---------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------

    async def run(self) -> Dict[str, Any]:
        def _pause_step(label: str) -> None:
            debug_log_event(self.policy_id, "phase2_pause", {"label": label})
            debug_pause(label, self.config.pause)

        def _dump(label: str, **kwargs: Any) -> None:
            debug_dump_state(
                self.policy_id,
                self.cache_dir,
                self.primitives,
                self.gate_library,
                self.tag_store,
                self.score_store,
                self.calibration,
                label,
                review_pool=self.review_pool,
                last_batch_debug=self._last_batch_debug,
                **kwargs,
            )

        self._set_step_prefix("Step 1/3: GateInit")
        _pause_step("Step 1/3: GateInit")
        await self.initialize_gates()
        _dump("after_gate_init")

        self._set_step_prefix("Step 2/3: GateScan")
        _pause_step("Step 2/3: GateScan")
        self.scan_gates()
        _dump("after_gate_scan", save_arrays=True)

        self._set_step_prefix("Step 3/3: Calibration")
        _pause_step("Step 3/3: Calibration")
        self._recompute_calibration()
        _dump("after_calibration")

        active_restaurants = list(range(len(self.restaurants)))
        per_restaurant_counts: Dict[int, Dict[str, int]] = self._compute_counts_by_restaurant()
        restaurant_verdicts: Dict[int, Dict[str, Any]] = {}
        rho_by_restaurant: Dict[int, float] = {}
        stall_counter = 0

        iteration = 0
        while active_restaurants:
            iteration += 1
            self._current_iter = iteration
            debug_log_event(
                self.policy_id,
                "phase2_iteration_start",
                {"iteration": iteration, "active_restaurants": len(active_restaurants)},
            )
            restaurant_batch = active_restaurants[: self.config.batch_size]
            self._set_step_prefix(f"Iter {iteration} Step 1/4: BatchSelect")
            _pause_step(f"Iter {iteration} Step 1/4: BatchSelect")
            batch = self._select_batch(
                restaurant_batch,
                per_restaurant_counts,
                self.config.explore_frac,
                self.config.verifier_batch_size,
                stall_mode=stall_counter >= self.config.stall_iters,
                rho_by_restaurant=rho_by_restaurant,
            )
            if self._last_batch_debug:
                summary = self._last_batch_debug.get("summary", {})
                self._status(
                    "BatchSelect: "
                    f"candidates={summary.get('candidates', 0)} "
                    f"selected={summary.get('selected', 0)} "
                    f"coverage={summary.get('coverage_added', 0)} "
                    f"explore_target={summary.get('explore_target', 0)} "
                    f"score[min/mean/max]="
                    f"{summary.get('score_min', 0.0):.3f}/"
                    f"{summary.get('score_mean', 0.0):.3f}/"
                    f"{summary.get('score_max', 0.0):.3f}"
                )
            _dump("batch_selected", batch=batch, iteration=iteration)
            if not batch:
                break
            self._set_step_prefix(f"Iter {iteration} Step 2/4: Verify")
            _pause_step(f"Iter {iteration} Step 2/4: Verify")
            await self._verify_batch(batch)
            self.tag_store.save()
            _dump("after_verify", iteration=iteration)
            self._set_step_prefix(f"Iter {iteration} Step 3/4: Calibration")
            _pause_step(f"Iter {iteration} Step 3/4: Calibration")
            self._recompute_calibration()
            _dump("after_calibration_iter", iteration=iteration)

            # Recompute counts to avoid double counting
            per_restaurant_counts = self._compute_counts_by_restaurant()

            # Check stopping for each restaurant
            self._set_step_prefix(f"Iter {iteration} Step 4/4: StopCheck")
            _pause_step(f"Iter {iteration} Step 4/4: StopCheck")
            still_active = []
            finalized_now = 0
            rho_by_restaurant = {}
            for ridx in active_restaurants:
                counts = per_restaurant_counts.get(ridx, {})
                verdict = self._evaluate_verdict(counts)
                if verdict:
                    restaurant_verdicts[ridx] = {
                        "verdict": verdict,
                        "stop_reason": "rule_satisfied",
                    }
                    finalized_now += 1
                    business_id = self.restaurants[ridx].get("business", {}).get("business_id", "")
                    debug_log_event(
                        self.policy_id,
                        "phase2_stop",
                        {
                            "verdict": verdict,
                            "stop_reason": "rule_satisfied",
                            "counts": counts,
                        },
                        sample_id=business_id or self.policy_id,
                    )
                    continue
                review_indices = self._restaurant_review_indices(ridx)
                rho = self._compute_default_bound(review_indices, counts)
                rho_by_restaurant[ridx] = rho
                if rho <= self.config.epsilon:
                    restaurant_verdicts[ridx] = {
                        "verdict": self.default_label or "Default",
                        "stop_reason": "bound",
                        "rho": rho,
                    }
                    finalized_now += 1
                    business_id = self.restaurants[ridx].get("business", {}).get("business_id", "")
                    debug_log_event(
                        self.policy_id,
                        "phase2_stop",
                        {
                            "verdict": self.default_label or "Default",
                            "stop_reason": "bound",
                            "rho": rho,
                            "counts": counts,
                        },
                        sample_id=business_id or self.policy_id,
                    )
                    continue
                still_active.append(ridx)
            active_restaurants = still_active
            _dump("after_stop_check", iteration=iteration)
            if finalized_now > 0:
                stall_counter = 0
            else:
                stall_counter += 1

            if (
                (self.config.gate_discover_every or self.config.gate_discover_period) > 0
                and iteration % (self.config.gate_discover_every or self.config.gate_discover_period) == 0
            ):
                self._set_step_prefix(f"Iter {iteration} Step 5/5: GateDiscover")
                _pause_step(f"Iter {iteration} Step 5/5: GateDiscover")
                await self._gate_discover()
                # Only scan new gates
                if self.score_store:
                    self._set_step_prefix(f"Iter {iteration} Step 5/5: GateDiscover Scan")
                    _pause_step(f"Iter {iteration} Step 5/5: GateDiscover Scan")
                    self.scan_gates()
                    _dump("after_gate_scan_discover", save_arrays=True, iteration=iteration)
                    self._set_step_prefix(f"Iter {iteration} Step 5/5: GateDiscover Calibration")
                    _pause_step(f"Iter {iteration} Step 5/5: GateDiscover Calibration")
                    self._recompute_calibration()
                    _dump("after_calibration_discover", iteration=iteration)

        # Any remaining restaurants default out
        for ridx in active_restaurants:
            restaurant_verdicts[ridx] = {
                "verdict": self.default_label or "Default",
                "stop_reason": "exhausted",
            }
            business_id = self.restaurants[ridx].get("business", {}).get("business_id", "")
            debug_log_event(
                self.policy_id,
                "phase2_stop",
                {
                    "verdict": self.default_label or "Default",
                    "stop_reason": "exhausted",
                },
                sample_id=business_id or self.policy_id,
            )

        results = []
        for idx, restaurant in enumerate(self.restaurants):
            business = restaurant.get("business", {})
            biz_id = business.get("business_id", "")
            verdict = restaurant_verdicts.get(idx, {}).get("verdict", self.default_label)
            evidence_items = self._collect_evidence_for_restaurant(idx, verdict)
            stop_reason = restaurant_verdicts.get(idx, {}).get("stop_reason")
            rho = restaurant_verdicts.get(idx, {}).get("rho")
            result = {
                "business_id": biz_id,
                "name": business.get("name", biz_id),
                "verdict": verdict,
                "risk_score": None,
                "response": json.dumps(
                    {
                        "verdict": verdict,
                        "stop_reason": stop_reason,
                        "rho": rho,
                        "policy_id": self.policy_id,
                        "evidences": evidence_items,
                    }
                ),
                "parsed": {
                    "verdict": verdict,
                    "evidences": evidence_items,
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
        _dump("final")

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

    def _collect_evidence_for_restaurant(self, ridx: int, verdict: str) -> List[Dict[str, Any]]:
        if verdict == self.default_label:
            return []
        rule = next((r for r in self.rules if r.label == verdict), None)
        if not rule:
            return []
        primitive_set = set(rule.primitive_ids)
        evidence_items: List[Dict[str, Any]] = []
        for record in self.tag_store.records():
            if record.review_index >= len(self.review_pool):
                continue
            if self.review_pool[record.review_index]["restaurant_index"] != ridx:
                continue
            for pid in primitive_set:
                if record.tags.get(pid, 0) > 0:
                    ev_list = record.evidence.get(pid, [])
                    if isinstance(ev_list, dict):
                        ev_list = [ev_list]
                    for ev in ev_list:
                        fields = ev.get("fields", {}) or {}
                        if not fields:
                            evidence_items.append(
                                {
                                    "review_id": record.review_id,
                                    "business_id": record.business_id,
                                    "primitive_id": pid,
                                    "event_id": ev.get("event_id"),
                                    "snippet": ev.get("snippet"),
                                    "span": ev.get("span"),
                                }
                            )
                            continue
                        for field_id, value in fields.items():
                            values = value if isinstance(value, list) else [value]
                            for v in values:
                                evidence_items.append(
                                    {
                                        "review_id": record.review_id,
                                        "business_id": record.business_id,
                                        "primitive_id": pid,
                                        "event_id": ev.get("event_id"),
                                        "field": field_id,
                                        "judgement": v,
                                        "snippet": ev.get("snippet"),
                                        "span": ev.get("span"),
                                    }
                                )
        return evidence_items

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
