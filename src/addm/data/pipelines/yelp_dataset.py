"""Build Yelp datasets with reviews per business."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class DatasetBuildConfig:
    raw_dir: Path
    selection_path: Path
    output_dir: Path
    scales: List[int]
    include_user: bool = True
    verbose: bool = True


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_selection(selection_path: Path) -> List[Dict]:
    data = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Selection file must be a JSON list")
    return data


def _load_reviews(review_path: Path, business_ids: set[str], verbose: bool) -> Dict[str, List[Dict]]:
    reviews_by_biz = {bid: [] for bid in business_ids}
    if verbose:
        print(f"Scanning reviews from {review_path}...")
    for row in _iter_jsonl(review_path):
        bid = row.get("business_id")
        if bid in reviews_by_biz:
            reviews_by_biz[bid].append(row)
    for bid in reviews_by_biz:
        reviews_by_biz[bid].sort(key=lambda x: x.get("date", ""), reverse=True)
    return reviews_by_biz


def _load_users(user_path: Path, user_ids: set[str], verbose: bool) -> Dict[str, Dict]:
    users = {}
    if verbose:
        print(f"Scanning users from {user_path}...")
    for row in _iter_jsonl(user_path):
        uid = row.get("user_id")
        if uid in user_ids:
            users[uid] = {
                "name": row.get("name"),
                "review_count": row.get("review_count"),
                "yelping_since": row.get("yelping_since"),
                "elite": row.get("elite"),
                "average_stars": row.get("average_stars"),
                "fans": row.get("fans"),
            }
    return users


def build_datasets(config: DatasetBuildConfig) -> None:
    review_path = config.raw_dir / "yelp_academic_dataset_review.json"
    user_path = config.raw_dir / "yelp_academic_dataset_user.json"
    selection = _load_selection(config.selection_path)
    business_ids = {b["business_id"] for b in selection if "business_id" in b}
    if not business_ids:
        raise ValueError("Selection file contains no business_id values")

    reviews_by_biz = _load_reviews(review_path, business_ids, config.verbose)

    users = {}
    if config.include_user:
        user_ids = {r.get("user_id") for reviews in reviews_by_biz.values() for r in reviews if r.get("user_id")}
        users = _load_users(user_path, user_ids, config.verbose)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    for k in config.scales:
        out_path = config.output_dir / f"dataset_K{k}.jsonl"
        if config.verbose:
            print(f"Building K={k} -> {out_path}")
        with out_path.open("w", encoding="utf-8") as handle:
            for business in selection:
                bid = business.get("business_id")
                if not bid:
                    continue
                top_reviews = reviews_by_biz.get(bid, [])[:k]
                enriched = []
                for review in top_reviews:
                    review_copy = dict(review)
                    uid = review_copy.get("user_id")
                    if config.include_user and uid in users:
                        review_copy["user"] = users[uid]
                    enriched.append(review_copy)
                record = {
                    "business_id": bid,
                    "stratification": business.get("stratification_tag"),
                    "business": business,
                    "reviews": enriched,
                    "review_count_actual": len(enriched),
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

        if config.verbose:
            print(f"Wrote {out_path}")
