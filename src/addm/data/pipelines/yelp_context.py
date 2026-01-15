"""Build context blocks from Yelp raw data."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class YelpBusiness:
    business_id: str
    name: str
    city: str
    state: str
    categories: List[str]
    stars: float
    review_count: int


@dataclass
class YelpReview:
    business_id: str
    review_id: str
    stars: float
    text: str
    date: str


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_businesses(path: Path, city: Optional[str] = None, categories: Optional[List[str]] = None,
                    max_businesses: Optional[int] = None) -> Dict[str, YelpBusiness]:
    businesses: Dict[str, YelpBusiness] = {}
    for row in _iter_jsonl(path):
        if city and row.get("city") != city:
            continue
        cats = row.get("categories") or ""
        cat_list = [c.strip() for c in cats.split(",") if c.strip()]
        if categories:
            if not any(c.lower() in {x.lower() for x in cat_list} for c in categories):
                continue
        business = YelpBusiness(
            business_id=row["business_id"],
            name=row.get("name", ""),
            city=row.get("city", ""),
            state=row.get("state", ""),
            categories=cat_list,
            stars=float(row.get("stars", 0.0)),
            review_count=int(row.get("review_count", 0)),
        )
        businesses[business.business_id] = business
        if max_businesses and len(businesses) >= max_businesses:
            break
    return businesses


def load_reviews(path: Path, business_ids: set[str], max_reviews: int) -> Dict[str, List[YelpReview]]:
    reviews: Dict[str, List[YelpReview]] = {bid: [] for bid in business_ids}
    for row in _iter_jsonl(path):
        bid = row.get("business_id")
        if bid not in reviews:
            continue
        if len(reviews[bid]) >= max_reviews:
            continue
        reviews[bid].append(
            YelpReview(
                business_id=bid,
                review_id=row.get("review_id", ""),
                stars=float(row.get("stars", 0.0)),
                text=row.get("text", ""),
                date=row.get("date", ""),
            )
        )
    return reviews


def build_context_text(business: YelpBusiness, reviews: List[YelpReview]) -> str:
    header = [
        f"Business: {business.name}",
        f"City: {business.city}, {business.state}",
        f"Categories: {', '.join(business.categories)}",
        f"Rating: {business.stars} ({business.review_count} reviews)",
    ]
    review_lines = []
    for idx, review in enumerate(reviews, start=1):
        review_lines.append(f"Review {idx} ({review.stars} stars, {review.date}): {review.text}")
    return "\n".join(header + [""] + review_lines).strip()


def build_context_dataset(
    business_path: Path,
    review_path: Path,
    output_path: Path,
    city: Optional[str] = None,
    categories: Optional[List[str]] = None,
    max_businesses: Optional[int] = None,
    max_reviews: int = 20,
    min_reviews: int = 1,
    seed: int = 42,
) -> None:
    random.seed(seed)
    businesses = load_businesses(
        business_path,
        city=city,
        categories=categories,
        max_businesses=max_businesses,
    )
    reviews = load_reviews(review_path, set(businesses.keys()), max_reviews)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for bid, business in businesses.items():
            business_reviews = reviews.get(bid, [])
            if len(business_reviews) < min_reviews:
                continue
            context = build_context_text(business, business_reviews)
            record = {
                "id": bid,
                "context": context,
                "metadata": {
                    "business": {
                        "name": business.name,
                        "city": business.city,
                        "state": business.state,
                        "categories": business.categories,
                        "stars": business.stars,
                        "review_count": business.review_count,
                    },
                    "reviews": [
                        {
                            "review_id": r.review_id,
                            "stars": r.stars,
                            "date": r.date,
                            "text": r.text,
                        }
                        for r in business_reviews
                    ],
                },
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_context_dataset_from_dir(
    raw_dir: Path,
    output_path: Path,
    city: Optional[str] = None,
    categories: Optional[List[str]] = None,
    max_businesses: Optional[int] = None,
    max_reviews: int = 20,
    min_reviews: int = 1,
    seed: int = 42,
) -> None:
    business_path = raw_dir / "yelp_academic_dataset_business.json"
    review_path = raw_dir / "yelp_academic_dataset_review.json"
    if not business_path.exists():
        raise FileNotFoundError(f"Missing business file: {business_path}")
    if not review_path.exists():
        raise FileNotFoundError(f"Missing review file: {review_path}")
    build_context_dataset(
        business_path=business_path,
        review_path=review_path,
        output_path=output_path,
        city=city,
        categories=categories,
        max_businesses=max_businesses,
        max_reviews=max_reviews,
        min_reviews=min_reviews,
        seed=seed,
    )
