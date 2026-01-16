"""
Keyword search utilities for finding restaurants with relevant reviews.

Usage:
    from addm.data.keyword_search import search_and_rank_restaurants

    # Find restaurants with allergy-related reviews
    results = search_and_rank_restaurants(
        topic="allergy",
        review_file="data/raw/yelp/yelp_academic_dataset_review.json",
        business_file="data/raw/yelp/yelp_academic_dataset_business.json",
        min_hits=10,
        top_n=100,
    )
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import json
import orjson

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


# Keywords for each topic
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    # G1: Health & Safety
    "allergy": [
        r"\ballerg", r"\bpeanut", r"\btree.?nut", r"\bshellfish",
        r"\banaphyla", r"\bepipen", r"\bcross.?contam",
        r"\bnut.?free", r"\ballergy.?friendly", r"\ballergy.?safe",
    ],
    "dietary": [
        r"\bgluten.?free", r"\bceliac", r"\bdairy.?free", r"\blactose",
        r"\bvegan\b", r"\bvegetarian", r"\bplant.?based",
        r"\bdietary.?restrict", r"\bspecial.?diet",
    ],
    "hygiene": [
        r"\bdirty", r"\bfilthy", r"\broach", r"\bcockroach", r"\bbug\b",
        r"\bhair.?in.?food", r"\bfood.?poison", r"\bgot.?sick",
        r"\bunsanitary", r"\bhealth.?code", r"\bgross", r"\bdisgusting",
    ],

    # G2: Social Context
    "romance": [
        r"\bdate.?night", r"\bromantic", r"\banniversary", r"\bintimate",
        r"\bproposal", r"\bcandle", r"\bcozy", r"\bcouples?\b",
        r"\bvalentine", r"\bspecial.?occasion",
    ],
    "business": [
        r"\bbusiness.?lunch", r"\bbusiness.?dinner", r"\bmeeting\b",
        r"\bclient\b", r"\bprofessional", r"\binterview",
        r"\bwork.?lunch", r"\bcorporate", r"\bnetworking",
    ],
    "group": [
        r"\blarge.?group", r"\bparty\b", r"\bbirthday", r"\bcelebrat",
        r"\bgroup.?dinner", r"\bbig.?table", r"\breservation.?for",
        r"\bfamily.?friendly", r"\bkids?\b", r"\bchildren", r"\bhigh.?chair",
    ],

    # G3: Economic Value
    "price_worth": [
        r"\bexpensive", r"\boverpriced", r"\bpricey", r"\bworth.?it",
        r"\bgood.?value", r"\bcheap", r"\baffordable",
        r"\bportion.?size", r"\bhappy.?hour", r"\blunch.?special",
        r"\bdeal\b", r"\bdiscount", r"\bprix.?fixe",
    ],
    "hidden_costs": [
        r"\bservice.?charge", r"\bauto.?gratuit", r"\bcorkage",
        r"\bhidden.?fee", r"\bsurprised.?by.?bill", r"\bextra.?charge",
        r"\bfine.?print", r"\btip.?included", r"\bmandatory.?tip",
    ],
    "time_value": [
        r"\bwait.?time", r"\blong.?wait", r"\bwaited.?forever",
        r"\bworth.?the.?wait", r"\bhour.?wait", r"\bwalk.?in",
        r"\bwait.?list", r"\bseated.?late", r"\btook.?forever",
    ],

    # G4: Talent & Performance
    "server": [
        r"\bwaiter", r"\bwaitress", r"\bour.?server", r"\bservice.?was",
        r"\battentive", r"\bignored", r"\brude.?server",
        r"\bfriendly.?staff", r"\bhelpful", r"\bknowledgeable",
    ],
    "kitchen": [
        r"\bovercooked", r"\bundercooked", r"\bburnt", r"\braw\b",
        r"\bcold.?food", r"\bchef\b", r"\bkitchen\b", r"\bfood.?quality",
        r"\bpresentation", r"\bsent.?back", r"\binedible", r"\bbland",
    ],
    "environment": [
        r"\bloud\b", r"\bnoisy", r"\bquiet", r"\batmosphere", r"\bambiance",
        r"\bdecor", r"\bcomfortable", r"\bcramped", r"\bseating",
        r"\btemperature", r"\bAC\b", r"\btoo.?cold", r"\btoo.?hot",
    ],

    # G5: Operational Efficiency
    "capacity": [
        r"\bunderstaffed", r"\bshort.?staffed", r"\boverwhelmed",
        r"\btoo.?busy", r"\bswamped", r"\bpacked", r"\bcouldn.?t.?handle",
        r"\bbusy.?night", r"\bweekend", r"\brush.?hour", r"\bslammed",
    ],
    "execution": [
        r"\bwrong.?order", r"\bforgot.?my", r"\bnever.?came",
        r"\bmissing", r"\bnot.?what.?I.?ordered", r"\bmodification",
        r"\bas.?described", r"\bnot.?as.?pictured", r"\bfalse.?advertis",
    ],
    "consistency": [
        r"\bused.?to.?be", r"\bwent.?downhill", r"\bnot.?like.?before",
        r"\binconsistent", r"\bhit.?or.?miss", r"\bdepends.?on.?the.?day",
        r"\balways.?reliable", r"\bnew.?management", r"\bnew.?owner",
    ],

    # G6: Competitive Strategy
    "uniqueness": [
        r"\bunique", r"\bone.?of.?a.?kind", r"\bonly.?place",
        r"\bcan.?t.?find.?elsewhere", r"\bspecial\b", r"\bnothing.?like",
        r"\bmust.?try", r"\bsignature.?dish", r"\bfamous.?for",
        r"\bhidden.?gem", r"\bstandout", r"\binnovative",
    ],
    "comparison": [
        r"\bbetter.?than", r"\bcompared.?to", r"\bprefer\b",
        r"\binstead.?of", r"\breminds.?me.?of", r"\bnot.?as.?good.?as",
        r"\bbeats\b", r"\bsimilar.?to", r"\balternative",
    ],
    "loyalty": [
        r"\bcome.?back", r"\breturn", r"\bregular\b", r"\bfavorite",
        r"\balways.?go", r"\bevery.?week", r"\bhighly.?recommend",
        r"\btell.?everyone", r"\bloyal", r"\bmy.?go.?to",
        r"\blove.?this.?place", r"\bobsessed", r"\baddicted",
    ],
}


def compile_topic_pattern(topic: str) -> re.Pattern:
    """Compile regex pattern for a topic."""
    if topic not in TOPIC_KEYWORDS:
        raise ValueError(f"Unknown topic: {topic}. Available: {list(TOPIC_KEYWORDS.keys())}")

    patterns = TOPIC_KEYWORDS[topic]
    combined = "|".join(f"({p})" for p in patterns)
    return re.compile(combined, re.IGNORECASE)


def count_lines(file_path: str) -> int:
    """Fast line count using wc -l."""
    import subprocess
    result = subprocess.run(["wc", "-l", file_path], capture_output=True, text=True)
    return int(result.stdout.strip().split()[0])


def search_reviews_for_keywords(
    review_file: str,
    topics: List[str],
    business_file: str = "data/raw/yelp/yelp_academic_dataset_business.json",
    min_reviews: int = 20,
    max_reviews_per_biz: int = 500,
    output_dir: Optional[Path] = None,
    save_interval: int = 500_000,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Search reviews for keyword matches across multiple topics.

    Args:
        review_file: Path to JSONL review file
        topics: List of topic names to search for
        business_file: Path to business JSONL file (for filtering)
        min_reviews: Minimum reviews a restaurant must have (default: 20)
        max_reviews_per_biz: Max reviews to process per restaurant (default: 500)
        output_dir: If provided, save partial results every save_interval lines
        save_interval: Save partial results every N lines (default: 500k)

    Returns:
        Dict[topic][business_id] = list of matching reviews
    """
    # Pre-load valid restaurant IDs
    console.print(f"[bold]Loading valid restaurants (>={min_reviews} reviews)...[/bold]")
    valid_biz_ids = get_valid_restaurant_ids(business_file, min_reviews)
    console.print(f"[green]Found {len(valid_biz_ids):,} valid restaurants[/green]")

    # Compile patterns
    patterns = {topic: compile_topic_pattern(topic) for topic in topics}

    # Results: topic -> business_id -> list of review info
    results: Dict[str, Dict[str, List[Dict]]] = {
        topic: defaultdict(list) for topic in topics
    }

    # Track review count per business (for capping)
    biz_review_count: Dict[str, int] = defaultdict(int)

    console.print(f"[bold]Scanning reviews for {len(topics)} topics...[/bold]")
    total_lines = count_lines(review_file)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    def save_partial():
        """Save current results to disk."""
        if not output_dir:
            return
        for topic in topics:
            out_file = output_dir / f"{topic}_partial.json"
            with open(out_file, "w") as f:
                json.dump(dict(results[topic]), f)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed:,}/{task.total:,})"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning", total=total_lines)
        line_count = 0

        with open(review_file, "rb") as f:
            for line in f:
                line_count += 1
                progress.advance(task)
                review = orjson.loads(line)
                biz_id = review["business_id"]

                # Skip if not a valid restaurant
                if biz_id not in valid_biz_ids:
                    continue

                # Skip if already at max reviews for this business
                if biz_review_count[biz_id] >= max_reviews_per_biz:
                    continue

                biz_review_count[biz_id] += 1
                text = review.get("text", "")

                for topic, pattern in patterns.items():
                    match = pattern.search(text)
                    if match:
                        results[topic][biz_id].append({
                            "review_id": review["review_id"],
                            "stars": review["stars"],
                            "date": review["date"],
                            "match": match.group(),
                            "snippet": text[:200],
                        })

                # Save partial results periodically
                if output_dir and line_count % save_interval == 0:
                    save_partial()
                    console.print(f"[dim]Saved partial results at {line_count:,} lines[/dim]")

    # Final save
    save_partial()
    return {topic: dict(hits) for topic, hits in results.items()}


def count_hits_by_business(
    hits: Dict[str, List[Dict]],
) -> Dict[str, int]:
    """Count total keyword hits per business."""
    return {biz_id: len(reviews) for biz_id, reviews in hits.items()}


def get_top_businesses(
    hits: Dict[str, List[Dict]],
    min_hits: int = 1,
    top_n: Optional[int] = None,
) -> List[tuple]:
    """
    Get top businesses by hit count.

    Returns:
        List of (business_id, hit_count) sorted by count descending
    """
    counts = count_hits_by_business(hits)
    filtered = [(biz_id, count) for biz_id, count in counts.items() if count >= min_hits]
    sorted_biz = sorted(filtered, key=lambda x: -x[1])

    if top_n:
        return sorted_biz[:top_n]
    return sorted_biz


def get_valid_restaurant_ids(
    business_file: str,
    min_reviews: int = 20,
) -> set:
    """
    Get IDs of restaurants with at least min_reviews.

    Args:
        business_file: Path to business JSONL file
        min_reviews: Minimum review count to include

    Returns:
        Set of valid business IDs
    """
    valid_ids = set()
    with open(business_file) as f:
        for line in f:
            biz = json.loads(line)
            cats = biz.get("categories", "") or ""

            # Must be a restaurant
            if not any(c in cats for c in ["Restaurant", "Food", "Cafe", "Bakery", "Bar"]):
                continue

            # Must have enough reviews
            if biz.get("review_count", 0) < min_reviews:
                continue

            valid_ids.add(biz["business_id"])

    return valid_ids


def get_business_info(
    business_file: str,
    business_ids: set,
    restaurant_only: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Load business info for specified business IDs.

    Args:
        business_file: Path to business JSONL file
        business_ids: Set of business IDs to load
        restaurant_only: Only include restaurants (has Restaurant/Food/Cafe in categories)

    Returns:
        Dict[business_id] = business info
    """
    businesses = {}

    with open(business_file) as f:
        for line in f:
            biz = json.loads(line)
            if biz["business_id"] not in business_ids:
                continue

            cats = biz.get("categories", "") or ""

            if restaurant_only:
                if not any(c in cats for c in ["Restaurant", "Food", "Cafe", "Bakery", "Bar"]):
                    continue

            businesses[biz["business_id"]] = {
                "business_id": biz["business_id"],
                "name": biz["name"],
                "categories": cats,
                "review_count": biz["review_count"],
                "stars": biz["stars"],
                "city": biz.get("city", ""),
                "state": biz.get("state", ""),
            }

    return businesses


def search_and_rank_restaurants(
    topics: List[str],
    review_file: str = "data/raw/yelp/yelp_academic_dataset_review.json",
    business_file: str = "data/raw/yelp/yelp_academic_dataset_business.json",
    min_hits: int = 10,
    min_reviews: int = 20,
    max_reviews_per_biz: int = 500,
    top_n: int = 100,
    output_dir: Optional[Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search reviews and rank restaurants by keyword hits for multiple topics.

    Args:
        topics: List of topic names (e.g., ["allergy", "dietary", "hygiene"])
        review_file: Path to review JSONL
        business_file: Path to business JSONL
        min_hits: Minimum keyword hits to include
        min_reviews: Minimum reviews a restaurant must have (default: 20)
        max_reviews_per_biz: Max reviews to process per restaurant (default: 500)
        top_n: Number of top restaurants to return per topic
        output_dir: If provided, save partial results during scanning

    Returns:
        Dict[topic] = list of restaurant info with hit counts, sorted by hits descending
    """
    # Search reviews
    all_hits = search_reviews_for_keywords(
        review_file,
        topics,
        business_file=business_file,
        min_reviews=min_reviews,
        max_reviews_per_biz=max_reviews_per_biz,
        output_dir=output_dir,
    )

    # Collect all business IDs that need info
    all_biz_ids = set()
    for topic_hits in all_hits.values():
        for biz_id, reviews in topic_hits.items():
            if len(reviews) >= min_hits:
                all_biz_ids.add(biz_id)

    console.print(f"\n[bold]Loading info for {len(all_biz_ids)} businesses...[/bold]")
    biz_info = get_business_info(business_file, all_biz_ids)

    # Build results per topic
    results = {}
    for topic, hits in all_hits.items():
        topic_results = []
        for biz_id, reviews in hits.items():
            if len(reviews) < min_hits:
                continue
            if biz_id not in biz_info:
                continue

            info = biz_info[biz_id].copy()
            info["keyword_hits"] = len(reviews)
            info["sample_matches"] = [r["match"] for r in reviews[:5]]
            topic_results.append(info)

        # Sort by hits
        topic_results.sort(key=lambda x: -x["keyword_hits"])
        results[topic] = topic_results[:top_n]

        console.print(f"\n[cyan]{topic}[/cyan]: {len(topic_results)} restaurants with >= {min_hits} hits")
        if topic_results:
            console.print("  [dim]Top 5:[/dim]")
            for r in topic_results[:5]:
                console.print(f"    {r['name'][:40]:<40} [green]{r['keyword_hits']:>4}[/green] hits")

    return results


def print_topic_summary(results: Dict[str, List[Dict]]) -> None:
    """Print summary of search results using rich tables."""
    console.print("\n[bold]KEYWORD SEARCH RESULTS SUMMARY[/bold]")
    console.print("=" * 70)

    for topic, restaurants in results.items():
        console.print(f"\n[bold cyan]{topic.upper()}[/bold cyan]: {len(restaurants)} restaurants")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Name", style="white", width=40)
        table.add_column("Hits", justify="right", style="green")
        table.add_column("Reviews", justify="right")
        table.add_column("Stars", justify="right", style="yellow")

        for r in restaurants[:10]:
            table.add_row(
                r["name"][:38],
                str(r["keyword_hits"]),
                str(r["review_count"]),
                f"{r['stars']:.1f}",
            )

        console.print(table)


if __name__ == "__main__":
    # Example usage: search for G1 topics
    results = search_and_rank_restaurants(
        topics=["allergy", "dietary", "hygiene"],
        min_hits=10,
        top_n=50,
    )
    print_topic_summary(results)
