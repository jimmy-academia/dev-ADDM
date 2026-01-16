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

import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Queue
from threading import Thread
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

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


def _process_batch(
    args: Tuple[List[bytes], Dict[str, str]]
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Process a batch of review lines (worker function for parallel processing).

    Args:
        args: (list of raw JSONL lines, topic_patterns_dict)

    Returns:
        Dict[topic][business_id] = list of matching reviews
    """
    lines, topic_patterns_str = args

    # Compile patterns in worker process
    patterns = {
        topic: re.compile(pattern_str, re.IGNORECASE)
        for topic, pattern_str in topic_patterns_str.items()
    }

    results: Dict[str, Dict[str, List[Dict]]] = {
        topic: defaultdict(list) for topic in patterns
    }

    for line in lines:
        review = orjson.loads(line)
        text = review.get("text", "")
        biz_id = review["business_id"]

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

    # Convert defaultdicts to regular dicts for pickling
    return {topic: dict(hits) for topic, hits in results.items()}


def search_reviews_for_keywords(
    review_file: str,
    topics: List[str],
    n_workers: Optional[int] = None,
    chunk_size: int = 100_000,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Search reviews for keyword matches across multiple topics (parallel).

    Reads file in a separate thread while processing chunks in parallel.

    Args:
        review_file: Path to JSONL review file
        topics: List of topic names to search for
        n_workers: Number of parallel workers (default: 2x CPU cores)
        chunk_size: Lines per chunk (default: 100k)

    Returns:
        Dict[topic][business_id] = list of matching reviews
    """
    if n_workers is None:
        n_workers = os.cpu_count() * 2  # 2x cores is usually optimal

    # Build pattern strings (can't pickle compiled patterns)
    topic_patterns_str = {}
    for topic in topics:
        patterns = TOPIC_KEYWORDS[topic]
        combined = "|".join(f"({p})" for p in patterns)
        topic_patterns_str[topic] = combined

    total_lines = count_lines(review_file)
    console.print(f"[bold]Processing {total_lines:,} reviews with {n_workers} workers...[/bold]")

    # Queue for chunks (reader thread -> main thread)
    chunk_queue: Queue = Queue(maxsize=n_workers * 2)
    read_done = {"done": False, "lines_read": 0}

    def reader_thread():
        """Read file and push chunks to queue."""
        with open(review_file, "rb") as f:
            batch = []
            for line in f:
                batch.append(line)
                read_done["lines_read"] += 1
                if len(batch) >= chunk_size:
                    chunk_queue.put((batch, topic_patterns_str))
                    batch = []
            if batch:
                chunk_queue.put((batch, topic_patterns_str))
        read_done["done"] = True

    # Start reader thread
    reader = Thread(target=reader_thread, daemon=True)
    reader.start()

    # Process chunks as they arrive
    merged: Dict[str, Dict[str, List[Dict]]] = {topic: defaultdict(list) for topic in topics}
    futures = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed:,}/{task.total:,})"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing", total=total_lines)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            lines_submitted = 0
            lines_processed = 0

            while True:
                # Submit new chunks from queue
                while not chunk_queue.empty():
                    chunk = chunk_queue.get()
                    lines_submitted += len(chunk[0])
                    futures.append((executor.submit(_process_batch, chunk), len(chunk[0])))

                # Collect completed results
                still_pending = []
                for future, n_lines in futures:
                    if future.done():
                        chunk_result = future.result()
                        for topic, hits in chunk_result.items():
                            for biz_id, reviews in hits.items():
                                merged[topic][biz_id].extend(reviews)
                        lines_processed += n_lines
                        progress.update(task, completed=lines_processed)
                    else:
                        still_pending.append((future, n_lines))
                futures = still_pending

                # Check if done
                if read_done["done"] and chunk_queue.empty() and not futures:
                    break

    reader.join()
    return {topic: dict(hits) for topic, hits in merged.items()}


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
    top_n: int = 100,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search reviews and rank restaurants by keyword hits for multiple topics.

    Args:
        topics: List of topic names (e.g., ["allergy", "dietary", "hygiene"])
        review_file: Path to review JSONL
        business_file: Path to business JSONL
        min_hits: Minimum keyword hits to include
        top_n: Number of top restaurants to return per topic

    Returns:
        Dict[topic] = list of restaurant info with hit counts, sorted by hits descending
    """
    # Search reviews
    all_hits = search_reviews_for_keywords(review_file, topics)

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
