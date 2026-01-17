#!/usr/bin/env python3
"""Test new allergy safety risk assessment query with a single restaurant.

Usage:
    .venv/bin/python scripts/test_allergy_query.py
    .venv/bin/python scripts/test_allergy_query.py --model claude-sonnet
    .venv/bin/python scripts/test_allergy_query.py --business-id "abc123"
    .venv/bin/python scripts/test_allergy_query.py --name "my_experiment"
    .venv/bin/python scripts/test_allergy_query.py --benchmark  # Use benchmark/ instead of dev/

Output structure:
    results/dev/{timestamp}_{name}/
    ├── results.jsonl      # Lean results (verdict, evidence, usage)
    ├── config.json        # Run configuration
    └── debug/
        └── {sample_id}.json  # Full prompt + response
"""

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from addm.llm import LLMService

ALLERGY_ASSESSMENT_STANDARD = '''
# Allergy Safety Risk Assessment Standard (Review-Based)

## Purpose
Assign a restaurant an Allergy Safety Risk label (Low / High / Critical) using customer reviews as evidence.

## Evidence Scope
A review is allergy-relevant if it discusses:
- An allergic reaction or suspected cross-contact
- Requesting allergy accommodation (e.g., "peanut allergy," "nut-free")
- Staff communication about allergy safety
- Explicit safety guarantees (e.g., "they assured it was nut-free")

Ingredient mentions without allergy context are NOT allergy-relevant.

## Definitions

### Account Type
- **firsthand**: Reviewer or their party experienced it directly
- **secondhand**: Reviewer reports someone else's experience
- **hypothetical**: Concern/preference without describing an incident

Only firsthand accounts count as confirmed incident evidence.

### Incident Severity
- **mild**: Discomfort/minor symptoms, no swelling/hives, no urgent care
- **moderate**: Visible symptoms (hives/swelling) and/or required medication, not life-threatening
- **severe**: Life-threatening (anaphylaxis, EpiPen, ER visit, hospitalization)

### Assurance Failure
When staff explicitly confirmed safety ("guaranteed nut-free") but an incident still occurred.

### Staff Response
- **accommodated**: Took allergy seriously, made successful adjustment
- **refused**: Would not accommodate
- **dismissive**: Minimized the concern

### Cuisine Baseline Risk
Higher for cuisines using peanuts/tree nuts (Thai, Vietnamese, Chinese). Lower for American/Italian/pizza.

## Verdict Rules

**Critical** if ANY:
- Single severe firsthand incident
- Moderate+ incident after explicit assurance
- Pattern of multiple firsthand incidents

**High** if Critical doesn't apply but ANY:
- At least one moderate firsthand incident (especially recent)
- Mild incident after explicit assurance (process failure)
- Staff repeatedly refusing/dismissive
- High cuisine risk + inconsistent handling

**Low** otherwise (no firsthand incidents, consistent accommodation)

## Confidence
- **limited**: Very few allergy-relevant reviews
- **substantial**: Many allergy-relevant reviews
'''

PROMPT_TEMPLATE = '''You are an expert at analyzing restaurant reviews for allergy safety concerns.

{standard}

---

# Task

Analyze the reviews for **{restaurant_name}** ({categories}) and provide an Allergy Safety Risk Assessment.

## Reviews

{reviews}

---

# Required Output Format

Return ONLY valid JSON with no other text before or after:

```json
{{
  "verdict": "Low" | "High" | "Critical",
  "confidence": "limited" | "substantial",
  "allergy_relevant_count": <number of allergy-relevant reviews>,
  "cuisine_baseline_risk": "low" | "medium" | "high",
  "supporting_evidence": [
    {{
      "review_num": <int>,
      "type": "incident" | "assurance_failure" | "staff_concern" | "accommodation",
      "account_type": "firsthand" | "secondhand" | "hypothetical",
      "severity": "mild" | "moderate" | "severe" | null,
      "staff_response": "accommodated" | "refused" | "dismissive" | null,
      "key_quote": "<brief quote, max 50 words>"
    }}
  ],
  "verdict_reasoning": "<1-2 sentences explaining why this verdict based on the rules>"
}}
```

Only include reviews in supporting_evidence that directly contribute to the verdict decision. Do not list every allergy-relevant review—only the ones that matter for the final risk determination.
'''


def find_allergy_relevant_restaurant() -> str | None:
    """Find a restaurant with G1_allergy topic coverage that's in the dataset."""
    selection_file = Path("data/selected/yelp/topic_100.json")
    dataset_file = Path("data/processed/yelp/dataset_K50.jsonl")

    if not selection_file.exists():
        print(f"Warning: {selection_file} not found")
        return None

    # Load dataset IDs
    dataset_ids = set()
    if dataset_file.exists():
        with open(dataset_file) as f:
            for line in f:
                record = json.loads(line)
                dataset_ids.add(record["business_id"])

    # Find restaurants with G1_allergy coverage
    with open(selection_file) as f:
        selected = json.load(f)

    # Sort by allergy hits descending
    allergy_restaurants = [
        r for r in selected
        if "G1_allergy" in r.get("topic_coverage", {}) and r["business_id"] in dataset_ids
    ]
    allergy_restaurants.sort(key=lambda r: r["topic_coverage"]["G1_allergy"], reverse=True)

    if allergy_restaurants:
        return allergy_restaurants[0]["business_id"]
    return None


def load_restaurant(business_id: str) -> dict | None:
    """Load restaurant data from dataset."""
    dataset_path = Path("data/processed/yelp/dataset_K50.jsonl")
    if not dataset_path.exists():
        print(f"Warning: {dataset_path} not found")
        return None

    with open(dataset_path) as f:
        for line in f:
            record = json.loads(line)
            if record["business_id"] == business_id:
                return record
    return None


def parse_json_response(response: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to extract from markdown code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in response
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def setup_output_dirs(name: str, benchmark: bool = False) -> tuple[Path, str]:
    """Create output directory structure, return (run_dir, timestamp)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "benchmark" if benchmark else "dev"
    run_name = f"{timestamp}_{name}" if name else timestamp

    run_dir = Path(f"results/{mode}/{run_name}")
    debug_dir = run_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, timestamp


def save_debug(run_dir: Path, sample_id: str, prompt: str, response: str,
               model: str, usage: dict):
    """Save full prompt/response to debug file."""
    debug_file = run_dir / "debug" / f"{sample_id}.json"
    debug_data = {
        "timestamp": datetime.now().isoformat(),
        "sample_id": sample_id,
        "model": model,
        "usage": usage,
        "prompt": prompt,
        "response": response,
    }
    with open(debug_file, "w") as f:
        json.dump(debug_data, f, indent=2)


def save_result(run_dir: Path, result: dict):
    """Append lean result to results.jsonl."""
    results_file = run_dir / "results.jsonl"
    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def save_config(run_dir: Path, config: dict):
    """Save run configuration."""
    config_file = run_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


async def main():
    parser = argparse.ArgumentParser(description="Test allergy safety risk assessment query")
    parser.add_argument("--model", default="gpt-5-nano", help="LLM model to use")
    parser.add_argument("--business-id", default=None, help="Specific restaurant business ID")
    parser.add_argument("--name", default="allergy_test", help="Run name for output directory")
    parser.add_argument("--benchmark", action="store_true", help="Save to benchmark/ instead of dev/")
    args = parser.parse_args()

    # 1. Find restaurant
    business_id = args.business_id or find_allergy_relevant_restaurant()
    if not business_id:
        print("Error: Could not find a restaurant to test")
        return

    record = load_restaurant(business_id)
    if not record:
        print(f"Error: Could not load restaurant {business_id}")
        return

    restaurant_name = record["business"]["name"]
    categories = record["business"].get("categories", "Unknown")
    sample_id = business_id[:8]  # Short ID for filenames

    print(f"Restaurant: {restaurant_name}")
    print(f"Business ID: {business_id}")
    print(f"Categories: {categories}")
    print(f"Reviews: {len(record['reviews'])}")
    print(f"Model: {args.model}")
    print()

    # 2. Setup output directories
    run_dir, timestamp = setup_output_dirs(args.name, args.benchmark)
    print(f"Output dir: {run_dir}")

    # Save config
    config = {
        "timestamp": timestamp,
        "model": args.model,
        "task": "allergy_safety_assessment",
        "domain": "yelp",
        "k": len(record["reviews"]),
    }
    save_config(run_dir, config)

    # 3. Format reviews
    reviews_text = "\n\n".join([
        f"**Review {j+1}** (Rating: {r.get('stars', 'N/A')}/5, Date: {r.get('date', 'N/A')}):\n{r['text']}"
        for j, r in enumerate(record["reviews"])
    ])

    # 4. Build prompt
    prompt = PROMPT_TEMPLATE.format(
        standard=ALLERGY_ASSESSMENT_STANDARD,
        reviews=reviews_text,
        restaurant_name=restaurant_name,
        categories=categories
    )

    # 5. Call LLM
    llm = LLMService()
    # Note: gpt-5-nano doesn't support temperature=0.0, must use 1.0
    if "nano" in args.model.lower():
        llm.configure(model=args.model, temperature=1.0)
    else:
        llm.configure(model=args.model, temperature=0.0)

    messages = [{"role": "user", "content": prompt}]

    print("Calling LLM...")
    response, usage = await llm.call_async_with_usage(messages, context={"sample_id": sample_id})

    # 6. Save debug (full prompt + response)
    save_debug(run_dir, sample_id, prompt, response, args.model, usage)
    print(f"Debug saved to: {run_dir}/debug/{sample_id}.json")

    # 7. Parse and save lean result
    parsed = parse_json_response(response)

    result = {
        "sample_id": sample_id,
        "business_id": business_id,
        "name": restaurant_name,
        "categories": categories,
        "num_reviews": len(record["reviews"]),
        "model": args.model,
        # From parsed response (or null if parse failed)
        "verdict": parsed.get("verdict") if parsed else None,
        "confidence": parsed.get("confidence") if parsed else None,
        "allergy_relevant_count": parsed.get("allergy_relevant_count") if parsed else None,
        "supporting_evidence": parsed.get("supporting_evidence") if parsed else None,
        "verdict_reasoning": parsed.get("verdict_reasoning") if parsed else None,
        "parse_success": parsed is not None,
        # Usage
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost_usd": usage.get("cost_usd"),
        "latency_ms": usage.get("latency_ms"),
    }

    save_result(run_dir, result)
    print(f"Result saved to: {run_dir}/results.jsonl")

    # 8. Print summary
    print(f"\n{'='*60}")
    print("RESULT SUMMARY:")
    print('='*60)

    if parsed:
        print(f"Verdict: {parsed.get('verdict')}")
        print(f"Confidence: {parsed.get('confidence')}")
        print(f"Allergy-relevant reviews: {parsed.get('allergy_relevant_count')}")
        print(f"Cuisine baseline risk: {parsed.get('cuisine_baseline_risk')}")
        print(f"\nSupporting evidence:")
        for ev in parsed.get("supporting_evidence", []):
            print(f"  - Review {ev.get('review_num')}: {ev.get('type')} "
                  f"({ev.get('account_type')}, severity={ev.get('severity')})")
            print(f"    \"{ev.get('key_quote', '')[:80]}...\"")
        print(f"\nReasoning: {parsed.get('verdict_reasoning')}")
    else:
        print("ERROR: Failed to parse JSON response")
        print("\nRaw response (first 500 chars):")
        print(response[:500])

    print(f"\n{'='*60}")
    print(f"Cost: ${usage.get('cost_usd', 0):.4f} | "
          f"Tokens: {usage.get('prompt_tokens', 0)} + {usage.get('completion_tokens', 0)} | "
          f"Latency: {usage.get('latency_ms', 0)/1000:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
