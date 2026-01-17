#!/usr/bin/env python3
"""Test new allergy safety risk assessment query with a single restaurant.

Usage:
    .venv/bin/python scripts/test_allergy_query.py
    .venv/bin/python scripts/test_allergy_query.py --model claude-sonnet
    .venv/bin/python scripts/test_allergy_query.py --business-id "abc123"
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from addm.llm import LLMService

ALLERGY_ASSESSMENT_STANDARD = '''
# Allergy Safety Risk Assessment Standard (Review-Based)

## Purpose / overall goal
This standard defines how we assign a restaurant an Allergy Safety Risk label (Low / High / Critical) for allergy safety, using customer reviews as evidence.

## Evidence scope
A review is considered allergy-relevant if it meaningfully discusses any of the following:
- an allergic reaction or suspected cross-contact,
- requesting allergy accommodation (e.g., "peanut allergy," "nut-free"),
- staff communication about allergy safety, or
- explicit statements about allergy safety guarantees (e.g., "they assured it was nut-free").

Ingredient mentions without allergy context (e.g., "peanut sauce is great") are not considered allergy-relevant.

## Definitions of terms

### Account type
- **Firsthand**: the reviewer or their dining party experienced the event directly (e.g., "I had…," "my child…").
- **Secondhand**: the reviewer reports someone else's experience ("my friend…").
- **Hypothetical / preference**: the reviewer expresses concern or preference without describing an incident ("I'm allergic so I avoid…").

Only firsthand accounts are treated as confirmed incident evidence.

### Incident severity
If a review describes an allergic reaction, classify severity as follows:
- **Mild incident**: discomfort or minor symptoms that do not include visible swelling/hives and do not require urgent medical care.
- **Moderate incident**: visible symptoms (e.g., hives or swelling) and/or symptoms that required medication (e.g., antihistamine), but the situation is not described as life-threatening.
- **Severe incident**: a life-threatening reaction, including any mention of anaphylaxis, EpiPen use, emergency room visit, hospitalization, or an explicitly described medical emergency.

If no reaction occurred, the review is treated as non-incident evidence.

### Assurance of safety
An assurance is present when staff are described as explicitly confirming safety for peanut/tree-nut allergy (e.g., "guaranteed nut-free," "confirmed no peanuts," "assured it was safe for my allergy").

If a firsthand incident occurs after such an assurance, this is treated as a high-severity policy failure signal, because it indicates breakdown of allergy-handling procedures or communication.

### Staff response
When described, staff response should be recorded as:
- **Accommodated**: staff took the allergy seriously and made a successful adjustment (ingredient check, substitution, separate preparation, clear warnings).
- **Refused**: staff stated they could not or would not accommodate allergies.
- **Dismissive**: staff minimized the concern or treated it as not important.
- **Not described**: no staff interaction about allergy is described.

### Cuisine baseline risk
Some cuisines have higher baseline likelihood of nut cross-contact due to common ingredients and preparation practices. When cuisine is known, treat baseline risk as higher for cuisines commonly using peanuts/tree nuts and shared sauces/oils (e.g., Thai, Vietnamese, Chinese/Asian, etc.), and lower for cuisines where nut exposure is less central (e.g., many American/Italian/pizza contexts). When multiple cuisines apply, use the higher-risk baseline.

### Recency principle
More recent incidents should weigh more heavily than older incidents. However, severe incidents remain important even if older.

## Final verdict rules (Low / High / Critical)

**Critical Risk** if any of the following is true:
- A single severe firsthand incident is reported.
- A firsthand incident is reported after an explicit assurance of safety, and the incident is described as moderate-or-severe or medically significant.
- Multiple firsthand incidents indicate an ongoing pattern (e.g., repeated moderate incidents, or repeated incidents with dismissive/refusal behavior).

**High Risk** if Critical Risk does not apply, but any of the following is true:
- At least one moderate firsthand incident is reported (especially if recent).
- A firsthand incident occurs after an explicit assurance of safety, even if symptoms are described as mild (because the assurance represents a process failure).
- Reviews repeatedly describe staff as refusing or being dismissive about allergy concerns.
- Baseline cuisine risk is high and evidence suggests inconsistent allergy handling.

**Low Risk** otherwise, especially when:
- no firsthand incidents are reported, and
- allergy-relevant reviews describe consistent accommodation behavior or careful handling.

## Confidence note (must be reported with verdict)
Alongside the verdict, report whether the decision is based on limited evidence (very few allergy-relevant reviews) or substantial evidence (many allergy-relevant reviews). This is not the risk level itself, but an indicator of how stable the conclusion is.
'''

PROMPT_TEMPLATE = '''You are an expert at analyzing restaurant reviews for allergy safety concerns.

{standard}

---

# Task

Analyze the following reviews for **{restaurant_name}** ({categories}) and provide an Allergy Safety Risk Assessment.

## Reviews

{reviews}

---

# Instructions

1. First, identify all allergy-relevant reviews from the set above (cite by review number).
2. For each allergy-relevant review, extract:
   - Account type (firsthand/secondhand/hypothetical)
   - Whether an incident occurred and its severity (if applicable)
   - Whether an assurance of safety was given before an incident
   - Staff response (if described)
3. Consider the cuisine baseline risk.
4. Apply the verdict rules to determine: **Low Risk**, **High Risk**, or **Critical Risk**.
5. State your confidence level: **limited evidence** or **substantial evidence**.

Provide your analysis in a structured format, ending with a clear final verdict and confidence level.
'''


def find_allergy_relevant_restaurant() -> str | None:
    """Find a restaurant with allergy keyword hits."""
    hits_file = Path("data/keyword_hits/yelp/G1_allergy.json")
    if not hits_file.exists():
        print(f"Warning: {hits_file} not found")
        return None

    with open(hits_file) as f:
        hits = json.load(f)

    # Return first restaurant with good hit count (skip first which may be peanut-themed)
    # Prefer restaurants with "allerg" in sample_matches for better test
    for hit in hits:
        if any("allerg" in m.lower() for m in hit.get("sample_matches", [])):
            return hit["business_id"]

    # Fallback to first with reasonable hits
    if hits:
        return hits[0]["business_id"]
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


async def main():
    parser = argparse.ArgumentParser(description="Test allergy safety risk assessment query")
    parser.add_argument("--model", default="gpt-5-nano", help="LLM model to use")
    parser.add_argument("--business-id", default=None, help="Specific restaurant business ID")
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

    print(f"Restaurant: {restaurant_name}")
    print(f"Business ID: {business_id}")
    print(f"Categories: {categories}")
    print(f"Reviews: {len(record['reviews'])}")
    print(f"Model: {args.model}")
    print()

    # 2. Format reviews
    reviews_text = "\n\n".join([
        f"**Review {j+1}** (Rating: {r.get('stars', 'N/A')}/5, Date: {r.get('date', 'N/A')}):\n{r['text']}"
        for j, r in enumerate(record["reviews"])
    ])

    # 3. Build prompt
    prompt = PROMPT_TEMPLATE.format(
        standard=ALLERGY_ASSESSMENT_STANDARD,
        reviews=reviews_text,
        restaurant_name=restaurant_name,
        categories=categories
    )

    # 4. Call LLM
    llm = LLMService()
    llm.configure(model=args.model, temperature=0.0)

    messages = [{"role": "user", "content": prompt}]

    print("Calling LLM...")
    response, usage = await llm.call_async_with_usage(messages, context={"test": "allergy_query"})

    # 5. Save output
    output_dir = Path("results/test_queries")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"allergy_test_{timestamp}.json"

    result = {
        "timestamp": timestamp,
        "model": args.model,
        "restaurant": {
            "name": restaurant_name,
            "business_id": business_id,
            "categories": categories,
        },
        "num_reviews": len(record["reviews"]),
        "prompt": prompt,
        "response": response,
        "usage": usage,
    }

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to: {output_file}")
    print(f"\n{'='*60}")
    print("RESPONSE:")
    print('='*60)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
