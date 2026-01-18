#!/usr/bin/env python3
"""Create restaurant ID filter list from Yelp business dataset."""

import json
from pathlib import Path

RESTAURANT_CATS = {'Restaurant', 'Restaurants', 'Food', 'Cafe', 'Bakery', 'Bakeries', 'Bar', 'Bars'}

def is_restaurant(categories_str):
    if not categories_str:
        return False
    cats = [c.strip() for c in categories_str.split(',')]
    return any(c in RESTAURANT_CATS for c in cats)

def main():
    raw_path = Path('data/raw/yelp/yelp_academic_dataset_business.json')
    output_dir = Path('data/selection/yelp')
    output_path = output_dir / 'restaurant_ids.json'

    restaurant_ids = []
    with open(raw_path) as f:
        for line in f:
            b = json.loads(line)
            if is_restaurant(b.get('categories', '')):
                restaurant_ids.append(b['business_id'])

    output = {
        "description": "Restaurant business IDs filtered from yelp_academic_dataset_business.json",
        "filter_categories": sorted(RESTAURANT_CATS),
        "count": len(restaurant_ids),
        "ids": sorted(restaurant_ids)
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'Saved {len(restaurant_ids)} restaurant IDs to {output_path}')

if __name__ == '__main__':
    main()
