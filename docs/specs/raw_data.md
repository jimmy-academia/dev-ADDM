# Raw Data Sources

Raw datasets are stored outside the repo and are not tracked in git.

## Yelp

Source files (from `../anot/preprocessing/raw/`):

- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_checkin.json`
- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_tip.json`
- `yelp_academic_dataset_user.json`
- `Dataset_User_Agreement.pdf`

Local placement (not committed):

```
data/raw/yelp/
```

The repository ignores `data/raw/` to avoid storing large files.

## Amazon

Source: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD)

### Download

```bash
# Download a category (reviews + metadata)
./scripts/download_amazon.sh Electronics both

# Reviews only
./scripts/download_amazon.sh Books reviews

# Metadata only
./scripts/download_amazon.sh All_Beauty meta
```

Files are downloaded and extracted to:

```
data/raw/amazon/
├── {Category}.jsonl        # reviews
└── meta_{Category}.jsonl   # product metadata
```

### Available Categories (33)

All_Beauty, Amazon_Fashion, Appliances, Arts_Crafts_and_Sewing, Automotive, Baby_Products, Books, CDs_and_Vinyl, Cell_Phones_and_Accessories, Clothing_Shoes_and_Jewelry, Digital_Music, Electronics, Gift_Cards, Grocery_and_Gourmet_Food, Handmade_Products, Health_and_Household, Health_and_Personal_Care, Home_and_Kitchen, Industrial_and_Scientific, Kindle_Store, Magazine_Subscriptions, Movies_and_TV, Musical_Instruments, Office_Products, Patio_Lawn_and_Garden, Pet_Supplies, Software, Sports_and_Outdoors, Subscription_Boxes, Tools_and_Home_Improvement, Toys_and_Games, Video_Games, Unknown

### File Format

Both files are JSON Lines (`.jsonl`). Load with:

```python
import json

with open("data/raw/amazon/Electronics.jsonl") as f:
    for line in f:
        review = json.loads(line)
```
