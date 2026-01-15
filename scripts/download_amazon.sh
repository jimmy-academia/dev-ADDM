#!/bin/bash
# Amazon Reviews 2023 Dataset Downloader
# Usage: ./download_amazon.sh <category> [reviews|meta|both]
#
# Example:
#   ./download_amazon.sh Electronics both
#   ./download_amazon.sh Books reviews
#   ./download_amazon.sh All_Beauty meta
#
# Available categories (33 total):
#   All_Beauty, Amazon_Fashion, Appliances, Arts_Crafts_and_Sewing,
#   Automotive, Baby_Products, Books, CDs_and_Vinyl,
#   Cell_Phones_and_Accessories, Clothing_Shoes_and_Jewelry,
#   Digital_Music, Electronics, Gift_Cards, Grocery_and_Gourmet_Food,
#   Handmade_Products, Health_and_Household, Health_and_Personal_Care,
#   Home_and_Kitchen, Industrial_and_Scientific, Kindle_Store,
#   Magazine_Subscriptions, Movies_and_TV, Musical_Instruments,
#   Office_Products, Patio_Lawn_and_Garden, Pet_Supplies, Software,
#   Sports_and_Outdoors, Subscription_Boxes, Tools_and_Home_Improvement,
#   Toys_and_Games, Video_Games, Unknown

set -e

BASE_URL="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
OUTPUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../data/raw/amazon" && pwd)"

if [ -z "$1" ]; then
    echo "Usage: $0 <category> [reviews|meta|both]"
    echo "Example: $0 Electronics both"
    exit 1
fi

CATEGORY="$1"
TYPE="${2:-both}"

download_and_extract() {
    local url="$1"
    local gz_file="$2"
    local jsonl_file="${gz_file%.gz}"

    if [ -f "$jsonl_file" ]; then
        echo "File already exists: $jsonl_file"
        return 0
    fi

    if [ ! -f "$gz_file" ]; then
        echo "Downloading: $url"
        echo "  -> $gz_file"
        curl -L --progress-bar -o "$gz_file" "$url"
    fi

    echo "Extracting: $gz_file"
    gunzip -v "$gz_file"
    echo "Done: $(du -h "$jsonl_file" | cut -f1)"
}

echo "=== Amazon Reviews 2023 Downloader ==="
echo "Category: $CATEGORY"
echo "Type: $TYPE"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

if [ "$TYPE" = "reviews" ] || [ "$TYPE" = "both" ]; then
    REVIEW_URL="${BASE_URL}/review_categories/${CATEGORY}.jsonl.gz"
    REVIEW_FILE="${OUTPUT_DIR}/${CATEGORY}.jsonl.gz"
    download_and_extract "$REVIEW_URL" "$REVIEW_FILE"
fi

if [ "$TYPE" = "meta" ] || [ "$TYPE" = "both" ]; then
    META_URL="${BASE_URL}/meta_categories/meta_${CATEGORY}.jsonl.gz"
    META_FILE="${OUTPUT_DIR}/meta_${CATEGORY}.jsonl.gz"
    download_and_extract "$META_URL" "$META_FILE"
fi

echo ""
echo "=== Download complete ==="
ls -lh "${OUTPUT_DIR}"/*.jsonl 2>/dev/null || echo "No files downloaded yet"
