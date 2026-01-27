#!/bin/bash
# Ground Truth Pipeline for T* System
#
# Usage:
#     ./scripts/run_gt_pipeline.sh              # Full: extract + compute GT for all T* policies
#     ./scripts/run_gt_pipeline.sh --aggregate  # Skip extract, just aggregate + compute
#     ./scripts/run_gt_pipeline.sh --tier T1    # Only compute GT for T1 tier
#
# Pipeline:
# 1. Extract L0 judgments for topics (G1_allergy, etc.) - reuses existing extractions
# 2. Compute GT for T* policies (35 total: 5 tiers × 7 variants)
#
# T* System:
#   - T1 (allergy), T2 (price_worth), T3 (environment), T4 (execution), T5 (server)
#   - P1-P3: Rule variants (base, extended, ALL logic)
#   - P4-P7: Format variants (reorder v1, reorder v2, XML, prose)

set -e

source .venv/bin/activate

TIER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --aggregate)
            AGGREGATE=true
            shift
            ;;
        --tier)
            TIER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--aggregate] [--tier T1|T2|T3|T4|T5]"
            exit 1
            ;;
    esac
done

# Step 1: Extract (if not --aggregate)
if [ "$AGGREGATE" = true ]; then
    echo "============================================================"
    echo "STEP 1: Aggregate cached raw data (--aggregate mode)"
    echo "============================================================"
    python -m addm.tasks.cli.extract --k 200 --all --aggregate
else
    echo "============================================================"
    echo "STEP 1: Extract L0 judgments (topics for T* tiers)"
    echo "============================================================"
    # Extract for topics used by T* system
    for TOPIC in G1_allergy G3_price_worth G4_environment G5_execution G4_server; do
        echo ""
        echo "--- Extracting $TOPIC ---"
        python -m addm.tasks.cli.extract --topic $TOPIC --k 200 --mode batch
    done
fi

echo ""
echo "============================================================"
echo "STEP 2: Compute Ground Truth for T* policies"
echo "============================================================"

# Determine which policies to compute
if [ -n "$TIER" ]; then
    POLICIES="--tier $TIER"
    echo "Computing GT for tier: $TIER (7 policies)"
else
    POLICIES=""
    echo "Computing GT for all 35 T* policies"
fi

for K in 25 50 100 200; do
    echo ""
    echo "--- Computing GT for K=$K ---"
    if [ -n "$TIER" ]; then
        # Expand tier to policy list
        for P in 1 2 3 4 5 6 7; do
            python -m addm.tasks.cli.compute_gt --policy "${TIER}P${P}" --k $K
        done
    else
        # All tiers
        for T in T1 T2 T3 T4 T5; do
            for P in 1 2 3 4 5 6 7; do
                python -m addm.tasks.cli.compute_gt --policy "${T}P${P}" --k $K
            done
        done
    fi
done

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Generated GT files in data/answers/yelp/"
echo "  Format: {policy}_K{k}_groundtruth.json"
echo "  Example: T1P1_K200_groundtruth.json"
