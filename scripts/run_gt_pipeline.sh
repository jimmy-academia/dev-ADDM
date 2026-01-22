#!/bin/bash
# Ground Truth Pipeline: Extract + Compute GT for all topics.
#
# Usage:
#     ./scripts/run_gt_pipeline.sh              # Normal: extract + aggregate + compute
#     ./scripts/run_gt_pipeline.sh --aggregate  # Skip extract, just aggregate + compute
#
# Pipeline:
# 1. Extract L0 judgments for all 18 topics (K=200, batch mode)
# 2. Aggregate raw extractions (runs automatically, or use --aggregate to force)
# 3. Compute GT for all 72 policies across K=25/50/100/200

set -e

source .venv/bin/activate

if [ "$1" = "--aggregate" ]; then
    echo "============================================================"
    echo "STEP 1: Aggregate cached raw data (--aggregate mode)"
    echo "============================================================"
    python -m addm.tasks.cli.extract --k 200 --all --aggregate
else
    echo "============================================================"
    echo "STEP 1: Extract L0 judgments (all 18 topics, K=200)"
    echo "============================================================"
    python -m addm.tasks.cli.extract --k 200 --mode batch --all
fi

echo ""
echo "============================================================"
echo "STEP 2: Compute Ground Truth (all 72 policies)"
echo "============================================================"

for K in 25 50 100 200; do
    echo ""
    echo "--- Computing GT for K=$K ---"
    python -m addm.tasks.cli.compute_gt --k $K
done

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
