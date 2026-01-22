#!/bin/bash
# Ground Truth Pipeline: Extract + Compute GT for all topics.
#
# Usage:
#     ./scripts/run_gt_pipeline.sh
#
# Pipeline:
# 1. Extract L0 judgments for all 18 topics (K=200, batch mode)
# 2. Compute GT for all 72 policies across K=25/50/100/200
#    (compute_gt will skip if cache incomplete)

set -e

echo "============================================================"
echo "STEP 1: Extract L0 judgments (all 18 topics, K=200)"
echo "============================================================"

source .venv/bin/activate 

python -m addm.tasks.cli.extract --k 200 --mode batch --all

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
