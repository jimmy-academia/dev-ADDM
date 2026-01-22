#!/bin/bash
# Temporary script for G1_allergy GT extraction
# Later: expand to full pipeline (baselines, AMOS) as code is implemented
set -e

# Configuration
POLL_INTERVAL=300  # 5 minutes
LOG_DIR="results/logs/extraction"
LOG_FILE="$LOG_DIR/g1_allergy.log"  # Fixed name - appends on re-run

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Activate venv
source .venv/bin/activate

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to run extraction and wait for completion
run_extraction() {
    local topic=$1
    local k=$2

    log "Starting extraction for $topic (K=$k)"

    while true; do
        # Run extraction (submits batch if needed, processes if complete)
        output=$(.venv/bin/python -m addm.tasks.cli.extract --topic "$topic" --k "$k" 2>&1)
        echo "$output" | tee -a "$LOG_FILE"

        # Check if complete
        if echo "$output" | grep -q "All extractions complete\|No reviews need extraction"; then
            log "✓ $topic extraction complete"
            return 0
        fi

        # Check if batch is still processing
        if echo "$output" | grep -q "still processing\|status:\|STILL PROCESSING"; then
            log "Batch processing... waiting ${POLL_INTERVAL}s (Ctrl+C to stop)"
            sleep $POLL_INTERVAL
            continue
        fi

        # If we get here, something unexpected - continue anyway
        log "Checking again in ${POLL_INTERVAL}s..."
        sleep $POLL_INTERVAL
    done
}

# Main
log "=== G1_allergy GT Extraction ==="

# Step 1: Extract G1_allergy (waits for batch)
run_extraction "G1_allergy" 200

# Step 2: Compute GT for all policy variants
log "Computing ground truth..."
.venv/bin/python -m addm.tasks.cli.compute_gt \
    --policy G1_allergy_V0,G1_allergy_V1,G1_allergy_V2,G1_allergy_V3 \
    --k 200 2>&1 | tee -a "$LOG_FILE"

log "=== G1_allergy GT Complete ==="
log "Results: data/tasks/yelp/G1_allergy_*_groundtruth.json"

# TODO: Later phases (when implemented)
# - Run baselines (direct, cot, react, rag)
# - Run AMOS
# - Run advanced baselines (LOTUS, ToT, GoT)
# - Generate analysis
