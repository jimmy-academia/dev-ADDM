# ATKD Fix Log (Long-Running)

This is a running engineering log for improving AMOS Phase 2 (ATKD) on `T1P1` (Yelp, K=25) until it reliably finds the rare positives and matches the verify-all baseline verdicts.

## Baseline (Before Fixes)

**Observed failure mode (from run `results/dev/20260128_232850_yelp_T1P1_K25`):**
- Ground truth distribution (`data/answers/yelp/T1P1_K25_groundtruth.json`): 98 Low Risk, 1 High Risk, 1 Critical Risk.
- After 2 iterations (64 verified reviews): **0 positive tags** across all primitives; and **0 verified reviews** came from the 2 GT-positive restaurants.
- GateScan Z rankings for the GT-positive restaurants were poor (not near top of 100).

**Root cause hypothesis:**
- BM25 scoring treated a multi-token query as an OR-of-tokens (sum of token contributions), so gates like `emergency_room visit` matched on common tokens like `room`, producing high false-positive Z and steering scheduling away from true signal.
- Negative BM25 gates were similarly vulnerable (e.g., `allergy accommodation` firing on `allergy` alone), suppressing true signals.
- Scheduler’s VOI score was dominated by global calibration (`v_global`) early (theta_hat flat at 0.5), so selection did not prioritize high-suspicion reviews when calibration had no supervision yet.

## Change 1: BM25 Multi-Token AND Semantics

**Code change:** `src/addm/methods/amos/phase2_atkd.py` (`BM25Index.score`)
- If a BM25 query has multiple tokens, require ALL tokens to exist in the document before scoring.

**Why:**
- Prevents single common token hits from dominating (e.g., `"emergency room"` matching any review containing only `"room"`).

**Observation (GateScan-only):**
- Still saw top-Z reviews that were unrelated to the policy topic, suggesting another issue beyond BM25 OR semantics.

## Change 2: Tie-Aware Rank Normalization (Fix Z Randomness)

**Code change:** `src/addm/methods/amos/phase2_atkd.py` (`rank_normalize`)
- Fixed tie handling: equal scores now get the same (average) rank.

**Why:**
- Large tie groups (especially BM25 scores being exactly 0 for most reviews) were previously assigned arbitrary distinct ranks.
- That made `Z` effectively random for non-matching reviews and polluted scheduling.

**Observation (GateScan-only):**
- Z distributions became much more stable.
- The GT-positive restaurants started to rise substantially in Z ranking (especially on `High Risk_clause_1`).

## Change 3: GateInit Prompt Tightening (+ topic_anchors output)

**Code change:** `src/addm/methods/amos/phase2_prompts.py`
- Added stricter instructions to avoid generic restaurant-quality gates and sentiment-only keywords.
- Added `topic_anchors` field to the GateInit JSON schema (for future inspection/control).

**Observation (GateScan-only, run example):**
- With the rank normalization fix, GateScan produced meaningful Z:
  - `High Risk_clause_1`: the GT High Risk restaurant (`Front Street Cafe`) ranked **#1/100** by max-Z, and the GT Critical Risk restaurant (`Cafe Blue Moose`) ranked in the top ~10 by max-Z.
- Top-Z reviews for severe clauses still include non-topic “medical-ish” confusers (e.g., food poisoning) because severe gates contain general medical terms (ER/hospital). This is acceptable short-term if scheduler hunts by Z and verifier rejects confusers.

## Change 4: Scheduler Uses Z for Hunt + Scale-Stable VOI

**Code change:** `src/addm/methods/amos/phase2_atkd.py` (`_select_batch`)
- `v_hunt` now uses per-review `Z` directly (works even when calibration is uninformative early).
- `v_local` and `v_global` are scaled by proportions (restaurant-local and global unverified fractions) to avoid raw-count domination.
- Forced float casting for VOI components to keep debug JSON dumps serializable.

**Expected impact:**
- Early iterations should actually verify the highest-suspicion reviews (including those from the 2 GT-positive restaurants) instead of spending most budget on global calibration bins.
- Confuser reviews that score high on generic “medical” language should be rejected by the review-level verifier, tightening calibration and reducing future wasted verifications.
