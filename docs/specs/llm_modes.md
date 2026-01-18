# LLM Modes: ondemand vs 24hrbatch

This document records the design and implementation plan for supporting two
LLM execution modes:

- `ondemand`: current online calls (default).
- `24hrbatch`: offline batch submission with periodic fetch via cron.

## Design goals

- Keep existing ondemand behavior unchanged.
- Support batch runs without local reminder files.
- Use a cron job to fetch results automatically.
- Encode routing info in `custom_id` so outputs can be applied without a
  separate local mapping file.

## CLI behavior

New/updated flags:
- `--llm-mode {ondemand,24hrbatch}` (default: `ondemand`)
- `--batch-id <id>` optional; if present, fetch-only mode for cron runs.

No explicit `--batch-submit`/`--batch-fetch` flags. The presence of
`--batch-id` triggers fetch-only; otherwise a batch is submitted.

## Cron scheduling

- Default schedule: every 5 minutes.
- Cron line includes a marker comment: `# ADDM_BATCH_<batch_id>`.
- The fetch step removes the cron entry only after outputs are fetched,
  applied, and finalized.
- On Windows: print a warning and skip cron creation.

## Custom ID encoding (no local mapping files)

All batch requests must include a stable `custom_id` encoding that lets us
route outputs without local state.

Per-review judgments:
- `review::<task_or_policy>::<business_id>::<review_id>`

Per-restaurant judgments (baseline):
- `sample::<run_name>::<sample_id>`

The fetch step parses `custom_id` to determine where the result is stored.

## Integration surfaces

Ondemand flow stays in `src/addm/llm.py`.

Batch flow adds a small OpenAI batch client module (new file), used by:
- `src/addm/tasks/cli/extract.py` (per-review judgments)
- `src/addm/tasks/cli/run_baseline.py` (per-restaurant judgments)

## 24hrbatch flow

Submit:
1) Build batch request items with `custom_id`.
2) Submit batch and get `batch_id`.
3) Install cron job that runs the same CLI with `--batch-id <id>`.
4) Exit.

Fetch (cron run):
1) Check batch status.
2) If complete, download output and error files.
3) Parse lines, decode `custom_id`, and write to cache/results.
4) Finalize outputs (same post-response logic as ondemand).
5) Remove cron entry by marker.

If not complete, exit quietly; cron will retry.

## Error handling

- Transient errors: log and let cron retry.
- Batch expired (30 days): log and remove cron entry.
- Partial failures: requeue only failed items (future enhancement).
