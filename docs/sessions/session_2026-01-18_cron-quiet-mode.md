# Session Log: Cron Quiet Mode & Log Redirect

**Date**: 2026-01-18
**Status**: completed

## Summary

Implemented quiet mode for cron jobs and sidecar log files. Cron polling now defaults to quiet (no "in progress" spam) and redirects output to log files that are auto-deleted on batch completion.

## Decisions Made

- `--quiet` is now default behavior (no in-progress status messages)
- `--show-status` flag to see in-progress messages if needed (not `--no-quiet`)
- Cron output redirects to sidecar log files next to manifest/batch files
- Log files auto-deleted on batch completion
- Use `output` singleton from `output.py` instead of raw `print()` (best practice)

## Changes Made

### `src/addm/tasks/cli/extract.py`
- Added `--show-status` flag (default: quiet)
- Added `_get_manifest_log_path()` and `_get_batch_log_path()` helpers
- Added `_delete_batch_log()` cleanup function
- Updated `_delete_manifest()` to also delete sidecar log
- All cron commands now redirect to log files: `>> {log_path} 2>&1`
- Fixed bug: `command` variable was inside `if args.verbose` block
- Mac users hint: "Click 'Allow' if a permission popup appears"

### `.claude/CLAUDE.md`
- Updated project structure with `data/tasks/{dataset}/` contents
- Added `results/` directory structure
- Added best practice: use `output` singleton vs raw `print()`

## File Locations

```
data/tasks/{domain}/
├── policy_cache.json       # L0 judgment cache
├── batch_manifest_*.json   # Multi-batch tracking
├── batch_manifest_*.log    # Manifest sidecar log (auto-deleted)
└── batch_*.log             # Single-batch log (auto-deleted)
```

## Key Files Modified

- `src/addm/tasks/cli/extract.py` - Cron quiet mode, log redirect
- `.claude/CLAUDE.md` - Project memory updates

## Next Steps

- Test cron with new quiet mode + log redirect
- Clear existing mail: `> /var/mail/jimmyyeh`

## Context

User was getting "You have mail" messages every 5 minutes from cron outputting batch status. Solution: quiet by default + log to file instead of mail.
