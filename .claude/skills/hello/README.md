# /hello - Session Startup

## What It Does

Recovers context from previous sessions by reading logs written by `/bye`. Produces a briefing with recent work, suggested todos, and key files.

## How to Use

```
/hello
```

Or naturally: "hello", "what was I working on?", "catch me up"

## What You Get

1. **Recent Sessions** - Summary of last 2-3 sessions
2. **Suggested Todos** - Next steps from in-progress work
3. **Key Files** - Important files from recent sessions

## Example

```
📋 RECENT SESSIONS

1. [2025-01-18] Topic: Ground Truth Generation
   Status: in-progress
   Summary: Designed batch mode approach, decided Option A.
   Next: Modify compute_gt.py to add batch support

2. [2025-01-17] Topic: Fix Auth Bug
   Status: completed

─────────────────────────────────────────

🎯 SUGGESTED TODOS

- [ ] Continue GT generation: add --mode batch to compute_gt.py
- [ ] Open question: PolicyIR scoring implementation?

─────────────────────────────────────────

Ready to continue. What would you like to work on?
```

## Pairing with /bye

| Start Session | End Session |
|---------------|-------------|
| `/hello` | `/bye` |
| Reads docs/logs/ | Writes to docs/logs/ |
| Recovers context | Saves context |

## When to Use

- **Starting a new session** - Get caught up
- **Returning after a break** - Remember where you left off
- **Feeling lost** - Recover context from logs

## Requirements

Session logs must exist in `docs/logs/`. These are created by `/bye`.

## Related

- `/bye` - End session, save context
- `/orient` - General project orientation
