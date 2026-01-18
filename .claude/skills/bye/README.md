# /bye - Clean Exit with Session Persistence

## What It Does

The `/bye` command captures your session context so the next Claude session can seamlessly continue where you left off.

**Key feature**: Automatically summarizes what was discussed, decisions made, and next steps - no manual note-taking required.

## How It Works

1. **Auto-summarizes** the session conversation
2. **Writes session log** to `docs/logs/session_YYYY-MM-DD_<topic>.md`
3. **Updates CLAUDE.md** with a pointer to the log
4. **Checks** git status, todos, background processes
5. **Displays** exit instructions

## How to Use

```
/bye
```

Or naturally: "bye", "goodbye", "I'm done for today"

## What Gets Captured

The session log includes:

- **Summary**: What was worked on (2-3 sentences)
- **Decisions Made**: Conclusions reached
- **Current State**: Where you left off
- **Next Steps**: What to do next session
- **Open Questions**: Unresolved issues
- **Key Files**: Relevant files discussed/modified

## Example Flow

```
You: /bye

Analyzing session context...

Session log written to: docs/logs/session_2025-01-18_gt-generation.md
CLAUDE.md updated with resume pointer.

Git status: 2 uncommitted files
  M src/addm/tasks/cli/compute_gt.py
  M .claude/CLAUDE.md

Todos: 1 incomplete
  - "Add batch mode to compute_gt" (pending)

Exit with Ctrl+C (safest) or Ctrl+D.

Goodbye!
```

## Next Session

When you start a new session, Claude will:
1. Read `.claude/CLAUDE.md`
2. See the "Active Work" pointer
3. Read the session log
4. Have full context to continue

## Why This Matters

**Without /bye**:
- Next session starts from scratch
- You repeat explanations
- Decisions get re-discussed
- Context is lost

**With /bye**:
- Next session has full context
- Work continues seamlessly
- Decisions are preserved
- No repeated exploration

## Session Log Location

```
docs/logs/
├── session_2025-01-18_gt-generation.md
├── session_2025-01-17_fix-auth-bug.md
└── session_2025-01-16_refactor-query.md
```

## Proper Exit Methods

After running `/bye`:

- `Ctrl+C` - Safest, sends proper cleanup signals
- `Ctrl+D` - Standard EOF exit

## Related Commands

- `/hello` - Session startup, reads logs, produces overview
- `/orient` - General project orientation
- `/doc` - Quick documentation capture
