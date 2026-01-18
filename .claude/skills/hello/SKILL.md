---
name: hello
description: Session startup - reads recent session logs, produces overview, suggests next steps. Pairs with /bye. Use at the start of a session or when returning after a break.
user-invocable: true
allowed-tools: Read, Bash, Glob, Grep, TodoWrite
---

# /hello - Session Startup with Context Recovery

Reads recent session logs and produces an actionable briefing for the current session.

## Core Principle

**You have NO memory of previous sessions.** This command recovers context by reading session logs written by `/bye`.

## Protocol

### 1. Discover Recent Session Logs

```bash
ls -t docs/logs/*.md 2>/dev/null | head -5
```

If no logs exist, skip to step 4.

### 2. Read Latest Session Logs

Read the **2-3 most recent** session logs. For each, extract:
- Topic/summary
- Status (in-progress, completed, blocked)
- Key decisions
- Next steps
- Open questions

### 3. Produce Overview

Display a concise briefing:

```
📋 RECENT SESSIONS

1. [DATE] Topic: <topic>
   Status: <status>
   Summary: <1-2 sentences>
   Next: <key next step>

2. [DATE] Topic: <topic>
   ...

─────────────────────────────────────────

🎯 SUGGESTED TODOS

Based on recent sessions, consider:
- [ ] <Next step from most recent in-progress session>
- [ ] <Open question that needs resolution>
- [ ] <Pending task>

─────────────────────────────────────────

📁 KEY FILES (from recent sessions)

- path/to/file.py - <context>
- path/to/other.py - <context>

─────────────────────────────────────────

Ready to continue. What would you like to work on?
```

### 4. If No Session Logs Exist

```
📋 NO SESSION LOGS FOUND

This appears to be a fresh start or logs haven't been created yet.

To enable session persistence:
- Use /bye when ending sessions to save context
- Session logs will be stored in docs/logs/

Ready to start. What would you like to work on?
```

### 5. Populate Todo List (Optional)

If the user wants to continue previous work, use TodoWrite to populate the todo list with next steps from the session log.

## Output Format Guidelines

**DO:**
- Be concise - this is a briefing, not a novel
- Prioritize in-progress work over completed
- Highlight blockers or open questions
- List concrete next steps

**DON'T:**
- Dump entire session logs
- Include completed work details (just note it's done)
- Overwhelm with too many todos
- Skip the overview and jump into work

## Example Output

```
📋 RECENT SESSIONS

1. [2025-01-18] Topic: Ground Truth Generation Batch Mode
   Status: in-progress
   Summary: Designed batch approach for GT generation. Decided Option A
   (single batch). Found extract.py already supports --mode 24hrbatch.
   Next: Decide whether to modify compute_gt.py or create new script

2. [2025-01-17] Topic: Fix Auth Token Refresh
   Status: completed
   Summary: Fixed token refresh race condition in auth middleware.

─────────────────────────────────────────

🎯 SUGGESTED TODOS

Based on recent sessions, consider:
- [ ] Continue GT generation: decide compute_gt.py vs new script
- [ ] Open question: How to handle PolicyIR scoring (V2/V3)?

─────────────────────────────────────────

📁 KEY FILES (from recent sessions)

- src/addm/tasks/cli/extract.py - has batch mode implementation
- src/addm/tasks/cli/compute_gt.py - current GT computation
- src/addm/llm_batch.py - BatchClient

─────────────────────────────────────────

Ready to continue. What would you like to work on?
```

## Integration with /bye

| /bye (session end) | /hello (session start) |
|--------------------|------------------------|
| Analyzes conversation | Reads session logs |
| Writes to docs/logs/ | Discovers latest logs |
| Captures decisions, state, next steps | Extracts and displays them |
| No CLAUDE.md pointer (concurrent sessions) | Auto-discovers logs |

## When to Use

- **Session start**: Run `/hello` to recover context
- **After a break**: Run `/hello` if you've been away
- **Context lost**: Run `/hello` if conversation seems confused
- **New topic**: Consider `/hello` to see what else is in-progress

## Related

- `/bye` - Session end, writes session logs
- `/orient` - General project orientation (reads CLAUDE.md)
