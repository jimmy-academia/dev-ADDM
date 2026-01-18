---
name: bye
description: Clean exit with session context capture. Writes detailed resume log to docs/logs/, updates CLAUDE.md with pointer. Use when ending a session.
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Grep, Glob, TodoWrite, Task, AskUserQuestion
---

# /bye - Clean Exit with Session Persistence

Ensures session context is captured for seamless resume in next session.

## Core Principle

**The next session's Claude has NO memory of this session.** Your job is to create a "briefing document" that lets future-Claude immediately understand and continue the work.

## Exit Protocol

### 1. Auto-Summarize Session Context

**This is the most critical step.** Analyze the conversation and extract:

- **Topic**: What was the main focus? (1-2 sentences)
- **Decisions Made**: Any conclusions reached
- **Current State**: Where did we leave off?
- **Next Steps**: What should happen next session?
- **Open Questions**: Unresolved issues or choices
- **Key Files**: Files that were discussed or modified

Create a session log at `docs/logs/session_YYYY-MM-DD_<topic-slug>.md`:

```markdown
# Session Log: <Topic>

**Date**: YYYY-MM-DD HH:MM
**Status**: <in-progress | completed | blocked>

## Summary

<2-3 sentence summary of what was worked on>

## Decisions Made

- <Decision 1>
- <Decision 2>

## Current State

<Where we left off - be specific enough that someone can continue>

## Next Steps

1. <Next step 1>
2. <Next step 2>

## Open Questions

- <Question or unresolved choice>

## Key Files

- `path/to/file.py` - <why relevant>

## Context & Background

<Any important context, links to docs, or background info>
```

### 2. Session Log Location

Session logs are stored in `docs/logs/`. **Do NOT update CLAUDE.md with a pointer** - the user runs multiple concurrent sessions, so a single pointer would cause conflicts.

At session startup, Claude automatically reads the latest session logs from `docs/logs/` to understand recent context.

### 3. Git Status Check

```bash
git status --short
```

Display if there are uncommitted changes. Do NOT offer to commit - just inform.

### 4. Todo Check

Check current todo list:
- List any incomplete todos
- These should be captured in the session log's "Next Steps"

### 5. Background Process Check

```bash
ps aux | grep -E '(python|node|npm|pytest)' | grep -v grep | head -5
```

Alert if processes are running.

### 6. Final Message

```
Session context saved to: docs/logs/session_YYYY-MM-DD_<topic>.md
CLAUDE.md updated with resume pointer.

Next session: Read the log file to continue where we left off.

Exit with Ctrl+C (safest) or Ctrl+D.

Goodbye!
```

## Important Guidelines

### Session Summarization

**DO**:
- Be specific about technical decisions (e.g., "Decided to use Option A: single batch mode")
- Include code snippets or file paths if relevant
- Capture the "why" behind decisions
- Note any blockers or dependencies

**DON'T**:
- Be vague (e.g., "worked on some stuff")
- Skip technical details
- Assume future-Claude remembers anything
- Write a novel - be concise but complete

### Log File Naming

Use descriptive topic slugs:
- `session_2025-01-18_gt-generation-batch-mode.md`
- `session_2025-01-18_fix-auth-bug.md`
- `session_2025-01-18_refactor-query-system.md`

### Multiple Sessions Same Day

If multiple sessions on same day, append time or number:
- `session_2025-01-18_gt-generation.md`
- `session_2025-01-18_gt-generation-2.md`

### Completed Work

If work is completed (not just paused):
- Set status to "completed"
- Remove "Active Work" section from CLAUDE.md
- Or update it to point to next priority

## Example Session Log

```markdown
# Session Log: Ground Truth Generation Script

**Date**: 2025-01-18 14:30
**Status**: in-progress

## Summary

Designed the batch mode approach for ground truth generation. Decided on Option A
(single OpenAI batch) over Option C (multiple concurrent batches) for simplicity.

## Decisions Made

- Use Option A: single batch submission via OpenAI Batch API
- Reuse existing `extract.py` batch infrastructure (already implements Option A)
- Goal: unify judgment extraction + GT computation into one script

## Current State

- Discovered `extract.py` already has `--mode 24hrbatch` support
- `compute_gt.py` exists but uses old formula modules (G1a/b/c/d), not new PolicyIR (V0-V3)
- Need to either:
  - Add `--mode 24hrbatch` to `compute_gt.py`, OR
  - Create new unified script

## Next Steps

1. Decide: modify `compute_gt.py` or create new script
2. Implement batch submission for judgment extraction
3. Implement collection + GT computation on batch completion
4. Test with G1_allergy_V2 policy

## Open Questions

- Should compute_gt.py gain --mode support, or create compute_gt_batch.py?
- How to handle PolicyIR scoring (V2/V3) vs old formula modules?

## Key Files

- `src/addm/tasks/cli/extract.py` - existing batch mode implementation
- `src/addm/tasks/cli/compute_gt.py` - current GT computation (old formulas)
- `src/addm/llm_batch.py` - BatchClient, build_chat_batch_item
- `src/addm/query/policies/G1/allergy/V2.yaml` - example PolicyIR with scoring

## Context & Background

- Project uses PolicyIR system (V0-V3) replacing old formula modules (a/b/c/d)
- See `docs/specs/query_construction.md` for PolicyIR details
- Batch API gives 50% cost discount, 24hr completion window
```

## Decision Tree

```
User invokes /bye
│
├─► 1. Analyze conversation
│   └─► Extract: topic, decisions, state, next steps, questions
│
├─► 2. Write session log
│   └─► docs/logs/session_YYYY-MM-DD_<topic>.md
│
├─► 3. (No CLAUDE.md update - concurrent sessions)
│   └─► Session logs auto-discovered at startup
│
├─► 4. Check git status (inform only)
│
├─► 5. Check todos (captured in log)
│
├─► 6. Check processes
│
└─► 7. Display exit message
```

## Related

- `/hello` - Session startup, reads session logs, produces overview
- `/orient` - General project orientation (reads CLAUDE.md)
- `/doc` - Quick documentation capture
