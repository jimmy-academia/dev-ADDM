---
name: bye
description: Clean exit with session context capture. Writes detailed resume log to docs/sessions/, updates CLAUDE.md with pointer. Use when ending a session.
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

Create a session log at `docs/sessions/session_YYYY-MM-DD_<topic-slug>.md`:

**Use compact format (max 20 lines):**

```markdown
# [YYYY-MM-DD] <Topic> <✅|🔄|⛔>

**What**: One-sentence summary of what was worked on.

**Impact**: Key outcome or result (optional, if significant).

**Files**:
- `path/to/file.py` - Brief note
- `path/to/other.py` - Brief note

**Next**:
- Next step 1
- Next step 2

**Status**: <Uncommitted: X files | Blocked: reason | Ready: next action>

**Context**: Any critical background (1-2 sentences max, optional)
```

**Format guidelines**:
- Use emojis: ✅ (completed), 🔄 (in-progress), ⛔ (blocked)
- Keep total length under 20 lines
- Be specific but concise
- Focus on actionable information

### 2. Session Log Location

Session logs are stored in `docs/sessions/`. **Do NOT update CLAUDE.md with a pointer** - the user runs multiple concurrent sessions, so a single pointer would cause conflicts.

At session startup, Claude automatically reads the latest session logs from `docs/sessions/` to understand recent context.

### 3. Git Status Check

```bash
git status --short
```

Display if there are uncommitted changes. Do NOT offer to commit - just inform.

### 4. Todo Check & Clear

Check current todo list:
- List any incomplete todos
- Capture these in the session log's "Next Steps"
- **Clear the todo list after archiving**: `TodoWrite(todos=[])`
- This ensures fresh start for next session (todos persisted in session log only)

### 5. Background Process Check

```bash
ps aux | grep -E '(python|node|npm|pytest)' | grep -v grep | head -5
```

Alert if processes are running.

### 6. Documentation Reminder

If significant changes were made this session, remind about doc maintenance:

```
📝 You made changes this session. Consider:
   /sync  - Check doc consistency (.claude/ ↔ docs/)
   /audit - Verify docs match code (deep check)
```

### 7. Final Message with Two Paths

```
✓ Session context saved to: docs/sessions/session_YYYY-MM-DD_<topic>.md

─────────────────────────────────────────

What's next?

  Continue working?  →  /compact  (clears context, keeps session)
  Done for now?      →  /exit     (ends session)

─────────────────────────────────────────

📊 Project progress: /roadmap
📋 Session history:  /hello
```

## Two Exit Paths

```
/bye
  │
  ├─► Save session context
  │
  └─► Two options:
      │
      ├─► /compact
      │   └─► Clear context, stay in session
      │       (good for switching to unrelated task)
      │
      └─► /exit
          └─► End session entirely
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
# [2025-01-18] Ground Truth Generation Script 🔄

**What**: Designed batch mode approach for GT generation. Chose Option A (single OpenAI batch) over concurrent batches for simplicity.

**Files**:
- `src/addm/tasks/cli/extract.py` - Has `--mode batch` support
- `src/addm/tasks/cli/compute_gt.py` - Needs PolicyIR support (currently uses old formulas)
- `src/addm/llm_batch.py` - BatchClient infrastructure

**Next**:
- Decide: modify compute_gt.py or create new unified script
- Implement batch submission for judgment extraction
- Test with G1_allergy_V2 policy

**Status**: Uncommitted: 3 files

**Context**: Project now uses PolicyIR (V0-V3) instead of old formula modules (a/b/c/d). Batch API gives 50% cost discount, 24hr window.
```

## Decision Tree

```
User invokes /bye
│
├─► 1. Analyze conversation
│   └─► Extract: topic, decisions, state, next steps, questions
│
├─► 2. Write session log
│   └─► docs/sessions/session_YYYY-MM-DD_<topic>.md
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
├─► 7. Show doc maintenance reminder (if changes made)
│
└─► 8. Display two options:
    ├─► /compact (continue working, fresh context)
    └─► /exit (done for now)
```

## Related

- `/hello` - Session startup, reads session logs, produces overview
- `/orient` - General project orientation (reads CLAUDE.md)
- `/doc` - Quick documentation capture
- `/roadmap` - Project progress tracking
- `/sync` - Quick doc consistency check
- `/audit` - Deep doc verification
