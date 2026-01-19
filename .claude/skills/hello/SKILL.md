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

## Protocol (Optimized for Token Efficiency)

### 1. Read Project Roadmap Summary

```bash
cat docs/ROADMAP_SUMMARY.md 2>/dev/null
```

**If ROADMAP_SUMMARY.md exists**, it contains:
- Current phase, timeline, days to deadline
- Today's focus, critical path status, blockers

**If it doesn't exist**, fall back to reading `docs/ROADMAP.md` but extract ONLY:
- Current phase line
- Timeline section (goal/hard deadline dates)
- Today's focus line
- Critical blockers (if any)

### 2. Discover Recent Session Logs

```bash
ls -t docs/sessions/*.md 2>/dev/null | head -2
```

Read only **1-2 most recent** session logs (not 5, not 3-5).

If no logs exist, skip to step 4.

### 3. Read Latest Session Logs

Session logs now use compact format (max 20 lines each). Extract:
- Topic/summary from title
- Status emoji (✅/🔄/⛔)
- Next steps
- Uncommitted file count (if any)

### 4. Produce Overview

Display a concise briefing:

```
🗓️ PROJECT STATUS (from ROADMAP.md)

Phase: <current phase>
Timeline: <days remaining> to goal, <days remaining> to hard deadline
Today's Focus: <tasks from weekly schedule>
Blockers: <critical blockers if any>

─────────────────────────────────────────

📋 RECENT SESSIONS

1. [DATE] Topic: <topic>
   Status: <status>
   Summary: <1-2 sentences>
   Next: <key next step>

2. [DATE] Topic: <topic>
   ...

─────────────────────────────────────────

🎯 SUGGESTED TODOS

Based on roadmap + recent sessions:
- [ ] <Today's focus from roadmap>
- [ ] <Next step from most recent in-progress session>
- [ ] <Critical path task if applicable>
- [ ] <Open question that needs resolution>

─────────────────────────────────────────

📁 KEY FILES (from recent sessions)

- path/to/file.py - <context>
- path/to/other.py - <context>

─────────────────────────────────────────

Ready to continue. What would you like to work on?
```

### 5. If No Session Logs Exist

Still show roadmap status if available:

```
🗓️ PROJECT STATUS (from ROADMAP.md)

Phase: <current phase>
Timeline: <days remaining> to goal
Today's Focus: <tasks from weekly schedule>

─────────────────────────────────────────

📋 NO SESSION LOGS FOUND

This appears to be a fresh start or logs haven't been created yet.

To enable session persistence:
- Use /bye when ending sessions to save context
- Session logs will be stored in docs/sessions/

─────────────────────────────────────────

🎯 SUGGESTED TODOS

Based on roadmap:
- [ ] <Today's focus from roadmap>
- [ ] <Critical path task>

─────────────────────────────────────────

Ready to start. What would you like to work on?
```

### 6. Populate Todo List (Optional)

If the user wants to continue previous work, use TodoWrite to populate the todo list with next steps from roadmap and session logs.

## Output Format Guidelines

**DO:**
- **Be ultra-concise** - target 30-40 lines total output
- Use ROADMAP_SUMMARY.md (not full ROADMAP.md)
- Read only 1-2 session logs (not 3-5)
- Session logs are now compact (max 20 lines each)
- Show status emojis (✅/🔄/⛔) not verbose descriptions
- List 3-5 todos max (prioritize critical path)
- List 3-5 key files max (most relevant only)

**DON'T:**
- Run git status (adds tokens, low value)
- Include completed work details (just show status emoji)
- Read full ROADMAP.md (528 lines → use summary instead)
- List all files from session logs (pick 3-5 most critical)

**Token budget target**: <5,000 tokens for entire /hello execution

**If roadmap doesn't exist:**
- Omit the PROJECT STATUS section
- Fall back to session logs only
- Note that `/roadmap` can be used to track milestones

## Example Output

```
🗓️ PROJECT STATUS

Phase: Phase I - G1_allergy Pipeline Validation
Timeline: 13 days to goal (Feb 1), 20 days to hard deadline (Feb 8)
Today's Focus: A4 (AMOS), B2 (Related Work)
Blockers: None

─────────────────────────────────────────

📋 RECENT SESSIONS

1. [2026-01-19] AMOS Generalization Fixes ✅
   Next: Validate on diverse policies, commit changes

2. [2026-01-19] Evaluation Docs & CLI ✅
   Next: Run AMOS evaluation, verify >75% accuracy

─────────────────────────────────────────

🎯 SUGGESTED TODOS

- [ ] A4: Run AMOS evaluation on G1_allergy_V2 (K=50)
- [ ] Commit AMOS generalization fixes (8 modified, 2 new)
- [ ] Validate AMOS on G2_romance_V2, G3_price_worth_V2
- [ ] B2: Draft Related Work (informs baseline selection)

─────────────────────────────────────────

📁 KEY FILES

- src/addm/methods/amos/ - Phase 1 & 2 (temporal support added)
- docs/AMOS_GENERALIZATION.md - Generalizability analysis
- docs/specs/evaluation.md - 3-score evaluation system

─────────────────────────────────────────

Ready to continue. What would you like to work on?
```

## Integration with Other Commands

| Command | Purpose | What /hello uses |
|---------|---------|------------------|
| `/bye` | Session end | Writes to docs/sessions/ → /hello reads them |
| `/roadmap` | Milestone tracking | Updates ROADMAP.md → /hello shows project status |
| `/orient` | Project overview | Reads CLAUDE.md for structure |

**Session workflow:**
1. `/hello` at start → roadmap + session logs → suggested todos
2. Work on tasks
3. `/roadmap` to update milestones (if needed)
4. `/bye` at end → writes session log for next time

## When to Use

- **Session start**: Run `/hello` to recover context
- **After a break**: Run `/hello` if you've been away
- **Context lost**: Run `/hello` if conversation seems confused
- **New topic**: Consider `/hello` to see what else is in-progress

## Related

- `/bye` - Session end, writes session logs
- `/roadmap` - Project milestone tracking (updates ROADMAP.md)
- `/orient` - General project orientation (reads CLAUDE.md)
- `/doc` - Quick context capture
