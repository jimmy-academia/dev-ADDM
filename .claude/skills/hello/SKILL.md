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

### 1. Read Project Roadmap

```bash
cat docs/ROADMAP.md 2>/dev/null
```

If roadmap exists, extract:
- Current phase and status
- Days remaining to deadline
- Today's focus (from weekly schedule)
- Critical path tasks
- Blockers/risks

### 2. Discover Recent Session Logs

```bash
ls -t docs/sessions/*.md 2>/dev/null | head -5
```

If no logs exist, skip to step 5.

### 3. Read Latest Session Logs

Read the **2-3 most recent** session logs. For each, extract:
- Topic/summary
- Status (in-progress, completed, blocked)
- Key decisions
- Next steps
- Open questions

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
- Be concise - this is a briefing, not a novel
- **Start with roadmap status** (phase, timeline, today's focus)
- **Integrate roadmap + session logs** in suggested todos
- Prioritize in-progress work over completed
- Highlight blockers or open questions from both roadmap and sessions
- List concrete next steps aligned with project timeline

**DON'T:**
- Dump entire session logs or roadmap
- Include completed work details (just note it's done)
- Overwhelm with too many todos (4-6 max)
- Skip the overview and jump into work

**If roadmap doesn't exist:**
- Omit the PROJECT STATUS section
- Fall back to session logs only
- Note that `/roadmap` can be used to track milestones

## Example Output

```
🗓️ PROJECT STATUS (from ROADMAP.md)

Phase: Phase I - G1_allergy Pipeline Validation
Timeline: 14 days to goal (Feb 1), 21 days to hard deadline (Feb 8)
Today's Focus: A1 (Aggregate GT), B1 (Polish Introduction)
Blockers: gpt-5-mini batch needs re-submission

─────────────────────────────────────────

📋 RECENT SESSIONS

1. [2026-01-18] Topic: Batch Pipeline Refactor
   Status: completed
   Summary: Removed cron, created run_g1_allergy.sh polling script.
   Next: Test pipeline script, commit changes

2. [2026-01-18] Topic: Temperature Fix
   Status: completed
   Summary: Fixed gpt-5-mini batch failures (temperature=0 unsupported).
   Next: Re-submit missing gpt-5-mini extractions

─────────────────────────────────────────

🎯 SUGGESTED TODOS

Based on roadmap + recent sessions:
- [ ] A1: Aggregate G1_allergy raw judgments → consensus L0
- [ ] Re-submit gpt-5-mini batch (18,440 requests)
- [ ] Test run_g1_allergy.sh pipeline script
- [ ] B1: Polish Introduction, define key claims for Discussion

─────────────────────────────────────────

📁 KEY FILES (from recent sessions)

- scripts/run_g1_allergy.sh - New polling script
- src/addm/tasks/cli/extract.py - Batch extraction CLI
- src/addm/tasks/extraction.py - PolicyJudgmentCache
- data/tasks/yelp/policy_cache.json - L0 judgment cache

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
