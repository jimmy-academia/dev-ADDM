---
name: bye
description: Clean exit with pre-exit checks - documentation sync, git status, todos, session cleanup. Use when ending a session, saying goodbye, or before closing Claude Code.
user-invocable: true
allowed-tools: Read, Bash, Grep, Glob, TodoWrite, Task, AskUserQuestion
---

# /bye - Clean Exit Command

Ensures clean session closure by checking documentation, git status, todos, and capturing final session notes.

## Exit Protocol Checklist

When invoked, execute ALL steps below systematically:

### 1. Documentation Sync Check

**Proactively run doc-sync agent** to harmonize `.claude/` and `docs/`:

```
Use Task tool with subagent_type="doc-sync" to synchronize documentation
```

**Purpose**: Ensure all significant code changes are documented before session ends.

### 2. Git Status Review

Check for uncommitted work:

```bash
git status
git diff --stat
```

**Display if present:**
- Uncommitted changes
- Debug code detected (print statements, breakpoints, TODO comments)
- Large number of modified files

**Note**: Just inform the user of the status. Do NOT offer to commit or run dogit/smart-commit. The user will handle commits separately if needed.

### 3. Todo Verification

Check current todo list status:

- List any **in_progress** or **pending** todos
- Ask: "You have incomplete todos. Should I document these for your next session?"
- If yes: Append to `.claude/CLAUDE.md` under a "## Session Resume Notes" section

### 4. Background Process Check

Verify no processes are running:

```bash
ps aux | grep -E '(python|node|npm|pytest)' | grep -v grep
```

**If found**:
- List active processes
- Ask: "These processes are still running. Should I terminate them?"

### 5. Session Memory Capture

Ask the user:

> "Before exiting, is there anything important to remember for the next session?"
>
> Examples:
> - Current blockers or issues
> - Next steps or priorities
> - Decisions made this session
> - Context about what you were working on

**If user provides notes**: Append to `.claude/CLAUDE.md` under:

```markdown
## Session Notes - [YYYY-MM-DD HH:MM]

[User's notes here]
```

### 6. Final Exit Message

**Display this message**:

```
✓ All checks complete. Ready to exit cleanly.

Recommended exit methods:
  • Ctrl+C - Safest, sends proper cleanup signals
  • Ctrl+D - Standard EOF exit
  • /exit - Built-in command (may cause MCP issues)

Goodbye!
```

## Exit Decision Tree

```
User invokes /bye
│
├─► 1. Run doc-sync agent (proactive)
│
├─► 2. Check git status
│   ├─► Has changes? → Display status (no commit)
│   └─► Clean? → Continue
│
├─► 3. Check todos
│   ├─► Has incomplete? → Ask to document
│   └─► All done? → Continue
│
├─► 4. Check processes
│   ├─► Found? → Ask to terminate
│   └─► None? → Continue
│
├─► 5. Capture session notes
│   └─► Ask user for important notes
│
└─► 6. Display exit instructions
    └─► Ready for Ctrl+C
```

## Example Output

```
🔍 PRE-EXIT CHECKLIST

✅ Documentation sync: Running doc-sync agent...
   → Updated 2 files in .claude/

⚠️  Git status: You have 3 uncommitted files
   → scripts/new_feature.py
   → tests/test_feature.py
   → .claude/CLAUDE.md

📋 Todos: 2 incomplete items
   → "Add error handling to parser" (in_progress)
   → "Update tests for edge cases" (pending)

🔧 Processes: Clean (no background tasks)

💭 Session notes captured to .claude/CLAUDE.md

─────────────────────────────────────────

✓ All checks complete. Ready to exit cleanly.

Recommended exit methods:
  • Ctrl+C (safest)
  • Ctrl+D
  • /exit

Goodbye!
```

## Implementation Notes

- **Always run doc-sync proactively** - don't ask permission first
- **Be concise** - show only relevant information
- **Use TodoWrite** if todos need to be updated/saved
- **Never execute exit** - only prepare and instruct
- **Update CLAUDE.md** atomically (read → append → write)

## Related Commands

- `/doc` - Quick documentation capture
- `/orient` - Load project context (session startup)

## Success Criteria

Exit agent succeeds when:
1. Documentation is synchronized
2. User is aware of uncommitted work
3. Incomplete todos are documented
4. Background processes are handled
5. Session notes are captured
6. User knows how to exit cleanly (Ctrl+C)
