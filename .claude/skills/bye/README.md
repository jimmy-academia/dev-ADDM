# /bye - Clean Exit Command

## What It Does

The `/bye` command performs a comprehensive pre-exit checklist to ensure:
- Documentation is synchronized
- Git status is clean (or you're aware of changes)
- Todos are documented for next session
- No background processes are left running
- Session notes are captured
- You know how to exit cleanly

## How to Use

Simply type when you're ready to end your session:

```
/bye
```

Or naturally say:
```
"bye"
"goodbye"
"I'm done for today"
"help me exit cleanly"
```

## What Happens

The agent will:

1. **Run doc-sync** (automatically) - Synchronizes `.claude/` and `doc/`
2. **Check git status** - Alerts if uncommitted changes exist
3. **Review todos** - Lists incomplete items, offers to document them
4. **Scan processes** - Detects any running background tasks
5. **Capture notes** - Asks if you want to save session context
6. **Show exit instructions** - Displays proper exit method (Ctrl+C)

## Example Session

```
You: /bye

🔍 PRE-EXIT CHECKLIST

✅ Documentation sync: Running doc-sync agent...
   → Updated .claude/CLAUDE.md

⚠️  Git status: You have 2 uncommitted files
   → Would you like to commit them? [y/n]

📋 Todos: 1 incomplete item
   → "Refactor data loader" (pending)
   → Should I save this to CLAUDE.md for next session? [y/n]

🔧 Processes: Clean

💭 Session notes: Anything to remember for next session?
   (You can share context, blockers, or next steps)

─────────────────────────────────────────

✓ All checks complete. Ready to exit cleanly.

Recommended exit methods:
  • Ctrl+C (safest)
  • Ctrl+D
  • /exit

Goodbye!
```

## Why Use This

**Prevents:**
- MCP server failures on exit
- Lost session context
- Forgotten uncommitted work
- Orphaned background processes
- Stale documentation

**Ensures:**
- Clean handoff to next session
- Proper resource cleanup
- Memory preservation
- Documentation hygiene

## Proper Exit Methods

After running `/bye`:

✅ **Safest:**
- `Ctrl+C` - Sends proper cleanup signals to MCP servers
- `Ctrl+D` - Standard terminal EOF

⚠️ **May cause MCP issues:**
- `/exit` - Built-in command, but can fail MCP server cleanup

## Related Commands

- `/smart-commit` - Create meaningful git commits
- `/doc` - Quick documentation capture
- `/orient` - Load project context at session start

## Tips

- Run `/bye` even for "quick" sessions - it's fast and thorough
- Let it run doc-sync automatically (keeps docs fresh)
- Provide session notes when prompted (helps future you!)
- Commit changes if prompted (good Git hygiene)
- Use `Ctrl+C` to exit for cleanest MCP shutdown
