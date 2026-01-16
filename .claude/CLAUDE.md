# ADDM Project Memory

## Git Rules

### NEVER do these:
- **NEVER add "Co-Authored-By: Claude" or similar co-author lines to commits**
- Never use vague commit messages like "just another push for progress"

### Commit message format:
```
<type>: <short description>

[optional body explaining WHY]
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `perf`, `chore`

## Project Structure

- `src/addm/` - Main source code
- `data/tasks/` - Task data (prompts, ground truth, cache)
- `.claude/` - Claude Code configuration
