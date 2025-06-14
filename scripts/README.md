# Documentation Management Scripts

This directory contains the new simplified documentation management system for ModelSEEDagent.

## Quick Start

```bash
# Update documentation when needed
./claude-code review-docs

# Check if documentation needs updating
./claude-code review-docs --check

# Show current status
./claude-code reminder
```

## Files

- **`claude-code`** - Main CLI wrapper script (run from repo root)
- **`claude_code_cli.py`** - Python CLI interface
- **`docs_review_simple.py`** - Simplified documentation review engine
- **`setup_git_hooks.py`** - Sets up smart reminder git hooks
- **`docs_review.py`** - Legacy complex system (replaced)

## System Overview

The new system is **trigger-based** rather than automatic:

1. **You control when documentation gets updated** - no automatic failures
2. **Smart reminders** - git hooks tell you when updates might be needed
3. **Simple commands** - just run `./claude-code review-docs`
4. **Reliable operation** - no complex git analysis to debug

## Workflow

1. Make code changes as usual
2. Commit - git hooks show reminder status
3. When reminded, run `./claude-code review-docs`
4. Commit documentation updates

## Setup

```bash
# Setup git hooks for reminders
python scripts/setup_git_hooks.py
```

## Migration

This system replaces the previous complex automatic system that was causing reliability issues. The new approach gives you full control while providing helpful reminders.

For detailed documentation, see: [docs/operations/documentation-automation.md](../docs/operations/documentation-automation.md)
