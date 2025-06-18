# Documentation Automation System

ModelSEEDagent features a simplified, trigger-based documentation automation system that maintains comprehensive, up-to-date documentation through manual reviews with intelligent reminders.

## Overview

The documentation automation system provides:

- **Manual Trigger Control** - Full control over when documentation gets updated
- **Intelligent Change Analysis** - Understands what changed since last review
- **Smart Reminders** - Git hooks that remind you when documentation might need updating
- **Comprehensive Updates** - Maintains consistency across all documentation files
- **Tool Count Tracking** - Automatically tracks and updates tool inventories
- **Change History Management** - Maintains detailed logs of all documentation changes

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Manual        │    │   Analysis      │    │  Documentation  │
│   Trigger       │    │   Engine        │    │   Updates       │
│                 │    │                 │    │                 │
│ • claude-code   │───▶│ • Change detect │───▶│ • File updates  │
│ • User control  │    │ • Tool counting │    │ • Content sync  │
│ • Smart remind  │    │ • Impact eval   │    │ • History log   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Git Hooks     │
                    │   Integration   │
                    │                 │
                    │ • Post-commit   │
                    │ • Reminders     │
                    │ • Status check  │
                    │ • User prompt   │
                    └─────────────────┘
```

## Core Components

### 1. Claude Code CLI

**Files**: `claude-code`, `scripts/claude_code_cli.py`, `scripts/docs_review_simple.py`

**Purpose**: Simple, trigger-based documentation management interface.

**Usage**:
```bash
# Review and update all documentation
./claude-code review-docs

# Check what needs updating without making changes
./claude-code review-docs --check

# Show reminder status
./claude-code reminder
```

**Key Features**:
- Manual trigger for full control
- Intelligent change detection since last review
- Automatic tool count updates
- Comprehensive file updates
- Review session tracking

### 2. Smart Reminder System

**File**: `scripts/setup_git_hooks.py`

**Purpose**: Git hooks that provide intelligent reminders about documentation updates.

**Setup**:
```bash
python scripts/setup_git_hooks.py
```

**Features**:
- **Post-commit hooks** - Show documentation status after each commit
- **Intelligent suggestions** - Recommend updates based on file changes
- **Tool change detection** - Highlight when tools are modified
- **Non-intrusive reminders** - Helpful without being annoying

### 3. Documentation Review State

**File**: `.docs_review_state.json` (auto-created)

**Purpose**: Tracks when documentation was last reviewed and what was changed.

**Contents**:
```json
{
  "timestamp": "2025-06-14 14:30:00",
  "commit_hash": "a1b2c3d4",
  "files_analyzed": ["src/tools/..."],
  "files_updated": ["README.md", "docs/..."],
  "tool_count": 29,
  "changes_summary": "Reviewed 15 changed files"
}
```

## Workflow

### Daily Development Workflow

1. **Make code changes** as usual
2. **Commit changes** - git hooks show reminder status
3. **When reminded**, run `./claude-code review-docs`
4. **Commit documentation updates** if any were made

### Example Session

```bash
# Make some code changes
git add src/tools/new_tool.py
git commit -m "feat: Add new analysis tool"

# Git hook shows:
# RECOMMENDATION: Consider running 'claude-code review-docs'
#    Changes detected that may affect documentation
# Tool files changed:
#    • src/tools/new_tool.py

# Update documentation
./claude-code review-docs

# Output:
# Starting documentation review...
# Analyzed 3 changed files
# Updated 2 documentation files:
#    • README.md
#    • docs/TOOL_REFERENCE.md
# Documentation review completed and saved

# Commit the updates
git add docs/ README.md
git commit -m "docs: Update documentation for new analysis tool"
```

### Checking Status

```bash
# Check what needs review
./claude-code review-docs --check

# Output:
# Documentation Review Status
#    • Files changed since last review: 5
#    • Tool files changed: 1
#    • Current tool count: 30
#    • Needs review: Yes
#    • Tool files: src/tools/enhanced_fba.py

# Show reminder status
./claude-code reminder

# Output:
# 📚 Documentation Review Status
# ========================================
# Last Review: 2025-06-14 14:30:00
# Last Commit: a1b2c3d4
# Files Changed: 5
# Tool Files: 1
# Current Tool Count: 30
#
# RECOMMENDATION: Consider running 'claude-code review-docs'
#    Changes detected that may affect documentation
```

## Files Updated Automatically

The system automatically updates these files when you run `./claude-code review-docs`:

1. **README.md** - Tool counts and overview information
2. **docs/TOOL_REFERENCE.md** - Complete tool documentation and counts
3. **docs/TOOL_TESTING_STATUS.md** - Tool implementation and testing status
4. **docs/documentation-updates.md** - History of documentation changes

## Benefits of This Approach

### Advantages
- **Full Control** - You decide when documentation gets updated
- **No Complexity** - Simple commands, no complex automation to debug
- **Smart Reminders** - Helpful hints without being intrusive
- **Reliable** - No automatic systems that can fail or misbehave
- **Fast Development** - Focus on coding, update docs when convenient

### Comparison with Previous System
- **Before**: Complex automatic git analysis, frequent failures, debugging overhead
- **After**: Simple trigger-based system, reliable operation, developer control

## Configuration

### Disable Git Hooks
```bash
# Remove hooks if you don't want reminders
rm .git/hooks/post-commit
rm .git/hooks/commit-msg
```

### Re-enable Git Hooks
```bash
# Re-run setup script
python scripts/setup_git_hooks.py
```

### Manual Documentation Updates
If you prefer not to use the automated system at all, you can manually update:
- Tool counts in README.md, docs/TOOL_REFERENCE.md, and docs/TOOL_TESTING_STATUS.md
- Tool documentation in docs/TOOL_REFERENCE.md
- Change history in docs/documentation-updates.md

## Migration from Previous System

The previous complex system in `scripts/docs_review.py` has been replaced with:
- `scripts/docs_review_simple.py` - Simplified core functionality
- `scripts/claude_code_cli.py` - CLI interface
- `claude-code` - Shell wrapper for easy access

The GitHub Actions automatic triggers have been removed to prevent the complexity and reliability issues we experienced.

## Troubleshooting

### Command Not Found
```bash
# Make sure the script is executable
chmod +x claude-code

# Run from repository root
./claude-code review-docs
```

### Hook Not Working
```bash
# Re-setup hooks
python scripts/setup_git_hooks.py

# Check if hooks exist
ls -la .git/hooks/post-commit
```

### Review State Issues
```bash
# Reset review state if needed
rm .docs_review_state.json

# Next review will analyze recent commits
./claude-code review-docs
```

## Best Practices

1. **Run reviews frequently** - After major changes or weekly
2. **Check status before releases** - Ensure documentation is current
3. **Review tool changes immediately** - New/modified tools should be documented quickly
4. **Commit documentation updates** - Keep documentation changes in version control
5. **Use meaningful commit messages** - Help others understand what was updated

This system provides the reliability and control needed for maintaining comprehensive documentation without the complexity overhead of automatic systems.
