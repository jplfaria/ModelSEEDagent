#!/usr/bin/env python3
"""
Claude Code CLI - Simple interface for documentation management

Usage:
    python scripts/claude_code_cli.py review-docs          # Review and update documentation
    python scripts/claude_code_cli.py review-docs --check  # Check what needs updating
    python scripts/claude_code_cli.py reminder             # Show reminder status
"""

import argparse
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from docs_review_simple import SimpleDocumentationReviewer


def analyze_change_scope(reviewer):
    """Analyze the scope of changes to determine review type needed"""
    analysis = reviewer.analyze_changes_since_last_review()

    # Check for indicators of major changes requiring comprehensive review
    major_indicators = [
        len(analysis["tool_files"]) >= 5,  # 5+ tool files changed
        len(analysis["script_files"]) >= 3,  # 3+ scripts changed
        len(analysis["config_files"]) >= 2,  # 2+ config files changed
        analysis["total_files_changed"] >= 20,  # 20+ total files
        any("performance" in f.lower() for f in analysis["changed_files"]),
        any("optimization" in f.lower() for f in analysis["changed_files"]),
        any("multiprocessing" in f.lower() for f in analysis["changed_files"]),
        any("connection_pool" in f.lower() for f in analysis["changed_files"]),
        any("cache" in f.lower() for f in analysis["changed_files"]),
        any("factory" in f.lower() for f in analysis["changed_files"]),
    ]

    needs_comprehensive = any(major_indicators)

    return {
        "needs_comprehensive": needs_comprehensive,
        "analysis": analysis,
        "indicators": [
            f"Tool files changed: {len(analysis['tool_files'])}",
            f"Script files changed: {len(analysis['script_files'])}",
            f"Config files changed: {len(analysis['config_files'])}",
            f"Total files changed: {analysis['total_files_changed']}",
            f"Performance-related files: {any('performance' in f.lower() for f in analysis['changed_files'])}",
            f"Optimization-related files: {any('optimization' in f.lower() for f in analysis['changed_files'])}",
        ],
    }


def create_comprehensive_documentation_updates(analysis):
    """Create comprehensive documentation updates for major changes"""

    # This is where we'd implement the intelligent documentation updates
    # For now, let's identify what needs updating based on the changes

    updates_needed = []
    changed_files = analysis["changed_files"]

    # Performance optimization documentation
    if any(
        "performance" in f.lower() or "optimization" in f.lower() for f in changed_files
    ):
        updates_needed.append(
            {
                "file": "docs/performance.md",
                "section": "Performance Optimizations",
                "reason": "Performance optimization work detected",
            }
        )

    # Connection pooling documentation
    if any(
        "connection_pool" in f.lower() or "pool" in f.lower() for f in changed_files
    ):
        updates_needed.append(
            {
                "file": "docs/configuration.md",
                "section": "Connection Pooling",
                "reason": "Connection pooling implementation detected",
            }
        )

    # COBRA multiprocessing documentation
    if any(
        "multiprocessing" in f.lower() or "cobra" in f.lower() for f in changed_files
    ):
        updates_needed.append(
            {
                "file": "docs/configuration.md",
                "section": "COBRA Multiprocessing Configuration",
                "reason": "COBRA multiprocessing changes detected",
            }
        )

    # Model caching documentation
    if any("cache" in f.lower() or "model_cache" in f.lower() for f in changed_files):
        updates_needed.append(
            {
                "file": "docs/performance.md",
                "section": "Model Caching",
                "reason": "Model caching implementation detected",
            }
        )

    return updates_needed


def apply_comprehensive_updates(updates_needed, analysis):
    """Apply comprehensive documentation updates"""
    changes_made = False

    for update in updates_needed:
        file_path = Path(update["file"])
        section = update["section"]
        reason = update["reason"]

        print(f"   📝 Updating {file_path} - {section}")

        try:
            # Create or update the documentation file
            content = create_documentation_content(section, analysis, reason)

            if file_path.exists():
                # Read existing content
                with open(file_path, "r") as f:
                    existing_content = f.read()

                # Update or add the section
                updated_content = update_documentation_section(
                    existing_content, section, content
                )
            else:
                # Create new file
                file_path.parent.mkdir(parents=True, exist_ok=True)
                updated_content = create_new_documentation_file(section, content)

            # Write the updated content
            with open(file_path, "w") as f:
                f.write(updated_content)

            print(f"   ✅ Updated {file_path}")
            changes_made = True

        except Exception as e:
            print(f"   ❌ Failed to update {file_path}: {e}")

    return changes_made


def create_documentation_content(section, analysis, reason):
    """Create documentation content based on the section and changes"""

    if section == "Performance Optimizations":
        return f"""
## Performance Optimizations

ModelSEEDagent includes several performance optimizations to improve analysis speed and reduce resource usage.

### Connection Pooling

HTTP connection pooling has been implemented to reduce the overhead of creating new connections for each LLM request:

- **Benefit**: Eliminates 40+ redundant ArgoLLM initializations per session
- **Implementation**: Session-level HTTP client reuse via `LLMConnectionPool`
- **Configuration**: Automatic - no user configuration required

### Model Caching

COBRA model caching prevents repeated model loading from disk:

- **Benefit**: 4.7x speedup for repeated model access
- **Implementation**: File modification tracking with intelligent cache invalidation
- **Storage**: In-memory cache with configurable size limits

### COBRA Multiprocessing Control

COBRA tools now default to single-process mode to prevent connection pool fragmentation:

- **Environment Variables**:
  - `COBRA_DISABLE_MULTIPROCESSING=1` - Force single process mode
  - `COBRA_PROCESSES=N` - Set process count for all COBRA tools
  - `COBRA_FVA_PROCESSES=N` - Set process count for flux variability analysis
  - `COBRA_SAMPLING_PROCESSES=N` - Set process count for flux sampling

### Performance Monitoring

Built-in performance monitoring tracks optimization effectiveness:

- **Session metrics**: Total runtime, tool execution times
- **Connection tracking**: LLM initialization counts
- **Model access patterns**: Cache hit rates and load times

*Last updated: {analysis.get('current_commit', 'unknown')[:8]} - {reason}*
"""

    elif section == "Connection Pooling":
        return f"""
### Connection Pooling Configuration

ModelSEEDagent automatically manages HTTP connection pooling for optimal performance.

#### LLM Connection Pooling

**Automatic Configuration**:
- HTTP clients are pooled per configuration key
- Connections are reused across tool executions
- Timeout and connection limits are automatically managed

**Benefits**:
- Eliminates redundant connection setup overhead
- Reduces memory usage for LLM communications
- Improves overall session performance

**Monitoring**:
Connection pool statistics are logged at DEBUG level:
```
LLM Connection Pool initialized
Created new HTTP client for config: dev_120.0
Reusing existing LLM instance: argo|gpto1|prod|jplfaria|30.0
```

*Configuration is automatic and requires no user intervention.*

*Last updated: {analysis.get('current_commit', 'unknown')[:8]} - {reason}*
"""

    elif section == "COBRA Multiprocessing Configuration":
        return f"""
### COBRA Multiprocessing Configuration

COBRA tools support both single-process and multiprocess execution modes.

#### Default Behavior

**Single Process Mode** (Default):
- All COBRA tools default to `processes=1`
- Prevents connection pool fragmentation
- Recommended for most use cases

#### Multiprocessing Control

**Global Environment Variables**:
```bash
# Disable multiprocessing for all COBRA tools
export COBRA_DISABLE_MULTIPROCESSING=1

# Set process count for all COBRA tools
export COBRA_PROCESSES=4
```

**Tool-Specific Environment Variables**:
```bash
# Flux Variability Analysis
export COBRA_FVA_PROCESSES=8

# Flux Sampling
export COBRA_SAMPLING_PROCESSES=4

# Gene Deletion Analysis
export COBRA_GENE_DELETION_PROCESSES=2

# Essentiality Analysis
export COBRA_ESSENTIALITY_PROCESSES=2
```

#### Performance Considerations

**Single Process (Default)**:
- ✅ No connection pool fragmentation
- ✅ Lower memory usage
- ✅ Simpler debugging
- ❌ Slower for large analyses

**Multiprocess**:
- ✅ Faster for large-scale analyses
- ❌ Higher memory usage
- ❌ Connection pool overhead
- ❌ Complex error handling

*Last updated: {analysis.get('current_commit', 'unknown')[:8]} - {reason}*
"""

    elif section == "Model Caching":
        return f"""
### Model Caching

Intelligent COBRA model caching reduces file I/O overhead and improves performance.

#### How It Works

**File Modification Tracking**:
- Models are cached with file modification timestamps
- Cache is invalidated when source files change
- Automatic cleanup prevents memory bloat

**Cache Benefits**:
- **4.7x speedup** for repeated model access
- Eliminates redundant SBML parsing
- Reduces disk I/O during analysis workflows

#### Cache Configuration

**Automatic Management**:
- Cache size is automatically managed
- LRU eviction for memory efficiency
- Debug logging shows cache hit/miss rates

**Cache Statistics** (in debug logs):
```
📁 Loading model from disk: /path/to/model.xml
🔥 Using cached model: /path/to/model.xml
✅ Cached model: /path/to/model.xml (ID: model_id)
```

#### Memory Usage

**Cache Efficiency**:
- Models are deep-copied when retrieved
- Original cached models remain unmodified
- Memory usage scales with model complexity

*No user configuration required - caching is automatic and transparent.*

*Last updated: {analysis.get('current_commit', 'unknown')[:8]} - {reason}*
"""

    return f"## {section}\n\n*Documentation for {section} - {reason}*\n"


def update_documentation_section(existing_content, section, new_content):
    """Update a specific section in existing documentation"""
    import re

    # Try to find and replace existing section
    section_pattern = rf"(##\s+{re.escape(section)}.*?)(?=##|\Z)"

    if re.search(section_pattern, existing_content, re.DOTALL):
        # Replace existing section
        return re.sub(
            section_pattern, new_content.strip(), existing_content, flags=re.DOTALL
        )
    else:
        # Append new section
        return existing_content + "\n\n" + new_content.strip() + "\n"


def create_new_documentation_file(section, content):
    """Create a new documentation file"""
    return f"""# {section}

{content.strip()}

## Additional Information

For more details on ModelSEEDagent configuration and usage, see the main documentation.
"""


def show_reminder_status():
    """Show reminder status for documentation review"""
    reviewer = SimpleDocumentationReviewer()
    analysis = reviewer.analyze_changes_since_last_review()
    last_review = reviewer.get_last_review()

    print("📚 Documentation Review Status")
    print("=" * 40)

    if last_review:
        print(f"Last Review: {last_review.timestamp}")
        print(f"Last Commit: {last_review.commit_hash[:8]}")
    else:
        print("Last Review: Never")

    print(f"Files Changed: {analysis['total_files_changed']}")
    print(f"Tool Files: {len(analysis['tool_files'])}")
    print(f"Current Tool Count: {reviewer.get_current_tool_count()}")

    if analysis["needs_review"]:
        print("\n🟡 RECOMMENDATION: Consider running 'claude-code review-docs'")
        print("   Changes detected that may affect documentation")
    else:
        print("\n✅ Documentation appears to be up to date")

    if analysis["tool_files"]:
        print(f"\n🔧 Tool files changed:")
        for f in analysis["tool_files"]:
            print(f"   • {f}")


def review_docs(check_only=False):
    """Review and update documentation"""
    reviewer = SimpleDocumentationReviewer()

    # Analyze scope of changes to determine review type
    scope_analysis = analyze_change_scope(reviewer)

    if check_only:
        print("📊 Documentation Review Analysis")
        print("=" * 40)
        for indicator in scope_analysis["indicators"]:
            print(f"   • {indicator}")

        if scope_analysis["needs_comprehensive"]:
            print("\n🔥 COMPREHENSIVE REVIEW NEEDED")
            print("   Major changes detected requiring detailed documentation updates")

            updates_needed = create_comprehensive_documentation_updates(
                scope_analysis["analysis"]
            )
            if updates_needed:
                print("\n📝 Documentation Updates Needed:")
                for update in updates_needed:
                    print(f"   • {update['file']} - {update['section']}")
                    print(f"     Reason: {update['reason']}")
        else:
            print("\n✅ SIMPLE REVIEW SUFFICIENT")
            print("   Standard tool count and metadata updates needed")

        reviewer.check_what_needs_review()
    else:
        if scope_analysis["needs_comprehensive"]:
            print("🔥 Performing comprehensive documentation review...")
            print("   Major changes detected - updating documentation content")

            # Run simple review first for basic updates
            changes_made = reviewer.run_full_review()

            # Then apply comprehensive updates
            updates_needed = create_comprehensive_documentation_updates(
                scope_analysis["analysis"]
            )
            if updates_needed:
                print("\n📝 Applying comprehensive documentation updates...")
                comprehensive_changes = apply_comprehensive_updates(
                    updates_needed, scope_analysis["analysis"]
                )
                changes_made = changes_made or comprehensive_changes
        else:
            print("✅ Performing standard documentation review...")
            changes_made = reviewer.run_full_review()

        if changes_made:
            print("\n🎉 Documentation successfully updated!")
            print("💡 Consider committing these changes:")
            print("   git add docs/ README.md")
            print(
                "   git commit -m 'docs: Update documentation via claude-code review'"
            )
        else:
            print("\n✅ Documentation is already up to date.")


def main():
    parser = argparse.ArgumentParser(
        description="Claude Code CLI - Documentation Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/claude_code_cli.py review-docs          # Update documentation
  python scripts/claude_code_cli.py review-docs --check  # Check status only
  python scripts/claude_code_cli.py reminder             # Show reminder status
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # review-docs command
    review_parser = subparsers.add_parser(
        "review-docs", help="Review and update documentation"
    )
    review_parser.add_argument(
        "--check", action="store_true", help="Check status without updating"
    )

    # reminder command
    subparsers.add_parser("reminder", help="Show documentation review reminder status")

    args = parser.parse_args()

    if args.command == "review-docs":
        review_docs(check_only=args.check)
    elif args.command == "reminder":
        show_reminder_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
