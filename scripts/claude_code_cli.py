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

    if check_only:
        reviewer.check_what_needs_review()
    else:
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
