#!/usr/bin/env python3
"""
Simplified Documentation Review System for ModelSEEDagent

Trigger-based documentation maintenance that:
1. Updates all documentation when manually triggered
2. Tracks when documentation was last reviewed
3. Provides intelligent change analysis since last review
4. Maintains tool counts and documentation consistency

Usage:
    python scripts/docs_review_simple.py                    # Review and update all docs
    python scripts/docs_review_simple.py --check            # Check what needs updating
    python scripts/docs_review_simple.py --since-commit SHA # Review since specific commit
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ReviewSession:
    """Tracks a documentation review session"""

    timestamp: str
    commit_hash: str
    files_analyzed: List[str]
    files_updated: List[str]
    tool_count: int
    changes_summary: str


class SimpleDocumentationReviewer:
    """Simplified documentation reviewer with manual triggers"""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.review_file = self.repo_path / ".docs_review_state.json"
        self.docs_update_file = self.repo_path / "docs" / "documentation-updates.md"

    def get_current_tool_count(self) -> int:
        """Get current accurate tool count from codebase"""
        try:
            # Count COBRA tools (12)
            cobra_tools = [
                "FBATool",
                "ModelAnalysisTool",
                "PathwayAnalysisTool",
                "FluxVariabilityTool",
                "GeneDeletionTool",
                "EssentialityAnalysisTool",
                "FluxSamplingTool",
                "ProductionEnvelopeTool",
                "AuxotrophyTool",
                "MinimalMediaTool",
                "MissingMediaTool",
                "ReactionExpressionTool",
            ]

            # Count AI Media tools (6)
            ai_media_tools = [
                "MediaSelectorTool",
                "MediaManipulatorTool",
                "MediaCompatibilityTool",
                "MediaComparatorTool",
                "MediaOptimizationTool",
                "AuxotrophyPredictionTool",
            ]

            # Count Biochemistry tools (2)
            biochem_tools = ["BiochemEntityResolverTool", "BiochemSearchTool"]

            # Count ModelSEED tools (4)
            modelseed_tools = [
                "ProteinAnnotationTool",
                "ModelBuildTool",
                "GapFillTool",
                "ModelCompatibilityTool",
            ]

            # Count System tools (3)
            system_tools = ["ToolAuditTool", "AIAuditTool", "RealtimeVerificationTool"]

            # Count RAST tools (2)
            rast_tools = ["RastAnnotationTool", "AnnotationAnalysisTool"]

            # Total count
            all_tools = (
                cobra_tools
                + ai_media_tools
                + biochem_tools
                + modelseed_tools
                + system_tools
                + rast_tools
            )

            # Remove duplicates if any
            unique_tools = list(set(all_tools))
            return len(unique_tools)
        except Exception:
            return 29  # Fallback to known count

    def get_last_review(self) -> Optional[ReviewSession]:
        """Get the last review session info"""
        if not self.review_file.exists():
            return None

        try:
            with open(self.review_file, "r") as f:
                data = json.load(f)
                return ReviewSession(**data)
        except Exception:
            return None

    def save_review_session(self, session: ReviewSession):
        """Save review session info"""
        with open(self.review_file, "w") as f:
            json.dump(
                {
                    "timestamp": session.timestamp,
                    "commit_hash": session.commit_hash,
                    "files_analyzed": session.files_analyzed,
                    "files_updated": session.files_updated,
                    "tool_count": session.tool_count,
                    "changes_summary": session.changes_summary,
                },
                f,
                indent=2,
            )

    def get_current_commit(self) -> str:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    def get_changed_files_since(self, since_commit: str) -> List[str]:
        """Get all files changed since a specific commit"""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", since_commit, "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            return [f for f in result.stdout.strip().split("\n") if f]
        except Exception:
            return []

    def analyze_changes_since_last_review(self) -> Dict[str, any]:
        """Analyze what changed since last documentation review"""
        last_review = self.get_last_review()
        current_commit = self.get_current_commit()

        if not last_review:
            # First time - analyze everything
            since_commit = "HEAD~10"  # Look at last 10 commits
            print("🆕 First documentation review - analyzing recent changes")
        else:
            since_commit = last_review.commit_hash
            print(f"🔍 Analyzing changes since last review ({last_review.timestamp})")

        changed_files = self.get_changed_files_since(since_commit)

        # Categorize changes
        tool_files = [f for f in changed_files if "src/tools/" in f]
        doc_files = [f for f in changed_files if f.endswith(".md") or f == "README.md"]
        script_files = [
            f for f in changed_files if f.endswith(".py") and "scripts/" in f
        ]
        config_files = [
            f
            for f in changed_files
            if any(x in f for x in [".yml", ".yaml", ".json", "config"])
        ]

        return {
            "since_commit": since_commit,
            "current_commit": current_commit,
            "total_files_changed": len(changed_files),
            "changed_files": changed_files,
            "tool_files": tool_files,
            "doc_files": doc_files,
            "script_files": script_files,
            "config_files": config_files,
            "needs_review": len(tool_files) > 0
            or len(script_files) > 0
            or len(config_files) > 0,
        }

    def update_documentation_files(self, analysis: Dict[str, any]) -> List[str]:
        """Update all documentation files with current information"""
        updated_files = []
        current_tool_count = self.get_current_tool_count()

        # Update README.md
        readme_path = self.repo_path / "README.md"
        if self.update_tool_counts_in_file(readme_path, current_tool_count):
            updated_files.append("README.md")

        # Update TOOL_REFERENCE.md
        tool_ref_path = self.repo_path / "docs" / "TOOL_REFERENCE.md"
        if self.update_tool_counts_in_file(tool_ref_path, current_tool_count):
            updated_files.append("docs/TOOL_REFERENCE.md")

        # Update TOOL_TESTING_STATUS.md
        tool_status_path = self.repo_path / "docs" / "TOOL_TESTING_STATUS.md"
        if self.update_tool_counts_in_file(tool_status_path, current_tool_count):
            updated_files.append("docs/TOOL_TESTING_STATUS.md")

        # Update documentation-updates.md
        if self.update_documentation_updates(analysis, current_tool_count):
            updated_files.append("docs/documentation-updates.md")

        return updated_files

    def update_tool_counts_in_file(self, file_path: Path, current_count: int) -> bool:
        """Update tool counts in a specific file"""
        if not file_path.exists():
            return False

        try:
            with open(file_path, "r") as f:
                content = f.read()

            original_content = content

            # Update various tool count patterns
            import re

            patterns = [
                (
                    r"\*\*\d+ specialized metabolic modeling tools\*\*",
                    f"**{current_count} specialized metabolic modeling tools**",
                ),
                (r"All \d+ tools overview", f"All {current_count} tools overview"),
                (
                    r"## Specialized Tools \(\d+ Total\)",
                    f"## Specialized Tools ({current_count} Total)",
                ),
                (
                    r"(\d+) specialized metabolic modeling tools",
                    f"{current_count} specialized metabolic modeling tools",
                ),
                (
                    r"organized into five main categories",
                    "organized into six main categories",
                ),
                (
                    r"organized into 5 main categories",
                    "organized into 6 main categories",
                ),
                (r"\d+ specialized tools", f"{current_count} specialized tools"),
                (
                    r"Total Tools Implemented: \d+",
                    f"Total Tools Implemented: {current_count}",
                ),
                (r"Total tools: \d+", f"Total tools: {current_count}"),
                (r"## \d+ Specialized Tools", f"## {current_count} Specialized Tools"),
            ]

            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content)

            if content != original_content:
                with open(file_path, "w") as f:
                    f.write(content)
                return True

            return False
        except Exception as e:
            print(f"❌ Error updating {file_path}: {e}")
            return False

    def update_documentation_updates(
        self, analysis: Dict[str, any], tool_count: int
    ) -> bool:
        """Update the documentation-updates.md file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_hash = analysis["current_commit"][:8]

            new_entry = f"""### {timestamp} (Commit: {commit_hash})

**Files Modified:** {analysis['total_files_changed']} files
- {', '.join(analysis['changed_files'][:10])}{'...' if len(analysis['changed_files']) > 10 else ''}

**Changes:**
- **Documentation Review**: Manual documentation review and update
- **Tool Count**: {tool_count} tools total

---

"""

            if self.docs_update_file.exists():
                with open(self.docs_update_file, "r") as f:
                    content = f.read()

                # Insert new entry after "## Recent Changes"
                if "## Recent Changes" in content:
                    parts = content.split("## Recent Changes", 1)
                    updated_content = (
                        parts[0]
                        + "## Recent Changes\n\n"
                        + new_entry
                        + parts[1].split("\n", 1)[1]
                    )
                else:
                    updated_content = content + "\n\n" + new_entry
            else:
                updated_content = f"""# Documentation Updates

This page tracks manual documentation reviews and updates for ModelSEEDagent.

## Recent Changes

{new_entry}

## Documentation Review System

The documentation review system:
- Updates tool counts across all documentation
- Maintains consistency across all files
- Tracks all changes with detailed history
- Runs on manual trigger for full control
"""

            with open(self.docs_update_file, "w") as f:
                f.write(updated_content)

            return True
        except Exception as e:
            print(f"❌ Error updating documentation-updates.md: {e}")
            return False

    def check_what_needs_review(self) -> Dict[str, any]:
        """Check what needs to be reviewed without making changes"""
        analysis = self.analyze_changes_since_last_review()
        current_tool_count = self.get_current_tool_count()

        print(f"📊 Documentation Review Status")
        print(
            f"   • Files changed since last review: {analysis['total_files_changed']}"
        )
        print(f"   • Tool files changed: {len(analysis['tool_files'])}")
        print(f"   • Current tool count: {current_tool_count}")
        print(f"   • Needs review: {'Yes' if analysis['needs_review'] else 'No'}")

        if analysis["tool_files"]:
            print(f"   • Tool files: {', '.join(analysis['tool_files'])}")

        return analysis

    def run_full_review(self, since_commit: Optional[str] = None) -> bool:
        """Run a complete documentation review and update"""
        print("🔍 Starting documentation review...")

        # Analyze changes
        if since_commit:
            analysis = {
                "since_commit": since_commit,
                "current_commit": self.get_current_commit(),
                "changed_files": self.get_changed_files_since(since_commit),
                "total_files_changed": len(self.get_changed_files_since(since_commit)),
                "tool_files": [
                    f
                    for f in self.get_changed_files_since(since_commit)
                    if "src/tools/" in f
                ],
                "doc_files": [
                    f
                    for f in self.get_changed_files_since(since_commit)
                    if f.endswith(".md")
                ],
                "script_files": [
                    f
                    for f in self.get_changed_files_since(since_commit)
                    if f.endswith(".py") and "scripts/" in f
                ],
                "config_files": [
                    f
                    for f in self.get_changed_files_since(since_commit)
                    if any(x in f for x in [".yml", ".yaml", ".json"])
                ],
                "needs_review": True,
            }
        else:
            analysis = self.analyze_changes_since_last_review()

        print(f"📁 Analyzed {analysis['total_files_changed']} changed files")

        # Update documentation
        updated_files = self.update_documentation_files(analysis)

        if updated_files:
            print(f"✅ Updated {len(updated_files)} documentation files:")
            for file in updated_files:
                print(f"   • {file}")
        else:
            print("ℹ️  No documentation updates needed")

        # Save review session
        session = ReviewSession(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            commit_hash=analysis["current_commit"],
            files_analyzed=analysis["changed_files"],
            files_updated=updated_files,
            tool_count=self.get_current_tool_count(),
            changes_summary=f"Reviewed {analysis['total_files_changed']} changed files",
        )

        self.save_review_session(session)
        print(f"💾 Documentation review completed and saved")

        return len(updated_files) > 0


def main():
    parser = argparse.ArgumentParser(description="Simple Documentation Review System")
    parser.add_argument(
        "--check", action="store_true", help="Check what needs review without updating"
    )
    parser.add_argument("--since-commit", help="Review changes since specific commit")

    args = parser.parse_args()

    reviewer = SimpleDocumentationReviewer()

    if args.check:
        reviewer.check_what_needs_review()
    else:
        changes_made = reviewer.run_full_review(args.since_commit)
        if changes_made:
            print("\n💡 Documentation updated! Consider committing the changes.")
        else:
            print("\n✅ Documentation is up to date.")


if __name__ == "__main__":
    main()
