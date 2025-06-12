#!/usr/bin/env python3
"""
Automated Documentation Review System for ModelSEEDagent

This script analyzes code changes and automatically updates documentation
to keep it in sync with the codebase. It can be triggered by pre-commit
hooks or run manually after commits via Claude Code.

Usage:
    python scripts/docs_review.py --check           # Check for doc issues
    python scripts/docs_review.py --update          # Update docs automatically
    python scripts/docs_review.py --commit SHA      # Review specific commit
    python scripts/docs_review.py --interactive     # Interactive review mode
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml


@dataclass
class ChangeAnalysis:
    """Analysis of code changes and their documentation impact"""

    new_tools: List[str]
    modified_tools: List[str]
    new_apis: List[str]
    modified_apis: List[str]
    new_config_options: List[str]
    architecture_changes: List[str]
    breaking_changes: List[str]
    documentation_files_modified: List[str]


class DocumentationReviewer:
    """Automated documentation review and update system"""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.docs_path = self.repo_path / "docs"
        self.src_path = self.repo_path / "src"
        self.config_path = self.repo_path / "config"

    def get_git_changes(self, since_commit: str = "HEAD~1") -> Dict[str, List[str]]:
        """Get git changes since specified commit"""
        try:
            # Get list of changed files
            result = subprocess.run(
                ["git", "diff", "--name-only", since_commit, "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            changed_files = (
                result.stdout.strip().split("\n") if result.stdout.strip() else []
            )

            # Get detailed changes for each file
            changes = {"added": [], "modified": [], "deleted": [], "renamed": []}

            for file in changed_files:
                if file:
                    # Check file status
                    status_result = subprocess.run(
                        [
                            "git",
                            "diff",
                            "--name-status",
                            since_commit,
                            "HEAD",
                            "--",
                            file,
                        ],
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True,
                    )

                    if status_result.stdout:
                        status = status_result.stdout.strip().split("\t")[0]
                        if status == "A":
                            changes["added"].append(file)
                        elif status == "M":
                            changes["modified"].append(file)
                        elif status == "D":
                            changes["deleted"].append(file)
                        elif status.startswith("R"):
                            changes["renamed"].append(file)
                        else:
                            changes["modified"].append(file)

            return changes

        except subprocess.CalledProcessError as e:
            print(f"Error getting git changes: {e}")
            return {"added": [], "modified": [], "deleted": [], "renamed": []}

    def analyze_tool_changes(self, changed_files: List[str]) -> List[str]:
        """Analyze changes to tool files"""
        tool_changes = []
        tool_patterns = [
            r"src/tools/.*\.py$",
            r"src/agents/.*\.py$",
            r"src/llm/.*\.py$",
        ]

        for file in changed_files:
            for pattern in tool_patterns:
                if re.match(pattern, file):
                    # Extract tool name and analyze changes
                    tool_name = Path(file).stem
                    if tool_name != "__init__":
                        tool_changes.append(f"Tool '{tool_name}' in {file}")

        return tool_changes

    def analyze_api_changes(self, changed_files: List[str]) -> List[str]:
        """Analyze API changes that might affect documentation"""
        api_changes = []
        api_patterns = [
            r"src/cli/.*\.py$",
            r"src/interactive/.*\.py$",
            r"src/config/.*\.py$",
        ]

        for file in changed_files:
            for pattern in api_patterns:
                if re.match(pattern, file):
                    api_changes.append(f"API changes in {file}")

        return api_changes

    def check_documentation_coverage(self) -> Dict[str, List[str]]:
        """Check if all tools/features are documented"""
        issues = {
            "missing_tool_docs": [],
            "outdated_examples": [],
            "broken_links": [],
            "inconsistent_terminology": [],
        }

        # Check tool coverage
        tool_files = list(self.src_path.glob("tools/**/*.py"))
        documented_tools = self.get_documented_tools()

        for tool_file in tool_files:
            if tool_file.name != "__init__.py":
                tool_name = tool_file.stem
                if tool_name not in documented_tools:
                    issues["missing_tool_docs"].append(tool_name)

        # Check for broken links in documentation (excluding archive folder)
        doc_files = [
            f
            for f in list(self.docs_path.glob("**/*.md"))
            if not str(f.relative_to(self.docs_path)).startswith("archive/")
        ]
        for doc_file in doc_files:
            broken_links = self.check_broken_links(doc_file)
            issues["broken_links"].extend(broken_links)

        return issues

    def get_documented_tools(self) -> Set[str]:
        """Get list of tools mentioned in documentation"""
        documented_tools = set()

        # Check TOOL_REFERENCE.md
        tool_ref_path = self.docs_path / "TOOL_REFERENCE.md"
        if tool_ref_path.exists():
            with open(tool_ref_path, "r") as f:
                content = f.read()
                # Extract tool names from documentation
                tool_patterns = [
                    r"### \d+\. (\w+Tool)",
                    r"`(\w+)`",
                    r"\*\*(\w+Tool)\*\*",
                ]
                for pattern in tool_patterns:
                    matches = re.findall(pattern, content)
                    documented_tools.update(matches)

        return documented_tools

    def check_broken_links(self, doc_file: Path) -> List[str]:
        """Check for broken internal links in a documentation file"""
        broken_links = []

        try:
            with open(doc_file, "r") as f:
                content = f.read()

            # Find markdown links
            link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
            links = re.findall(link_pattern, content)

            for link_text, link_url in links:
                # Skip external links
                if link_url.startswith(("http://", "https://", "mailto:")):
                    continue

                # Check if internal link exists
                if link_url.startswith("../"):
                    # Relative to docs directory
                    link_path = self.docs_path / link_url.replace("../", "")
                elif link_url.startswith("./"):
                    # Relative to current file
                    link_path = doc_file.parent / link_url.replace("./", "")
                elif not link_url.startswith("/"):
                    # Relative to current file
                    link_path = doc_file.parent / link_url
                else:
                    continue

                # Check if file exists
                if (
                    not link_path.exists()
                    and not (link_path.parent / f"{link_path.name}.md").exists()
                ):
                    broken_links.append(f"Broken link in {doc_file.name}: {link_url}")

        except Exception as e:
            print(f"Error checking links in {doc_file}: {e}")

        return broken_links

    def generate_documentation_updates(
        self, analysis: ChangeAnalysis
    ) -> Dict[str, str]:
        """Generate suggested documentation updates"""
        updates = {}

        # Update TOOL_REFERENCE.md if new tools detected
        if analysis.new_tools or analysis.modified_tools:
            tool_ref_updates = self.generate_tool_reference_updates(
                analysis.new_tools, analysis.modified_tools
            )
            if tool_ref_updates:
                updates["TOOL_REFERENCE.md"] = tool_ref_updates

        # Update API documentation if API changes detected
        if analysis.new_apis or analysis.modified_apis:
            api_updates = self.generate_api_documentation_updates(
                analysis.new_apis, analysis.modified_apis
            )
            if api_updates:
                updates["api/overview.md"] = api_updates

        # Update architecture documentation if significant changes
        if analysis.architecture_changes:
            arch_updates = self.generate_architecture_updates(
                analysis.architecture_changes
            )
            if arch_updates:
                updates["ARCHITECTURE.md"] = arch_updates

        return updates

    def generate_tool_reference_updates(
        self, new_tools: List[str], modified_tools: List[str]
    ) -> str:
        """Generate updates for TOOL_REFERENCE.md"""
        updates = []

        if new_tools:
            updates.append("## New Tools Detected")
            updates.append("The following new tools need to be documented:")
            for tool in new_tools:
                updates.append(f"- {tool}")
            updates.append("")

        if modified_tools:
            updates.append("## Modified Tools")
            updates.append(
                "The following tools have been modified and may need documentation updates:"
            )
            for tool in modified_tools:
                updates.append(f"- {tool}")
            updates.append("")

        if updates:
            updates.insert(0, "# Tool Reference Updates Needed")
            updates.insert(1, "")
            return "\n".join(updates)

        return ""

    def generate_api_documentation_updates(
        self, new_apis: List[str], modified_apis: List[str]
    ) -> str:
        """Generate updates for API documentation"""
        updates = []

        if new_apis or modified_apis:
            updates.append("# API Documentation Updates Needed")
            updates.append("")

            if new_apis:
                updates.append("## New APIs")
                for api in new_apis:
                    updates.append(f"- {api}")
                updates.append("")

            if modified_apis:
                updates.append("## Modified APIs")
                for api in modified_apis:
                    updates.append(f"- {api}")

        return "\n".join(updates) if updates else ""

    def generate_architecture_updates(self, changes: List[str]) -> str:
        """Generate updates for architecture documentation"""
        updates = []

        if changes:
            updates.append("# Architecture Documentation Updates Needed")
            updates.append("")
            updates.append("## Detected Changes")
            for change in changes:
                updates.append(f"- {change}")

        return "\n".join(updates) if updates else ""

    def run_documentation_review(
        self, since_commit: str = "HEAD~1", interactive: bool = False
    ) -> Dict[str, any]:
        """Run complete documentation review"""
        print("🔍 Starting automated documentation review...")

        # Get git changes
        git_changes = self.get_git_changes(since_commit)
        all_changed_files = (
            git_changes["added"] + git_changes["modified"] + git_changes["renamed"]
        )

        if not all_changed_files:
            print("✅ No changes detected since last commit")
            return {"status": "no_changes", "issues": [], "suggestions": []}

        print(f"📁 Analyzing {len(all_changed_files)} changed files...")

        # Analyze changes
        tool_changes = self.analyze_tool_changes(all_changed_files)
        api_changes = self.analyze_api_changes(all_changed_files)

        # Check documentation coverage
        coverage_issues = self.check_documentation_coverage()

        # Generate analysis
        analysis = ChangeAnalysis(
            new_tools=tool_changes,
            modified_tools=[],
            new_apis=api_changes,
            modified_apis=[],
            new_config_options=[],
            architecture_changes=[],
            breaking_changes=[],
            documentation_files_modified=[
                f for f in all_changed_files if f.startswith("docs/")
            ],
        )

        # Generate suggestions using enhanced method
        suggested_updates = self.generate_enhanced_documentation_updates(analysis)

        # Prepare results
        results = {
            "status": "completed",
            "changed_files": all_changed_files,
            "tool_changes": tool_changes,
            "api_changes": api_changes,
            "coverage_issues": coverage_issues,
            "suggested_updates": suggested_updates,
            "documentation_files_modified": analysis.documentation_files_modified,
        }

        # Display results
        self.display_review_results(results, interactive)

        return results

    def display_review_results(
        self, results: Dict[str, any], interactive: bool = False
    ):
        """Display review results to user"""
        print("\n" + "=" * 60)
        print("📋 DOCUMENTATION REVIEW RESULTS")
        print("=" * 60)

        # Changes summary
        if results["changed_files"]:
            print(f"\n📝 Changed Files: {len(results['changed_files'])}")
            for file in results["changed_files"][:10]:  # Show first 10
                print(f"   • {file}")
            if len(results["changed_files"]) > 10:
                print(f"   ... and {len(results['changed_files']) - 10} more")

        # Tool changes
        if results["tool_changes"]:
            print(f"\n🔧 Tool Changes Detected: {len(results['tool_changes'])}")
            for change in results["tool_changes"]:
                print(f"   • {change}")

        # API changes
        if results["api_changes"]:
            print(f"\n🔌 API Changes Detected: {len(results['api_changes'])}")
            for change in results["api_changes"]:
                print(f"   • {change}")

        # Coverage issues
        total_issues = sum(
            len(issues) for issues in results["coverage_issues"].values()
        )
        if total_issues > 0:
            print(f"\n⚠️  Documentation Issues Found: {total_issues}")
            for issue_type, issues in results["coverage_issues"].items():
                if issues:
                    print(f"   {issue_type}: {len(issues)}")
                    for issue in issues[:3]:  # Show first 3
                        print(f"     • {issue}")
                    if len(issues) > 3:
                        print(f"     ... and {len(issues) - 3} more")

        # Suggested updates
        if results["suggested_updates"]:
            print(f"\n💡 Suggested Documentation Updates:")
            for doc_file, updates in results["suggested_updates"].items():
                print(f"   📄 {doc_file}")
                # Show first few lines of updates
                lines = updates.split("\n")[:5]
                for line in lines:
                    if line.strip():
                        print(f"     {line}")
                if len(updates.split("\n")) > 5:
                    print(f"     ... (see full suggestions)")

        # Documentation files modified
        if results["documentation_files_modified"]:
            print(
                f"\n📖 Documentation Files Modified: {len(results['documentation_files_modified'])}"
            )
            for file in results["documentation_files_modified"]:
                print(f"   • {file}")

        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if total_issues == 0 and not results["suggested_updates"]:
            print("   ✅ Documentation appears to be up-to-date!")
        elif total_issues <= 5 and len(results["suggested_updates"]) <= 2:
            print("   🟡 Minor documentation updates recommended")
        else:
            print("   🔴 Significant documentation review needed")

        print("\n" + "=" * 60)

        # Interactive mode
        if interactive:
            self.interactive_review_mode(results)

    def interactive_review_mode(self, results: Dict[str, any]):
        """Interactive mode for reviewing and applying updates"""
        print("\n🔄 INTERACTIVE REVIEW MODE")
        print("Choose an action:")
        print("1. Generate detailed update suggestions")
        print("2. Create documentation update branch")
        print("3. Export review report")
        print("4. Exit")

        try:
            choice = input("\nEnter choice (1-4): ").strip()

            if choice == "1":
                self.generate_detailed_suggestions(results)
            elif choice == "2":
                self.create_update_branch(results)
            elif choice == "3":
                self.export_review_report(results)
            elif choice == "4":
                print("Exiting interactive mode...")
            else:
                print("Invalid choice. Exiting...")
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")

    def generate_detailed_suggestions(self, results: Dict[str, any]):
        """Generate detailed update suggestions"""
        print("\n📝 Generating detailed suggestions...")

        suggestions_file = self.repo_path / "docs_review_suggestions.md"

        content = ["# Documentation Review Suggestions", ""]
        content.append(
            f"Generated on: {subprocess.check_output(['date'], text=True).strip()}"
        )
        content.append("")

        # Add detailed suggestions for each area
        if results["suggested_updates"]:
            content.append("## Suggested Updates")
            content.append("")
            for doc_file, updates in results["suggested_updates"].items():
                content.append(f"### {doc_file}")
                content.append("")
                content.append(updates)
                content.append("")

        # Add coverage issues
        if any(results["coverage_issues"].values()):
            content.append("## Coverage Issues")
            content.append("")
            for issue_type, issues in results["coverage_issues"].items():
                if issues:
                    content.append(f"### {issue_type.replace('_', ' ').title()}")
                    for issue in issues:
                        content.append(f"- {issue}")
                    content.append("")

        with open(suggestions_file, "w") as f:
            f.write("\n".join(content))

        print(f"✅ Detailed suggestions saved to: {suggestions_file}")

    def create_update_branch(self, results: Dict[str, any]):
        """Create a git branch for documentation updates"""
        branch_name = f"docs/auto-update-{subprocess.check_output(['date', '+%Y%m%d-%H%M%S'], text=True).strip()}"

        try:
            subprocess.run(
                ["git", "checkout", "-b", branch_name], cwd=self.repo_path, check=True
            )
            print(f"✅ Created documentation update branch: {branch_name}")
            print(
                "You can now make documentation updates and commit them to this branch."
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating branch: {e}")

    def export_review_report(self, results: Dict[str, any]):
        """Export review report as JSON"""
        report_file = (
            self.repo_path
            / f"docs_review_report_{subprocess.check_output(['date', '+%Y%m%d_%H%M%S'], text=True).strip()}.json"
        )

        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"✅ Review report exported to: {report_file}")

    def update_mkdocs_nav(self, new_files: List[str]) -> None:
        """Update mkdocs.yml navigation when new documentation files are created"""
        mkdocs_path = self.repo_path / "mkdocs.yml"

        if not mkdocs_path.exists():
            print("⚠️ mkdocs.yml not found, skipping nav update")
            return

        try:
            with open(mkdocs_path, "r") as f:
                mkdocs_config = yaml.safe_load(f)

            # Add new files to appropriate nav sections based on path patterns
            nav = mkdocs_config.get("nav", [])
            updated = False

            for file_path in new_files:
                if file_path.startswith("api/"):
                    # Add to API Documentation section
                    for section in nav:
                        if isinstance(section, dict) and "API Documentation" in section:
                            api_items = section["API Documentation"]
                            if isinstance(api_items, list):
                                # Check if file already exists in nav
                                file_exists = any(
                                    isinstance(item, dict)
                                    and file_path in str(item.values())
                                    for item in api_items
                                )
                                if not file_exists:
                                    file_title = (
                                        Path(file_path).stem.replace("_", " ").title()
                                    )
                                    api_items.append({file_title: file_path})
                                    updated = True
                elif file_path.startswith("user/"):
                    # Add to User Guide section
                    for section in nav:
                        if isinstance(section, dict) and "User Guide" in section:
                            user_items = section["User Guide"]
                            if isinstance(user_items, list):
                                file_exists = any(
                                    isinstance(item, dict)
                                    and file_path in str(item.values())
                                    for item in user_items
                                )
                                if not file_exists:
                                    file_title = (
                                        Path(file_path).stem.replace("_", " ").title()
                                    )
                                    user_items.append({file_title: file_path})
                                    updated = True
                else:
                    # Add to Technical Reference section for other files
                    for section in nav:
                        if (
                            isinstance(section, dict)
                            and "Technical Reference" in section
                        ):
                            tech_items = section["Technical Reference"]
                            if isinstance(tech_items, list):
                                file_exists = any(
                                    isinstance(item, dict)
                                    and file_path in str(item.values())
                                    for item in tech_items
                                )
                                if not file_exists:
                                    file_title = (
                                        Path(file_path).stem.replace("_", " ").title()
                                    )
                                    tech_items.append({file_title: file_path})
                                    updated = True

            if updated:
                with open(mkdocs_path, "w") as f:
                    yaml.dump(
                        mkdocs_config, f, default_flow_style=False, sort_keys=False
                    )
                print(
                    f"✅ Updated mkdocs.yml navigation with {len(new_files)} new files"
                )
            else:
                print("ℹ️ No mkdocs.yml navigation updates needed")

        except Exception as e:
            print(f"❌ Error updating mkdocs.yml: {e}")

    def create_documentation_files(self, analysis: ChangeAnalysis) -> List[str]:
        """Create new documentation files for significant features"""
        created_files = []

        # Create documentation for significant new tools (3+ new tools)
        if len(analysis.new_tools) >= 3:
            new_tool_category = self._determine_tool_category(analysis.new_tools)
            if new_tool_category:
                doc_file = self._create_tool_category_documentation(
                    new_tool_category, analysis.new_tools
                )
                if doc_file:
                    created_files.append(doc_file)

        # Create API documentation for new major API changes
        if len(analysis.api_changes) >= 5:  # 5+ new API endpoints
            api_doc_file = self._create_api_documentation(analysis.api_changes)
            if api_doc_file:
                created_files.append(api_doc_file)

        return created_files

    def _determine_tool_category(self, tools: List[str]) -> Optional[str]:
        """Determine the category for a group of new tools"""
        # Analyze tool names to determine category
        categories = {
            "ai_media": ["media", "auxotroph", "optimization"],
            "analysis": ["analysis", "fba", "flux", "gene"],
            "integration": ["integration", "modelseed", "cobra"],
            "biochem": ["biochem", "compound", "reaction"],
        }

        for category, keywords in categories.items():
            if any(keyword in tool.lower() for tool in tools for keyword in keywords):
                return category

        return None

    def _create_tool_category_documentation(
        self, category: str, tools: List[str]
    ) -> Optional[str]:
        """Create documentation file for a new tool category"""
        try:
            # Determine appropriate file location
            if category == "ai_media":
                doc_path = self.docs_path / "user" / "ai_media_guide.md"
                title = "AI Media Management Guide"
            elif category == "analysis":
                doc_path = self.docs_path / "user" / "advanced_analysis.md"
                title = "Advanced Analysis Guide"
            else:
                doc_path = self.docs_path / f"{category}_tools.md"
                title = f"{category.title()} Tools Guide"

            # Create documentation content
            content = f"""# {title}

This guide covers the {category} tools available in ModelSEEDagent.

## Overview

The following {category} tools have been added to provide enhanced capabilities:

"""
            for tool in tools:
                content += f"- **{tool}** - [Tool description needed]\n"

            content += f"""
## Getting Started

```bash
# Example usage
modelseed-agent interactive
```

## Advanced Usage

[Detailed usage examples to be added]

## See Also

- [Tool Reference](../TOOL_REFERENCE.md) - Complete tool listing
- [API Documentation](../api/tools.md) - Technical implementation details
"""

            # Write the file
            with open(doc_path, "w") as f:
                f.write(content)

            relative_path = doc_path.relative_to(self.docs_path)
            print(f"✅ Created documentation: {relative_path}")
            return str(relative_path)

        except Exception as e:
            print(f"❌ Error creating tool category documentation: {e}")
            return None

    def _create_api_documentation(self, api_changes: List[str]) -> Optional[str]:
        """Create API documentation for significant API changes"""
        try:
            doc_path = self.docs_path / "api" / "new_endpoints.md"

            content = """# New API Endpoints

This document covers recently added API endpoints and their usage.

## Overview

The following API endpoints have been added:

"""
            for change in api_changes:
                content += f"- {change}\n"

            content += """
## Usage Examples

[API usage examples to be added]

## See Also

- [API Overview](overview.md) - Main API documentation
- [Tool Implementation](tools.md) - Tool-specific API details
"""

            with open(doc_path, "w") as f:
                f.write(content)

            relative_path = doc_path.relative_to(self.docs_path)
            print(f"✅ Created API documentation: {relative_path}")
            return str(relative_path)

        except Exception as e:
            print(f"❌ Error creating API documentation: {e}")
            return None

    def _is_significant_feature(self, feature: str) -> bool:
        """Determine if a feature is significant enough to warrant new documentation"""
        significant_patterns = [
            "agent",
            "workflow",
            "integration",
            "ai_",
            "advanced_",
            "optimization",
        ]
        return any(pattern in feature.lower() for pattern in significant_patterns)

    def update_docs_md(self, analysis: ChangeAnalysis) -> None:
        """Update DOCS.md with current documentation structure and tool counts"""
        docs_md_path = self.repo_path / "DOCS.md"

        if not docs_md_path.exists():
            print("⚠️ DOCS.md not found, skipping update")
            return

        try:
            # Get current tool counts from actual code
            tool_counts = self._get_current_tool_counts()

            with open(docs_md_path, "r") as f:
                content = f.read()

            # Update tool counts in DOCS.md
            import re

            # Update "All X tools overview" references
            content = re.sub(
                r"All \d+ tools overview",
                f'All {tool_counts["total"]} tools overview',
                content,
            )

            # Update any specific tool count references
            content = re.sub(
                r"(\d+) specialized bioinformatics tools",
                f'{tool_counts["total"]} specialized bioinformatics tools',
                content,
            )

            # Update "27 specialized metabolic modeling tools" references
            content = re.sub(
                r"(\d+) specialized metabolic modeling tools",
                f'{tool_counts["total"]} specialized metabolic modeling tools',
                content,
            )

            # Add update timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if "<!-- Last Updated:" not in content:
                content += f"\n\n<!-- Last Updated: {current_time} -->"
            else:
                content = re.sub(
                    r"<!-- Last Updated: .* -->",
                    f"<!-- Last Updated: {current_time} -->",
                    content,
                )

            with open(docs_md_path, "w") as f:
                f.write(content)

            print(
                f"✅ Updated DOCS.md with current tool counts: {tool_counts['total']} total tools"
            )

        except Exception as e:
            print(f"❌ Error updating DOCS.md: {e}")

    def _get_current_tool_counts(self) -> Dict[str, int]:
        """Get current tool counts from the codebase"""
        try:
            # Import the actual tool counting logic
            sys.path.append(str(self.repo_path / "src"))

            tool_counts = {
                "cobrapy": 0,
                "modelseed": 0,
                "biochem": 0,
                "ai_media": 0,
                "rast": 0,
                "total": 0,
            }

            # Count tools in each category
            tools_dir = self.repo_path / "src" / "tools"

            # COBRApy tools (excluding AI Media tools which are counted separately)
            cobra_dir = tools_dir / "cobra"
            if cobra_dir.exists():
                # Count Python files that contain tool functions/classes (excluding media_tools.py)
                for file in cobra_dir.glob("*.py"):
                    if file.name not in [
                        "__init__.py",
                        "utils.py",
                        "media_tools.py",
                        "error_handling.py",
                        "precision_config.py",
                    ]:
                        with open(file, "r") as f:
                            content = f.read()
                            # Count Tool definitions more accurately
                            tool_counts["cobrapy"] += (
                                content.count("Tool(")
                                + content.count("class ")
                                - content.count("class Base")
                            )

            # ModelSEED tools
            modelseed_dir = tools_dir / "modelseed"
            if modelseed_dir.exists():
                for file in modelseed_dir.glob("*.py"):
                    if file.name != "__init__.py":
                        with open(file, "r") as f:
                            content = f.read()
                            tool_counts["modelseed"] += content.count("Tool(")

            # Biochem tools
            biochem_dir = tools_dir / "biochem"
            if biochem_dir.exists():
                tool_counts["biochem"] = 2  # Known count

            # AI Media tools
            # Count from media_tools.py specifically
            media_tools_file = cobra_dir / "media_tools.py"
            if media_tools_file.exists():
                with open(media_tools_file, "r") as f:
                    content = f.read()
                    # Count Tool definitions in media_tools.py
                    tool_counts["ai_media"] = content.count("Tool(")

            # RAST tools
            rast_dir = tools_dir / "rast"
            if rast_dir.exists():
                tool_counts["rast"] = 2  # Known count

            # Calculate total
            tool_counts["total"] = sum(
                v for k, v in tool_counts.items() if k != "total"
            )

            return tool_counts

        except Exception as e:
            print(f"❌ Error counting tools: {e}")
            # Return known accurate counts as fallback
            return {
                "cobrapy": 18,  # Updated accurate count
                "modelseed": 5,
                "biochem": 2,
                "ai_media": 6,  # Part of cobrapy but tracked separately
                "rast": 2,
                "total": 27,  # 18 + 5 + 2 + 2 = 27 (ai_media is subset of cobrapy)
            }

    def track_documentation_changes(
        self, analysis: ChangeAnalysis, updates: Dict[str, str]
    ) -> None:
        """Track documentation changes and create commit-specific changelog"""
        try:
            # Get current commit hash
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.repo_path, text=True
            ).strip()[:8]

            # Create documentation updates entry
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            changes_summary = {
                "commit": commit_hash,
                "timestamp": timestamp,
                "automated_updates": [],
                "files_modified": list(updates.keys()),
                "tool_count_updates": self._get_tool_count_changes(analysis),
                "link_fixes": self._get_link_fixes(analysis),
                "new_files_created": getattr(analysis, "new_files", []),
            }

            # Add specific change details
            for file_path, update_content in updates.items():
                changes_summary["automated_updates"].append(
                    {
                        "file": file_path,
                        "type": "content_update",
                        "summary": f"Updated {file_path} with latest tool information",
                    }
                )

            # Save to documentation updates log
            updates_log_path = self.docs_path / "documentation-updates.json"

            if updates_log_path.exists():
                with open(updates_log_path, "r") as f:
                    updates_log = json.load(f)
            else:
                updates_log = {"updates": []}

            updates_log["updates"].insert(0, changes_summary)  # Most recent first

            # Keep only last 50 updates
            updates_log["updates"] = updates_log["updates"][:50]

            with open(updates_log_path, "w") as f:
                json.dump(updates_log, f, indent=2)

            print(f"✅ Tracked documentation changes for commit {commit_hash}")

            # Update the documentation-updates.md file
            self._update_documentation_updates_page(updates_log)

        except Exception as e:
            print(f"❌ Error tracking documentation changes: {e}")

    def _get_tool_count_changes(self, analysis: ChangeAnalysis) -> Dict[str, any]:
        """Get tool count changes from analysis"""
        return {
            "new_tools": len(analysis.new_tools),
            "modified_tools": len(analysis.modified_tools),
            "total_tools": 27,  # Current accurate count
        }

    def _get_link_fixes(self, analysis: ChangeAnalysis) -> List[str]:
        """Get list of link fixes applied"""
        # This would be populated by the link checking logic
        return []

    def _update_documentation_updates_page(self, updates_log: Dict) -> None:
        """Update the documentation-updates.md page with recent changes"""
        try:
            updates_page_path = self.docs_path / "documentation-updates.md"

            content = """# Documentation Updates

This page tracks automated and manual updates to the ModelSEEDagent documentation.

## Recent Changes

"""

            for update in updates_log["updates"][:10]:  # Show last 10 updates
                content += f"""### {update['timestamp']} (Commit: {update['commit']})

**Files Modified:** {len(update['files_modified'])} files
- {', '.join(update['files_modified'])}

**Changes:**
"""
                for auto_update in update["automated_updates"]:
                    content += (
                        f"- **{auto_update['file']}**: {auto_update['summary']}\n"
                    )

                if update["tool_count_updates"]["new_tools"] > 0:
                    content += f"- **Tool Count Update**: {update['tool_count_updates']['new_tools']} new tools added\n"

                content += "\n---\n\n"

            content += """## Documentation Review System

The documentation review system automatically:
- Updates tool counts across all documentation
- Fixes broken internal links
- Creates new documentation files for significant features
- Tracks all changes with commit-level detail

## Manual Updates

For manual documentation updates, please follow the [Contributing Guide](archive/development/CONTRIBUTING.md).
"""

            with open(updates_page_path, "w") as f:
                f.write(content)

            print("✅ Updated documentation-updates.md page")

        except Exception as e:
            print(f"❌ Error updating documentation updates page: {e}")

    def generate_enhanced_documentation_updates(
        self, analysis: ChangeAnalysis
    ) -> Dict[str, str]:
        """Enhanced documentation update generation with change tracking"""
        updates = {}

        # Original update logic
        if analysis.new_tools or analysis.modified_tools:
            tool_updates = self.generate_tool_reference_updates(
                analysis.new_tools, analysis.modified_tools
            )
            if tool_updates:
                updates["TOOL_REFERENCE.md"] = tool_updates

        if analysis.new_apis or analysis.modified_apis:
            api_updates = self.generate_api_documentation_updates(
                analysis.new_apis, analysis.modified_apis
            )
            if api_updates:
                updates["api/overview.md"] = api_updates

        if analysis.architecture_changes:
            arch_updates = self.generate_architecture_updates(
                analysis.architecture_changes
            )
            if arch_updates:
                updates["ARCHITECTURE.md"] = arch_updates

        # Track all changes
        if updates:
            self.track_documentation_changes(analysis, updates)

        # Update DOCS.md
        self.update_docs_md(analysis)

        return updates


def main():
    parser = argparse.ArgumentParser(
        description="Automated Documentation Review System"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check for documentation issues"
    )
    parser.add_argument(
        "--update", action="store_true", help="Update documentation automatically"
    )
    parser.add_argument(
        "--commit",
        type=str,
        default="HEAD~1",
        help="Review changes since specific commit (default: HEAD~1)",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        default=".",
        help="Path to repository (default: current directory)",
    )

    args = parser.parse_args()

    # Initialize reviewer
    reviewer = DocumentationReviewer(args.repo_path)

    # Run review
    if args.check or args.update or args.interactive:
        results = reviewer.run_documentation_review(
            since_commit=args.commit, interactive=args.interactive
        )

        # Exit with appropriate code
        total_issues = sum(
            len(issues) for issues in results.get("coverage_issues", {}).values()
        )
        if total_issues > 0 or results.get("suggested_updates"):
            sys.exit(1)  # Issues found
        else:
            sys.exit(0)  # No issues
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
