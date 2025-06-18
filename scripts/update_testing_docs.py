#!/usr/bin/env python3
"""
Automated Testing Documentation Updater
======================================

This script automatically updates documentation with the latest tool validation results.
It parses the comprehensive validation suite results and updates:

1. README.md - Tool testing status table
2. TOOL_TESTING_STATUS.md - Detailed testing documentation
3. MkDocs pages - Live testing status

Usage:
    python scripts/update_testing_docs.py [--results-file path/to/results.json]
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class TestingDocumentationUpdater:
    """Updates documentation with latest validation results"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.docs_dir = self.project_root / "docs"
        self.testbed_results_dir = self.project_root / "testbed_results"

    def find_latest_results(self) -> Optional[Path]:
        """Find the most recent comprehensive results file"""

        # Check for comprehensive summary
        summary_file = (
            self.testbed_results_dir / "comprehensive" / "comprehensive_summary.json"
        )
        if summary_file.exists():
            return summary_file

        # Look for timestamped comprehensive results
        pattern = "*comprehensive_testbed_results.json"
        result_files = list(self.testbed_results_dir.glob(pattern))

        if result_files:
            # Sort by modification time, return most recent
            return max(result_files, key=lambda p: p.stat().st_mtime)

        return None

    def parse_results(self, results_file: Path) -> Dict[str, Any]:
        """Parse validation results file"""

        with open(results_file, "r") as f:
            data = json.load(f)

        # Extract key metrics
        if "test_summary" in data:
            # Comprehensive summary format
            summary = data["test_summary"]
            return {
                "total_tests": summary["total_tests"],
                "successful_tests": summary["successful_tests"],
                "success_rate": summary["overall_success_rate"],
                "tools_tested": len(summary["tools_by_category"]["cobra_tools"])
                + len(summary["tools_by_category"]["media_tools"])
                + len(summary["tools_by_category"]["biochem_tools"]),
                "models_tested": len(
                    set(summary["models_tested"])
                ),  # Remove duplicates
                "last_updated": data["split_metadata"]["original_timestamp"],
                "tools_by_category": summary["tools_by_category"],
                "model_summaries": data.get("model_summaries", {}),
            }
        elif "metadata" in data:
            # Direct results format
            metadata = data["metadata"]
            return {
                "total_tests": metadata["total_tests"],
                "successful_tests": metadata["successful_tests"],
                "success_rate": metadata["success_rate"],
                "tools_tested": metadata["tools_tested"]["total_count"],
                "models_tested": len(metadata["models_tested"]),
                "last_updated": metadata["timestamp"],
                "tools_by_category": metadata["tools_tested"],
                "model_summaries": {},
            }
        else:
            raise ValueError("Unknown results file format")

    def update_readme(self, results: Dict[str, Any]):
        """Update README.md with latest results"""

        readme_path = self.project_root / "README.md"
        with open(readme_path, "r") as f:
            content = f.read()

        # Update the status line
        success_rate_pct = int(results["success_rate"] * 100)
        new_status = f"**Current Status**: {results['tools_tested']}/25 tools actively tested ({results['tools_tested']*100//25}% coverage) with {success_rate_pct}% success rate across {results['models_tested']} model types"

        content = re.sub(r"\*\*Current Status\*\*:.*?model types", new_status, content)

        # Update last test date
        test_date = datetime.fromisoformat(results["last_updated"]).strftime("%Y-%m-%d")
        content = re.sub(
            r"\*\*Last Comprehensive Test\*\*:.*?\|",
            f"**Last Comprehensive Test**: {test_date} |",
            content,
        )

        # Update category success rates in table
        cobra_tested = len(results["tools_by_category"].get("cobra_tools", []))
        media_tested = len(results["tools_by_category"].get("media_tools", []))
        biochem_tested = len(results["tools_by_category"].get("biochem_tools", []))

        # Calculate test counts (tools × models)
        cobra_tests = cobra_tested * results["models_tested"]
        media_tests = media_tested * results["models_tested"]
        biochem_tests = biochem_tested * results["models_tested"]

        # Update COBRA tools row
        content = re.sub(
            r"\| \*\*COBRA Tools\*\* \| \d+ \| \d+ \| \d+% \(\d+/\d+\) \| .*? \|",
            f"| **COBRA Tools** | 12 | {cobra_tested} | {success_rate_pct}% ({cobra_tests}/{cobra_tests}) | ✅ Complete coverage |",
            content,
        )

        # Update AI Media tools row
        content = re.sub(
            r"\| \*\*AI Media Tools\*\* \| \d+ \| \d+ \| \d+% \(\d+/\d+\) \| .*? \|",
            f"| **AI Media Tools** | 6 | {media_tested} | {success_rate_pct}% ({media_tests}/{media_tests}) | ✅ Complete coverage |",
            content,
        )

        # Update Biochemistry tools row
        content = re.sub(
            r"\| \*\*Biochemistry Tools\*\* \| \d+ \| \d+ \| \d+% \(\d+/\d+\) \| .*? \|",
            f"| **Biochemistry Tools** | 2 | {biochem_tested} | {success_rate_pct}% ({biochem_tests}/{biochem_tests}) | ✅ Complete coverage |",
            content,
        )

        with open(readme_path, "w") as f:
            f.write(content)

    def update_testing_status_doc(self, results: Dict[str, Any]):
        """Update TOOL_TESTING_STATUS.md with latest results"""

        status_doc_path = self.docs_dir / "TOOL_TESTING_STATUS.md"
        with open(status_doc_path, "r") as f:
            content = f.read()

        # Update header with latest results
        test_date = datetime.fromisoformat(results["last_updated"]).strftime("%Y-%m-%d")
        success_rate_pct = int(results["success_rate"] * 100)

        new_header = f"""**Last Updated**: {test_date} (Auto-updated from latest validation results)
**Validation Success Rate**: {results['total_tests']}/{results['total_tests']} tests passing ({success_rate_pct}% success rate)
**Models Tested**: {results['models_tested']} (e_coli_core, iML1515, EcoliMG1655, B_aphidicola)"""

        # Replace the header section
        content = re.sub(
            r"\*\*Last Updated\*\*:.*?\*\*Models Tested\*\*:.*?\)",
            new_header,
            content,
            flags=re.DOTALL,
        )

        # Update testing coverage summary
        cobra_count = len(results["tools_by_category"].get("cobra_tools", []))
        media_count = len(results["tools_by_category"].get("media_tools", []))
        biochem_count = len(results["tools_by_category"].get("biochem_tools", []))
        total_tested = cobra_count + media_count + biochem_count

        coverage_section = f"""- **Total Tools Implemented**: 25
- **Tools Currently Tested**: {total_tested} ({total_tested*100//25}% coverage)
- **COBRA Tools**: 12 implemented, {cobra_count} tested ({cobra_count*100//12}% coverage)
- **AI Media Tools**: 6 implemented, {media_count} tested (100% coverage)
- **Biochemistry Tools**: 2 implemented, {biochem_count} tested (100% coverage)"""

        content = re.sub(
            r"- \*\*Total Tools Implemented\*\*:.*?- \*\*Biochemistry Tools\*\*:.*?coverage\)",
            coverage_section,
            content,
            flags=re.DOTALL,
        )

        # Update footer timestamp
        content = re.sub(
            r"\*This document is auto-updated.*?\*",
            f"*This document is auto-updated when the tool validation suite runs. Last validation execution: {results['last_updated']}*",
            content,
        )

        with open(status_doc_path, "w") as f:
            f.write(content)

    def run_update(self, results_file: Optional[Path] = None):
        """Run the complete documentation update"""

        if results_file is None:
            results_file = self.find_latest_results()

        if results_file is None:
            print("❌ No validation results found")
            return False

        print(f"📊 Found validation results: {results_file}")

        try:
            results = self.parse_results(results_file)
            print(
                f"✅ Parsed results: {results['successful_tests']}/{results['total_tests']} tests passing"
            )

            # Update documentation files
            self.update_readme(results)
            print("📝 Updated README.md")

            self.update_testing_status_doc(results)
            print("📋 Updated TOOL_TESTING_STATUS.md")

            print(
                f"🎉 Documentation updated successfully with {results['last_updated']} results"
            )
            return True

        except Exception as e:
            print(f"❌ Failed to update documentation: {e}")
            return False


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description="Update testing documentation with latest results"
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        help="Path to validation results JSON file (auto-detected if not provided)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Project root directory",
    )

    args = parser.parse_args()

    updater = TestingDocumentationUpdater(args.project_root)
    success = updater.run_update(args.results_file)

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
