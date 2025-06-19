#!/usr/bin/env python3
"""
Development Validation Helper

Quick validation tool for development workflows. Provides shortcuts for
common validation tasks and immediate feedback on changes.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict


def run_command(cmd: list) -> tuple:
    """Run command and return (success, output)"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def load_latest_results() -> Dict[str, Any]:
    """Load latest validation results"""
    results_file = Path("results/reasoning_validation/latest_validation_summary.json")
    if not results_file.exists():
        return {}

    try:
        with open(results_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def print_status(message: str, status: str = "info") -> None:
    """Print status message with emoji"""
    emojis = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "running": "🔄",
    }
    print(f"{emojis.get(status, 'ℹ️')} {message}")


def print_summary(results: Dict[str, Any]) -> None:
    """Print quick results summary"""
    if not results:
        print_status("No validation results found", "warning")
        return

    total_tests = results.get("total_tests", 0)
    passed_tests = results.get("passed_tests", 0)
    quality_score = results.get("average_quality_score", 0) * 100
    exec_time = results.get("average_execution_time", 0)
    success_rate = (
        results.get("system_performance", {}).get("overall_success_rate", 0) * 100
    )

    print()
    print("📊 VALIDATION SUMMARY")
    print("-" * 40)
    print(f"Tests Passed:    {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    print(f"Quality Score:   {quality_score:.1f}%")
    print(f"Execution Time:  {exec_time:.1f}s")

    # Status indicators
    if passed_tests == total_tests:
        print_status("All tests passing!", "success")
    else:
        print_status(f"{total_tests - passed_tests} tests failing", "error")

    if quality_score >= 90:
        print_status(f"Excellent quality ({quality_score:.1f}%)", "success")
    elif quality_score >= 85:
        print_status(f"Good quality ({quality_score:.1f}%)", "info")
    else:
        print_status(f"Quality below target ({quality_score:.1f}%)", "warning")


def quick_validate():
    """Run quick validation"""
    print_status("Running quick validation...", "running")

    success, stdout, stderr = run_command(
        ["python", "scripts/integrated_intelligence_validator.py", "--mode=quick"]
    )

    if success:
        print_status("Quick validation completed", "success")
        results = load_latest_results()
        print_summary(results)
    else:
        print_status("Validation failed", "error")
        if stderr:
            print(f"Error: {stderr}")
        return False

    return True


def full_validate():
    """Run full validation"""
    print_status("Running full validation...", "running")

    success, stdout, stderr = run_command(
        ["python", "scripts/integrated_intelligence_validator.py", "--mode=full"]
    )

    if success:
        print_status("Full validation completed", "success")
        results = load_latest_results()
        print_summary(results)
    else:
        print_status("Validation failed", "error")
        if stderr:
            print(f"Error: {stderr}")
        return False

    return True


def component_validate(component: str):
    """Run component-specific validation"""
    print_status(f"Running {component} validation...", "running")

    success, stdout, stderr = run_command(
        [
            "python",
            "scripts/integrated_intelligence_validator.py",
            "--mode=component",
            "--component",
            component,
        ]
    )

    if success:
        print_status(f"{component} validation completed", "success")
        if stdout:
            try:
                results = json.loads(stdout)
                tests_run = results.get("tests_run", 0)
                tests_passed = results.get("tests_passed", 0)
                success_rate = results.get("success_rate", 0) * 100
                avg_quality = results.get("average_quality", 0) * 100

                print()
                print(f"📊 {component.upper()} COMPONENT RESULTS")
                print("-" * 40)
                print(
                    f"Tests Passed:    {tests_passed}/{tests_run} ({success_rate:.1f}%)"
                )
                print(f"Average Quality: {avg_quality:.1f}%")

                if tests_passed == tests_run:
                    print_status("Component validation passed!", "success")
                else:
                    print_status("Some component tests failed", "warning")

            except json.JSONDecodeError:
                print("Raw output:")
                print(stdout)
    else:
        print_status(f"{component} validation failed", "error")
        if stderr:
            print(f"Error: {stderr}")
        return False

    return True


def compare_with_previous():
    """Compare current results with previous run"""
    print_status("Comparing with previous validation...", "running")

    success, stdout, stderr = run_command(
        ["python", "scripts/validation_comparison.py", "--mode=latest"]
    )

    if success:
        print(stdout)
    else:
        print_status("Comparison failed", "error")
        if stderr:
            print(f"Error: {stderr}")


def watch_mode():
    """Watch mode - run validation on file changes"""
    print_status("Watch mode not implemented yet", "warning")
    print("Suggestion: Use a file watcher like 'entr' or 'watchman'")
    print(
        "Example: find src -name '*.py' | entr -r python scripts/dev_validate.py --quick"
    )


def main():
    parser = argparse.ArgumentParser(description="Development validation helper")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (recommended for development)",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Run full validation (recommended before commit)",
    )
    group.add_argument(
        "--component", type=str, help="Run component-specific validation"
    )
    group.add_argument(
        "--compare",
        action="store_true",
        help="Compare current results with previous run",
    )
    group.add_argument(
        "--status", action="store_true", help="Show current validation status"
    )
    group.add_argument(
        "--watch", action="store_true", help="Watch mode (run validation on changes)"
    )

    args = parser.parse_args()

    # Ensure we're in the right directory
    if not Path("scripts/integrated_intelligence_validator.py").exists():
        print_status("Please run from ModelSEEDagent root directory", "error")
        sys.exit(1)

    start_time = time.time()
    success = True

    if args.quick:
        success = quick_validate()
    elif args.full:
        success = full_validate()
    elif args.component:
        success = component_validate(args.component)
    elif args.compare:
        compare_with_previous()
    elif args.status:
        results = load_latest_results()
        if results:
            print_summary(results)
        else:
            print_status(
                "No validation results found. Run validation first.", "warning"
            )
    elif args.watch:
        watch_mode()

    elapsed = time.time() - start_time

    if args.quick or args.full or args.component:
        print()
        print(f"⏱️  Completed in {elapsed:.1f} seconds")

        if success:
            print_status("Validation successful! 🎉", "success")
        else:
            print_status("Validation issues detected", "error")
            sys.exit(1)


if __name__ == "__main__":
    main()
