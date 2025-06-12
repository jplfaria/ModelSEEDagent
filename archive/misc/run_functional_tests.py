#!/usr/bin/env python3
"""
Comprehensive Functional Test Suite Runner

This script runs all functional correctness tests for ModelSEEDagent using pytest.
Tests validate that the system produces biologically meaningful and mathematically
correct results.

Test Categories:
- Metabolic Tool Correctness: Biological output validation
- AI Reasoning Quality: LLM decision intelligence
- Workflow Integration: End-to-end system validation
- Biochemistry Tools: ID resolution and search accuracy
- Advanced COBRA Tools: Gene deletion, flux sampling, production envelope
"""

import subprocess
import sys


def run_functional_test_suite():
    """Run complete functional correctness test suite"""
    print("🧪 ModelSEEDagent Comprehensive Functional Test Suite")
    print("=" * 60)
    print("Testing biological correctness, AI intelligence, and workflow integration")
    print()

    test_modules = [
        (
            "📊 Metabolic Tool Correctness",
            "tests/functional/test_metabolic_tools_correctness.py",
        ),
        (
            "🧠 AI Reasoning Quality",
            "tests/functional/test_ai_reasoning_correctness.py",
        ),
        (
            "🧬 Biochemistry Tools",
            "tests/functional/test_biochemistry_tools_correctness.py",
        ),
        (
            "⚡ Advanced COBRA Tools",
            "tests/functional/test_advanced_cobra_tools_correctness.py",
        ),
        ("🔄 Workflow Integration", "tests/functional/test_workflow_correctness.py"),
    ]

    results = {}

    for category, test_file in test_modules:
        print(f"\n{category}")
        print("-" * len(category))

        # Run pytest on the specific test file
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_file,
            "-v",
            "--tb=short",
            "--disable-warnings",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print("✅ PASSED")
                results[category] = True
            else:
                print("❌ FAILED")
                print("STDOUT:", result.stdout[-500:] if result.stdout else "No output")
                print("STDERR:", result.stderr[-500:] if result.stderr else "No errors")
                results[category] = False

        except subprocess.TimeoutExpired:
            print("⏰ TIMEOUT (300s)")
            results[category] = False
        except Exception as e:
            print(f"🚨 ERROR: {e}")
            results[category] = False

    # Summary
    print(f"\n📋 FUNCTIONAL TEST SUITE SUMMARY")
    print("=" * 40)

    passed_categories = sum(1 for success in results.values() if success)
    total_categories = len(results)

    for category, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED/SKIPPED"
        category_name = category.split(" ", 1)[1]  # Remove emoji
        print(f"  {category_name}: {status}")

    print()

    if passed_categories == total_categories:
        print("🎉 ALL FUNCTIONAL TESTS PASSED!")
        print("✨ System produces biologically meaningful and intelligent results")
        return True
    else:
        print(f"⚠️  {passed_categories}/{total_categories} test categories passed")
        print("🔍 Review failed categories for system correctness issues")
        return False


if __name__ == "__main__":
    success = run_functional_test_suite()
    sys.exit(0 if success else 1)
