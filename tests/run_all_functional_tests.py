#!/usr/bin/env python3
"""
Comprehensive Functional Test Suite Runner

This script runs all functional correctness tests organized by capability
rather than development phases. Tests validate that the system produces
biologically meaningful and mathematically correct results.

Test Categories:
- Metabolic Tool Correctness: Biological output validation
- AI Reasoning Quality: LLM decision intelligence
- Workflow Integration: End-to-end system validation
"""

import sys
from pathlib import Path

# Add src and tests to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from functional.test_advanced_cobra_tools_correctness import (
    run_advanced_cobra_functional_tests,
)
from functional.test_ai_reasoning_correctness import run_ai_reasoning_tests
from functional.test_biochemistry_tools_correctness import (
    run_biochemistry_functional_tests,
)

# Import test modules
from functional.test_metabolic_tools_correctness import (
    run_comprehensive_functional_tests,
)
from functional.test_workflow_correctness import run_workflow_correctness_tests


def run_all_functional_tests():
    """Run complete functional correctness test suite"""
    print("🧪 ModelSEEDagent Comprehensive Functional Test Suite")
    print("=" * 60)
    print("Testing biological correctness, AI intelligence, and workflow integration")
    print()

    test_results = {}

    # Test 1: Metabolic Tool Correctness
    print("📊 1. METABOLIC TOOL CORRECTNESS TESTS")
    print("-" * 45)
    print("Validating biological realism and mathematical correctness")
    test_results["metabolic_tools"] = run_comprehensive_functional_tests()
    print()

    # Test 2: AI Reasoning Quality
    print("🧠 2. AI REASONING QUALITY TESTS")
    print("-" * 35)
    print("Validating AI decision intelligence and reasoning coherence")
    test_results["ai_reasoning"] = run_ai_reasoning_tests()
    print()

    # Test 3: Biochemistry Tools
    print("🧬 3. BIOCHEMISTRY TOOLS CORRECTNESS TESTS")
    print("-" * 45)
    print("Validating ID resolution accuracy and biochemistry search quality")
    test_results["biochemistry_tools"] = run_biochemistry_functional_tests()
    print()

    # Test 4: Advanced COBRA Tools
    print("⚡ 4. ADVANCED COBRA TOOLS CORRECTNESS TESTS")
    print("-" * 48)
    print("Validating gene deletion, flux sampling, and production envelope")
    test_results["advanced_cobra_tools"] = run_advanced_cobra_functional_tests()
    print()

    # Test 5: Workflow Integration
    print("🔄 5. WORKFLOW INTEGRATION TESTS")
    print("-" * 36)
    print("Validating end-to-end system workflows and data flow")
    test_results["workflow_integration"] = run_workflow_correctness_tests()
    print()

    # Summary
    print("📋 FUNCTIONAL TEST SUITE SUMMARY")
    print("=" * 40)

    passed_categories = sum(1 for success in test_results.values() if success)
    total_categories = len(test_results)

    for category, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED/SKIPPED"
        category_name = category.replace("_", " ").title()
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
    success = run_all_functional_tests()
    sys.exit(0 if success else 1)
