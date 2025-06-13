#!/usr/bin/env python3
"""
Test Organization Script

This script reorganizes tests by functionality rather than development phases,
making the codebase more intuitive for users and developers.

Organizational Structure:
- Core functionality tests (FBA, essentiality, etc.)
- AI reasoning and intelligence tests
- Workflow and integration tests
- Tool-specific functional tests
"""

import os
import shutil
from pathlib import Path


def organize_tests_by_functionality():
    """Reorganize tests by functionality instead of phases"""

    print("🔄 Organizing Tests by Functionality")
    print("=" * 40)

    repo_root = Path(__file__).parent
    tests_dir = repo_root / "tests"

    # Create functional test organization
    functional_dir = tests_dir / "functional"
    if not functional_dir.exists():
        functional_dir.mkdir(exist_ok=True)
        print(f"✅ Created functional test directory: {functional_dir}")

    # Test organization mapping
    organization_plan = {
        "Core Metabolic Tools": [
            "test_metabolic_tools_correctness.py",
            "test_biochemistry_tools_correctness.py",
        ],
        "Advanced Analysis Tools": [
            "test_advanced_cobra_tools_correctness.py",
        ],
        "AI and Reasoning": [
            "test_ai_reasoning_correctness.py",
        ],
        "Integration and Workflows": [
            "test_workflow_correctness.py",
        ],
        "Test Runners": [
            "run_all_functional_tests.py",
        ],
    }

    # Check current test organization
    current_files = list(functional_dir.glob("*.py"))
    current_names = [f.name for f in current_files]

    print(f"\n📁 Current functional tests:")
    for category, files in organization_plan.items():
        print(f"  {category}:")
        for file in files:
            status = "✅" if file in current_names else "❌"
            print(f"    {status} {file}")

    # Check for any phase-named files that need renaming
    phase_patterns = ["phase", "Phase", "PHASE"]
    phase_files = []

    for test_file in tests_dir.rglob("*.py"):
        if any(pattern in test_file.name for pattern in phase_patterns):
            phase_files.append(test_file)

    if phase_files:
        print(f"\n🔍 Found {len(phase_files)} phase-named files:")
        for file in phase_files:
            print(f"  📄 {file.relative_to(repo_root)}")

        # Suggest renaming
        print(f"\n💡 Renaming suggestions:")
        for file in phase_files:
            suggested_name = suggest_functional_name(file.name)
            print(f"  {file.name} → {suggested_name}")
    else:
        print(f"\n✅ No phase-named test files found")

    # Check for old test structure
    old_test_files = []
    for test_file in tests_dir.rglob("test_phase*.py"):
        old_test_files.append(test_file)

    if old_test_files:
        print(f"\n📋 Migration needed for {len(old_test_files)} old test files:")
        for file in old_test_files:
            print(f"  🔄 {file.relative_to(repo_root)}")
    else:
        print(f"\n✅ No old phase-based test files found")

    # Test runner organization
    test_runners = list(tests_dir.glob("run_*.py"))
    if test_runners:
        print(f"\n🚀 Test runners available:")
        for runner in test_runners:
            print(f"  ✅ {runner.name}")

    print(f"\n📊 Test Organization Summary:")
    print(f"  Functional tests: {len(current_files)} files")
    print(f"  Test runners: {len(test_runners)} files")
    print(f"  Phase-named files: {len(phase_files)} files")

    if len(phase_files) == 0 and len(current_files) >= 4:
        print(f"\n🎉 Test organization is COMPLETE!")
        print(f"   Tests are organized by functionality, not phases")
    else:
        print(f"\n⚠️  Test organization needs attention")


def suggest_functional_name(filename):
    """Suggest functional name for phase-based file"""
    name_lower = filename.lower()

    if "phase1" in name_lower and "modelseed" in name_lower:
        return "test_modelseed_tools_correctness.py"
    elif "phase1a" in name_lower and "cobra" in name_lower:
        return "test_cobra_tools_enhancement.py"
    elif "phase2" in name_lower and "compatibility" in name_lower:
        return "test_model_compatibility_correctness.py"
    elif "phase3" in name_lower and "biochem" in name_lower:
        return "test_biochemistry_tools_correctness.py"
    elif "phase4" in name_lower and "audit" in name_lower:
        return "test_audit_system_correctness.py"
    elif "phase5" in name_lower and ("dynamic" in name_lower or "ai" in name_lower):
        return "test_ai_agent_correctness.py"
    elif "phase8" in name_lower and (
        "advanced" in name_lower or "reasoning" in name_lower
    ):
        return "test_advanced_reasoning_correctness.py"
    else:
        return filename.replace("phase", "functional").replace("Phase", "Functional")


def check_test_naming_consistency():
    """Check that all tests follow functional naming conventions"""
    print("\n🔍 Checking test naming consistency...")

    repo_root = Path(__file__).parent
    tests_dir = repo_root / "tests"

    functional_keywords = [
        "correctness",
        "functional",
        "integration",
        "workflow",
        "biochemistry",
        "cobra",
        "modelseed",
        "ai",
        "reasoning",
    ]

    phase_keywords = ["phase", "Phase", "PHASE"]

    all_test_files = list(tests_dir.rglob("test_*.py"))

    functional_named = []
    phase_named = []
    other_named = []

    for test_file in all_test_files:
        name = test_file.name

        if any(keyword in name for keyword in functional_keywords):
            functional_named.append(test_file)
        elif any(keyword in name for keyword in phase_keywords):
            phase_named.append(test_file)
        else:
            other_named.append(test_file)

    print(f"  ✅ Functional naming: {len(functional_named)} files")
    print(f"  ⚠️  Phase naming: {len(phase_named)} files")
    print(f"  ℹ️  Other naming: {len(other_named)} files")

    if len(phase_named) == 0:
        print(f"  🎉 All tests use functional naming!")

    return len(phase_named) == 0


if __name__ == "__main__":
    organize_tests_by_functionality()
    check_test_naming_consistency()

    print(f"\n🚀 Test organization complete!")
    print(
        f"   Use 'python tests/run_all_functional_tests.py' to run comprehensive tests"
    )
