#!/usr/bin/env python3
"""
Phase 3 Completion Validation Script

This script validates that all aspects of ModelSEEDagent development
have been completed successfully across all three phases.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n🔍 Testing: {description}")
    print(f"   Command: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print(f"   ✅ PASS: {description}")
            return True
        else:
            print(f"   ❌ FAIL: {description}")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {description} - {e}")
        return False


def test_imports():
    """Test that all critical imports work"""
    print("\n🐍 Testing Python API Imports")

    imports_to_test = [
        "from src.agents.langgraph_metabolic import LangGraphMetabolicAgent",
        "from src.llm.argo import ArgoLLM",
        "from src.tools.cobra.fba import FBATool",
        "from src.interactive.interactive_cli import InteractiveCLI",
        "from src.config.settings import ConfigManager",
    ]

    success_count = 0
    for import_stmt in imports_to_test:
        try:
            exec(import_stmt)
            print(f"   ✅ {import_stmt}")
            success_count += 1
        except Exception as e:
            print(f"   ❌ {import_stmt} - Error: {e}")

    return success_count == len(imports_to_test)


def main():
    """Run comprehensive validation"""
    print("🧬 ModelSEEDagent Phase 3 Completion Validation")
    print("=" * 60)

    # Track results
    results = {}

    # Phase 1 Validation: Import and Test Issues Resolved
    print("\n📋 PHASE 1 VALIDATION: Import Fixes and Test Suite")

    results["test_suite"] = run_command(
        "python -m pytest tests/ --tb=no -q",
        "Complete test suite (should be 47/47 passing)",
    )

    results["imports"] = test_imports()

    # Phase 2 Validation: CLI Functionality and Configuration
    print("\n📋 PHASE 2 VALIDATION: CLI and Configuration")

    results["cli_help"] = run_command(
        "modelseed-agent --help", "CLI help command (should show formatted help)"
    )

    results["cli_status"] = run_command(
        "modelseed-agent status", "CLI status command (should show configuration)"
    )

    results["cli_logs"] = run_command(
        "modelseed-agent logs --last 1", "CLI logs command (should show recent runs)"
    )

    results["setup_help"] = run_command(
        "modelseed-agent setup --help", "Setup command help (should show options)"
    )

    results["analyze_help"] = run_command(
        "modelseed-agent analyze --help", "Analyze command help (should show options)"
    )

    # Phase 3 Validation: Documentation Examples
    print("\n📋 PHASE 3 VALIDATION: Documentation and Examples")

    results["workflow_example"] = run_command(
        "python examples/complete_workflow_example.py",
        "Complete workflow example (should demonstrate all features)",
    )

    results["config_file"] = Path.home().joinpath(".modelseed-agent-cli.json").exists()
    if results["config_file"]:
        print("\n   ✅ Configuration persistence: CLI config file exists")
    else:
        print("\n   ❌ Configuration persistence: CLI config file missing")

    # Summary
    print("\n" + "=" * 60)
    print("🏆 VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)
    success_rate = (passed / total) * 100

    print(f"\n📊 Results: {passed}/{total} tests passed ({success_rate:.1f}%)")

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")

    # Final determination
    if success_rate >= 90:
        print(f"\n🎉 PHASE 3 COMPLETION: SUCCESS!")
        print("🧬 ModelSEEDagent is production ready!")
        print("\n✅ All phases completed successfully:")
        print("   Phase 1: Import fixes and test suite - COMPLETE")
        print("   Phase 2: CLI functionality and configuration - COMPLETE")
        print("   Phase 3: Documentation polish and validation - COMPLETE")
        print("\n🚀 Ready for production use!")
        return 0
    else:
        print(f"\n⚠️ PHASE 3 COMPLETION: NEEDS ATTENTION")
        print("Some validation tests failed. Review the results above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
