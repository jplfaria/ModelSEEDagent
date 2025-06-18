#!/usr/bin/env python3
"""
Quick Tool Validation Test
=========================

Runs a subset of tools for quick validation testing.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.tool_validation_suite import ModelSEEDToolValidationSuite


def quick_test():
    """Run a quick test with just essential tools"""

    print("🧪 Quick Tool Validation Test")
    print("=" * 40)

    # Initialize validation suite
    suite = ModelSEEDToolValidationSuite()

    # Test just a few key tools on e_coli_core
    test_tools = ["FBA", "ModelAnalysis", "PathwayAnalysis"]
    model_name = "e_coli_core"
    model_path = "data/examples/e_coli_core.xml"

    results = {}

    for tool_name in test_tools:
        if tool_name in suite.all_tools:
            print(f"\n🔬 Testing {tool_name}")
            tool = suite.all_tools[tool_name]
            result = suite.test_tool_on_model(tool_name, tool, model_name, model_path)
            results[tool_name] = result

            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            print(f"  {status}")
            if not result["success"] and result.get("error"):
                print(f"    Error: {result['error']['message']}")
        else:
            print(f"❌ Tool {tool_name} not found")

    print(
        f"\n📊 Results: {sum(1 for r in results.values() if r['success'])}/{len(results)} passed"
    )
    return results


if __name__ == "__main__":
    quick_test()
