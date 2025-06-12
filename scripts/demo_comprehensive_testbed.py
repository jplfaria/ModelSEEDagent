#!/usr/bin/env python3
"""
Demo Comprehensive Testbed - Quick Demonstration
==============================================

This script demonstrates the comprehensive testbed with biological validation
by testing a subset of tools on one model.
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.biochem import BiochemEntityResolverTool

# Import tools
from src.tools.cobra import EssentialityAnalysisTool, FBATool
from src.tools.cobra.media_tools import MediaCompatibilityTool, MediaSelectorTool


def demo_biological_validation():
    """Demonstrate biological validation with a few tools"""

    print("🧪 Demo: Comprehensive Testbed with Biological Validation")
    print("=" * 60)
    print("Testing 5 tools on e_coli_core with biological validation\n")

    # Basic config
    basic_config = {"fba_config": {}, "model_config": {}}
    model_path = "data/examples/e_coli_core.xml"

    # Test tools
    tools_to_test = {
        "FBA": FBATool(basic_config),
        "Essentiality": EssentialityAnalysisTool(basic_config),
        "MediaSelector": MediaSelectorTool(basic_config),
        "MediaCompatibility": MediaCompatibilityTool(basic_config),
        "BiochemEntityResolver": BiochemEntityResolverTool(basic_config),
    }

    results = {}

    for tool_name, tool in tools_to_test.items():
        print(f"🔬 Testing {tool_name}...")

        try:
            # Get appropriate parameters
            if tool_name == "FBA":
                params = {"model_path": model_path}
            elif tool_name == "Essentiality":
                params = {"model_path": model_path, "threshold": 0.01}
            elif tool_name == "MediaSelector":
                params = {"model_path": model_path, "target_growth": 0.1}
            elif tool_name == "MediaCompatibility":
                params = {"model_path": model_path, "media_names": ["GMM", "AuxoMedia"]}
            elif tool_name == "BiochemEntityResolver":
                params = {
                    "entity_id": "glc__D",
                    "entity_type": "compound",
                    "include_aliases": True,
                }

            # Run tool
            start_time = datetime.now()
            result = tool._run_tool(params)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Extract key info
            if hasattr(result, "model_dump"):
                output = result.model_dump()
            elif isinstance(result, dict):
                output = result
            else:
                output = {"raw_result": str(result)}

            # Simple biological validation demonstration
            validation = validate_result(tool_name, output)

            results[tool_name] = {
                "success": output.get("success", True),
                "execution_time": execution_time,
                "output": output,
                "biological_validation": validation,
            }

            # Display results
            success_icon = "✅" if output.get("success", True) else "❌"
            print(f"  {success_icon} {tool_name}: {execution_time:.2f}s")

            # Show biological insights
            if validation.get("biological_insights"):
                for insight in validation["biological_insights"][:2]:  # Show top 2
                    print(f"    🔬 {insight}")

            if validation.get("warnings"):
                for warning in validation["warnings"][:1]:  # Show top 1
                    print(f"    ⚠️  {warning}")

        except Exception as e:
            print(f"  ❌ {tool_name}: Error - {e}")
            results[tool_name] = {
                "success": False,
                "error": str(e),
                "execution_time": 0,
            }

    # Summary
    print(f"\n📊 Demo Summary:")
    successful = sum(1 for r in results.values() if r.get("success", False))
    total = len(results)
    print(f"  Success Rate: {successful}/{total} ({100*successful/total:.0f}%)")
    print(
        f"  Total Time: {sum(r.get('execution_time', 0) for r in results.values()):.2f}s"
    )

    # Save demo results
    demo_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "demo_version": "v1",
            "model_tested": "e_coli_core",
            "tools_tested": list(tools_to_test.keys()),
        },
        "results": results,
    }

    output_file = (
        Path("testbed_results")
        / f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(demo_results, f, indent=2, default=str)

    print(f"  📄 Demo results saved: {output_file}")
    print(f"\n🎉 Demo complete! This shows the comprehensive testbed framework.")
    print(f"📋 Run 'python scripts/comprehensive_tool_testbed.py' for full testing.")


def validate_result(tool_name: str, output: dict) -> dict:
    """Simple biological validation for demo"""
    validation = {
        "is_valid": True,
        "warnings": [],
        "biological_insights": [],
        "scores": {},
    }

    if not output.get("success", True):
        validation["is_valid"] = False
        validation["warnings"].append("Tool execution failed")
        return validation

    data = output.get("data", {})

    if tool_name == "FBA":
        growth_rate = data.get("objective_value", 0)
        if 0.1 <= growth_rate <= 1.5:
            validation["biological_insights"].append(
                f"Growth rate {growth_rate:.3f} h⁻¹ is biologically realistic"
            )
            validation["scores"]["growth_feasibility"] = 1.0
        elif growth_rate == 0:
            validation["warnings"].append("No growth detected - check media conditions")
        else:
            validation["warnings"].append(f"Unusual growth rate: {growth_rate:.3f} h⁻¹")

    elif tool_name == "Essentiality":
        essential_genes = data.get("essential_genes", [])
        total_genes = data.get("total_genes", len(essential_genes) * 5)
        if total_genes > 0:
            essential_percent = (len(essential_genes) / total_genes) * 100
            if 10 <= essential_percent <= 20:
                validation["biological_insights"].append(
                    f"Essential gene percentage {essential_percent:.1f}% is typical"
                )
                validation["scores"]["essentiality"] = 1.0
            else:
                validation["warnings"].append(
                    f"Unusual essential gene percentage: {essential_percent:.1f}%"
                )

    elif tool_name == "MediaSelector":
        best_media = data.get("best_media")
        if best_media:
            validation["biological_insights"].append(
                f"Selected {best_media} as optimal media"
            )
            validation["scores"]["media_selection"] = 1.0

    elif tool_name == "MediaCompatibility":
        compatibility_results = data.get("compatibility_results", {})
        if compatibility_results:
            avg_compatibility = sum(
                r.get("compatibility_score", 0) for r in compatibility_results.values()
            ) / len(compatibility_results)
            if avg_compatibility > 0.7:
                validation["biological_insights"].append(
                    f"High media compatibility: {avg_compatibility:.2f}"
                )
                validation["scores"]["compatibility"] = avg_compatibility
            else:
                validation["warnings"].append(
                    f"Low media compatibility: {avg_compatibility:.2f}"
                )

    elif tool_name == "BiochemEntityResolver":
        resolved = data.get("resolved", False)
        primary_name = data.get("primary_name", "")
        if resolved and primary_name:
            validation["biological_insights"].append(
                f"Successfully resolved to: {primary_name}"
            )
            validation["scores"]["resolution"] = 1.0
        else:
            validation["warnings"].append("Failed to resolve biochemistry entity")

    return validation


if __name__ == "__main__":
    demo_biological_validation()
