#!/usr/bin/env python3
"""
Quick Output Analysis - Analyze tool output sizes
=================================================

This script tests key tools and analyzes output sizes to determine
which tools need summarization for LLM consumption.
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
from src.tools.cobra import (
    EssentialityAnalysisTool,
    FBATool,
    FluxSamplingTool,
    FluxVariabilityTool,
)
from src.tools.cobra.media_tools import MediaSelectorTool


def analyze_output_sizes():
    """Test key tools and analyze output sizes"""

    print("🔍 Analyzing Tool Output Sizes for LLM Consumption")
    print("=" * 60)

    # Basic config
    basic_config = {"fba_config": {}, "model_config": {}}
    model_path = "data/examples/e_coli_core.xml"

    # Test tools with different output patterns
    tools_to_test = {
        "FBA": (FBATool(basic_config), {"model_path": model_path}),
        "FluxVariability": (
            FluxVariabilityTool(basic_config),
            {"model_path": model_path, "fraction_of_optimum": 0.9},
        ),
        "FluxSampling": (
            FluxSamplingTool(basic_config),
            {"model_path": model_path, "n_samples": 100, "method": "optgp"},
        ),
        "Essentiality": (
            EssentialityAnalysisTool(basic_config),
            {"model_path": model_path, "threshold": 0.01},
        ),
        "MediaSelector": (
            MediaSelectorTool(basic_config),
            {"model_path": model_path, "target_growth": 0.1},
        ),
        "BiochemEntityResolver": (
            BiochemEntityResolverTool(basic_config),
            {"entity_id": "glc__D", "entity_type": "compound", "include_aliases": True},
        ),
    }

    results = {}

    for tool_name, (tool, params) in tools_to_test.items():
        print(f"\n🔬 Testing {tool_name}...")

        try:
            # Run tool
            start_time = datetime.now()
            result = tool._run_tool(params)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Convert to JSON to measure size
            if hasattr(result, "model_dump"):
                output = result.model_dump()
            elif isinstance(result, dict):
                output = result
            else:
                output = {"raw_result": str(result)}

            json_str = json.dumps(output, default=str)
            size_bytes = len(json_str.encode("utf-8"))
            size_kb = size_bytes / 1024

            # Analyze structure
            data = output.get("data", {}) if isinstance(output, dict) else {}

            # Count key elements
            element_counts = {}
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        element_counts[key] = len(value)
                    elif isinstance(value, dict):
                        element_counts[key] = len(value)
                    else:
                        element_counts[key] = 1

            results[tool_name] = {
                "execution_time": execution_time,
                "size_bytes": size_bytes,
                "size_kb": size_kb,
                "element_counts": element_counts,
                "success": (
                    output.get("success", True) if isinstance(output, dict) else True
                ),
                "requires_summarization": size_kb
                > 50,  # > 50KB likely too large for LLM
                "data_keys": list(data.keys()) if isinstance(data, dict) else [],
            }

            # Status
            status = "✅" if results[tool_name]["success"] else "❌"
            size_flag = "🔴" if size_kb > 50 else "🟡" if size_kb > 10 else "🟢"
            print(
                f"  {status} {tool_name}: {execution_time:.2f}s | {size_kb:.1f} KB {size_flag}"
            )

            # Show element counts for large outputs
            if size_kb > 10:
                for key, count in element_counts.items():
                    if count > 10:
                        print(f"    📊 {key}: {count} items")

        except Exception as e:
            print(f"  ❌ {tool_name}: Error - {e}")
            results[tool_name] = {
                "error": str(e),
                "success": False,
                "requires_summarization": False,
            }

    # Analysis summary
    print(f"\n📊 Output Size Analysis Summary:")
    print("-" * 40)

    large_outputs = [
        name
        for name, result in results.items()
        if result.get("requires_summarization", False)
    ]

    if large_outputs:
        print(f"🔴 Tools with large outputs (>50KB): {', '.join(large_outputs)}")
        print("   These tools need summarization for LLM consumption")

    medium_outputs = [
        name
        for name, result in results.items()
        if result.get("size_kb", 0) > 10
        and not result.get("requires_summarization", False)
    ]

    if medium_outputs:
        print(f"🟡 Tools with medium outputs (10-50KB): {', '.join(medium_outputs)}")
        print("   These tools might benefit from summarization")

    small_outputs = [
        name
        for name, result in results.items()
        if result.get("size_kb", 0) <= 10 and result.get("success", False)
    ]

    if small_outputs:
        print(f"🟢 Tools with small outputs (<10KB): {', '.join(small_outputs)}")
        print("   These tools are fine for direct LLM consumption")

    # Recommendations
    print(f"\n💡 Summarization Recommendations:")
    print("-" * 40)

    for tool_name, result in results.items():
        if not result.get("success", False):
            continue

        size_kb = result.get("size_kb", 0)
        if size_kb > 50:
            print(f"🔴 {tool_name}:")
            print(f"   - Create summary extraction (key metrics only)")
            print(f"   - Implement search/filter interface for detailed results")

            # Specific recommendations based on data structure
            data_keys = result.get("data_keys", [])
            if "significant_fluxes" in data_keys:
                print(f"   - Extract top 10 highest fluxes")
                print(f"   - Group by pathway/subsystem")
            if "flux_samples" in data_keys:
                print(f"   - Statistical summary (mean, std, min, max)")
                print(f"   - Correlation analysis with growth rate")
            if "essential_genes" in data_keys:
                print(f"   - Count by functional category")
                print(f"   - Essential pathway summary")

        elif size_kb > 10:
            print(f"🟡 {tool_name}: Consider optional detailed/summary modes")

    # Save results
    output_file = (
        Path("testbed_results")
        / f"output_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 Detailed analysis saved: {output_file}")
    return results


if __name__ == "__main__":
    analyze_output_sizes()
