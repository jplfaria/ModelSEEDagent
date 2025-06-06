#!/usr/bin/env python3
"""
Quick E. coli Analysis - Tests core tools
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.tools import ToolRegistry


def quick_analysis():
    """Run quick analysis of E. coli core model"""

    print("🧬 ModelSEEDagent Quick Analysis")
    print("=" * 40)

    model_path = str(project_root / "data" / "examples" / "e_coli_core.xml")

    # Quick analysis plan
    analyses = [
        ("run_metabolic_fba", "FBA Analysis", {"model_path": model_path}),
        ("analyze_essentiality", "Essential Genes", {"model_path": model_path}),
        ("find_minimal_media", "Minimal Media", {"model_path": model_path}),
        ("resolve_biochem_entity", "Glucose Lookup", {"id": "cpd00027"}),
        ("search_biochem", "ATP Search", {"query": "ATP"}),
    ]

    for i, (tool_name, desc, inputs) in enumerate(analyses, 1):
        print(f"\n{i}. {desc}")
        try:
            tool = ToolRegistry.create_tool(tool_name, {})
            result = tool._run_tool(inputs)

            if result.success:
                print(f"   ✅ {result.message}")

                # Show key results
                if isinstance(result.data, dict):
                    if "objective_value" in result.data:
                        print(
                            f"      Growth rate: {result.data['objective_value']:.3f}"
                        )
                    if "essential_genes" in result.data:
                        print(
                            f"      Essential genes: {len(result.data['essential_genes'])}"
                        )
                    if "essential_reactions" in result.data:
                        print(
                            f"      Essential reactions: {len(result.data['essential_reactions'])}"
                        )
                    if "minimal_media" in result.data:
                        print(
                            f"      Minimal media components: {len(result.data['minimal_media'])}"
                        )
                    if "name" in result.data:
                        print(f"      Result: {result.data['name']}")
                    if "results" in result.data and isinstance(
                        result.data["results"], list
                    ):
                        print(f"      Found {len(result.data['results'])} matches")

            else:
                print(f"   ❌ {result.error}")

        except Exception as e:
            print(f"   💥 {e}")

    print(f"\n🎉 Quick analysis complete!")
    print(f"\nFor comprehensive analysis, run: python run_comprehensive_analysis.py")


if __name__ == "__main__":
    quick_analysis()
