#!/usr/bin/env python3
"""
Comprehensive E. coli Core Model Analysis Script
Demonstrates all available ModelSEEDagent tools systematically
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.tools import ToolRegistry


def run_comprehensive_analysis():
    """Run comprehensive analysis of E. coli core model using all available tools"""

    print("🧬 ModelSEEDagent Comprehensive Analysis")
    print("=" * 50)

    # Model path
    model_path = str(project_root / "data" / "examples" / "e_coli_core.xml")

    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return

    print(f"📊 Model: {model_path}")
    print(f"📅 Started: {datetime.now().isoformat()}")

    # Output directory
    output_dir = project_root / "comprehensive_analysis_results"
    output_dir.mkdir(exist_ok=True)

    print(f"📁 Output: {output_dir}")
    print()

    # Analysis plan: (tool_name, description, inputs)
    analysis_plan = [
        ("run_metabolic_fba", "Flux Balance Analysis", {"model_path": model_path}),
        ("analyze_metabolic_model", "Basic Model Analysis", {"model_path": model_path}),
        (
            "run_flux_variability_analysis",
            "Flux Variability Analysis",
            {"model_path": model_path},
        ),
        (
            "run_gene_deletion_analysis",
            "Gene Deletion Analysis",
            {"model_path": model_path},
        ),
        ("analyze_essentiality", "Essentiality Analysis", {"model_path": model_path}),
        ("find_minimal_media", "Minimal Media Analysis", {"model_path": model_path}),
        ("identify_auxotrophies", "Auxotrophy Analysis", {"model_path": model_path}),
        ("run_flux_sampling", "Flux Sampling Analysis", {"model_path": model_path}),
        (
            "run_production_envelope",
            "Production Envelope Analysis",
            {"model_path": model_path},
        ),
        (
            "analyze_reaction_expression",
            "Reaction Expression Analysis",
            {"model_path": model_path},
        ),
        ("check_missing_media", "Missing Media Check", {"model_path": model_path}),
        ("resolve_biochem_entity", "Resolve Glucose ID", {"id": "cpd00027"}),
        ("search_biochem", "Search ATP", {"query": "ATP"}),
        (
            "test_modelseed_cobra_compatibility",
            "Compatibility Test",
            {"model_path": model_path},
        ),
    ]

    results = {}
    successful_analyses = 0

    for i, (tool_name, description, inputs) in enumerate(analysis_plan, 1):
        print(f"{i:2d}. {description} ({tool_name})")

        try:
            # Create and run tool
            tool = ToolRegistry.create_tool(tool_name, {})
            result = tool._run_tool(inputs)

            if result.success:
                print(f"    ✅ Success: {result.message}")
                successful_analyses += 1

                # Save result
                result_data = {
                    "tool_name": tool_name,
                    "description": description,
                    "success": True,
                    "message": result.message,
                    "data_type": str(type(result.data)),
                    "data_keys": (
                        list(result.data.keys())
                        if isinstance(result.data, dict)
                        else None
                    ),
                    "timestamp": datetime.now().isoformat(),
                }

                # Save tool-specific data
                if result.data and isinstance(result.data, dict):
                    # Extract key metrics
                    if "growth_rate" in result.data:
                        result_data["growth_rate"] = result.data["growth_rate"]
                    if "objective_value" in result.data:
                        result_data["objective_value"] = result.data["objective_value"]
                    if "model_stats" in result.data:
                        result_data["model_stats"] = result.data["model_stats"]
                    if "essential_genes" in result.data:
                        result_data["essential_genes_count"] = len(
                            result.data["essential_genes"]
                        )
                    if "essential_reactions" in result.data:
                        result_data["essential_reactions_count"] = len(
                            result.data["essential_reactions"]
                        )

                results[tool_name] = result_data

                # Save individual result file
                with open(output_dir / f"{tool_name}_result.json", "w") as f:
                    json.dump(result_data, f, indent=2, default=str)

            else:
                print(f"    ❌ Failed: {result.error}")
                results[tool_name] = {
                    "tool_name": tool_name,
                    "description": description,
                    "success": False,
                    "error": result.error,
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            print(f"    💥 Exception: {e}")
            results[tool_name] = {
                "tool_name": tool_name,
                "description": description,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    # Generate comprehensive summary
    summary = {
        "analysis_info": {
            "model_path": model_path,
            "total_tools_tested": len(analysis_plan),
            "successful_analyses": successful_analyses,
            "success_rate": successful_analyses / len(analysis_plan),
            "analysis_time": datetime.now().isoformat(),
            "output_directory": str(output_dir),
        },
        "results_by_category": {
            "metabolic_analysis": {
                "tools": [
                    "run_metabolic_fba",
                    "analyze_metabolic_model",
                    "run_flux_variability_analysis",
                ],
                "results": {
                    k: v
                    for k, v in results.items()
                    if k
                    in [
                        "run_metabolic_fba",
                        "analyze_metabolic_model",
                        "run_flux_variability_analysis",
                    ]
                },
            },
            "gene_analysis": {
                "tools": ["run_gene_deletion_analysis", "analyze_essentiality"],
                "results": {
                    k: v
                    for k, v in results.items()
                    if k in ["run_gene_deletion_analysis", "analyze_essentiality"]
                },
            },
            "media_analysis": {
                "tools": [
                    "find_minimal_media",
                    "identify_auxotrophies",
                    "check_missing_media",
                ],
                "results": {
                    k: v
                    for k, v in results.items()
                    if k
                    in [
                        "find_minimal_media",
                        "identify_auxotrophies",
                        "check_missing_media",
                    ]
                },
            },
            "advanced_analysis": {
                "tools": [
                    "run_flux_sampling",
                    "run_production_envelope",
                    "analyze_reaction_expression",
                ],
                "results": {
                    k: v
                    for k, v in results.items()
                    if k
                    in [
                        "run_flux_sampling",
                        "run_production_envelope",
                        "analyze_reaction_expression",
                    ]
                },
            },
            "biochemistry_database": {
                "tools": ["resolve_biochem_entity", "search_biochem"],
                "results": {
                    k: v
                    for k, v in results.items()
                    if k in ["resolve_biochem_entity", "search_biochem"]
                },
            },
            "compatibility": {
                "tools": ["test_modelseed_cobra_compatibility"],
                "results": {
                    k: v
                    for k, v in results.items()
                    if k in ["test_modelseed_cobra_compatibility"]
                },
            },
        },
        "all_results": results,
    }

    # Save comprehensive summary
    with open(output_dir / "comprehensive_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n📈 Analysis Summary:")
    print(f"   • Total Tools Tested: {len(analysis_plan)}")
    print(f"   • Successful Analyses: {successful_analyses}")
    print(f"   • Success Rate: {successful_analyses/len(analysis_plan):.1%}")
    print(f"   • Results Saved: {output_dir}")

    print(f"\n✅ Successful Tools:")
    for tool_name, result in results.items():
        if result["success"]:
            print(f"   • {tool_name}: {result['description']}")

    failed_tools = [tool for tool, result in results.items() if not result["success"]]
    if failed_tools:
        print(f"\n❌ Failed Tools:")
        for tool_name in failed_tools:
            print(f"   • {tool_name}: {results[tool_name]['error'][:60]}...")

    print(f"\n💾 Check results in: {output_dir}/")
    print(f"🔍 Main summary: comprehensive_analysis_summary.json")


if __name__ == "__main__":
    run_comprehensive_analysis()
