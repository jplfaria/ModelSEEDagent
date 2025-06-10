#!/usr/bin/env python3
"""
Comprehensive COBRApy Tool Testbed
=================================

This script tests all COBRApy tools against E. coli models with detailed output,
debugging information, and error analysis.

Models tested:
- data/examples/e_coli_core.xml (core model)
- data/examples/iML1515.xml (genome-scale model)

Tools tested:
- FBATool (Flux Balance Analysis)
- ModelAnalysisTool (Model structure analysis)
- PathwayAnalysisTool (Pathway analysis)
- FluxVariabilityTool (Flux Variability Analysis)
- GeneDeletionTool (Gene deletion analysis)
- EssentialityAnalysisTool (Gene essentiality)
- FluxSamplingTool (Monte Carlo flux sampling)
- ProductionEnvelopeTool (Production envelope)
- AuxotrophyTool (Auxotrophy analysis)
- MinimalMediaTool (Minimal media finder)
- MissingMediaTool (Missing media components)
- ReactionExpressionTool (Reaction expression analysis)
"""

import json
import logging
import os
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.cobra import (
    AuxotrophyTool,
    EssentialityAnalysisTool,
    FBATool,
    FluxSamplingTool,
    FluxVariabilityTool,
    GeneDeletionTool,
    MinimalMediaTool,
    MissingMediaTool,
    ModelAnalysisTool,
    PathwayAnalysisTool,
    ProductionEnvelopeTool,
    ReactionExpressionTool,
)


class ToolTestbed:
    """Comprehensive testbed for COBRApy tools"""

    def __init__(self):
        self.results = {}
        self.models = {
            "e_coli_core": "data/examples/e_coli_core.xml",
            "iML1515": "data/examples/iML1515.xml",
            "Mycoplasma_G37": "data/examples/Mycoplasma_G37.GMM.mdl.xml",
        }

        # Configure detailed logging
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Initialize all tools with basic config
        basic_config = {"fba_config": {}, "model_config": {}}

        self.tools = {
            "FBA": FBATool(basic_config),
            "ModelAnalysis": ModelAnalysisTool(basic_config),
            "PathwayAnalysis": PathwayAnalysisTool(basic_config),
            "FluxVariability": FluxVariabilityTool(basic_config),
            "GeneDeletion": GeneDeletionTool(basic_config),
            "Essentiality": EssentialityAnalysisTool(basic_config),
            "FluxSampling": FluxSamplingTool(basic_config),
            "ProductionEnvelope": ProductionEnvelopeTool(basic_config),
            "Auxotrophy": AuxotrophyTool(basic_config),
            "MinimalMedia": MinimalMediaTool(basic_config),
            "MissingMedia": MissingMediaTool(basic_config),
            "ReactionExpression": ReactionExpressionTool(basic_config),
        }

        print(f"🧪 COBRApy Tool Testbed Initialized")
        print(f"📊 Testing {len(self.tools)} tools on {len(self.models)} models")
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def test_tool_on_model(self, tool_name, tool, model_name, model_path):
        """Test a single tool on a single model with comprehensive output"""

        print(f"\n🔬 Testing {tool_name} on {model_name}")
        print(f"📁 Model: {model_path}")
        print("-" * 60)

        test_result = {
            "tool": tool_name,
            "model": model_name,
            "model_path": model_path,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "execution_time": 0,
            "output": None,
            "error": None,
            "warnings": [],
            "debug_info": {},
            "solver_messages": [],
            "detailed_output": "",
        }

        start_time = datetime.now()

        try:
            # Capture detailed output
            captured_output = []

            # Tool-specific test parameters
            tool_params = self._get_tool_params(tool_name, model_path)

            print(f"⚙️  Parameters: {tool_params}")
            captured_output.append(f"Parameters: {tool_params}")

            # Execute tool with detailed logging
            print("🚀 Executing tool...")
            captured_output.append("Executing tool...")

            result = tool._run_tool(tool_params)

            # Parse and analyze results
            if isinstance(result, dict):
                test_result["output"] = result
                captured_output.append(
                    f"Result type: dictionary with {len(result)} keys"
                )
                captured_output.append(f"Result keys: {list(result.keys())}")

                # Extract key metrics
                if "growth_rate" in result:
                    captured_output.append(f"Growth rate: {result['growth_rate']}")
                    print(f"  📈 Growth rate: {result['growth_rate']}")

                if "objective_value" in result:
                    captured_output.append(
                        f"Objective value: {result['objective_value']}"
                    )
                    print(f"  🎯 Objective value: {result['objective_value']}")

                if "status" in result:
                    captured_output.append(f"Solver status: {result['status']}")
                    print(f"  ⚡ Solver status: {result['status']}")

                if "essential_genes" in result:
                    essential_count = len(result.get("essential_genes", []))
                    captured_output.append(f"Essential genes found: {essential_count}")
                    print(f"  🧬 Essential genes: {essential_count}")

                if "flux_ranges" in result:
                    flux_count = len(result.get("flux_ranges", {}))
                    captured_output.append(f"Flux ranges calculated: {flux_count}")
                    print(f"  🔄 Flux ranges: {flux_count}")

                if "samples" in result:
                    sample_count = len(result.get("samples", []))
                    captured_output.append(f"Flux samples generated: {sample_count}")
                    print(f"  🎲 Flux samples: {sample_count}")

                if "minimal_media" in result:
                    media_components = len(result.get("minimal_media", []))
                    captured_output.append(
                        f"Minimal media components: {media_components}"
                    )
                    print(f"  🧪 Minimal media components: {media_components}")

            else:
                test_result["output"] = str(result)
                captured_output.append(f"Result: {result}")
                print(f"  📋 Result: {result}")

            test_result["success"] = True
            print("  ✅ SUCCESS")

        except Exception as e:
            test_result["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            captured_output.append(f"ERROR: {type(e).__name__}: {str(e)}")
            print(f"  ❌ ERROR: {type(e).__name__}: {str(e)}")

            # Check for specific solver issues
            error_msg = str(e).lower()
            if "infeasible" in error_msg:
                test_result["solver_messages"].append("Model infeasible")
                print("  🚫 Model appears to be infeasible")
            elif "unbounded" in error_msg:
                test_result["solver_messages"].append("Model unbounded")
                print("  ♾️  Model appears to be unbounded")
            elif "solver" in error_msg:
                test_result["solver_messages"].append(f"Solver issue: {str(e)}")
                print("  ⚠️  Solver-related issue detected")

        end_time = datetime.now()
        test_result["execution_time"] = (end_time - start_time).total_seconds()
        test_result["detailed_output"] = "\n".join(captured_output)

        print(f"  ⏱️  Execution time: {test_result['execution_time']:.2f}s")

        return test_result

    def _get_tool_params(self, tool_name, model_path):
        """Get appropriate parameters for each tool type"""

        base_params = {"model_path": model_path}

        tool_specific_params = {
            "FBA": {},
            "ModelAnalysis": {},
            "PathwayAnalysis": {"pathway_name": "glycolysis"},
            "FluxVariability": {"fraction_of_optimum": 0.9},
            "GeneDeletion": {"genes": ["b0008", "b0009"], "method": "single"},
            "Essentiality": {"threshold": 0.1},
            "FluxSampling": {"n_samples": 100, "method": "optgp"},
            "ProductionEnvelope": {
                "objective_rxn": "BIOMASS_Ecoli_core_w_GAM",
                "carbon_sources": ["EX_glc__D_e"],
            },
            "Auxotrophy": {},
            "MinimalMedia": {},
            "MissingMedia": {"target_metabolites": ["cpd00027", "cpd00067"]},
            "ReactionExpression": {"expression_data": {"b0008": 1.5, "b0009": 0.8}},
        }

        # Adjust parameters for iML1515 model
        if "iML1515" in model_path:
            if tool_name == "ProductionEnvelope":
                tool_specific_params["ProductionEnvelope"][
                    "objective_rxn"
                ] = "BIOMASS_Ec_iML1515_core_75p37M"
            elif tool_name == "GeneDeletion":
                tool_specific_params["GeneDeletion"]["genes"] = ["b0008", "b0116"]

        # Adjust parameters for Mycoplasma model
        elif "Mycoplasma" in model_path:
            if tool_name == "ProductionEnvelope":
                # Use generic biomass objective for ModelSEEDpy models
                tool_specific_params["ProductionEnvelope"][
                    "objective_rxn"
                ] = "bio1"  # Common ModelSEEDpy biomass reaction ID
            elif tool_name == "GeneDeletion":
                # Use ModelSEEDpy gene IDs (will need to check what's available)
                tool_specific_params["GeneDeletion"]["genes"] = [
                    "83331.1.peg.1",
                    "83331.1.peg.2",
                ]

        return {**base_params, **tool_specific_params.get(tool_name, {})}

    def run_comprehensive_test(self):
        """Run all tools on all models with comprehensive reporting"""

        total_tests = len(self.tools) * len(self.models)
        current_test = 0

        for model_name, model_path in self.models.items():
            self.results[model_name] = {}

            print(f"\n🧬 TESTING MODEL: {model_name}")
            print(f"📍 Path: {model_path}")

            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                continue

            print(f"📏 File size: {os.path.getsize(model_path) / 1024:.1f} KB")
            print("=" * 60)

            for tool_name, tool in self.tools.items():
                current_test += 1
                print(f"\n[{current_test}/{total_tests}] Testing {tool_name}")

                test_result = self.test_tool_on_model(
                    tool_name, tool, model_name, model_path
                )
                self.results[model_name][tool_name] = test_result

    def generate_summary_report(self):
        """Generate comprehensive summary report"""

        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)

        # Overall statistics
        total_tests = sum(len(model_results) for model_results in self.results.values())
        successful_tests = sum(
            1
            for model_results in self.results.values()
            for test_result in model_results.values()
            if test_result["success"]
        )

        print(
            f"📈 Overall Success Rate: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)"
        )

        # Per-model summary
        for model_name, model_results in self.results.items():
            print(f"\n🧬 {model_name.upper()}")
            print("-" * 40)

            model_successes = sum(
                1 for result in model_results.values() if result["success"]
            )
            model_total = len(model_results)

            print(
                f"Success Rate: {model_successes}/{model_total} ({100*model_successes/model_total:.1f}%)"
            )

            # Tool-by-tool results
            for tool_name, result in model_results.items():
                status = "✅" if result["success"] else "❌"
                time_str = f"{result['execution_time']:.2f}s"
                print(f"  {status} {tool_name:<18} {time_str:>8}")

                if not result["success"] and result["error"]:
                    error_type = result["error"]["type"]
                    print(f"      💥 {error_type}")

        # Failed tests detail
        failed_tests = []
        for model_name, model_results in self.results.items():
            for tool_name, result in model_results.items():
                if not result["success"]:
                    failed_tests.append((model_name, tool_name, result))

        if failed_tests:
            print(f"\n❌ FAILED TESTS ANALYSIS ({len(failed_tests)} failures)")
            print("-" * 50)

            for model_name, tool_name, result in failed_tests:
                print(f"\n🔍 {model_name} → {tool_name}")
                print(f"   Error: {result['error']['type']}")
                print(f"   Message: {result['error']['message']}")

                # Show solver messages if any
                if result["solver_messages"]:
                    print(f"   Solver: {', '.join(result['solver_messages'])}")

    def save_detailed_results(self, output_file="tool_testbed_results.json"):
        """Save detailed results to JSON file"""

        # Create output directory
        output_dir = Path("testbed_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{timestamp}_{output_file}"

        # Add metadata
        full_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "models_tested": list(self.models.keys()),
                "tools_tested": list(self.tools.keys()),
                "total_tests": sum(
                    len(model_results) for model_results in self.results.values()
                ),
            },
            "results": self.results,
        }

        with open(output_path, "w") as f:
            json.dump(full_results, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: {output_path}")
        print(f"📄 File size: {os.path.getsize(output_path) / 1024:.1f} KB")

        return output_path


def main():
    """Main execution function"""

    print("🧪 COBRApy Tool Comprehensive Testbed")
    print("====================================")
    print("Testing all COBRApy tools against E. coli models")
    print("with detailed output, debugging, and error analysis.\n")

    testbed = ToolTestbed()

    try:
        # Run comprehensive tests
        testbed.run_comprehensive_test()

        # Generate summary
        testbed.generate_summary_report()

        # Save detailed results
        output_file = testbed.save_detailed_results()

        print(f"\n🎉 Testbed complete!")
        print(f"📊 View detailed results in: {output_file}")

    except KeyboardInterrupt:
        print("\n⚠️  Testbed interrupted by user")
    except Exception as e:
        print(f"\n💥 Testbed failed with error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
