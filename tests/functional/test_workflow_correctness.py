#!/usr/bin/env python3
"""
End-to-End Workflow Correctness Testing

This module tests complete workflows from start to finish, validating
that the entire system works together correctly and produces meaningful
scientific results.

Tests validate:
- Complete genome-to-model pipelines
- Multi-tool analysis workflows
- Data flow between tools
- Scientific coherence of results
"""

import json
import sys
import time
from pathlib import Path

import pytest

from src.tools.biochem.resolver import BiochemEntityResolverTool
from src.tools.cobra.auxotrophy import AuxotrophyTool
from src.tools.cobra.essentiality import EssentialityAnalysisTool
from src.tools.cobra.fba import FBATool
from src.tools.cobra.minimal_media import MinimalMediaTool
from src.tools.modelseed.annotation import RastAnnotationTool
from src.tools.modelseed.builder import ModelBuildTool
from src.tools.modelseed.gapfill import GapFillTool


class TestCompleteAnalysisWorkflows:
    """Test complete analysis workflows work end-to-end"""

    @pytest.fixture
    def analysis_tools(self):
        """Get standard analysis tools"""
        return {
            "fba": FBATool({"name": "workflow_fba"}),
            "minimal_media": MinimalMediaTool({"name": "workflow_mm"}),
            "essentiality": EssentialityAnalysisTool({"name": "workflow_ess"}),
            "auxotrophy": AuxotrophyTool({"name": "workflow_aux"}),
            "biochem": BiochemEntityResolverTool({"name": "workflow_biochem"}),
        }

    @pytest.fixture
    def test_model_path(self):
        return "data/examples/e_coli_core.xml"

    def test_comprehensive_model_characterization(
        self, analysis_tools, test_model_path
    ):
        """Test complete model characterization workflow"""
        print(f"\n🔬 Running comprehensive model characterization...")

        results = {}

        # Step 1: Baseline growth analysis
        print("Step 1: Analyzing baseline growth...")
        fba_result = analysis_tools["fba"]._run_tool({"model_path": test_model_path})
        assert fba_result.success, f"FBA failed: {fba_result.error}"
        results["fba"] = fba_result.data

        baseline_growth = fba_result.data["objective_value"]
        print(f"  Growth rate: {baseline_growth:.3f} h⁻¹")

        # Step 2: Nutritional requirements
        print("Step 2: Analyzing nutritional requirements...")
        mm_result = analysis_tools["minimal_media"]._run_tool(
            {"model_path": test_model_path}
        )
        assert mm_result.success, f"Minimal media failed: {mm_result.error}"
        results["minimal_media"] = mm_result.data

        essential_nutrients = mm_result.data.get("essential_nutrients", [])
        print(f"  Essential nutrients: {len(essential_nutrients)}")

        # Step 3: Essential components
        print("Step 3: Analyzing essential components...")
        ess_result = analysis_tools["essentiality"]._run_tool(
            {"model_path": test_model_path}
        )
        assert ess_result.success, f"Essentiality failed: {ess_result.error}"
        results["essentiality"] = ess_result.data

        essential_genes = ess_result.data.get("essential_genes", [])
        print(f"  Essential genes: {len(essential_genes)}")

        # Step 4: Auxotrophy analysis
        print("Step 4: Analyzing auxotrophies...")
        aux_result = analysis_tools["auxotrophy"]._run_tool(
            {"model_path": test_model_path}
        )
        if aux_result.success:  # Auxotrophy might not be available for all models
            results["auxotrophy"] = aux_result.data
            auxotrophies = aux_result.data.get("auxotrophies", [])
            print(f"  Auxotrophies: {len(auxotrophies)}")

        # Validate workflow coherence
        self._validate_workflow_coherence(results)

        print("✅ Comprehensive characterization completed successfully")
        return results

    def _validate_workflow_coherence(self, results):
        """Validate that workflow results are scientifically coherent"""

        # Extract key metrics
        growth_rate = results["fba"]["objective_value"]
        essential_nutrients = len(
            results["minimal_media"].get("essential_nutrients", [])
        )
        essential_genes = len(results["essentiality"].get("essential_genes", []))

        # Coherence checks

        # 1. Growth rate and nutritional complexity should be reasonable
        if growth_rate > 0.8:  # High growth
            assert (
                essential_nutrients <= 25
            ), f"High growth model ({growth_rate:.3f}) shouldn't need too many nutrients ({essential_nutrients})"

        # 2. Essential genes and growth should be consistent
        if growth_rate > 0.1:  # Model can grow
            assert (
                essential_genes < 100
            ), f"Growing model shouldn't have excessive essential genes ({essential_genes})"

        # 3. Nutritional and auxotrophy consistency
        if "auxotrophy" in results:
            auxotrophies = len(results["auxotrophy"].get("auxotrophies", []))
            if auxotrophies > 0:
                # If auxotrophic, should need external nutrients
                assert (
                    essential_nutrients >= auxotrophies
                ), f"Auxotrophies ({auxotrophies}) should be reflected in essential nutrients ({essential_nutrients})"

    def test_optimization_workflow(self, analysis_tools, test_model_path):
        """Test optimization-focused workflow"""
        print(f"\n⚡ Running optimization workflow...")

        # Step 1: Baseline assessment
        fba_result = analysis_tools["fba"]._run_tool({"model_path": test_model_path})
        assert fba_result.success

        baseline_growth = fba_result.data["objective_value"]
        print(f"  Baseline growth: {baseline_growth:.3f} h⁻¹")

        # Step 2: Identify limitations
        mm_result = analysis_tools["minimal_media"]._run_tool(
            {"model_path": test_model_path}
        )
        assert mm_result.success

        # Step 3: Find optimization targets
        ess_result = analysis_tools["essentiality"]._run_tool(
            {"model_path": test_model_path}
        )
        assert ess_result.success

        # Optimization potential assessment
        optimization_potential = self._assess_optimization_potential(
            baseline_growth, mm_result.data, ess_result.data
        )

        assert "growth_potential" in optimization_potential
        assert "limiting_factors" in optimization_potential

        print(f"  Optimization potential: {optimization_potential['growth_potential']}")
        print("✅ Optimization workflow completed")

        return optimization_potential

    def _assess_optimization_potential(self, growth_rate, mm_data, ess_data):
        """Assess optimization potential based on analysis results"""

        essential_nutrients = len(mm_data.get("essential_nutrients", []))
        essential_genes = len(ess_data.get("essential_genes", []))

        # Simple heuristic for optimization potential
        if growth_rate > 0.9:
            potential = "low"  # Already optimized
        elif growth_rate > 0.6:
            potential = "medium"
        else:
            potential = "high"  # Lots of room for improvement

        limiting_factors = []
        if essential_nutrients > 20:
            limiting_factors.append("high_nutritional_requirements")
        if essential_genes > 30:
            limiting_factors.append("high_gene_essentiality")
        if growth_rate < 0.3:
            limiting_factors.append("low_baseline_growth")

        return {
            "growth_potential": potential,
            "limiting_factors": limiting_factors,
            "current_growth": growth_rate,
            "nutritional_complexity": essential_nutrients,
            "genetic_constraints": essential_genes,
        }


class TestModelBuildingWorkflows:
    """Test complete model building workflows (if ModelSEED tools are available)"""

    @pytest.fixture
    def modeling_tools(self):
        """Get model building tools"""
        try:
            return {
                "annotation": RastAnnotationTool({"name": "workflow_rast"}),
                "builder": ModelBuildTool({"name": "workflow_build"}),
                "gapfill": GapFillTool({"name": "workflow_gapfill"}),
                "fba": FBATool({"name": "workflow_fba"}),
            }
        except Exception as e:
            pytest.skip(f"ModelSEED tools not available: {e}")

    @pytest.fixture
    def test_genome_path(self):
        genome_path = Path("data/examples/pputida.fna")
        if not genome_path.exists():
            pytest.skip("Test genome file not available")
        return str(genome_path)

    def test_genome_to_model_pipeline(self, modeling_tools, test_genome_path):
        """Test complete genome-to-model pipeline"""
        print(f"\n🧬 Running genome-to-model pipeline...")

        # This test might take a while and require external services
        # Mark as slow/integration test

        # Step 1: Genome annotation
        print("Step 1: Annotating genome...")
        try:
            annotation_result = modeling_tools["annotation"].run(
                {"genome_file": test_genome_path, "genome_name": "test_putida"}
            )

            if not annotation_result.success:
                pytest.skip(
                    f"Annotation failed (external service): {annotation_result.error}"
                )

            print("  Annotation completed")

        except Exception as e:
            pytest.skip(f"Annotation service unavailable: {e}")

        # Step 2: Model building
        print("Step 2: Building draft model...")
        try:
            build_result = modeling_tools["builder"].run(
                {
                    "genome_object": annotation_result.data["genome_object"],
                    "model_id": "test_putida_model",
                }
            )

            if not build_result.success:
                pytest.skip(f"Model building failed: {build_result.error}")

            print("  Draft model built")

        except Exception as e:
            pytest.skip(f"Model building unavailable: {e}")

        # Step 3: Test draft model growth
        print("Step 3: Testing draft model...")
        draft_model_path = build_result.data.get("model_path")
        if draft_model_path:
            fba_result = modeling_tools["fba"].run({"model_path": draft_model_path})
            if fba_result.success:
                draft_growth = fba_result.data["objective_value"]
                print(f"  Draft growth: {draft_growth:.3f} h⁻¹")

                # Step 4: Gapfilling if needed
                if draft_growth < 0.01:  # Very low growth, try gapfilling
                    print("Step 4: Gapfilling model...")
                    gapfill_result = modeling_tools["gapfill"].run(
                        {"model_object": build_result.data["model_object"]}
                    )

                    if gapfill_result.success:
                        gapfilled_model_path = gapfill_result.data.get("model_path")
                        if gapfilled_model_path:
                            final_fba = modeling_tools["fba"].run(
                                {"model_path": gapfilled_model_path}
                            )
                            if final_fba.success:
                                final_growth = final_fba.data["objective_value"]
                                print(f"  Final growth: {final_growth:.3f} h⁻¹")

                                # Gapfilling should improve growth
                                assert (
                                    final_growth > draft_growth
                                ), f"Gapfilling should improve growth: {draft_growth} → {final_growth}"

        print("✅ Genome-to-model pipeline completed")


class TestDataFlowValidation:
    """Test data flows correctly between tools"""

    def test_biochemistry_integration(self):
        """Test biochemistry data flows to analysis tools"""
        biochem_tool = BiochemEntityResolverTool({"name": "test_biochem"})

        # Test resolving common metabolites
        test_compounds = ["cpd00002", "cpd00001", "ATP", "H2O"]

        resolved_compounds = {}
        for compound in test_compounds:
            result = biochem_tool._run_tool({"entity_id": compound})
            if result.success:
                resolved_compounds[compound] = result.data

        # Should resolve at least some compounds
        assert len(resolved_compounds) > 0, "Should resolve some biochemistry entities"

        # Check that names are meaningful
        for compound, data in resolved_compounds.items():
            name = data.get("primary_name", "")
            assert len(name) > 0, f"Compound {compound} should have a name"

            # Allow case where input name matches output (e.g., "ATP" -> "ATP")
            # But for compound IDs, expect different names
            if compound.startswith("cpd"):
                assert (
                    name != compound
                ), f"Compound ID {compound} should resolve to a different name, got {name}"

            print(f"  {compound} → {name}")

    def test_result_format_consistency(self):
        """Test all tools return consistent result formats"""
        tools = [
            FBATool({"name": "test_fba"}),
            MinimalMediaTool({"name": "test_mm"}),
            EssentialityAnalysisTool({"name": "test_ess"}),
        ]

        model_path = "data/examples/e_coli_core.xml"

        for tool in tools:
            result = tool._run_tool({"model_path": model_path})

            # All results should have standard structure
            assert hasattr(
                result, "success"
            ), f"{tool.tool_name} missing success attribute"
            assert hasattr(result, "data"), f"{tool.tool_name} missing data attribute"
            assert hasattr(result, "error"), f"{tool.tool_name} missing error attribute"

            if result.success:
                assert isinstance(
                    result.data, dict
                ), f"{tool.tool_name} data should be dict"
                assert len(result.data) > 0, f"{tool.tool_name} should return data"
            else:
                assert result.error, f"{tool.tool_name} should provide error message"


class TestPerformanceAndScalability:
    """Test system performance with realistic workloads"""

    def test_analysis_performance(self):
        """Test analysis tools complete in reasonable time"""
        tools = [
            ("FBA", FBATool({"name": "perf_fba"})),
            ("Minimal Media", MinimalMediaTool({"name": "perf_mm"})),
            ("Essentiality", EssentialityAnalysisTool({"name": "perf_ess"})),
        ]

        model_path = "data/examples/e_coli_core.xml"

        performance_results = {}

        for tool_name, tool in tools:
            start_time = time.time()
            result = tool._run_tool({"model_path": model_path})
            end_time = time.time()

            duration = end_time - start_time
            performance_results[tool_name] = {
                "duration": duration,
                "success": result.success,
            }

            # Performance expectations
            if tool_name == "FBA":
                assert duration < 10, f"FBA took too long: {duration:.1f}s"
            elif tool_name == "Minimal Media":
                assert duration < 30, f"Minimal media took too long: {duration:.1f}s"
            elif tool_name == "Essentiality":
                assert duration < 60, f"Essentiality took too long: {duration:.1f}s"

            print(f"  {tool_name}: {duration:.1f}s")

        # Overall workflow should be reasonable
        total_time = sum(result["duration"] for result in performance_results.values())
        assert total_time < 120, f"Total workflow too slow: {total_time:.1f}s"

        return performance_results

    def test_memory_usage_reasonable(self):
        """Test tools don't use excessive memory"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run analysis tools
        fba_tool = FBATool({"name": "memory_test"})
        fba_tool._run_tool({"model_path": "data/examples/e_coli_core.xml"})

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 100 MB for simple analysis)
        assert memory_increase < 100, f"Memory usage too high: {memory_increase:.1f} MB"

        print(f"  Memory increase: {memory_increase:.1f} MB")


def run_workflow_correctness_tests():
    """Run all workflow correctness tests"""
    print("🔄 Running End-to-End Workflow Correctness Tests")
    print("=" * 55)
    print("⚠️  Note: Some tests require external services (RAST, ModelSEED)")
    print("   Tests will be skipped if services are unavailable")
    print()

    # Run pytest on this module
    test_file = __file__
    exit_code = pytest.main([test_file, "-v", "--tb=short"])

    if exit_code == 0:
        print("\n✅ All workflow correctness tests PASSED!")
        print("🔄 Complete workflows produce scientifically valid results")
    else:
        print("\n❌ Some workflow tests FAILED or were SKIPPED!")
        print("⚠️  Check external service availability and workflow logic")

    return exit_code == 0


if __name__ == "__main__":
    success = run_workflow_correctness_tests()
    sys.exit(0 if success else 1)
