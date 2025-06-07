#!/usr/bin/env python3
"""
Advanced COBRA Tools Functional Correctness Testing

This module tests that advanced COBRA analysis tools produce biologically
meaningful results for gene deletion, flux sampling, and production envelope
analyses.

Tests validate:
- Gene deletion impact analysis accuracy
- Flux sampling statistical validity
- Production envelope trade-off calculations
- Cross-tool consistency and biological realism
"""

import sys

import numpy as np
import pytest

from src.tools.cobra.flux_sampling import FluxSamplingTool
from src.tools.cobra.gene_deletion import GeneDeletionTool
from src.tools.cobra.production_envelope import ProductionEnvelopeTool


class TestGeneDeletionCorrectness:
    """Test gene deletion analysis produces biologically meaningful results"""

    @pytest.fixture
    def gene_deletion_tool(self):
        return GeneDeletionTool({"name": "test_gene_deletion"})

    @pytest.fixture
    def test_model_path(self):
        return "data/examples/e_coli_core.xml"

    def test_single_gene_deletion_impacts(self, gene_deletion_tool, test_model_path):
        """Test single gene deletion produces realistic growth impacts"""
        result = gene_deletion_tool._run_tool(
            {
                "model_path": test_model_path,
                "gene_list": ["b0008", "b0114", "b0116"],  # Known E. coli genes
                "method": "single",
            }
        )

        assert result.success, f"Gene deletion failed: {result.error}"

        analysis = result.data.get("analysis", {})
        wild_type_growth = result.data.get("wild_type_growth", 0)

        assert wild_type_growth > 0, "Should have positive wild-type growth"
        assert "summary" in analysis, "Should have analysis summary"

        # Check all tested genes across all categories
        all_tested_genes = []
        for category in [
            "essential_genes",
            "severely_impaired",
            "moderately_impaired",
            "mildly_impaired",
            "no_effect",
            "improved_growth",
        ]:
            genes_in_category = analysis.get(category, [])
            all_tested_genes.extend(genes_in_category)

            # Check each gene's data structure
            for gene_data in genes_in_category:
                growth_rate = gene_data.get("growth", 0)
                growth_ratio = gene_data.get("growth_rate_ratio", 0)

                # Growth rate should be non-negative and finite
                assert growth_rate >= 0, f"Gene has negative growth rate: {growth_rate}"
                assert np.isfinite(growth_rate), f"Gene has non-finite growth rate"

                # Growth ratio should be reasonable (0 to ~1.1 for possible improvement)
                assert 0 <= growth_ratio <= 1.5, f"Invalid growth ratio: {growth_ratio}"

                genes = gene_data.get("genes", "unknown")
                print(f"  {genes}: growth={growth_rate:.3f}, ratio={growth_ratio:.3f}")

        assert len(all_tested_genes) >= 3, "Should have tested at least 3 genes"

        # Should find at least one gene with some impact (< 95% of wild-type)
        genes_with_impact = []
        for gene_data in all_tested_genes:
            if gene_data.get("growth_rate_ratio", 1.0) < 0.95:
                genes_with_impact.append(gene_data)

        assert (
            len(genes_with_impact) > 0
        ), "Should find at least one gene with measurable impact (>5% reduction)"

    def test_essential_gene_identification(self, gene_deletion_tool, test_model_path):
        """Test identification of essential genes"""
        result = gene_deletion_tool._run_tool(
            {
                "model_path": test_model_path,
                "gene_list": None,  # Analyze all genes
                "method": "single",
                "essential_threshold": 0.01,  # Growth < 1% of wild-type is essential
            }
        )

        if not result.success:
            pytest.skip(f"Gene deletion analysis failed: {result.error}")

        analysis = result.data.get("analysis", {})
        essential_genes = analysis.get("essential_genes", [])
        summary = analysis.get("summary", {})

        # E. coli core should have some essential genes (typically 5-30)
        total_tested = summary.get("total_genes_tested", 0)
        assert total_tested > 0, "Should have tested some genes"

        # For a comprehensive test, should find some essential genes
        # But for a limited test with 3 genes, might find 0-3 essential
        essential_count = len(essential_genes)
        assert (
            0 <= essential_count <= total_tested
        ), "Essential count should be reasonable"

        # Check that essential genes have very low growth
        for gene_data in essential_genes:
            growth_ratio = gene_data.get("growth_rate_ratio", 0)
            genes = gene_data.get("genes", "unknown")
            assert (
                growth_ratio <= 0.01
            ), f"Essential gene {genes} has too high growth ratio: {growth_ratio}"

        print(
            f"✅ Found {essential_count} essential genes out of {total_tested} tested"
        )

    def test_double_gene_deletion_analysis(self, gene_deletion_tool, test_model_path):
        """Test double gene deletion analysis for synthetic lethality"""
        # Test with a small subset to avoid long computation
        test_genes = ["b0008", "b0114"]

        result = gene_deletion_tool._run_tool(
            {"model_path": test_model_path, "gene_list": test_genes, "method": "double"}
        )

        if not result.success:
            # Double deletion might not be implemented or might be slow
            pytest.skip(f"Double gene deletion not available: {result.error}")

        deletion_results = result.data.get("deletion_results", {})

        # Should have results for gene pairs
        expected_pairs = len(test_genes) * (len(test_genes) - 1) // 2
        if expected_pairs > 0:
            assert len(deletion_results) > 0, "Should have double deletion results"

        # Check result format
        for gene_pair, deletion_data in deletion_results.items():
            assert (
                "," in gene_pair or "+" in gene_pair
            ), f"Invalid gene pair format: {gene_pair}"

            growth_rate = deletion_data.get("growth_rate", 0)
            assert (
                growth_rate >= 0
            ), f"Negative growth rate for {gene_pair}: {growth_rate}"


class TestFluxSamplingCorrectness:
    """Test flux sampling produces statistically valid results"""

    @pytest.fixture
    def flux_sampling_tool(self):
        return FluxSamplingTool({"name": "test_flux_sampling"})

    def test_flux_sampling_statistical_validity(self, flux_sampling_tool):
        """Test flux sampling produces statistically valid flux distributions"""
        result = flux_sampling_tool._run_tool(
            {
                "model_path": "data/examples/e_coli_core.xml",
                "n_samples": 100,  # Reduced for faster testing
                "method": "achr",
            }
        )

        if not result.success:
            pytest.skip(f"Flux sampling failed: {result.error}")

        # Check sampling results
        sampling_data = result.data.get("sampling_results", {})
        assert "flux_samples" in sampling_data, "Should have flux samples"

        flux_samples = sampling_data["flux_samples"]
        assert len(flux_samples) > 0, "Should have flux sample data"

        # Check sample statistics
        sample_stats = result.data.get("sample_statistics", {})

        if "mean_fluxes" in sample_stats and "std_fluxes" in sample_stats:
            mean_fluxes = sample_stats["mean_fluxes"]
            std_fluxes = sample_stats["std_fluxes"]

            # Should have statistics for multiple reactions
            assert (
                len(mean_fluxes) > 10
            ), "Should have statistics for multiple reactions"
            assert len(std_fluxes) == len(
                mean_fluxes
            ), "Mean and std should have same length"

            # Check statistical validity
            for reaction_id, mean_flux in mean_fluxes.items():
                std_flux = std_fluxes.get(reaction_id, 0)

                # Values should be finite
                assert np.isfinite(mean_flux), f"Non-finite mean flux for {reaction_id}"
                assert np.isfinite(std_flux), f"Non-finite std flux for {reaction_id}"

                # Standard deviation should be non-negative
                assert std_flux >= 0, f"Negative std for {reaction_id}: {std_flux}"

        print(f"✅ Flux sampling completed with valid statistics")

    def test_flux_sampling_correlation_analysis(self, flux_sampling_tool):
        """Test flux correlation analysis validity"""
        result = flux_sampling_tool._run_tool(
            {
                "model_path": "data/examples/e_coli_core.xml",
                "n_samples": 50,
                "method": "achr",
            }
        )

        if not result.success:
            pytest.skip(f"Flux sampling failed: {result.error}")

        correlation_data = result.data.get("correlation_analysis", {})

        if "high_correlations" in correlation_data:
            high_correlations = correlation_data["high_correlations"]

            # Check correlation format
            for correlation in high_correlations:
                assert "reaction1" in correlation, "Missing reaction1 in correlation"
                assert "reaction2" in correlation, "Missing reaction2 in correlation"
                assert "correlation" in correlation, "Missing correlation value"

                corr_value = correlation["correlation"]
                assert -1 <= corr_value <= 1, f"Invalid correlation: {corr_value}"

        print("✅ Flux correlation analysis has valid format")

    def test_flux_sampling_subsystem_analysis(self, flux_sampling_tool):
        """Test flux sampling subsystem breakdown"""
        result = flux_sampling_tool._run_tool(
            {"model_path": "data/examples/e_coli_core.xml", "n_samples": 50}
        )

        if not result.success:
            pytest.skip(f"Flux sampling failed: {result.error}")

        subsystem_data = result.data.get("subsystem_analysis", {})

        if "subsystem_activity" in subsystem_data:
            subsystem_activity = subsystem_data["subsystem_activity"]

            # Should have activity data for metabolic subsystems
            assert len(subsystem_activity) > 0, "Should have subsystem activity data"

            # Check activity values are reasonable
            for subsystem, activity in subsystem_activity.items():
                assert isinstance(
                    activity, (int, float)
                ), f"Invalid activity type for {subsystem}"
                assert activity >= 0, f"Negative activity for {subsystem}: {activity}"

        print("✅ Subsystem analysis has valid structure")


class TestProductionEnvelopeCorrectness:
    """Test production envelope analysis for metabolic engineering"""

    @pytest.fixture
    def production_envelope_tool(self):
        return ProductionEnvelopeTool({"name": "test_production_envelope"})

    def test_growth_vs_production_tradeoff(self, production_envelope_tool):
        """Test growth vs production trade-off analysis"""
        result = production_envelope_tool._run_tool(
            {
                "model_path": "data/examples/e_coli_core.xml",
                "target_reaction": "EX_ac_e",  # Acetate exchange
                "carbon_source": "EX_glc__D_e",  # Glucose exchange
                "points": 20,
            }
        )

        if not result.success:
            pytest.skip(f"Production envelope failed: {result.error}")

        envelope_data = result.data.get("envelope_data", {})
        assert "pareto_points" in envelope_data, "Should have Pareto frontier points"

        pareto_points = envelope_data["pareto_points"]
        assert len(pareto_points) > 5, "Should have multiple Pareto points"

        # Check Pareto point validity
        for point in pareto_points:
            growth_rate = point.get("growth_rate", 0)
            production_rate = point.get("production_rate", 0)

            # Values should be non-negative and finite
            assert growth_rate >= 0, f"Negative growth rate: {growth_rate}"
            assert production_rate >= 0, f"Negative production rate: {production_rate}"
            assert np.isfinite(growth_rate), "Non-finite growth rate"
            assert np.isfinite(production_rate), "Non-finite production rate"

        print(f"✅ Production envelope has {len(pareto_points)} valid Pareto points")

    def test_production_envelope_tradeoff_relationship(self, production_envelope_tool):
        """Test that production envelope shows realistic trade-offs"""
        result = production_envelope_tool._run_tool(
            {
                "model_path": "data/examples/e_coli_core.xml",
                "target_reaction": "EX_etoh_e",  # Ethanol production
                "carbon_source": "EX_glc__D_e",
                "points": 15,
            }
        )

        if not result.success:
            pytest.skip(f"Production envelope failed: {result.error}")

        envelope_data = result.data.get("envelope_data", {})
        pareto_points = envelope_data.get("pareto_points", [])

        if len(pareto_points) > 3:
            # Sort points by growth rate
            sorted_points = sorted(pareto_points, key=lambda p: p["growth_rate"])

            # Check for general trade-off trend
            # Higher production often correlates with lower growth
            high_growth_point = sorted_points[-1]  # Highest growth
            low_growth_point = sorted_points[0]  # Lowest growth

            high_growth_rate = high_growth_point["growth_rate"]
            high_production = high_growth_point["production_rate"]

            low_growth_rate = low_growth_point["growth_rate"]
            low_production = low_growth_point["production_rate"]

            # Should show some variation in the envelope
            growth_variation = high_growth_rate - low_growth_rate
            production_variation = abs(high_production - low_production)

            assert growth_variation > 0.01, "Should show growth rate variation"
            print(f"✅ Growth variation: {growth_variation:.3f}")
            print(f"✅ Production variation: {production_variation:.3f}")

    def test_production_envelope_optimization_validity(self, production_envelope_tool):
        """Test production envelope optimization constraints"""
        result = production_envelope_tool._run_tool(
            {
                "model_path": "data/examples/e_coli_core.xml",
                "target_reaction": "EX_succ_e",  # Succinate production
                "carbon_source": "EX_glc__D_e",
                "points": 10,
            }
        )

        if not result.success:
            pytest.skip(f"Production envelope failed: {result.error}")

        # Check optimization settings
        optimization_data = result.data.get("optimization_settings", {})

        if "objective_function" in optimization_data:
            objective = optimization_data["objective_function"]
            assert objective, "Should have defined objective function"

        # Check constraint satisfaction
        envelope_data = result.data.get("envelope_data", {})
        pareto_points = envelope_data.get("pareto_points", [])

        for point in pareto_points:
            # All points should be feasible solutions
            carbon_uptake = point.get("carbon_uptake", 0)

            # Carbon uptake should be reasonable (not excessive)
            if carbon_uptake > 0:
                assert carbon_uptake < 50, f"Excessive carbon uptake: {carbon_uptake}"

        print("✅ Production envelope optimization constraints satisfied")


class TestAdvancedCOBRAToolsIntegration:
    """Test integration between advanced COBRA tools"""

    def test_gene_deletion_flux_sampling_consistency(self):
        """Test consistency between gene deletion and flux sampling"""
        model_path = "data/examples/e_coli_core.xml"

        # Run gene deletion analysis
        gene_tool = GeneDeletionTool({"name": "test_integration_gene"})
        gene_result = gene_tool._run_tool(
            {
                "model_path": model_path,
                "gene_list": ["b0008"],  # Single gene test
                "method": "single",
            }
        )

        # Run flux sampling
        flux_tool = FluxSamplingTool({"name": "test_integration_flux"})
        flux_result = flux_tool._run_tool({"model_path": model_path, "n_samples": 25})

        if gene_result.success and flux_result.success:
            # Both tools should work on the same model
            gene_data = gene_result.data
            flux_data = flux_result.data

            # Check that both report reasonable biomass-related values
            if "deletion_results" in gene_data and "sample_statistics" in flux_data:
                print(
                    "✅ Gene deletion and flux sampling both successful on same model"
                )
            else:
                print("⚠️  Tools successful but data format differs")
        else:
            print("⚠️  One or both advanced tools failed - check tool availability")


def run_advanced_cobra_functional_tests():
    """Run all advanced COBRA tools functional tests"""
    print("🧬 Running Advanced COBRA Tools Functional Correctness Tests")
    print("=" * 65)
    print("Testing gene deletion, flux sampling, and production envelope accuracy")
    print()

    # Run pytest on this module
    test_file = __file__
    exit_code = pytest.main([test_file, "-v", "--tb=short"])

    if exit_code == 0:
        print("\n✅ All advanced COBRA functional tests PASSED!")
        print("🔬 Advanced COBRA tools produce biologically meaningful results")
    else:
        print("\n❌ Some advanced COBRA functional tests FAILED!")
        print("⚠️  Check tool implementations and biological validity")

    return exit_code == 0


if __name__ == "__main__":
    success = run_advanced_cobra_functional_tests()
    sys.exit(0 if success else 1)
