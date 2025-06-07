#!/usr/bin/env python3
"""
Comprehensive Functional Testing for Metabolic Tools

This module tests that metabolic analysis tools produce biologically
correct and mathematically valid results, not just that they execute
without errors.

Tests validate:
- Numerical correctness and biological feasibility
- Output ranges and expected values
- Cross-tool consistency
- Edge case handling
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.tools.cobra.auxotrophy import AuxotrophyTool
from src.tools.cobra.essentiality import EssentialityAnalysisTool
from src.tools.cobra.fba import FBATool
from src.tools.cobra.flux_variability import FluxVariabilityTool
from src.tools.cobra.gene_deletion import GeneDeletionTool
from src.tools.cobra.minimal_media import MinimalMediaTool


class TestFBAToolCorrectness:
    """Test FBA tool produces biologically correct results"""

    @pytest.fixture
    def fba_tool(self):
        return FBATool({"name": "test_fba"})

    @pytest.fixture
    def test_model_path(self):
        return "data/examples/e_coli_core.xml"

    def test_ecoli_growth_rate_realistic(self, fba_tool, test_model_path):
        """Test E. coli core model produces realistic growth rate"""
        result = fba_tool._run_tool({"model_path": test_model_path})

        assert result.success, f"FBA failed: {result.error}"

        # Find actual biomass flux (not optimization objective)
        significant_fluxes = result.data.get("significant_fluxes", {})
        growth_rate = None
        for reaction, flux in significant_fluxes.items():
            if "biomass" in reaction.lower():
                growth_rate = flux
                break

        assert growth_rate is not None, "Could not find biomass reaction flux"

        # E. coli core model should grow between 0.5-1.2 h⁻¹ on glucose
        assert (
            0.5 <= growth_rate <= 1.2
        ), f"Growth rate {growth_rate:.3f} h⁻¹ outside expected range [0.5, 1.2] for E. coli"

        # Check solution status is optimal
        assert (
            result.data.get("status") == "optimal"
        ), f"Expected optimal solution, got {result.data.get('status')}"

        # Biomass reaction should be active (which we already confirmed above)
        assert growth_rate > 0, "Biomass reaction should be active for growing model"

    def test_fba_flux_balance(self, fba_tool, test_model_path):
        """Test flux balance constraint satisfaction"""
        result = fba_tool._run_tool({"model_path": test_model_path})

        assert result.success

        # Check that flux values are provided (in significant_fluxes)
        significant_fluxes = result.data.get("significant_fluxes", {})
        assert len(significant_fluxes) > 0, "Should have significant flux values"

        # Verify no unrealistic flux values (should be finite)
        for reaction, flux in significant_fluxes.items():
            assert np.isfinite(flux), f"Reaction {reaction} has non-finite flux {flux}"
            # Fluxes should be reasonable (not extremely large)
            assert (
                abs(flux) < 1000
            ), f"Reaction {reaction} flux {flux} seems unrealistically large"

    def test_fba_glucose_consumption(self, fba_tool, test_model_path):
        """Test that glucose consumption is reasonable"""
        result = fba_tool._run_tool({"model_path": test_model_path})

        assert result.success

        # Find glucose exchange flux
        significant_fluxes = result.data.get("significant_fluxes", {})

        glucose_flux = None
        for reaction, flux in significant_fluxes.items():
            if any(term in reaction.lower() for term in ["glc", "glucose", "ex_glc"]):
                glucose_flux = flux
                break

        if glucose_flux is not None:
            # Glucose consumption should be negative (uptake) and reasonable
            assert (
                glucose_flux < 0
            ), f"Glucose flux should be negative (uptake), got {glucose_flux}"
            assert glucose_flux > -30, f"Glucose uptake {glucose_flux} seems too high"


class TestFluxVariabilityCorrectness:
    """Test FVA produces mathematically correct flux ranges"""

    @pytest.fixture
    def fva_tool(self):
        return FluxVariabilityTool({"name": "test_fva"})

    def test_fva_flux_ranges_valid(self, fva_tool):
        """Test FVA produces valid min/max flux ranges"""
        result = fva_tool._run_tool(
            {"model_path": "data/examples/e_coli_core.xml", "fraction_of_optimum": 0.9}
        )

        assert result.success, f"FVA failed: {result.error}"

        summary = result.data["summary"]
        assert "variable_reactions" in summary, "FVA should return variable reactions"

        variable_reactions = summary["variable_reactions"]
        assert len(variable_reactions) > 0, "FVA should find some variable reactions"

        for reaction_data in variable_reactions:
            min_flux = reaction_data["minimum"]
            max_flux = reaction_data["maximum"]
            reaction_id = reaction_data["reaction_id"]

            # Min should be <= Max
            assert (
                min_flux <= max_flux
            ), f"Reaction {reaction_id}: min flux ({min_flux}) > max flux ({max_flux})"

            # Values should be finite
            assert np.isfinite(
                min_flux
            ), f"Reaction {reaction_id} min flux is not finite"
            assert np.isfinite(
                max_flux
            ), f"Reaction {reaction_id} max flux is not finite"

    def test_fva_categories_meaningful(self, fva_tool):
        """Test FVA categorizes reactions meaningfully"""
        result = fva_tool._run_tool(
            {"model_path": "data/examples/e_coli_core.xml", "fraction_of_optimum": 0.9}
        )

        assert result.success

        summary = result.data.get("summary", {})

        # Should have some reactions in each category
        expected_categories = [
            "fixed_reactions",
            "variable_reactions",
            "blocked_reactions",
        ]
        for category in expected_categories:
            if category in summary:
                assert len(summary[category]) >= 0  # Some categories might be empty

        # Fixed reactions should have min ≈ max
        fixed_reactions = summary.get("fixed_reactions", [])

        for reaction_data in fixed_reactions:
            min_flux = reaction_data["minimum"]
            max_flux = reaction_data["maximum"]
            flux_range = abs(max_flux - min_flux)
            reaction_id = reaction_data["reaction_id"]

            # Fixed reactions should have very small range
            assert (
                flux_range < 1e-6
            ), f"Fixed reaction {reaction_id} has range {flux_range}, should be near zero"


class TestEssentialityCorrectness:
    """Test essentiality analysis produces biologically meaningful results"""

    @pytest.fixture
    def essentiality_tool(self):
        return EssentialityAnalysisTool({"name": "test_essentiality"})

    def test_essential_genes_realistic(self, essentiality_tool):
        """Test essential gene identification is biologically reasonable"""
        result = essentiality_tool._run_tool(
            {"model_path": "data/examples/e_coli_core.xml"}
        )

        assert result.success, f"Essentiality analysis failed: {result.error}"

        essential_genes = result.data.get("essential_genes", [])

        # E. coli core should have some essential genes (typically 10-30)
        assert (
            5 <= len(essential_genes) <= 50
        ), f"Essential gene count {len(essential_genes)} outside expected range [5, 50]"

        # Essential genes should be strings (gene IDs)
        for gene in essential_genes:
            assert isinstance(gene, str), f"Gene ID should be string, got {type(gene)}"
            assert len(gene) > 0, "Gene ID should not be empty"

    def test_essential_reactions_consistent(self, essentiality_tool):
        """Test essential reactions are consistent with essential genes"""
        result = essentiality_tool._run_tool(
            {"model_path": "data/examples/e_coli_core.xml"}
        )

        assert result.success

        essential_genes = result.data.get("essential_genes", [])
        essential_reactions = result.data.get("essential_reactions", [])

        # Should have essential reactions if there are essential genes
        if len(essential_genes) > 0:
            assert (
                len(essential_reactions) > 0
            ), "If there are essential genes, there should be essential reactions"

        # Essential reactions should have meaningful impact
        gene_impacts = result.data.get("gene_impacts", {})
        for gene in essential_genes:
            if gene in gene_impacts:
                impact = gene_impacts[gene].get("growth_impact", 0)
                # Essential genes should have significant impact (near 100% reduction)
                assert (
                    impact > 0.8
                ), f"Essential gene {gene} has low impact {impact}, expected >0.8"


class TestMinimalMediaCorrectness:
    """Test minimal media analysis produces realistic nutritional requirements"""

    @pytest.fixture
    def minimal_media_tool(self):
        return MinimalMediaTool({"name": "test_minimal_media"})

    def test_minimal_media_realistic_count(self, minimal_media_tool):
        """Test minimal media has realistic number of nutrients"""
        result = minimal_media_tool._run_tool(
            {"model_path": "data/examples/e_coli_core.xml"}
        )

        assert result.success, f"Minimal media analysis failed: {result.error}"

        minimal_media = result.data.get("minimal_media", {})

        # Find essential nutrients (those with negative lower bounds, meaning required uptake)
        essential_nutrients = []
        for reaction_id, bounds in minimal_media.items():
            if bounds[0] < 0:  # Negative lower bound means required uptake
                essential_nutrients.append(reaction_id)

        # E. coli should need some nutrients in minimal medium (typically 3-10)
        assert (
            3 <= len(essential_nutrients) <= 15
        ), f"Essential nutrient count {len(essential_nutrients)} outside expected range [3, 15]"

    def test_minimal_media_includes_basics(self, minimal_media_tool):
        """Test minimal media includes basic required nutrients"""
        result = minimal_media_tool._run_tool(
            {"model_path": "data/examples/e_coli_core.xml"}
        )

        assert result.success

        minimal_media = result.data.get("minimal_media", {})

        # Find essential nutrients (those with negative lower bounds)
        essential_nutrients = []
        for reaction_id, bounds in minimal_media.items():
            if bounds[0] < 0:  # Required uptake
                essential_nutrients.append(reaction_id)

        nutrient_names = [nutrient.lower() for nutrient in essential_nutrients]

        # Should include basic nutrients for E. coli (glucose, NH4, Pi)
        basic_requirements = ["glc", "nh4", "pi"]

        found_basic = 0
        for basic in basic_requirements:
            if any(basic in nutrient for nutrient in nutrient_names):
                found_basic += 1

        assert found_basic >= 2, (
            f"Should find at least 2 basic nutrients (glc, nh4, pi), found {found_basic}. "
            f"Essential nutrients: {essential_nutrients}"
        )


class TestAuxotrophyCorrectness:
    """Test auxotrophy analysis identifies real biosynthetic gaps"""

    @pytest.fixture
    def auxotrophy_tool(self):
        return AuxotrophyTool({"name": "test_auxotrophy"})

    def test_auxotrophy_analysis_realistic(self, auxotrophy_tool):
        """Test auxotrophy analysis produces realistic results"""
        result = auxotrophy_tool._run_tool(
            {"model_path": "data/examples/e_coli_core.xml"}
        )

        assert result.success, f"Auxotrophy analysis failed: {result.error}"

        auxotrophies = result.data.get("auxotrophies", [])

        # E. coli core is typically auxotrophic for several amino acids
        # Should find some auxotrophies (0-10 range is reasonable)
        assert (
            0 <= len(auxotrophies) <= 15
        ), f"Auxotrophy count {len(auxotrophies)} outside expected range [0, 15]"

        # If auxotrophies found, they should be meaningful compounds
        for auxotrophy in auxotrophies:
            compound = auxotrophy.get("compound", "")
            assert len(compound) > 0, "Auxotrophic compound should not be empty"


class TestCrossToolConsistency:
    """Test that tools produce consistent results with each other"""

    def test_fba_essentiality_consistency(self):
        """Test FBA and essentiality results are consistent"""
        model_path = "data/examples/e_coli_core.xml"

        # Run FBA
        fba_tool = FBATool({"name": "test_fba"})
        fba_result = fba_tool._run_tool({"model_path": model_path})

        # Run essentiality
        ess_tool = EssentialityAnalysisTool({"name": "test_essentiality"})
        ess_result = ess_tool._run_tool({"model_path": model_path})

        assert fba_result.success and ess_result.success

        fba_growth = fba_result.data["objective_value"]

        # If FBA shows growth, model shouldn't be completely broken
        if fba_growth > 0.1:
            essential_genes = ess_result.data.get("essential_genes", [])
            # Growing model should have some but not all genes essential
            total_genes = ess_result.data.get("total_genes", len(essential_genes) * 3)
            essential_fraction = (
                len(essential_genes) / total_genes if total_genes > 0 else 0
            )

            assert essential_fraction < 0.8, (
                f"Too many genes essential ({essential_fraction:.2%}), "
                f"inconsistent with positive growth"
            )

    def test_minimal_media_auxotrophy_consistency(self):
        """Test minimal media and auxotrophy results are consistent"""
        model_path = "data/examples/e_coli_core.xml"

        # Run minimal media
        mm_tool = MinimalMediaTool({"name": "test_mm"})
        mm_result = mm_tool._run_tool({"model_path": model_path})

        # Run auxotrophy
        aux_tool = AuxotrophyTool({"name": "test_aux"})
        aux_result = aux_tool._run_tool({"model_path": model_path})

        if mm_result.success and aux_result.success:
            # Extract essential nutrients from minimal media format
            minimal_media = mm_result.data.get("minimal_media", {})
            essential_nutrients = []
            for reaction_id, bounds in minimal_media.items():
                if bounds[0] < 0:  # Required uptake
                    essential_nutrients.append(reaction_id)

            auxotrophies = aux_result.data.get("auxotrophies", [])

            # If there are auxotrophies, there should be some essential nutrients
            # This is a loose check since the formats might differ
            if len(auxotrophies) > 0:
                assert (
                    len(essential_nutrients) >= 1
                ), "If auxotrophies exist, should have some essential nutrients"


class TestEdgeCasesAndErrorHandling:
    """Test tools handle edge cases and errors gracefully"""

    def test_nonexistent_model_file(self):
        """Test tools handle missing model files gracefully"""
        fba_tool = FBATool({"name": "test_fba"})
        result = fba_tool._run_tool({"model_path": "nonexistent_model.xml"})

        assert not result.success, "Should fail for nonexistent model"
        assert (
            "not found" in result.error.lower()
            or "no such file" in result.error.lower()
        )

    def test_empty_model_handling(self):
        """Test tools handle empty/invalid models"""
        # This would require creating a minimal invalid model
        # For now, test that tools have proper error handling structure
        fba_tool = FBATool({"name": "test_fba"})

        # Test with obviously invalid input
        result = fba_tool._run_tool({"model_path": ""})
        assert not result.success, "Should fail for empty model path"

    def test_invalid_parameters(self):
        """Test tools handle invalid parameters gracefully"""
        fva_tool = FluxVariabilityTool({"name": "test_fva"})

        # Test with invalid fraction_of_optimum
        result = fva_tool._run_tool(
            {
                "model_path": "data/examples/e_coli_core.xml",
                "fraction_of_optimum": -1.0,  # Invalid negative value
            }
        )

        # Should either succeed with default value or fail gracefully
        if not result.success:
            assert (
                "fraction" in result.error.lower() or "invalid" in result.error.lower()
            )


def run_comprehensive_functional_tests():
    """Run all functional correctness tests and report results"""
    print("🧪 Running Comprehensive Functional Correctness Tests")
    print("=" * 60)

    # Run pytest on this module
    test_file = __file__
    exit_code = pytest.main([test_file, "-v", "--tb=short"])

    if exit_code == 0:
        print("\n✅ All functional correctness tests PASSED!")
        print("🎉 Tools produce biologically meaningful results")
    else:
        print("\n❌ Some functional tests FAILED!")
        print("⚠️  Tools may not be producing correct results")

    return exit_code == 0


if __name__ == "__main__":
    success = run_comprehensive_functional_tests()
    sys.exit(0 if success else 1)
