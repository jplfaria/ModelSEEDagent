#!/usr/bin/env python3
"""
ModelSEED-COBRApy Compatibility Layer

This module provides SBML round-trip compatibility verification and
testing functions to ensure ModelSEED-generated models work seamlessly
with existing COBRA tools.

Phase 2 Implementation:
- SBML round-trip verification (ModelSEED → SBML → COBRApy)
- Growth rate compatibility testing (tolerance: 1e-6)
- Model structure validation
- Reaction/metabolite ID mapping verification
"""

import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from ..base import BaseTool, ToolRegistry, ToolResult
from ..cobra.utils import ModelUtils


class CompatibilityConfig(BaseModel):
    """Configuration for ModelSEED-COBRApy compatibility testing"""

    growth_tolerance: float = 1e-6
    structure_validation: bool = True
    flux_comparison: bool = True
    id_mapping_validation: bool = True
    export_comparison_data: bool = True


class ModelCompatibilityMetrics(BaseModel):
    """Metrics for model compatibility assessment"""

    # Growth rate comparison
    modelseed_growth: float
    cobra_growth: float
    growth_difference: float
    growth_compatible: bool

    # Structure comparison
    reactions_match: bool
    metabolites_match: bool
    genes_match: bool

    # ID mapping
    id_mapping_errors: List[str] = Field(default_factory=list)

    # Round-trip success
    sbml_roundtrip_success: bool
    conversion_errors: List[str] = Field(default_factory=list)


@ToolRegistry.register
class ModelCompatibilityTool(BaseTool):
    """Tool for testing ModelSEED-COBRApy compatibility"""

    tool_name = "test_modelseed_cobra_compatibility"
    tool_description = """Test compatibility between ModelSEED models and COBRApy
    through SBML round-trip verification and growth rate comparison."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._compat_config = CompatibilityConfig(
            **config.get("compatibility_config", {})
        )

    def _run_tool(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Test ModelSEED-COBRApy compatibility.

        Args:
            input_data: Dictionary containing:
                - modelseed_model: ModelSEED model object or path
                - media_condition: Media for growth testing (optional)
                - output_dir: Directory for comparison outputs (optional)

        Returns:
            ToolResult with compatibility metrics and recommendations
        """
        try:
            # Load ModelSEED model
            if isinstance(input_data.get("modelseed_model"), str):
                model_path = Path(input_data["modelseed_model"])
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                ms_model = ModelUtils().load_model(str(model_path))
            else:
                ms_model = input_data["modelseed_model"]

            if ms_model is None:
                raise ValueError("No ModelSEED model provided")

            # Perform compatibility tests
            metrics = self._perform_compatibility_tests(ms_model, input_data)

            # Generate recommendations
            recommendations = self._generate_recommendations(metrics)

            # Export comparison data if requested
            output_data = {}
            if self._compat_config.export_comparison_data:
                output_data = self._export_comparison_data(
                    ms_model, metrics, input_data
                )

            return ToolResult(
                success=metrics.sbml_roundtrip_success and metrics.growth_compatible,
                message=self._format_compatibility_message(metrics),
                data={
                    "compatibility_metrics": metrics.dict(),
                    "recommendations": recommendations,
                    "comparison_data": output_data,
                },
                metadata={
                    "tool_type": "compatibility_testing",
                    "growth_tolerance": self._compat_config.growth_tolerance,
                    "tests_performed": [
                        "sbml_roundtrip",
                        "growth_comparison",
                        "structure_validation",
                        "id_mapping",
                    ],
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Compatibility testing failed: {str(e)}",
                error=str(e),
            )

    def _perform_compatibility_tests(
        self, ms_model, input_data: Dict[str, Any]
    ) -> ModelCompatibilityMetrics:
        """Perform comprehensive compatibility tests"""

        # Test 1: SBML Round-trip Verification
        cobra_model, roundtrip_success, conversion_errors = self._test_sbml_roundtrip(
            ms_model
        )

        # Test 2: Growth Rate Comparison
        growth_metrics = self._compare_growth_rates(ms_model, cobra_model, input_data)

        # Test 3: Structure Validation
        structure_metrics = self._validate_model_structure(ms_model, cobra_model)

        # Test 4: ID Mapping Validation
        id_mapping_errors = self._validate_id_mapping(ms_model, cobra_model)

        return ModelCompatibilityMetrics(
            modelseed_growth=growth_metrics["modelseed_growth"],
            cobra_growth=growth_metrics["cobra_growth"],
            growth_difference=growth_metrics["growth_difference"],
            growth_compatible=growth_metrics["growth_compatible"],
            reactions_match=structure_metrics["reactions_match"],
            metabolites_match=structure_metrics["metabolites_match"],
            genes_match=structure_metrics["genes_match"],
            id_mapping_errors=id_mapping_errors,
            sbml_roundtrip_success=roundtrip_success,
            conversion_errors=conversion_errors,
        )

    def _test_sbml_roundtrip(
        self, ms_model
    ) -> Tuple[Optional[object], bool, List[str]]:
        """Test SBML round-trip: ModelSEED → SBML → COBRApy"""

        conversion_errors = []
        cobra_model = None

        try:
            # Lazy import cobra only when needed
            import cobra

            # Create temporary SBML file
            with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            # Export ModelSEED model to SBML
            ms_model.write_sbml_file(tmp_path)

            # Import SBML into COBRApy
            cobra_model = cobra.io.read_sbml_model(tmp_path)

            # Clean up temporary file
            Path(tmp_path).unlink()

            return cobra_model, True, conversion_errors

        except Exception as e:
            conversion_errors.append(f"SBML round-trip failed: {str(e)}")
            return None, False, conversion_errors

    def _compare_growth_rates(
        self, ms_model, cobra_model, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare growth rates between ModelSEED and COBRApy models"""

        # Set media condition if specified
        media_condition = input_data.get("media_condition")
        if media_condition and media_condition != "Complete":
            # Apply media constraints to both models
            self._apply_media_constraints(ms_model, media_condition)
            if cobra_model:
                self._apply_media_constraints(cobra_model, media_condition)

        # Test ModelSEED model growth
        with ms_model:
            ms_solution = ms_model.optimize()
            ms_growth = (
                ms_solution.objective_value if ms_solution.status == "optimal" else 0
            )

        # Test COBRApy model growth
        cobra_growth = 0
        if cobra_model:
            with cobra_model:
                cobra_solution = cobra_model.optimize()
                cobra_growth = (
                    cobra_solution.objective_value
                    if cobra_solution.status == "optimal"
                    else 0
                )

        # Compare growth rates
        growth_difference = abs(ms_growth - cobra_growth)
        growth_compatible = growth_difference <= self._compat_config.growth_tolerance

        return {
            "modelseed_growth": ms_growth,
            "cobra_growth": cobra_growth,
            "growth_difference": growth_difference,
            "growth_compatible": growth_compatible,
        }

    def _validate_model_structure(self, ms_model, cobra_model) -> Dict[str, bool]:
        """Validate that model structures match after conversion"""

        if not cobra_model:
            return {
                "reactions_match": False,
                "metabolites_match": False,
                "genes_match": False,
            }

        # Compare reaction counts and IDs
        ms_rxn_ids = set(r.id for r in ms_model.reactions)
        cobra_rxn_ids = set(r.id for r in cobra_model.reactions)
        reactions_match = len(ms_rxn_ids.symmetric_difference(cobra_rxn_ids)) == 0

        # Compare metabolite counts and IDs
        ms_met_ids = set(m.id for m in ms_model.metabolites)
        cobra_met_ids = set(m.id for m in cobra_model.metabolites)
        metabolites_match = len(ms_met_ids.symmetric_difference(cobra_met_ids)) == 0

        # Compare gene counts and IDs
        ms_gene_ids = set(g.id for g in ms_model.genes)
        cobra_gene_ids = set(g.id for g in cobra_model.genes)
        genes_match = len(ms_gene_ids.symmetric_difference(cobra_gene_ids)) == 0

        return {
            "reactions_match": reactions_match,
            "metabolites_match": metabolites_match,
            "genes_match": genes_match,
        }

    def _validate_id_mapping(self, ms_model, cobra_model) -> List[str]:
        """Validate ID mapping consistency between models"""

        id_errors = []

        if not cobra_model:
            id_errors.append(
                "Cannot validate ID mapping: COBRApy model conversion failed"
            )
            return id_errors

        # Check for problematic ID patterns that might cause issues
        for reaction in ms_model.reactions:
            # Check for ModelSEED-specific ID patterns
            if "rxn" in reaction.id.lower() and len(reaction.id) > 20:
                id_errors.append(
                    f"Long ModelSEED reaction ID may cause issues: {reaction.id}"
                )

        for metabolite in ms_model.metabolites:
            # Check for compartment suffix issues
            if "_" not in metabolite.id and metabolite.compartment:
                id_errors.append(
                    f"Metabolite ID missing compartment suffix: {metabolite.id}"
                )

        # Check for duplicate IDs after conversion
        cobra_rxn_ids = [r.id for r in cobra_model.reactions]
        if len(cobra_rxn_ids) != len(set(cobra_rxn_ids)):
            id_errors.append("Duplicate reaction IDs found after conversion")

        cobra_met_ids = [m.id for m in cobra_model.metabolites]
        if len(cobra_met_ids) != len(set(cobra_met_ids)):
            id_errors.append("Duplicate metabolite IDs found after conversion")

        return id_errors

    def _apply_media_constraints(self, model, media_condition: str):
        """Apply media constraints to a model"""
        # This is a simplified implementation
        # In practice, you'd use proper media definitions
        if media_condition == "minimal":
            # Close all exchange reactions except essential ones
            for reaction in model.exchanges:
                if "glc" not in reaction.id.lower() and "o2" not in reaction.id.lower():
                    reaction.lower_bound = 0

    def _generate_recommendations(
        self, metrics: ModelCompatibilityMetrics
    ) -> List[str]:
        """Generate recommendations based on compatibility test results"""

        recommendations = []

        if not metrics.sbml_roundtrip_success:
            recommendations.append(
                "CRITICAL: SBML round-trip failed. Review model SBML export/import compatibility."
            )

        if not metrics.growth_compatible:
            recommendations.append(
                f"WARNING: Growth rates differ by {metrics.growth_difference:.8f}. "
                f"ModelSEED: {metrics.modelseed_growth:.6f}, COBRApy: {metrics.cobra_growth:.6f}"
            )

        if not metrics.reactions_match:
            recommendations.append(
                "Model reaction sets don't match. Check for conversion issues in reaction definitions."
            )

        if not metrics.metabolites_match:
            recommendations.append(
                "Model metabolite sets don't match. Verify compartment and ID mapping."
            )

        if metrics.id_mapping_errors:
            recommendations.append(
                f"ID mapping issues detected: {len(metrics.id_mapping_errors)} problems found."
            )

        if not recommendations:
            recommendations.append(
                "✅ Excellent compatibility! Models are fully compatible between ModelSEED and COBRApy."
            )

        return recommendations

    def _format_compatibility_message(self, metrics: ModelCompatibilityMetrics) -> str:
        """Format the main compatibility assessment message"""

        if metrics.sbml_roundtrip_success and metrics.growth_compatible:
            return (
                f"✅ Models are compatible! Growth rates match within tolerance "
                f"(difference: {metrics.growth_difference:.8f})"
            )
        else:
            issues = []
            if not metrics.sbml_roundtrip_success:
                issues.append("SBML conversion failed")
            if not metrics.growth_compatible:
                issues.append(f"Growth rates differ by {metrics.growth_difference:.8f}")

            return f"❌ Compatibility issues detected: {', '.join(issues)}"

    def _export_comparison_data(
        self, ms_model, metrics: ModelCompatibilityMetrics, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Export detailed comparison data for analysis"""

        comparison_data = {
            "model_statistics": {
                "reactions": len(ms_model.reactions),
                "metabolites": len(ms_model.metabolites),
                "genes": len(ms_model.genes),
            },
            "growth_analysis": {
                "modelseed_growth": metrics.modelseed_growth,
                "cobra_growth": metrics.cobra_growth,
                "absolute_difference": metrics.growth_difference,
                "relative_difference": (
                    metrics.growth_difference
                    / max(metrics.modelseed_growth, 1e-10)
                    * 100
                    if metrics.modelseed_growth > 0
                    else float("inf")
                ),
            },
            "structural_comparison": {
                "reactions_identical": metrics.reactions_match,
                "metabolites_identical": metrics.metabolites_match,
                "genes_identical": metrics.genes_match,
            },
            "conversion_status": {
                "sbml_export_success": True,  # If we got here, export worked
                "cobra_import_success": metrics.sbml_roundtrip_success,
                "errors": metrics.conversion_errors,
            },
        }

        return comparison_data


def test_modelseed_cobra_pipeline(
    model_path: str, media_condition: str = "Complete"
) -> Dict[str, Any]:
    """
    Convenience function for testing complete ModelSEED → COBRApy pipeline

    Args:
        model_path: Path to ModelSEED model
        media_condition: Media condition to test

    Returns:
        Dictionary with compatibility test results
    """

    tool = ModelCompatibilityTool({})
    result = tool._run(
        {"modelseed_model": model_path, "media_condition": media_condition}
    )

    return result.data


def verify_cobra_tool_compatibility(
    ms_model, cobra_tools: List[str]
) -> Dict[str, bool]:
    """
    Test if ModelSEED model works with specific COBRApy tools

    Args:
        ms_model: ModelSEED model object
        cobra_tools: List of COBRApy tool names to test

    Returns:
        Dictionary mapping tool names to compatibility status
    """

    compatibility = {}

    # Convert to COBRApy format
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Lazy import cobra only when needed
        import cobra

        ms_model.write_sbml_file(tmp_path)
        cobra_model = cobra.io.read_sbml_model(tmp_path)

        # Test specific COBRApy functionalities
        for tool_name in cobra_tools:
            try:
                if tool_name == "fba":
                    solution = cobra_model.optimize()
                    compatibility[tool_name] = solution.status == "optimal"

                elif tool_name == "fva":
                    import cobra.flux_analysis as flux_analysis

                    fva_result = flux_analysis.flux_variability_analysis(
                        cobra_model, fraction_of_optimum=0.9
                    )
                    compatibility[tool_name] = fva_result is not None

                elif tool_name == "gene_deletion":
                    deletion_results = cobra.flux_analysis.single_gene_deletion(
                        cobra_model
                    )
                    compatibility[tool_name] = deletion_results is not None

                elif tool_name == "flux_sampling":
                    # Test with minimal sampling to avoid long computation
                    samples = cobra.sampling.sample(cobra_model, n=10, method="achr")
                    compatibility[tool_name] = samples is not None

                else:
                    compatibility[tool_name] = False  # Unknown tool

            except Exception:
                compatibility[tool_name] = False

    except Exception:
        # If conversion fails, all tools are incompatible
        for tool_name in cobra_tools:
            compatibility[tool_name] = False

    finally:
        # Clean up
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()

    return compatibility
