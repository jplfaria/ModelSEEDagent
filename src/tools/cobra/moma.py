from typing import Any, Dict, List, Optional

import cobra
from cobra.flux_analysis import moma
from pydantic import BaseModel, Field, PrivateAttr

from ..base import BaseTool, ToolRegistry, ToolResult
from .precision_config import PrecisionConfig, is_significant_growth
from .utils import should_disable_auditing
from .utils_optimized import OptimizedModelUtils


class MOMAConfig(BaseModel):
    """Configuration for MOMA (Minimization of Metabolic Adjustment) Analysis"""

    model_config = {"protected_namespaces": ()}

    # Analysis parameters
    knockout_genes: Optional[List[str]] = Field(
        None, description="List of gene IDs to knock out"
    )
    knockout_reactions: Optional[List[str]] = Field(
        None, description="List of reaction IDs to knock out"
    )
    linear: bool = Field(True, description="Use linear MOMA (faster) vs quadratic MOMA")
    solver: str = Field("glpk", description="Solver to use for optimization")

    # Comparison settings
    compare_to_fba: bool = Field(
        True, description="Compare MOMA results to standard FBA"
    )

    # Numerical precision settings
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)


@ToolRegistry.register
class MOMATool(BaseTool):
    """Tool for MOMA (Minimization of Metabolic Adjustment) analysis"""

    tool_name = "run_moma_analysis"
    tool_description = """Run MOMA analysis to predict metabolic adjustments after genetic perturbations.
    MOMA finds the flux distribution that minimizes metabolic changes compared to wild-type."""

    _moma_config: MOMAConfig = PrivateAttr()
    _utils: OptimizedModelUtils = PrivateAttr()

    def __init__(self, config: Dict[str, Any]):
        # Disable auditing in subprocess to prevent connection pool issues
        if should_disable_auditing():
            config = config.copy()
            config["audit_enabled"] = False
        super().__init__(config)

        # Initialize MOMA configuration
        moma_config_dict = config.get("moma_config", {})
        self._moma_config = MOMAConfig(**moma_config_dict)

        # Initialize optimized utilities
        self._utils = OptimizedModelUtils()

    @property
    def moma_config(self) -> MOMAConfig:
        return self._moma_config

    def _run(self, inputs: Dict[str, Any]) -> ToolResult:
        """Execute MOMA analysis"""
        try:
            # Get inputs
            model_path = inputs.get("model_path")
            knockout_genes = inputs.get(
                "knockout_genes", self.moma_config.knockout_genes
            )
            knockout_reactions = inputs.get(
                "knockout_reactions", self.moma_config.knockout_reactions
            )
            linear = inputs.get("linear", self.moma_config.linear)
            solver = inputs.get("solver", self.moma_config.solver)
            compare_to_fba = inputs.get(
                "compare_to_fba", self.moma_config.compare_to_fba
            )

            if not model_path:
                return ToolResult(
                    success=False,
                    message="Model path is required",
                    error="No model_path provided in inputs",
                )

            if not knockout_genes and not knockout_reactions:
                return ToolResult(
                    success=False,
                    message="At least one gene or reaction knockout must be specified",
                    error="No knockouts provided",
                )

            # Load model
            model = self._utils.load_model(model_path)
            model.solver = solver

            # Get wild-type reference solution
            wt_solution = model.optimize()
            if wt_solution.status != "optimal":
                return ToolResult(
                    success=False,
                    message="Wild-type model optimization failed",
                    error=f"Solution status: {wt_solution.status}",
                )

            wt_growth = wt_solution.objective_value
            wt_fluxes = wt_solution.fluxes.copy()

            # Create knockout model
            ko_model = model.copy()

            knockouts_applied = []

            # Apply gene knockouts
            if knockout_genes:
                for gene_id in knockout_genes:
                    try:
                        gene = ko_model.genes.get_by_id(gene_id)
                        gene.knock_out()
                        knockouts_applied.append(f"gene:{gene_id}")
                    except KeyError:
                        return ToolResult(
                            success=False,
                            message=f"Gene {gene_id} not found in model",
                            error=f"Invalid gene ID: {gene_id}",
                        )

            # Apply reaction knockouts
            if knockout_reactions:
                for reaction_id in knockout_reactions:
                    try:
                        reaction = ko_model.reactions.get_by_id(reaction_id)
                        reaction.knock_out()
                        knockouts_applied.append(f"reaction:{reaction_id}")
                    except KeyError:
                        return ToolResult(
                            success=False,
                            message=f"Reaction {reaction_id} not found in model",
                            error=f"Invalid reaction ID: {reaction_id}",
                        )

            # Run MOMA analysis
            moma_solution = moma(ko_model, solution=wt_solution, linear=linear)

            if moma_solution.status != "optimal":
                return ToolResult(
                    success=False,
                    message="MOMA optimization failed",
                    error=f"Solution status: {moma_solution.status}",
                )

            moma_growth = moma_solution.objective_value
            moma_fluxes = moma_solution.fluxes.copy()

            # Calculate metabolic adjustment metrics
            flux_differences = (moma_fluxes - wt_fluxes).abs()
            total_adjustment = flux_differences.sum()
            max_adjustment = flux_differences.max()
            num_changed_reactions = (
                flux_differences > self.moma_config.precision.flux_tolerance
            ).sum()

            # Calculate growth metrics
            growth_change = moma_growth - wt_growth
            growth_fraction = moma_growth / wt_growth if wt_growth > 0 else 0
            is_viable = is_significant_growth(moma_growth, self.moma_config.precision)

            # Prepare results
            results = {
                "knockouts_applied": knockouts_applied,
                "wild_type_growth": wt_growth,
                "moma_growth": moma_growth,
                "growth_change": growth_change,
                "growth_fraction": growth_fraction,
                "is_viable": is_viable,
                "moma_method": "linear" if linear else "quadratic",
                "metabolic_adjustment": {
                    "total_flux_adjustment": total_adjustment,
                    "max_flux_adjustment": max_adjustment,
                    "reactions_changed": int(num_changed_reactions),
                    "total_reactions": len(model.reactions),
                },
                "solution_status": moma_solution.status,
            }

            # Compare to FBA if requested
            if compare_to_fba:
                fba_solution = ko_model.optimize()
                if fba_solution.status == "optimal":
                    fba_growth = fba_solution.objective_value
                    results["fba_comparison"] = {
                        "fba_growth": fba_growth,
                        "growth_difference_moma_vs_fba": moma_growth - fba_growth,
                        "moma_more_conservative": moma_growth < fba_growth,
                    }

            # Identify most changed reactions
            flux_changes = flux_differences.sort_values(ascending=False)
            top_changed_reactions = []
            for reaction_id in flux_changes.head(10).index:
                if (
                    flux_changes[reaction_id]
                    > self.moma_config.precision.flux_tolerance
                ):
                    top_changed_reactions.append(
                        {
                            "reaction_id": reaction_id,
                            "reaction_name": model.reactions.get_by_id(
                                reaction_id
                            ).name,
                            "wild_type_flux": wt_fluxes[reaction_id],
                            "moma_flux": moma_fluxes[reaction_id],
                            "flux_change": flux_changes[reaction_id],
                        }
                    )

            results["top_changed_reactions"] = top_changed_reactions

            # Add flux distributions if requested (for small models)
            if len(model.reactions) <= 100:
                results["flux_distributions"] = {
                    "wild_type_fluxes": wt_fluxes.to_dict(),
                    "moma_fluxes": moma_fluxes.to_dict(),
                }

            return ToolResult(
                success=True,
                message=f"MOMA analysis completed successfully. "
                f"Applied {len(knockouts_applied)} knockouts. "
                f"Growth: {wt_growth:.4f} → {moma_growth:.4f} "
                f"({growth_fraction:.1%} of wild-type)",
                data=results,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message="MOMA analysis failed",
                error=str(e),
            )

    def get_example_usage(self) -> Dict[str, Any]:
        """Get example usage for the MOMA tool"""
        return {
            "model_path": "path/to/your/model.xml",
            "knockout_genes": ["gene1", "gene2"],
            "linear": True,
            "compare_to_fba": True,
        }

    def get_tool_description(self) -> str:
        """Get a detailed description of the MOMA tool"""
        return """
        MOMA (Minimization of Metabolic Adjustment) Analysis Tool

        This tool performs MOMA analysis to predict how metabolic networks respond to genetic
        perturbations by finding flux distributions that minimize metabolic adjustments.

        Key Features:
        - Gene and reaction knockout simulations
        - Linear (faster) and quadratic MOMA variants
        - Comparison with standard FBA predictions
        - Detailed metabolic adjustment metrics
        - Identification of most affected reactions

        MOMA is particularly useful for:
        - Predicting realistic metabolic responses to genetic modifications
        - Understanding metabolic adaptation strategies
        - Comparing different knockout strategies
        - Identifying key reactions affected by perturbations

        Input Parameters:
        - model_path: Path to SBML model file
        - knockout_genes: List of gene IDs to knock out
        - knockout_reactions: List of reaction IDs to knock out
        - linear: Use linear (True) vs quadratic (False) MOMA
        - compare_to_fba: Compare results with standard FBA

        Output:
        - Growth rates (wild-type vs MOMA prediction)
        - Metabolic adjustment metrics
        - Most affected reactions
        - Optional comparison with FBA results
        """
