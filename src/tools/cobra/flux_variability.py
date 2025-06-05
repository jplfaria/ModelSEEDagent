from typing import Any, Dict, List, Optional

import cobra
from cobra.flux_analysis import flux_variability_analysis
from pydantic import BaseModel, Field, PrivateAttr

from ..base import BaseTool, ToolRegistry, ToolResult
from .utils import ModelUtils


class FluxVariabilityConfig(BaseModel):
    """Configuration for Flux Variability Analysis"""

    model_config = {"protected_namespaces": ()}
    reaction_list: Optional[List[str]] = None
    loopless: bool = False
    fraction_of_optimum: float = 1.0
    solver: str = "glpk"
    processes: Optional[int] = None
    pfba_factor: Optional[float] = None


@ToolRegistry.register
class FluxVariabilityTool(BaseTool):
    """Tool for running Flux Variability Analysis (FVA) on metabolic models"""

    tool_name = "run_flux_variability_analysis"
    tool_description = """Run Flux Variability Analysis (FVA) to determine minimum and maximum
    flux values for each reaction in the model while maintaining a specified fraction of the optimal objective."""

    _fva_config: FluxVariabilityConfig = PrivateAttr()
    _utils: ModelUtils = PrivateAttr()

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        fva_config_dict = config.get("fva_config", {})
        if isinstance(fva_config_dict, dict):
            self._fva_config = FluxVariabilityConfig(**fva_config_dict)
        else:
            self._fva_config = FluxVariabilityConfig(
                reaction_list=getattr(fva_config_dict, "reaction_list", None),
                loopless=getattr(fva_config_dict, "loopless", False),
                fraction_of_optimum=getattr(
                    fva_config_dict, "fraction_of_optimum", 1.0
                ),
                solver=getattr(fva_config_dict, "solver", "glpk"),
                processes=getattr(fva_config_dict, "processes", None),
                pfba_factor=getattr(fva_config_dict, "pfba_factor", None),
            )
        self._utils = ModelUtils()

    @property
    def fva_config(self) -> FluxVariabilityConfig:
        """Get the FVA configuration"""
        return self._fva_config

    def _run(self, input_data: Any) -> ToolResult:
        try:
            # Support both dict and string inputs
            if isinstance(input_data, dict):
                model_path = input_data.get("model_path")
                reaction_list = input_data.get(
                    "reaction_list", self.fva_config.reaction_list
                )
                loopless = input_data.get("loopless", self.fva_config.loopless)
                fraction_of_optimum = input_data.get(
                    "fraction_of_optimum", self.fva_config.fraction_of_optimum
                )
            else:
                model_path = input_data
                reaction_list = self.fva_config.reaction_list
                loopless = self.fva_config.loopless
                fraction_of_optimum = self.fva_config.fraction_of_optimum

            if not isinstance(model_path, str):
                raise ValueError("Model path must be a string")

            # Load model
            model = self._utils.load_model(model_path)
            model.solver = self.fva_config.solver

            # Set up reaction list
            if reaction_list is None:
                reaction_list = [rxn.id for rxn in model.reactions]

            # Run FVA
            fva_result = flux_variability_analysis(
                model=model,
                reaction_list=reaction_list,
                loopless=loopless,
                fraction_of_optimum=fraction_of_optimum,
                processes=self.fva_config.processes,
                pfba_factor=self.fva_config.pfba_factor,
            )

            # Process results
            fva_summary = {
                "fixed_reactions": [],  # reactions with min = max (no variability)
                "variable_reactions": [],  # reactions with min != max
                "blocked_reactions": [],  # reactions with min = max = 0
                "essential_reactions": [],  # reactions where min > 0 or max < 0
            }

            tolerance = 1e-6

            for reaction_id, row in fva_result.iterrows():
                min_flux = row["minimum"]
                max_flux = row["maximum"]

                flux_data = {
                    "reaction_id": reaction_id,
                    "minimum": float(min_flux),
                    "maximum": float(max_flux),
                    "range": float(max_flux - min_flux),
                }

                # Categorize reaction
                if abs(min_flux) < tolerance and abs(max_flux) < tolerance:
                    fva_summary["blocked_reactions"].append(flux_data)
                elif abs(max_flux - min_flux) < tolerance:
                    fva_summary["fixed_reactions"].append(flux_data)
                else:
                    fva_summary["variable_reactions"].append(flux_data)

                # Check if essential (always carries flux in same direction)
                if min_flux > tolerance or max_flux < -tolerance:
                    fva_summary["essential_reactions"].append(flux_data)

            # Sort by flux range (most variable first)
            fva_summary["variable_reactions"].sort(
                key=lambda x: x["range"], reverse=True
            )

            return ToolResult(
                success=True,
                message=f"FVA completed for {len(reaction_list)} reactions",
                data={
                    "fva_results": fva_result.to_dict(),
                    "summary": fva_summary,
                    "statistics": {
                        "total_reactions": len(reaction_list),
                        "blocked_reactions": len(fva_summary["blocked_reactions"]),
                        "fixed_reactions": len(fva_summary["fixed_reactions"]),
                        "variable_reactions": len(fva_summary["variable_reactions"]),
                        "essential_reactions": len(fva_summary["essential_reactions"]),
                    },
                },
                metadata={
                    "model_id": model.id,
                    "fraction_of_optimum": fraction_of_optimum,
                    "loopless": loopless,
                    "solver": self.fva_config.solver,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message="Error running Flux Variability Analysis",
                error=str(e),
            )
