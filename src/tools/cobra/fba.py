import json
import os
from datetime import datetime
from typing import Any, Dict

import cobra
import pandas as pd
from cobra.io import read_sbml_model
from pydantic import BaseModel, Field, PrivateAttr

from ..base import BaseTool, ToolRegistry, ToolResult
from .precision_config import PrecisionConfig, is_significant_flux
from .simulation_wrapper import run_simulation
from .utils import ModelUtils


# Configuration for FBA tool
class FBAConfig(BaseModel):
    """Configuration for FBA tool with enhanced numerical precision"""

    model_config = {"protected_namespaces": ()}
    default_objective: str = "biomass_reaction"
    solver: str = "glpk"
    additional_constraints: Dict[str, float] = Field(default_factory=dict)
    simulation_method: str = "pfba"  # Options: "fba", "pfba", "geometric", "slim", etc.

    # Numerical precision settings
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)


# --------------------------
# Simulation Results Store
# --------------------------
class SimulationResultsStore:
    """
    A simple store to save simulation results and export them to JSON and CSV.
    """

    def __init__(self):
        self.results = {}

    def save_results(
        self,
        tool_name: str,
        model_id: str,
        solution: cobra.Solution,
        additional_metadata=None,
    ) -> str:
        """Store simulation results with metadata and return a unique result ID."""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        result_id = f"{tool_name}_{model_id}_{timestamp}"
        result_data = {
            "objective_value": float(solution.objective_value),
            "fluxes": solution.fluxes.to_dict(),
            "status": solution.status,
            "timestamp": timestamp,
            "model_id": model_id,
            "tool": tool_name,
            "metadata": additional_metadata or {},
        }
        self.results[result_id] = result_data
        return result_id

    def export_results(self, result_id: str, output_dir: str):
        """Export simulation results to JSON and CSV files in the output directory."""
        if result_id not in self.results:
            return None, None
        os.makedirs(output_dir, exist_ok=True)
        result = self.results[result_id]
        json_path = os.path.join(output_dir, f"{result_id}.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        flux_df = pd.DataFrame.from_dict(
            result["fluxes"], orient="index", columns=["flux"]
        )
        csv_path = os.path.join(output_dir, f"{result_id}_fluxes.csv")
        flux_df.to_csv(csv_path)
        return json_path, csv_path


# --------------------------
# Updated FBATool Implementation
# --------------------------
@ToolRegistry.register
class FBATool(BaseTool):
    """Tool for running Flux Balance Analysis with configurable simulation method and result export."""

    tool_name = "run_metabolic_fba"
    tool_description = "Run Flux Balance Analysis (FBA) on a metabolic model using a configurable simulation method."

    _fba_config: FBAConfig = PrivateAttr()
    _utils: ModelUtils = PrivateAttr()

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        fba_config_dict = config.get("fba_config", {})
        if isinstance(fba_config_dict, dict):
            self._fba_config = FBAConfig(**fba_config_dict)
        else:
            self._fba_config = FBAConfig(
                default_objective=getattr(
                    fba_config_dict, "default_objective", "biomass_reaction"
                ),
                solver=getattr(fba_config_dict, "solver", "glpk"),
                precision=PrecisionConfig(),
                additional_constraints=getattr(
                    fba_config_dict, "additional_constraints", {}
                ),
                simulation_method=getattr(fba_config_dict, "simulation_method", "pfba"),
            )
        self._utils = ModelUtils()

    @property
    def fba_config(self) -> FBAConfig:
        """Get the FBA configuration"""
        return self._fba_config

    def validate_input(self, model_path: str) -> None:
        if not isinstance(model_path, str):
            raise ValueError("Model path must be a string")
        if not model_path.endswith((".xml", ".sbml")):
            raise ValueError("Model file must be in SBML format (.xml or .sbml)")

    def _run_tool(self, input_data: Any) -> ToolResult:
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"🔍 FBA TOOL DEBUG: input_data = {repr(input_data)}")
        logger.info(f"🔍 FBA TOOL DEBUG: type(input_data) = {type(input_data)}")

        try:
            # Support both dict and string inputs
            if isinstance(input_data, dict):
                logger.info(f"🔍 FBA TOOL DEBUG: Processing as dict")
                model_path = input_data.get("model_path")
                output_dir = input_data.get("output_dir", None)
                media_name = input_data.get("media", None)
                logger.info(f"🔍 FBA TOOL DEBUG: model_path = {repr(model_path)}, media = {repr(media_name)}")
            else:
                logger.info(f"🔍 FBA TOOL DEBUG: Processing as string/other")
                model_path = input_data
                output_dir = None
                media_name = None
                logger.info(f"🔍 FBA TOOL DEBUG: model_path = {repr(model_path)}")

            self.validate_input(model_path)
            model = self._utils.load_model(model_path)
            
            # Apply media if specified
            if media_name:
                logger.info(f"🔍 FBA TOOL DEBUG: Applying media: {media_name}")
                from .modelseedpy_integration import get_modelseedpy_enhancement
                enhancement = get_modelseedpy_enhancement()
                model = enhancement.apply_media_with_cobrakbase(model, media_name)
                logger.info(f"🔍 FBA TOOL DEBUG: Media {media_name} applied successfully")

            model.solver = self.fba_config.solver
            # Set solver tolerance (much smaller than flux significance threshold)
            model.tolerance = self.fba_config.precision.model_tolerance

            # Set objective if available
            if hasattr(model, self.fba_config.default_objective):
                model.objective = self.fba_config.default_objective

            # Apply additional constraints if specified
            for reaction_id, bound in self.fba_config.additional_constraints.items():
                if reaction_id in model.reactions:
                    model.reactions.get_by_id(reaction_id).bounds = (-bound, bound)

            # Run simulation using the specified method (FBA, pfba, geometric, or slim)
            simulation_method = self.fba_config.simulation_method
            solution = run_simulation(model, method=simulation_method)

            if solution.status != "optimal":
                return ToolResult(
                    success=False,
                    message="FBA simulation failed to produce an optimal solution",
                    error=f"Solution status: {solution.status}",
                )

            # Use precision-aware flux filtering
            flux_threshold = self.fba_config.precision.flux_threshold
            significant_fluxes = {
                rxn.id: float(solution.fluxes[rxn.id])
                for rxn in model.reactions
                if is_significant_flux(solution.fluxes[rxn.id], flux_threshold)
            }

            subsystem_fluxes = {}
            for rxn_id, flux in significant_fluxes.items():
                rxn = model.reactions.get_by_id(rxn_id)
                subsystem = rxn.subsystem or "Unknown"
                if subsystem not in subsystem_fluxes:
                    subsystem_fluxes[subsystem] = []
                subsystem_fluxes[subsystem].append((rxn_id, flux))

            # Export simulation results if an output directory is provided
            result_file = None
            if output_dir:
                print(f"DEBUG-FBA: Attempting to save results to {output_dir}")
                store = SimulationResultsStore()
                result_id = store.save_results(
                    self.tool_name,
                    model.id,
                    solution,
                    additional_metadata={"simulation_method": simulation_method},
                )
                json_path, csv_path = store.export_results(result_id, output_dir)
                result_file = {"json": json_path, "csv": csv_path}

            # Extract the actual growth rate from biomass reaction flux
            growth_rate = solution.objective_value
            biomass_reactions = [
                rxn for rxn in model.reactions if rxn.objective_coefficient != 0
            ]
            if biomass_reactions:
                # Use the actual biomass flux value (growth rate)
                biomass_rxn = biomass_reactions[0]
                growth_rate = float(solution.fluxes[biomass_rxn.id])

            return ToolResult(
                success=True,
                message="FBA simulation completed successfully",
                data={
                    "objective_value": growth_rate,  # Correct growth rate
                    "status": solution.status,
                    "significant_fluxes": significant_fluxes,
                    "subsystem_fluxes": subsystem_fluxes,
                    "result_file": result_file,
                },
                metadata={
                    "model_id": model.id,
                    "objective_reaction": str(model.objective.expression),
                    "num_reactions": len(model.reactions),
                    "num_metabolites": len(model.metabolites),
                    "num_significant_fluxes": len(significant_fluxes),
                },
            )

        except Exception as e:
            return ToolResult(
                success=False, message="Error running FBA simulation", error=str(e)
            )
