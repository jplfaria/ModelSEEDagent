from typing import Any, Dict

import cobra
from pydantic import BaseModel, Field, PrivateAttr

from ..base import BaseTool, ToolRegistry, ToolResult
from .fba import SimulationResultsStore  # Import the results store
from .simulation_wrapper import run_simulation
from .utils import ModelUtils


class ReactionExpressionConfig(BaseModel):
    """Configuration for reaction expression analysis tool."""

    flux_threshold: float = 1e-6


@ToolRegistry.register
class ReactionExpressionTool(BaseTool):
    """Tool to analyze reaction expression levels using parsimonious FBA (pFBA)."""

    tool_name = "analyze_reaction_expression"
    tool_description = "Analyze reaction expression levels using pFBA to obtain realistic flux distributions."

    _config: ReactionExpressionConfig

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        config_dict = config.get("reaction_expression_config", {})
        self._config = ReactionExpressionConfig(**config_dict)
        self._utils = ModelUtils()

    def _run(self, input_data: Any, media: dict = None) -> ToolResult:
        try:
            if isinstance(input_data, dict):
                model_path = input_data.get("model_path")
                output_dir = input_data.get("output_dir", None)
            else:
                model_path = input_data
                output_dir = None

            model = self._utils.load_model(model_path)
            if media:
                # Apply media conditions if such a method exists in ModelUtils.
                ModelUtils.apply_media_conditions(model, media)
            solution = run_simulation(model, method="pfba")

            # Optionally export simulation results
            result_file = None
            if output_dir:
                store = SimulationResultsStore()
                result_id = store.save_results(
                    self.tool_name,
                    model.id,
                    solution,
                    additional_metadata={"simulation_method": "pfba", "media": media},
                )
                json_path, csv_path = store.export_results(result_id, output_dir)
                result_file = {"json": json_path, "csv": csv_path}

            if solution.status != "optimal":
                return ToolResult(
                    success=False,
                    message="pFBA optimization failed",
                    error=f"Solution status: {solution.status}",
                )
            active_reactions = {
                rxn.id: float(solution.fluxes[rxn.id])
                for rxn in model.reactions
                if abs(solution.fluxes[rxn.id]) > self._config.flux_threshold
            }
            return ToolResult(
                success=True,
                message="Reaction expression analysis completed successfully",
                data={
                    "active_reactions": active_reactions,
                    "objective_value": solution.objective_value,
                    "result_file": result_file,
                },
                metadata={"num_active_reactions": len(active_reactions)},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message="Error analyzing reaction expression",
                error=str(e),
            )
