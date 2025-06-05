from typing import Any, Dict

import cobra
from pydantic import BaseModel, Field, PrivateAttr

from ..base import BaseTool, ToolRegistry, ToolResult
from .fba import SimulationResultsStore  # Import the results store
from .simulation_wrapper import run_simulation
from .utils import ModelUtils


class MinimalMediaConfig(BaseModel):
    """Configuration for minimal media finder tool."""

    growth_threshold: float = 1e-6


@ToolRegistry.register
class MinimalMediaTool(BaseTool):
    """Tool to determine the minimal set of media components required for growth using standard FBA."""

    tool_name = "find_minimal_media"
    tool_description = "Determine the minimal set of media components required for growth by iteratively removing nutrients using FBA."

    _config: MinimalMediaConfig

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        config_dict = config.get("minimal_media_config", {})
        self._config = MinimalMediaConfig(**config_dict)
        self._utils = ModelUtils()

    def _run_tool(self, input_data: Any) -> ToolResult:
        try:
            # Allow input to be a dict with "model_path" and optional "output_dir"
            if isinstance(input_data, dict):
                model_path = input_data.get("model_path")
                output_dir = input_data.get("output_dir", None)
            else:
                model_path = input_data
                output_dir = None

            model = self._utils.load_model(model_path)
            complete_solution = run_simulation(model, method="fba")
            if complete_solution.objective_value < self._config.growth_threshold:
                return ToolResult(
                    success=False,
                    message="Complete media does not support growth.",
                    error="No growth with provided media.",
                )

            # Optionally export the complete simulation results
            result_file = None
            if output_dir:
                store = SimulationResultsStore()
                result_id = store.save_results(
                    self.tool_name,
                    model.id,
                    complete_solution,
                    additional_metadata={"simulation_method": "fba"},
                )
                json_path, csv_path = store.export_results(result_id, output_dir)
                result_file = {"json": json_path, "csv": csv_path}

            minimal_media = {}
            for rxn in model.exchanges:
                original_bounds = rxn.bounds
                rxn.lower_bound = 0
                test_solution = run_simulation(model, method="fba")
                if test_solution.objective_value < self._config.growth_threshold:
                    minimal_media[rxn.id] = original_bounds
                else:
                    minimal_media[rxn.id] = (0, original_bounds[1])
                rxn.bounds = original_bounds
            return ToolResult(
                success=True,
                message="Minimal media determination completed.",
                data={
                    "minimal_media": minimal_media,
                    "complete_solution": complete_solution.objective_value,
                    "result_file": result_file,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False, message="Error determining minimal media.", error=str(e)
            )
