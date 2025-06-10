from typing import Any, Dict

import cobra
from pydantic import BaseModel, Field, PrivateAttr

from ..base import BaseTool, ToolRegistry, ToolResult
from .fba import SimulationResultsStore  # Import the results store
from .precision_config import PrecisionConfig, is_significant_growth
from .simulation_wrapper import run_simulation
from .utils import ModelUtils


class MinimalMediaConfig(BaseModel):
    """Configuration for minimal media finder tool with enhanced numerical precision."""

    # Numerical precision settings
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)


@ToolRegistry.register
class MinimalMediaTool(BaseTool):
    """Tool to determine the minimal set of media components required for growth using standard FBA."""

    tool_name = "find_minimal_media"
    tool_description = "Determine the minimal set of media components required for growth by iteratively removing nutrients using FBA."

    _config: MinimalMediaConfig

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        config_dict = config.get("minimal_media_config", {})
        if isinstance(config_dict, dict):
            self._config = MinimalMediaConfig(**config_dict)
        else:
            self._config = MinimalMediaConfig(precision=PrecisionConfig())
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
            if not is_significant_growth(
                complete_solution.objective_value,
                self._config.precision.growth_threshold,
            ):
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
            essential_nutrients = set()

            # Store original bounds for all exchange reactions
            original_bounds = {rxn.id: rxn.bounds for rxn in model.exchanges}

            # Test each exchange reaction individually
            for rxn in model.exchanges:
                # Only test uptake reactions (negative lower bound)
                if rxn.lower_bound >= 0:
                    continue

                # Save current bounds and block this nutrient uptake
                current_bounds = rxn.bounds
                rxn.lower_bound = 0

                # Test if growth is still possible without this nutrient
                test_solution = run_simulation(model, method="fba")

                if not is_significant_growth(
                    test_solution.objective_value,
                    self._config.precision.growth_threshold,
                ):
                    # Growth fails without this nutrient - it's essential
                    essential_nutrients.add(rxn.id)
                    minimal_media[rxn.id] = original_bounds[rxn.id]
                else:
                    # Growth succeeds without this nutrient - it's not essential
                    minimal_media[rxn.id] = (0, original_bounds[rxn.id][1])

                # Always restore original bounds for next test
                rxn.bounds = current_bounds
            # Create more detailed results
            essential_media = {
                rxn_id: bounds
                for rxn_id, bounds in minimal_media.items()
                if rxn_id in essential_nutrients
            }

            non_essential_media = {
                rxn_id: bounds
                for rxn_id, bounds in minimal_media.items()
                if rxn_id not in essential_nutrients
            }

            return ToolResult(
                success=True,
                message=f"Minimal media determination completed. Found {len(essential_nutrients)} essential nutrients.",
                data={
                    "minimal_media": minimal_media,
                    "essential_nutrients": essential_media,
                    "non_essential_nutrients": non_essential_media,
                    "complete_solution": complete_solution.objective_value,
                    "result_file": result_file,
                },
                metadata={
                    "model_id": model.id,
                    "num_exchange_reactions": len(
                        [rxn for rxn in model.exchanges if rxn.lower_bound < 0]
                    ),
                    "num_essential_nutrients": len(essential_nutrients),
                    "num_non_essential_nutrients": len(non_essential_media),
                    "growth_threshold": self._config.precision.growth_threshold,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False, message="Error determining minimal media.", error=str(e)
            )
