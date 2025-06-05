from typing import Any, Dict

import cobra
from pydantic import BaseModel, Field

from ..base import BaseTool, ToolRegistry, ToolResult
from .fba import SimulationResultsStore  # Import the results store
from .simulation_wrapper import run_simulation
from .utils import ModelUtils


class MissingMediaConfig(BaseModel):
    """Configuration for missing media analysis tool."""

    growth_threshold: float = 1e-6
    essential_metabolites: list = Field(
        default_factory=lambda: [
            "EX_glc__D_e",
            "EX_o2_e",
            "EX_nh4_e",
            "EX_pi_e",
            "EX_so4_e",
        ]
    )
    supplementation_amount: float = 10.0


@ToolRegistry.register
class MissingMediaTool(BaseTool):
    """Tool to check for missing media components using standard FBA."""

    tool_name = "check_missing_media"
    tool_description = (
        "Identify missing media components in a metabolic model using FBA."
    )

    _config: MissingMediaConfig

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        config_dict = config.get("missing_media_config", {})
        self._config = MissingMediaConfig(**config_dict)
        self._utils = ModelUtils()

    def _run_tool(self, input_data: Any) -> ToolResult:
        try:
            if isinstance(input_data, dict):
                model_path = input_data.get("model_path")
                output_dir = input_data.get("output_dir", None)
            else:
                model_path = input_data
                output_dir = None

            model = self._utils.load_model(model_path)
            solution = run_simulation(model, method="fba")

            # Optionally export simulation results
            result_file = None
            if output_dir:
                store = SimulationResultsStore()
                result_id = store.save_results(
                    self.tool_name,
                    model.id,
                    solution,
                    additional_metadata={"simulation_method": "fba"},
                )
                json_path, csv_path = store.export_results(result_id, output_dir)
                result_file = {"json": json_path, "csv": csv_path}

            if (
                solution.status != "optimal"
                or solution.objective_value < self._config.growth_threshold
            ):
                missing = []
                for met in self._config.essential_metabolites:
                    try:
                        rxn = model.reactions.get_by_id(met)
                        original_bounds = rxn.bounds
                        rxn.lower_bound = -self._config.supplementation_amount
                        test_solution = run_simulation(model, method="fba")
                        rxn.bounds = original_bounds
                        if (
                            test_solution.objective_value
                            >= self._config.growth_threshold
                        ):
                            missing.append(met)
                    except Exception:
                        continue
                return ToolResult(
                    success=True,
                    message="Missing media components identified.",
                    data={
                        "missing_components": missing,
                        "objective_value": solution.objective_value,
                        "result_file": result_file,
                    },
                )
            else:
                return ToolResult(
                    success=True,
                    message="Model grows under current media. No missing components detected.",
                    data={
                        "objective_value": solution.objective_value,
                        "result_file": result_file,
                    },
                )
        except Exception as e:
            return ToolResult(
                success=False,
                message="Error checking missing media components.",
                error=str(e),
            )
