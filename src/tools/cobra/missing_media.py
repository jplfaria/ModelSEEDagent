from typing import Any, Dict

import cobra
from pydantic import BaseModel, Field

from ..base import BaseTool, ToolRegistry, ToolResult
from .error_handling import (
    ModelValidationError,
    ParameterValidationError,
    create_error_result,
    create_progress_logger,
    safe_optimize,
    validate_model_path,
    validate_numerical_parameters,
)
from .fba import SimulationResultsStore  # Import the results store
from .simulation_wrapper import run_simulation
from .utils_optimized import OptimizedModelUtils


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
        self._utils = OptimizedModelUtils(use_cache=True)

    def _run_tool(self, input_data: Any) -> ToolResult:
        try:
            if isinstance(input_data, dict):
                model_path = input_data.get("model_path")
                output_dir = input_data.get("output_dir", None)
            else:
                model_path = input_data
                output_dir = None

            # Validate inputs
            try:
                model_path = validate_model_path(model_path)
                validate_numerical_parameters(
                    {
                        "growth_threshold": self._config.growth_threshold,
                        "supplementation_amount": self._config.supplementation_amount,
                    },
                    self.tool_name,
                )
            except (ParameterValidationError, ModelValidationError) as e:
                return create_error_result(
                    f"Input validation failed: {str(e)}",
                    str(e),
                    getattr(e, "suggestions", []),
                )

            model = self._utils.load_model(model_path)
            # Use safe optimization with better error handling
            solution = safe_optimize(model, "Initial FBA for missing media check")

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
                log_progress = create_progress_logger(
                    len(self._config.essential_metabolites),
                    "Testing essential metabolites",
                )

                for i, met in enumerate(self._config.essential_metabolites):
                    log_progress(i, f"Testing {met}")
                    try:
                        rxn = model.reactions.get_by_id(met)
                        original_bounds = rxn.bounds
                        rxn.lower_bound = -self._config.supplementation_amount
                        test_solution = safe_optimize(
                            model, f"Testing {met} supplementation"
                        )
                        rxn.bounds = original_bounds
                        if (
                            test_solution.objective_value
                            >= self._config.growth_threshold
                        ):
                            missing.append(met)
                    except Exception:
                        # Log specific metabolite test failure but continue
                        continue

                log_progress(len(self._config.essential_metabolites), "Complete")
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
            return create_error_result(
                "Failed to analyze missing media components",
                str(e),
                [
                    "Verify model file is valid SBML format",
                    "Check that model has proper exchange reactions (EX_*)",
                    "Ensure model can perform basic FBA before analysis",
                    "Try with different essential metabolite list if needed",
                ],
                {
                    "tool_name": self.tool_name,
                    "model_path": (
                        str(model_path) if "model_path" in locals() else "unknown"
                    ),
                },
            )
