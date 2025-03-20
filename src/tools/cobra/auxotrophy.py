from typing import Dict, Any
from pydantic import BaseModel, Field, PrivateAttr
import cobra
from ..base import BaseTool, ToolResult, ToolRegistry
from .utils import ModelUtils
from .simulation_wrapper import run_simulation
from .fba import SimulationResultsStore  # Import the results store

class AuxotrophyConfig(BaseModel):
    """Configuration for auxotrophy identification tool."""
    growth_threshold: float = 1e-6
    candidate_metabolites: list = Field(default_factory=lambda: [
        "EX_arg__L_e", "EX_leu__L_e", "EX_lys__L_e"
    ])

@ToolRegistry.register
class AuxotrophyTool(BaseTool):
    """Tool to identify potential auxotrophies by testing the removal of candidate nutrients using standard FBA."""
    tool_name = "identify_auxotrophies"
    tool_description = "Identify potential auxotrophies by testing the removal of candidate nutrients using FBA."
    
    _config: AuxotrophyConfig
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        config_dict = config.get("auxotrophy_config", {})
        self._config = AuxotrophyConfig(**config_dict)
        self._utils = ModelUtils()
    
    def _run(self, input_data: Any) -> ToolResult:
        try:
            if isinstance(input_data, dict):
                model_path = input_data.get("model_path")
                output_dir = input_data.get("output_dir", None)
            else:
                model_path = input_data
                output_dir = None
            
            model = self._utils.load_model(model_path)
            complete_solution = run_simulation(model, method="fba")
            complete_growth = complete_solution.objective_value
            
            # Optionally export the complete simulation results
            result_file = None
            if output_dir:
                store = SimulationResultsStore()
                result_id = store.save_results(self.tool_name, model.id, complete_solution, additional_metadata={"simulation_method": "fba"})
                json_path, csv_path = store.export_results(result_id, output_dir)
                result_file = {"json": json_path, "csv": csv_path}
            
            auxotrophies = []
            for met_id in self._config.candidate_metabolites:
                try:
                    rxn = model.reactions.get_by_id(met_id)
                    original_bounds = rxn.bounds
                    rxn.lower_bound = 0
                    test_solution = run_simulation(model, method="fba")
                    rxn.bounds = original_bounds
                    if test_solution.objective_value < self._config.growth_threshold:
                        auxotrophies.append(met_id)
                except Exception:
                    continue
            return ToolResult(
                success=True,
                message="Auxotrophy analysis completed.",
                data={"complete_growth_rate": complete_growth, "auxotrophies": auxotrophies, "result_file": result_file}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message="Error identifying auxotrophies.",
                error=str(e)
            )