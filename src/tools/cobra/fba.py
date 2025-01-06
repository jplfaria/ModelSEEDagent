from typing import Dict, Any
from pydantic import BaseModel, Field, PrivateAttr
import cobra
from ..base import BaseTool, ToolResult, ToolRegistry
from .utils import ModelUtils

class FBAConfig(BaseModel):
    """Configuration for FBA tool"""
    model_config = {"protected_namespaces": ()}
    default_objective: str = "biomass_reaction"
    solver: str = "glpk"
    tolerance: float = 1e-6
    additional_constraints: Dict[str, float] = Field(default_factory=dict)

@ToolRegistry.register
class FBATool(BaseTool):
    """Tool for running Flux Balance Analysis"""
    
    tool_name = "run_metabolic_fba"
    tool_description = """Run Flux Balance Analysis (FBA) on a metabolic model."""
    
    _fba_config: FBAConfig = PrivateAttr()
    _utils: ModelUtils = PrivateAttr()
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Initialize private attributes
        fba_config_dict = config.get("fba_config", {})
        if isinstance(fba_config_dict, dict):
            self._fba_config = FBAConfig(**fba_config_dict)
        else:
            # If fba_config is already a Pydantic model or similar
            self._fba_config = FBAConfig(
                default_objective=getattr(fba_config_dict, "default_objective", "biomass_reaction"),
                solver=getattr(fba_config_dict, "solver", "glpk"),
                tolerance=getattr(fba_config_dict, "tolerance", 1e-6),
                additional_constraints=getattr(fba_config_dict, "additional_constraints", {})
            )
        self._utils = ModelUtils()
        
    @property
    def fba_config(self) -> FBAConfig:
        """Get the FBA configuration"""
        return self._fba_config
    
    def validate_input(self, model_path: str) -> None:
        if not isinstance(model_path, str):
            raise ValueError("Model path must be a string")
        if not model_path.endswith(('.xml', '.sbml')):
            raise ValueError("Model file must be in SBML format (.xml or .sbml)")
    
    def _run(self, model_path: str) -> ToolResult:
        try:
            self.validate_input(model_path)
            model = self._utils.load_model(model_path)
            
            model.solver = self.fba_config.solver
            model.tolerance = self.fba_config.tolerance
            
            if hasattr(model, self.fba_config.default_objective):
                model.objective = self.fba_config.default_objective
            
            for reaction_id, bound in self.fba_config.additional_constraints.items():
                if reaction_id in model.reactions:
                    model.reactions.get_by_id(reaction_id).bounds = (-bound, bound)
            
            solution = model.optimize()
            
            if solution.status != 'optimal':
                return ToolResult(
                    success=False,
                    message="FBA optimization failed to find optimal solution",
                    error=f"Solution status: {solution.status}"
                )
            
            significant_fluxes = {
                rxn.id: solution.fluxes[rxn.id]
                for rxn in model.reactions
                if abs(solution.fluxes[rxn.id]) > self.fba_config.tolerance
            }
            
            subsystem_fluxes = {}
            for rxn_id, flux in significant_fluxes.items():
                rxn = model.reactions.get_by_id(rxn_id)
                subsystem = rxn.subsystem or "Unknown"
                if subsystem not in subsystem_fluxes:
                    subsystem_fluxes[subsystem] = []
                subsystem_fluxes[subsystem].append((rxn_id, flux))
            
            return ToolResult(
                success=True,
                message="FBA analysis completed successfully",
                data={
                    "objective_value": solution.objective_value,
                    "status": solution.status,
                    "significant_fluxes": significant_fluxes,
                    "subsystem_fluxes": subsystem_fluxes,
                    "solver_time": getattr(solution, "solver_time", None),
                },
                metadata={
                    "model_id": model.id,
                    "objective_reaction": str(model.objective.expression),
                    "num_reactions": len(model.reactions),
                    "num_metabolites": len(model.metabolites),
                    "num_significant_fluxes": len(significant_fluxes)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                message="Error running FBA analysis",
                error=str(e)
            )