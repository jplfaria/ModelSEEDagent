from typing import Any, Dict, List, Optional, Union

import cobra
import numpy as np
import pandas as pd
from cobra.flux_analysis import production_envelope
from pydantic import BaseModel, Field, PrivateAttr

from ..base import BaseTool, ToolRegistry, ToolResult
from .error_handling import (
    ModelValidationError,
    ParameterValidationError,
    create_error_result,
    create_progress_logger,
    validate_model_path,
    validate_numerical_parameters,
    validate_solver_availability,
)
from .utils_optimized import OptimizedModelUtils


class ProductionEnvelopeConfig(BaseModel):
    """Configuration for Production Envelope Analysis"""

    model_config = {"protected_namespaces": ()}
    points: int = 20
    solver: str = "glpk"
    tolerance: float = 1e-6


@ToolRegistry.register
class ProductionEnvelopeTool(BaseTool):
    """Tool for production envelope analysis to explore growth vs production trade-offs"""

    tool_name = "run_production_envelope"
    tool_description = """Calculate production envelope to analyze the relationship between
    growth rate and product formation, useful for metabolic engineering design."""

    _envelope_config: ProductionEnvelopeConfig = PrivateAttr()
    _utils: OptimizedModelUtils = PrivateAttr()

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        envelope_config_dict = config.get("envelope_config", {})
        if isinstance(envelope_config_dict, dict):
            self._envelope_config = ProductionEnvelopeConfig(**envelope_config_dict)
        else:
            self._envelope_config = ProductionEnvelopeConfig(
                points=getattr(envelope_config_dict, "points", 20),
                solver=getattr(envelope_config_dict, "solver", "glpk"),
                tolerance=getattr(envelope_config_dict, "tolerance", 1e-6),
            )
        self._utils = OptimizedModelUtils(use_cache=True)

    @property
    def envelope_config(self) -> ProductionEnvelopeConfig:
        """Get the production envelope configuration"""
        return self._envelope_config

    def _run_tool(self, input_data: Any) -> ToolResult:
        try:
            # Support both dict and string inputs (for consistency with other tools)
            if isinstance(input_data, dict):
                model_path = input_data.get("model_path")
                reactions = input_data.get("reactions")
                points = input_data.get("points", self.envelope_config.points)
                objective = input_data.get("objective", None)
                c_source = input_data.get("c_source", None)
                c_uptake_rates = input_data.get("c_uptake_rates", None)

                if not model_path or not reactions:
                    return create_error_result(
                        "Missing required parameters",
                        "Both model_path and reactions must be provided",
                        [
                            "Provide model_path: path to SBML model file",
                            "Provide reactions: list of reaction IDs to analyze",
                            "Example: {'model_path': 'model.xml', 'reactions': ['EX_ac_e', 'EX_etoh_e']}",
                        ],
                    )
            else:
                # For string input, we can't determine reactions, so raise an error
                return create_error_result(
                    "Invalid input format",
                    "ProductionEnvelopeTool requires dictionary input with model_path and reactions keys",
                    [
                        "Use dictionary format: {'model_path': 'path.xml', 'reactions': ['rxn1', 'rxn2']}",
                        "Cannot use string-only input for this tool",
                        "Specify which reactions to analyze in the envelope",
                    ],
                )

            # Validate inputs
            try:
                model_path = validate_model_path(model_path)
                validate_numerical_parameters(
                    {"points": points, "tolerance": self.envelope_config.tolerance},
                    self.tool_name,
                )
                solver = validate_solver_availability(self.envelope_config.solver)
            except (ParameterValidationError, ModelValidationError) as e:
                return create_error_result(
                    f"Input validation failed: {str(e)}",
                    str(e),
                    getattr(e, "suggestions", []),
                )

            # Load model
            model = self._utils.load_model(model_path)
            model.solver = solver

            # Validate reactions exist in model
            reaction_objects = []
            missing_reactions = []
            for rxn_id in reactions:
                if rxn_id not in model.reactions:
                    missing_reactions.append(rxn_id)
                else:
                    reaction_objects.append(model.reactions.get_by_id(rxn_id))

            if missing_reactions:
                # Find similar reaction names as suggestions
                similar_reactions = []
                for missing in missing_reactions:
                    for rxn in model.reactions:
                        if (
                            missing.lower() in rxn.id.lower()
                            or rxn.id.lower() in missing.lower()
                        ):
                            similar_reactions.append(rxn.id)
                            if len(similar_reactions) >= 5:  # Limit suggestions
                                break

                return create_error_result(
                    f"Reactions not found in model: {missing_reactions}",
                    f"Invalid reaction IDs: {missing_reactions}",
                    [
                        f"Available similar reactions: {similar_reactions[:5]}",
                        "Check reaction ID spelling and case sensitivity",
                        "Use model.reactions to list all available reactions",
                        "Common exchange reactions start with 'EX_'",
                    ],
                )

            # Set up carbon source constraints if provided
            if c_source and c_uptake_rates:
                self._setup_carbon_source_constraints(model, c_source, c_uptake_rates)

            # Set objective if provided
            if objective:
                if objective in model.reactions:
                    model.objective = objective
                else:
                    raise ValueError(
                        f"Objective reaction {objective} not found in model"
                    )

            # Calculate production envelope with progress logging
            log_progress = create_progress_logger(
                1, f"Production envelope ({points} points)"
            )
            log_progress(
                0, f"Calculating envelope for {len(reaction_objects)} reactions"
            )

            envelope_data = production_envelope(
                model=model, reactions=reaction_objects, points=points
            )

            log_progress(1, "Envelope calculation complete")

            # Analyze the envelope
            analysis = self._analyze_envelope(envelope_data, reaction_objects, model)

            # Create lightweight envelope summary instead of full DataFrame
            envelope_summary = self._create_envelope_summary(envelope_data, reactions)
            
            return ToolResult(
                success=True,
                message=f"Production envelope calculated for {len(reactions)} reaction(s) with {points} points",
                data={"envelope_summary": envelope_summary, "analysis": analysis},
                metadata={
                    "model_id": model.id,
                    "reactions": reactions,
                    "points": points,
                    "objective": str(model.objective.expression),
                },
            )

        except Exception as e:
            error_msg = str(e)
            suggestions = [
                "Verify model file is valid and can perform FBA",
                "Check that specified reactions exist in the model",
                "Ensure model has proper objective function set",
                "Try reducing number of points if memory issues occur",
            ]

            # Add specific suggestions based on error type
            if "solver" in error_msg.lower() or "optimization" in error_msg.lower():
                suggestions.extend(
                    [
                        "Install additional solvers: pip install python-glpk-cffi",
                        "Check if model constraints allow feasible solutions",
                    ]
                )
            elif "objective" in error_msg.lower():
                suggestions.extend(
                    [
                        "Set model objective: model.objective = 'biomass_reaction_id'",
                        "Verify objective reaction exists and has proper bounds",
                    ]
                )
            elif "infeasible" in error_msg.lower():
                suggestions.extend(
                    [
                        "Check medium constraints and exchange reaction bounds",
                        "Ensure essential nutrients are available for uptake",
                    ]
                )

            return create_error_result(
                "Failed to calculate production envelope",
                error_msg,
                suggestions,
                {
                    "tool_name": self.tool_name,
                    "reactions": reactions if "reactions" in locals() else "unknown",
                    "points": points if "points" in locals() else "unknown",
                    "model_path": (
                        str(model_path) if "model_path" in locals() else "unknown"
                    ),
                },
            )

    def _setup_carbon_source_constraints(
        self,
        model: cobra.Model,
        c_source: str,
        c_uptake_rates: Union[float, List[float]],
    ):
        """Set up carbon source uptake constraints"""

        # Handle single rate or list of rates
        if isinstance(c_uptake_rates, (int, float)):
            uptake_rates = [c_uptake_rates]
        else:
            uptake_rates = c_uptake_rates

        # Find carbon source exchange reaction
        carbon_exchange = None
        possible_names = [
            f"EX_{c_source}",
            f"EX_{c_source}_e",
            f"EX_{c_source}(e)",
            c_source,
        ]

        for name in possible_names:
            if name in model.reactions:
                carbon_exchange = model.reactions.get_by_id(name)
                break

        if carbon_exchange is None:
            raise ValueError(
                f"Carbon source exchange reaction for {c_source} not found"
            )

        # Set uptake rate (negative for uptake)
        uptake_rate = -abs(uptake_rates[0])  # Use first rate if multiple provided
        carbon_exchange.lower_bound = uptake_rate
        carbon_exchange.upper_bound = 0  # Only uptake, no secretion

    def _analyze_envelope(
        self,
        envelope_data: pd.DataFrame,
        reactions: List[cobra.Reaction],
        model: cobra.Model,
    ) -> Dict[str, Any]:
        """Streamlined envelope analysis focusing on key insights"""
        # Find objective column (usually flux_maximum or similar)
        objective_col = None
        for col in ['flux_maximum', 'flux_minimum', 'biomass', 'growth']:
            if col in envelope_data.columns:
                objective_col = col
                break
        
        if objective_col is None:
            # Fallback to first numeric column
            numeric_cols = envelope_data.select_dtypes(include=['float64', 'int64']).columns
            objective_col = numeric_cols[0] if len(numeric_cols) > 0 else envelope_data.columns[0]
        
        # Find reaction columns (columns that match our requested reactions)
        reaction_names = [rxn.id for rxn in reactions]
        reaction_cols = [col for col in envelope_data.columns if col in reaction_names]
        
        analysis = {
            "summary": {
                "total_points": len(envelope_data),
                "reactions_analyzed": len(reaction_cols),
                "objective_column": objective_col,
            },
            "key_insights": []
        }
        
        # Add objective max if numeric
        try:
            analysis["summary"]["objective_max"] = float(envelope_data[objective_col].max())
        except (ValueError, TypeError):
            analysis["summary"]["objective_max"] = "N/A"
        
        # Generate key insights for each reaction
        for col in reaction_cols:
            production_values = envelope_data[col]
            max_production = float(production_values.max())
            min_production = float(production_values.min())
            
            # Try correlation if objective is numeric
            try:
                correlation = envelope_data[objective_col].corr(production_values)
                
                # Classify trade-off relationship
                if correlation > 0.1:
                    trade_off = "synergistic"
                    insight = f"Production increases with {objective_col} (r={correlation:.2f})"
                elif correlation < -0.1:
                    trade_off = "competitive" 
                    insight = f"Production competes with {objective_col} (r={correlation:.2f})"
                else:
                    trade_off = "independent"
                    insight = f"Production independent of {objective_col} (r={correlation:.2f})"
            except (ValueError, TypeError):
                trade_off = "unknown"
                insight = f"Production analysis limited due to non-numeric objective"
            
            analysis["key_insights"].append({
                "reaction": col,
                "trade_off_type": trade_off,
                "max_production": max_production,
                "min_production": min_production,
                "insight": insight
            })
        
        return analysis

    def _create_envelope_summary(self, envelope_data, reactions):
        """Create lightweight summary of envelope data instead of full DataFrame"""
        # Find objective column
        objective_col = None
        for col in ['flux_maximum', 'flux_minimum', 'biomass', 'growth']:
            if col in envelope_data.columns:
                objective_col = col
                break
        
        if objective_col is None:
            numeric_cols = envelope_data.select_dtypes(include=['float64', 'int64']).columns
            objective_col = numeric_cols[0] if len(numeric_cols) > 0 else envelope_data.columns[0]
        
        # Find reaction columns from our requested reactions
        reaction_names = [r for r in reactions]  # reactions is already a list of strings
        reaction_cols = [col for col in envelope_data.columns if col in reaction_names]
        
        # Basic envelope statistics
        envelope_summary = {
            "total_points": len(envelope_data),
            "reactions": {}
        }
        
        # Add objective statistics if numeric
        try:
            envelope_summary["objective"] = {
                "column": objective_col,
                "min": float(envelope_data[objective_col].min()),
                "max": float(envelope_data[objective_col].max()),
                "mean": float(envelope_data[objective_col].mean()),
            }
        except (ValueError, TypeError):
            envelope_summary["objective"] = {"column": objective_col, "type": "non-numeric"}
        
        # Reaction-specific statistics  
        for col in reaction_cols:
            production_values = envelope_data[col]
            
            envelope_summary["reactions"][col] = {
                "production_range": {
                    "min": float(production_values.min()),
                    "max": float(production_values.max()),
                    "mean": float(production_values.mean()),
                }
            }
            
            # Add correlation if objective is numeric
            try:
                objective_values = envelope_data[objective_col]
                correlation = float(objective_values.corr(production_values))
                envelope_summary["reactions"][col]["correlation_with_objective"] = correlation
            except (ValueError, TypeError):
                envelope_summary["reactions"][col]["correlation_with_objective"] = "N/A"
        
        return envelope_summary
