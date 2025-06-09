from typing import Any, Dict, List, Optional, Union

import cobra
import numpy as np
import pandas as pd
from cobra.flux_analysis import production_envelope
from pydantic import BaseModel, Field, PrivateAttr

from ..base import BaseTool, ToolRegistry, ToolResult
from .utils import ModelUtils


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
    _utils: ModelUtils = PrivateAttr()

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
        self._utils = ModelUtils()

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
                    raise ValueError("Both model_path and reactions must be provided")
            else:
                # For string input, we can't determine reactions, so raise an error
                raise ValueError(
                    "ProductionEnvelopeTool requires dictionary input with model_path and reactions keys"
                )

            # Load model
            model = self._utils.load_model(model_path)
            model.solver = self.envelope_config.solver

            # Validate reactions exist in model
            reaction_objects = []
            for rxn_id in reactions:
                if rxn_id not in model.reactions:
                    raise ValueError(f"Reaction {rxn_id} not found in model")
                reaction_objects.append(model.reactions.get_by_id(rxn_id))

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

            # Calculate production envelope
            envelope_data = production_envelope(
                model=model, reactions=reaction_objects, points=points
            )

            # Analyze the envelope
            analysis = self._analyze_envelope(envelope_data, reaction_objects, model)

            return ToolResult(
                success=True,
                message=f"Production envelope calculated for {len(reactions)} reaction(s) with {points} points",
                data={"envelope_data": envelope_data.to_dict(), "analysis": analysis},
                metadata={
                    "model_id": model.id,
                    "reactions": reactions,
                    "points": points,
                    "objective": str(model.objective.expression),
                    "c_source": c_source,
                    "c_uptake_rates": c_uptake_rates,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message="Error calculating production envelope",
                error=str(e),
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
        """Analyze production envelope data"""

        analysis = {
            "envelope_summary": {},
            "optimization_analysis": {},
            "trade_offs": {},
            "design_points": {},
        }

        # Get objective column (usually growth rate)
        objective_col = envelope_data.columns[
            0
        ]  # First column is typically the objective
        reaction_cols = envelope_data.columns[
            1:
        ]  # Subsequent columns are the reactions

        # Basic envelope summary
        analysis["envelope_summary"] = {
            "total_points": len(envelope_data),
            "objective_range": {
                "min": float(envelope_data[objective_col].min()),
                "max": float(envelope_data[objective_col].max()),
                "mean": float(envelope_data[objective_col].mean()),
            },
            "production_ranges": {},
        }

        # Analyze each reaction's production range
        for col in reaction_cols:
            production_values = envelope_data[col]
            analysis["envelope_summary"]["production_ranges"][col] = {
                "min": float(production_values.min()),
                "max": float(production_values.max()),
                "mean": float(production_values.mean()),
            }

        # Optimization analysis
        max_objective_idx = envelope_data[objective_col].idxmax()
        max_production_points = {}

        for col in reaction_cols:
            max_prod_idx = envelope_data[col].idxmax()
            max_production_points[col] = {
                "max_production": float(envelope_data.loc[max_prod_idx, col]),
                "objective_at_max_production": float(
                    envelope_data.loc[max_prod_idx, objective_col]
                ),
                "production_at_max_objective": float(
                    envelope_data.loc[max_objective_idx, col]
                ),
            }

        analysis["optimization_analysis"] = {
            "max_objective": float(envelope_data.loc[max_objective_idx, objective_col]),
            "max_production_points": max_production_points,
        }

        # Trade-off analysis
        trade_offs = {}
        for col in reaction_cols:
            # Calculate correlation between objective and production
            correlation = envelope_data[objective_col].corr(envelope_data[col])

            # Find points with high production and reasonable growth
            high_production_threshold = envelope_data[col].quantile(0.8)
            reasonable_growth_threshold = envelope_data[objective_col].max() * 0.5

            good_points = envelope_data[
                (envelope_data[col] >= high_production_threshold)
                & (envelope_data[objective_col] >= reasonable_growth_threshold)
            ]

            trade_offs[col] = {
                "correlation_with_objective": float(correlation),
                "trade_off_severity": (
                    "positive"
                    if correlation > 0.1
                    else "negative" if correlation < -0.1 else "neutral"
                ),
                "high_production_high_growth_points": len(good_points),
                "feasible_design_space": (
                    len(good_points) / len(envelope_data)
                    if len(envelope_data) > 0
                    else 0
                ),
            }

        analysis["trade_offs"] = trade_offs

        # Identify promising design points
        design_points = []

        for col in reaction_cols:
            # Find Pareto-optimal points (high production, high growth)
            production_values = envelope_data[col]
            objective_values = envelope_data[objective_col]

            # Score each point as weighted sum of normalized production and objective
            norm_production = (production_values - production_values.min()) / (
                production_values.max() - production_values.min() + 1e-6
            )
            norm_objective = (objective_values - objective_values.min()) / (
                objective_values.max() - objective_values.min() + 1e-6
            )

            # Weight production and growth equally
            scores = 0.5 * norm_production + 0.5 * norm_objective

            # Find top 5 scoring points
            top_indices = scores.nlargest(5).index

            top_points = []
            for idx in top_indices:
                top_points.append(
                    {
                        "production": float(envelope_data.loc[idx, col]),
                        "objective": float(envelope_data.loc[idx, objective_col]),
                        "score": float(scores[idx]),
                        "production_efficiency": (
                            float(
                                envelope_data.loc[idx, col] / envelope_data[col].max()
                            )
                            if envelope_data[col].max() > 0
                            else 0
                        ),
                        "growth_efficiency": (
                            float(
                                envelope_data.loc[idx, objective_col]
                                / envelope_data[objective_col].max()
                            )
                            if envelope_data[objective_col].max() > 0
                            else 0
                        ),
                    }
                )

            design_points.append({"reaction": col, "top_design_points": top_points})

        analysis["design_points"] = design_points

        return analysis
