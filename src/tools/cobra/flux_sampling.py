from typing import Any, Dict, List, Optional

import cobra
import numpy as np
import pandas as pd
from cobra.sampling import sample
from pydantic import BaseModel, Field, PrivateAttr

from ..base import BaseTool, ToolRegistry, ToolResult
from .utils import ModelUtils


class FluxSamplingConfig(BaseModel):
    """Configuration for Flux Sampling"""

    model_config = {"protected_namespaces": ()}
    n_samples: int = 1000
    method: str = "optgp"  # "optgp" or "achr"
    thinning: int = 100
    processes: Optional[int] = None
    seed: Optional[int] = None
    solver: str = "glpk"


@ToolRegistry.register
class FluxSamplingTool(BaseTool):
    """Tool for statistical flux sampling to explore the solution space"""

    tool_name = "run_flux_sampling"
    tool_description = """Sample the feasible flux space to understand flux distributions
    and variability across the metabolic network."""

    _sampling_config: FluxSamplingConfig = PrivateAttr()
    _utils: ModelUtils = PrivateAttr()

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        sampling_config_dict = config.get("sampling_config", {})
        if isinstance(sampling_config_dict, dict):
            self._sampling_config = FluxSamplingConfig(**sampling_config_dict)
        else:
            self._sampling_config = FluxSamplingConfig(
                n_samples=getattr(sampling_config_dict, "n_samples", 1000),
                method=getattr(sampling_config_dict, "method", "optgp"),
                thinning=getattr(sampling_config_dict, "thinning", 100),
                processes=getattr(sampling_config_dict, "processes", None),
                seed=getattr(sampling_config_dict, "seed", None),
                solver=getattr(sampling_config_dict, "solver", "glpk"),
            )
        self._utils = ModelUtils()

    @property
    def sampling_config(self) -> FluxSamplingConfig:
        """Get the flux sampling configuration"""
        return self._sampling_config

    def _run_tool(self, input_data: Any) -> ToolResult:
        try:
            # Support both dict and string inputs
            if isinstance(input_data, dict):
                model_path = input_data.get("model_path")
                n_samples = input_data.get("n_samples", self.sampling_config.n_samples)
                method = input_data.get("method", self.sampling_config.method)
                thinning = input_data.get("thinning", self.sampling_config.thinning)
                seed = input_data.get("seed", self.sampling_config.seed)
            else:
                model_path = input_data
                n_samples = self.sampling_config.n_samples
                method = self.sampling_config.method
                thinning = self.sampling_config.thinning
                seed = self.sampling_config.seed

            if not isinstance(model_path, str):
                raise ValueError("Model path must be a string")

            # Load model
            model = self._utils.load_model(model_path)
            model.solver = self.sampling_config.solver

            # Set random seed if provided
            if seed is not None:
                np.random.seed(seed)

            # Run flux sampling
            samples = sample(
                model=model,
                n=n_samples,
                method=method,
                thinning=thinning,
                processes=self.sampling_config.processes,
                seed=seed,
            )

            # Analyze samples
            analysis = self._analyze_samples(samples, model)

            return ToolResult(
                success=True,
                message=f"Flux sampling completed with {n_samples} samples using {method} method",
                data={"samples": samples.to_dict(), "analysis": analysis},
                metadata={
                    "model_id": model.id,
                    "n_samples": n_samples,
                    "method": method,
                    "thinning": thinning,
                    "seed": seed,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False, message="Error running flux sampling", error=str(e)
            )

    def _analyze_samples(
        self, samples: pd.DataFrame, model: cobra.Model
    ) -> Dict[str, Any]:
        """Analyze flux samples to extract statistical insights"""

        analysis = {
            "statistics": {},
            "flux_patterns": {},
            "correlations": {},
            "subsystem_analysis": {},
            "distribution_analysis": {},
        }

        # Basic statistics
        analysis["statistics"] = {
            "mean_fluxes": samples.mean().to_dict(),
            "std_fluxes": samples.std().to_dict(),
            "median_fluxes": samples.median().to_dict(),
            "min_fluxes": samples.min().to_dict(),
            "max_fluxes": samples.max().to_dict(),
        }

        # Identify flux patterns
        tolerance = 1e-6

        # Always active reactions (non-zero in all samples)
        always_active = []
        # Variable reactions (standard deviation > threshold)
        variable_reactions = []
        # Rarely active reactions (active in <10% of samples)
        rarely_active = []

        for reaction_id in samples.columns:
            flux_values = samples[reaction_id]
            non_zero_fraction = (np.abs(flux_values) > tolerance).mean()
            std_dev = flux_values.std()

            if non_zero_fraction > 0.9:
                always_active.append(
                    {
                        "reaction_id": reaction_id,
                        "mean_flux": flux_values.mean(),
                        "std_dev": std_dev,
                        "active_fraction": non_zero_fraction,
                    }
                )
            elif non_zero_fraction < 0.1:
                rarely_active.append(
                    {
                        "reaction_id": reaction_id,
                        "mean_flux": flux_values.mean(),
                        "std_dev": std_dev,
                        "active_fraction": non_zero_fraction,
                    }
                )

            if std_dev > tolerance:
                variable_reactions.append(
                    {
                        "reaction_id": reaction_id,
                        "mean_flux": flux_values.mean(),
                        "std_dev": std_dev,
                        "coefficient_of_variation": (
                            std_dev / abs(flux_values.mean())
                            if abs(flux_values.mean()) > tolerance
                            else float("inf")
                        ),
                    }
                )

        # Sort by variability
        variable_reactions.sort(key=lambda x: x["std_dev"], reverse=True)

        analysis["flux_patterns"] = {
            "always_active": always_active[:20],  # Top 20
            "variable_reactions": variable_reactions[:20],  # Top 20 most variable
            "rarely_active": rarely_active[:20],  # Top 20
        }

        # Correlation analysis (top 50 most variable reactions to keep manageable)
        top_variable = [r["reaction_id"] for r in variable_reactions[:50]]
        if len(top_variable) > 1:
            correlation_matrix = samples[top_variable].corr()

            # Find highly correlated pairs
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # High correlation threshold
                        high_correlations.append(
                            {
                                "reaction_1": correlation_matrix.columns[i],
                                "reaction_2": correlation_matrix.columns[j],
                                "correlation": corr_value,
                            }
                        )

            # Sort by absolute correlation
            high_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            analysis["correlations"] = {
                "high_correlations": high_correlations[:20],  # Top 20
                "correlation_matrix_shape": correlation_matrix.shape,
            }

        # Subsystem analysis
        subsystem_stats = {}
        for reaction in model.reactions:
            if reaction.id in samples.columns:
                subsystem = reaction.subsystem or "Unknown"
                if subsystem not in subsystem_stats:
                    subsystem_stats[subsystem] = {
                        "reactions": [],
                        "mean_absolute_flux": 0,
                        "total_variance": 0,
                    }

                flux_values = samples[reaction.id]
                subsystem_stats[subsystem]["reactions"].append(reaction.id)
                subsystem_stats[subsystem]["mean_absolute_flux"] += abs(
                    flux_values.mean()
                )
                subsystem_stats[subsystem]["total_variance"] += flux_values.var()

        # Sort subsystems by activity
        subsystem_summary = []
        for subsystem, stats in subsystem_stats.items():
            subsystem_summary.append(
                {
                    "subsystem": subsystem,
                    "num_reactions": len(stats["reactions"]),
                    "mean_absolute_flux": stats["mean_absolute_flux"],
                    "total_variance": stats["total_variance"],
                    "avg_flux_per_reaction": (
                        stats["mean_absolute_flux"] / len(stats["reactions"])
                        if len(stats["reactions"]) > 0
                        else 0
                    ),
                }
            )

        subsystem_summary.sort(key=lambda x: x["mean_absolute_flux"], reverse=True)
        analysis["subsystem_analysis"] = {
            "subsystem_summary": subsystem_summary[:15],  # Top 15 most active
            "total_subsystems": len(subsystem_stats),
        }

        # Distribution analysis
        objective_samples = None
        if hasattr(model, "objective") and model.objective:
            # Try to find objective reaction in samples
            try:
                # For newer CobraP versions
                obj_coeffs = model.objective.get_linear_coefficients(model.variables)
                obj_reactions = [rxn.id for rxn in obj_coeffs.keys()]
            except TypeError:
                # For older CobraP versions or fallback
                try:
                    obj_reactions = [
                        rxn.id
                        for rxn in model.objective.get_linear_coefficients().keys()
                    ]
                except:
                    # Final fallback - get from objective expression
                    obj_reactions = [
                        var.name
                        for var in model.objective.variables
                        if hasattr(var, "name")
                    ]

            for obj_rxn in obj_reactions:
                if obj_rxn in samples.columns:
                    objective_samples = samples[obj_rxn]
                    break

        analysis["distribution_analysis"] = {
            "total_samples": len(samples),
            "total_reactions_sampled": len(samples.columns),
            "objective_stats": (
                {
                    "mean": (
                        objective_samples.mean()
                        if objective_samples is not None
                        else None
                    ),
                    "std": (
                        objective_samples.std()
                        if objective_samples is not None
                        else None
                    ),
                    "min": (
                        objective_samples.min()
                        if objective_samples is not None
                        else None
                    ),
                    "max": (
                        objective_samples.max()
                        if objective_samples is not None
                        else None
                    ),
                }
                if objective_samples is not None
                else None
            ),
        }

        return analysis
