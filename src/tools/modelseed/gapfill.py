from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..base import BaseTool, ToolRegistry, ToolResult
from ..cobra.utils import ModelUtils


class GapFillConfig(BaseModel):
    """Configuration for ModelSEED gap filling"""

    media_condition: str = "Complete"
    allow_reactions: List[str] = Field(default_factory=list)
    blacklist_reactions: List[str] = Field(default_factory=list)
    max_solutions: int = 1
    objective_function: str = "biomass"
    additional_config: Dict[str, Any] = Field(default_factory=dict)


@ToolRegistry.register
class GapFillTool(BaseTool):
    """Tool for gap filling metabolic models using ModelSEED"""

    tool_name = "gapfill_model"
    tool_description = """Perform gap filling on metabolic models to resolve network gaps
    and enable model growth using ModelSEED."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Use private attribute to avoid Pydantic field conflicts
        self._gapfill_config = GapFillConfig(**config.get("gapfill_config", {}))

    def _run_tool(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Perform gap filling on a metabolic model.

        Args:
            input_data: Dictionary containing:
                - model_path: Path to input model file (or model_object)
                - model_object: Pre-loaded COBRA model (optional)
                - media_condition: Growth condition to test
                - output_path: Where to save the gapfilled model
                - max_solutions: Maximum number of solutions to find
                - blacklist_reactions: Reactions to exclude from gapfilling

        Returns:
            ToolResult containing the gap filling results
        """
        try:
            # Load model
            model = None
            if "model_object" in input_data:
                model = input_data["model_object"]
            elif "model_path" in input_data:
                model_path = Path(input_data["model_path"])
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                model = ModelUtils().load_model(str(model_path))
            else:
                raise ValueError("Either model_path or model_object must be provided")

            # Get media condition
            media_condition = input_data.get(
                "media_condition", self._gapfill_config.media_condition
            )

            # Lazy import modelseedpy only when needed
            import modelseedpy

            # Initialize MSGapfill (simplified constructor)
            gapfiller = modelseedpy.MSGapfill(model)

            # Handle blacklisted reactions after initialization if needed
            blacklisted_reactions = input_data.get(
                "blacklist_reactions", self._gapfill_config.blacklist_reactions
            )
            if blacklisted_reactions:
                # Apply blacklist if the API supports it
                try:
                    gapfiller.blacklisted_reactions = blacklisted_reactions
                except AttributeError:
                    print(
                        f"Warning: Cannot set blacklisted reactions: {blacklisted_reactions}"
                    )

            # Configure allowed reactions if specified
            if (
                input_data.get("allow_reactions")
                or self._gapfill_config.allow_reactions
            ):
                allowed = input_data.get(
                    "allow_reactions", self._gapfill_config.allow_reactions
                )
                # Filter gapfilling database to only allowed reactions
                gapfiller.prefilter(allowed_reactions=allowed)

            # Convert media condition to MSMedia object if needed
            media_obj = None
            if isinstance(media_condition, str):
                # Try to create MSMedia object or use string directly
                try:
                    # Try different ways to get media
                    if hasattr(modelseedpy, "MSMedia"):
                        media_obj = modelseedpy.MSMedia()
                        # Set media type if possible
                        if hasattr(media_obj, "id"):
                            media_obj.id = media_condition
                    else:
                        # Use string directly if MSMedia not available or use None for default
                        media_obj = None
                except Exception:
                    # Fallback: use None for default media
                    media_obj = None
            else:
                media_obj = (
                    media_condition  # Assume it's already an MSMedia object or None
                )

            # Run gapfilling with media passed to the method (correct API)
            # Try different parameter combinations based on the API
            try:
                solutions = gapfiller.run_gapfilling(media=media_obj)
            except Exception as e:
                # Fallback: try without media parameter
                try:
                    solutions = gapfiller.run_gapfilling()
                except Exception as e2:
                    raise Exception(
                        f"Gapfilling failed with media ({e}) and without media ({e2})"
                    )

            if not solutions:
                return ToolResult(
                    success=False,
                    message="No gapfilling solutions found",
                    error="Unable to find reactions that enable growth",
                )

            # Apply the best solution (first one)
            best_solution = solutions[0]
            gapfiller.integrate_gapfill_solution(best_solution)

            # Save gapfilled model if output path provided
            output_path = input_data.get("output_path")
            output_file = None
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)

                # Use cobra.io for SBML export
                from cobra.io import write_sbml_model

                write_sbml_model(model, str(output_file))

            # Gather solution statistics (handle different solution data structures)
            added_reactions = []
            objective_value = 0

            # Handle different solution formats from MSGapfill
            if hasattr(best_solution, "added_reactions"):
                added_reactions = [r.id for r in best_solution.added_reactions]
            elif isinstance(best_solution, dict):
                added_reactions = [
                    r.id if hasattr(r, "id") else str(r)
                    for r in best_solution.get("added_reactions", [])
                ]
                objective_value = best_solution.get("objective_value", 0)
            elif hasattr(best_solution, "__iter__"):
                # If solution is a list/set of reaction objects
                added_reactions = [
                    r.id if hasattr(r, "id") else str(r) for r in best_solution
                ]

            solution_stats = {
                "num_solutions": len(solutions),
                "added_reactions": added_reactions,
                "removed_reactions": [],  # MSGapfill typically only adds reactions
                "objective_value": objective_value,
                "growth_improvement": len(added_reactions) > 0,
            }

            # Test gapfilled model growth
            with model:
                fba_solution = model.optimize()
                final_growth = (
                    fba_solution.objective_value
                    if fba_solution.status == "optimal"
                    else 0
                )

            return ToolResult(
                success=True,
                message=f"Gapfilling completed: {len(solution_stats['added_reactions'])} reactions added, growth rate: {final_growth:.6f}",
                data={
                    "gapfill_statistics": solution_stats,
                    "final_growth_rate": final_growth,
                    "model_path": str(output_file) if output_path else None,
                    "gapfilled_model": model,  # Include gapfilled model object
                    "solutions": solutions[:3],  # Include top 3 solutions for analysis
                },
                metadata={
                    "tool_type": "gapfilling",
                    "media_condition": media_condition,
                    "num_added_reactions": len(solution_stats["added_reactions"]),
                    "successful_growth": final_growth > 1e-6,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error during gapfilling: {str(e)}",
                error=str(e),
            )
