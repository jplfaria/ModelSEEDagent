from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..base import BaseTool, ToolRegistry, ToolResult
from ..cobra.utils import ModelUtils


class ModelBuildConfig(BaseModel):
    """Configuration for ModelSEED model building"""

    template_model: Optional[str] = None
    media_condition: str = "Complete"
    genome_domain: str = "Bacteria"
    gapfill_on_build: bool = True
    additional_config: Dict[str, Any] = Field(default_factory=dict)


@ToolRegistry.register
class ModelBuildTool(BaseTool):
    """Tool for building metabolic models using ModelSEED"""

    tool_name = "build_metabolic_model"
    tool_description = """Build a metabolic model from genome annotations using ModelSEED.
    Can use RAST annotations or other supported annotation formats."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Use private attribute to avoid Pydantic field conflicts
        self._build_config = ModelBuildConfig(**config.get("build_config", {}))

    def _run_tool(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Build a metabolic model using ModelSEED.

        Args:
            input_data: Dictionary containing:
                - annotation_file: Path to genome annotation file (optional)
                - genome_object: Pre-loaded MSGenome object (optional)
                - output_path: Where to save the model
                - model_id: Identifier for the new model
                - template_model: Template model to use (optional)

        Returns:
            ToolResult containing the built model information
        """
        try:
            # Validate inputs
            model_id = input_data.get("model_id", "model")
            output_path = input_data.get("output_path", f"{model_id}.xml")

            # Lazy import modelseedpy only when needed
            import modelseedpy

            # Initialize MSBuilder
            builder = modelseedpy.MSBuilder()

            # Handle genome input - either file path or MSGenome object
            genome = None
            if "genome_object" in input_data:
                genome = input_data["genome_object"]
            elif "annotation_file" in input_data:
                annotation_file = Path(input_data["annotation_file"])
                if not annotation_file.exists():
                    raise FileNotFoundError(
                        f"Annotation file not found: {annotation_file}"
                    )
                # Load genome from annotation file
                genome = modelseedpy.MSGenome.from_fasta(str(annotation_file))
            else:
                raise ValueError(
                    "Either annotation_file or genome_object must be provided"
                )

            # Configure builder
            if "template_model" in input_data and input_data["template_model"]:
                template_path = input_data["template_model"]
                if Path(template_path).exists():
                    template = ModelUtils().load_model(template_path)
                    builder.template = template
                else:
                    # Use auto-select if template path doesn't exist
                    builder.auto_select_template(genome)
            else:
                # Auto-select template based on genome
                builder.auto_select_template(genome)

            # Build the model
            model = builder.build(genome, model_id)

            # Optional gapfilling during build
            if self._build_config.gapfill_on_build:
                gapfiller = modelseedpy.MSGapfill(
                    model, media=self._build_config.media_condition
                )
                gapfill_solutions = gapfiller.run_gapfilling()
                if gapfill_solutions:
                    gapfiller.integrate_gapfill_solution(gapfill_solutions[0])

            # Save model
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            model.write_sbml_file(str(output_file))

            # Gather statistics
            stats = {
                "num_reactions": len(model.reactions),
                "num_metabolites": len(model.metabolites),
                "num_genes": len(model.genes),
                "objective_id": model.objective.direction,
                "template_used": getattr(builder.template, "id", "auto-selected"),
                "gapfilled": self._build_config.gapfill_on_build,
            }

            return ToolResult(
                success=True,
                message=f"Model {model_id} built successfully with {stats['num_reactions']} reactions",
                data={
                    "model_id": model_id,
                    "model_path": str(output_file),
                    "statistics": stats,
                    "model_object": model,  # Include model object for downstream tools
                },
                metadata={
                    "tool_type": "model_building",
                    "template_used": stats["template_used"],
                    "gapfilled": stats["gapfilled"],
                },
            )

        except Exception as e:
            return ToolResult(
                success=False, message=f"Error building model: {str(e)}", error=str(e)
            )
