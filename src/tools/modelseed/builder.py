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
    gapfill_on_build: bool = False  # Changed default to False
    annotate_with_rast: bool = False  # Option to skip RAST if already annotated
    export_sbml: bool = True
    export_json: bool = False
    additional_config: Dict[str, Any] = Field(default_factory=dict)


@ToolRegistry.register
class ModelBuildTool(BaseTool):
    """Tool for building metabolic models using ModelSEED"""

    tool_name = "build_metabolic_model"
    tool_description = """Build a metabolic model from genome annotations using ModelSEED MSBuilder.
    Supports MSGenome objects from RAST annotation or annotation files. Can export to SBML/JSON formats."""

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
            from modelseedpy import MSGenome, MSBuilder

            # Handle genome input - either file path or MSGenome object
            genome = None
            if "genome_object" in input_data:
                # Preferred: use pre-annotated MSGenome from RAST tool
                genome = input_data["genome_object"]
            elif "annotation_file" in input_data:
                annotation_file = Path(input_data["annotation_file"])
                if not annotation_file.exists():
                    raise FileNotFoundError(
                        f"Annotation file not found: {annotation_file}"
                    )
                # Create MSGenome from protein FASTA file
                genome = MSGenome.from_fasta(str(annotation_file))
                
                # Optional RAST annotation if requested
                if input_data.get("annotate_with_rast", self._build_config.annotate_with_rast):
                    rast_client = modelseedpy.RastClient()
                    rast_client.annotate_genome(genome)
            else:
                raise ValueError(
                    "Either annotation_file or genome_object must be provided"
                )

            # Initialize MSBuilder with genome (modern approach)
            builder = MSBuilder(genome)

            # Build the model with the specified approach
            annotate_with_rast = input_data.get(
                "annotate_with_rast", self._build_config.annotate_with_rast
            )
            model = builder.build(model_id, annotate_with_rast=annotate_with_rast)

            # Optional gapfilling during build (fixed API)
            gapfill_applied = False
            if input_data.get("gapfill_on_build", self._build_config.gapfill_on_build):
                try:
                    gapfiller = modelseedpy.MSGapfill(model)
                    
                    # Convert media condition to MSMedia object
                    media_condition = input_data.get(
                        "media_condition", self._build_config.media_condition
                    )
                    try:
                        media_obj = modelseedpy.MSMedia()
                        if hasattr(media_obj, 'id'):
                            media_obj.id = media_condition
                    except Exception:
                        media_obj = None
                    
                    gapfill_solutions = gapfiller.run_gapfilling(media=media_obj)
                    if gapfill_solutions:
                        gapfiller.integrate_gapfill_solution(gapfill_solutions[0])
                        gapfill_applied = True
                except Exception as gf_error:
                    # Don't fail the entire build if gapfilling fails
                    print(f"Warning: Gapfilling failed during build: {gf_error}")

            # Export model to requested formats
            output_files = {}
            base_path = Path(output_path)
            base_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Remove extension from base path if present
            base_name = base_path.stem if base_path.suffix else base_path.name
            base_dir = base_path.parent if base_path.suffix else base_path.parent
            
            # Export to SBML
            if input_data.get("export_sbml", self._build_config.export_sbml):
                from cobra.io import write_sbml_model
                sbml_path = base_dir / f"{base_name}.xml"
                write_sbml_model(model, str(sbml_path))
                output_files["sbml"] = str(sbml_path)
            
            # Export to JSON
            if input_data.get("export_json", self._build_config.export_json):
                from cobra.io import save_json_model
                json_path = base_dir / f"{base_name}.json"
                save_json_model(model, str(json_path))
                output_files["json"] = str(json_path)

            # Gather comprehensive model statistics
            stats = {
                "model_id": model.id,
                "num_reactions": len(model.reactions),
                "num_metabolites": len(model.metabolites),
                "num_genes": len(model.genes),
                "num_groups": len(getattr(model, 'groups', [])),
                "objective_expression": str(model.objective.expression),
                "compartments": list(model.compartments.keys()),
                "template_used": getattr(getattr(builder, 'template', None), 'id', 'default'),
                "gapfilled": gapfill_applied,
                "annotate_with_rast": annotate_with_rast,
            }

            return ToolResult(
                success=True,
                message=f"Model {model_id} built successfully: {stats['num_reactions']} reactions, {stats['num_metabolites']} metabolites, {stats['num_genes']} genes",
                data={
                    "model_id": model_id,
                    "output_files": output_files,
                    "statistics": stats,
                    "model_object": model,  # Include model object for downstream tools (gapfilling)
                    "genome_object": genome,  # Include genome for reference
                },
                metadata={
                    "tool_type": "model_building",
                    "template_used": stats["template_used"],
                    "gapfilled": stats["gapfilled"],
                    "num_reactions": stats["num_reactions"],
                    "num_metabolites": stats["num_metabolites"],
                    "num_genes": stats["num_genes"],
                },
            )

        except Exception as e:
            return ToolResult(
                success=False, message=f"Error building model: {str(e)}", error=str(e)
            )
