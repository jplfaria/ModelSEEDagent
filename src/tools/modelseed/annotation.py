from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..base import BaseTool, ToolRegistry, ToolResult


class RastAnnotationConfig(BaseModel):
    """Configuration for RAST genome annotation"""

    genome_domain: str = "Bacteria"
    genetic_code: int = 11
    kingdom: str = "Bacteria"
    include_quality_control: bool = True
    additional_config: Dict[str, Any] = Field(default_factory=dict)


@ToolRegistry.register
class RastAnnotationTool(BaseTool):
    """Tool for genome annotation using RAST"""

    tool_name = "annotate_genome_rast"
    tool_description = """Annotate a genome using the RAST (Rapid Annotation using Subsystem Technology) service.
    Provides functional annotation of protein-coding genes."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Use private attribute to avoid Pydantic field conflicts
        self._annotation_config = RastAnnotationConfig(
            **config.get("annotation_config", {})
        )

    def _run_tool(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Annotate a genome using RAST.

        Args:
            input_data: Dictionary containing:
                - genome_file: Path to genome FASTA file
                - genome_name: Name for the genome
                - output_path: Where to save the annotated genome
                - organism_type: Type of organism (Bacteria, Archaea, etc.)

        Returns:
            ToolResult containing the annotation results
        """
        try:
            # Validate inputs
            genome_file = Path(input_data["genome_file"])
            if not genome_file.exists():
                raise FileNotFoundError(f"Genome file not found: {genome_file}")

            genome_name = input_data.get("genome_name", genome_file.stem)
            output_path = input_data.get("output_path", f"{genome_name}_annotated")

            # Lazy import modelseedpy only when needed
            import modelseedpy

            # Initialize RAST client
            rast_client = modelseedpy.RastClient()

            # Configure annotation parameters
            organism_type = input_data.get(
                "organism_type", self._annotation_config.genome_domain
            )
            genetic_code = input_data.get(
                "genetic_code", self._annotation_config.genetic_code
            )

            # Submit annotation job
            annotation_result = rast_client.annotate_genome_from_fasta(
                str(genome_file),
                genome_name=genome_name,
                domain=organism_type,
                genetic_code=genetic_code,
            )

            # Create MSGenome object from annotation
            genome = modelseedpy.MSGenome()
            if hasattr(annotation_result, "features"):
                # Process annotation features
                for feature in annotation_result.features:
                    if feature.type == "CDS":
                        genome.add_gene(
                            feature.id,
                            feature.location.start,
                            feature.location.end,
                            feature.location.strand,
                            feature.function,
                        )

            # Save annotated genome if output path provided
            output_file = None
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                genome.save(str(output_file))

            # Gather annotation statistics
            stats = {
                "genome_name": genome_name,
                "num_contigs": len(genome.contigs) if hasattr(genome, "contigs") else 0,
                "num_genes": len(genome.genes) if hasattr(genome, "genes") else 0,
                "num_proteins": (
                    len([g for g in genome.genes if g.type == "protein_coding"])
                    if hasattr(genome, "genes")
                    else 0
                ),
                "annotation_source": "RAST",
                "organism_type": organism_type,
                "genetic_code": genetic_code,
            }

            return ToolResult(
                success=True,
                message=f"Genome {genome_name} annotated successfully: {stats['num_genes']} genes identified",
                data={
                    "genome_name": genome_name,
                    "annotation_statistics": stats,
                    "output_path": str(output_file) if output_file else None,
                    "genome_object": genome,  # Include genome object for downstream tools
                    "annotation_result": annotation_result,
                },
                metadata={
                    "tool_type": "genome_annotation",
                    "annotation_source": "RAST",
                    "organism_type": organism_type,
                    "num_genes": stats["num_genes"],
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error during genome annotation: {str(e)}",
                error=str(e),
            )


@ToolRegistry.register
class ProteinAnnotationTool(BaseTool):
    """Tool for protein sequence annotation using RAST"""

    tool_name = "annotate_proteins_rast"
    tool_description = """Annotate protein sequences using RAST functional annotation.
    Useful for annotating individual proteins or protein sets."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def _run_tool(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Annotate protein sequences using RAST.

        Args:
            input_data: Dictionary containing:
                - protein_sequences: List of protein sequences or path to FASTA file
                - output_path: Where to save the annotation results

        Returns:
            ToolResult containing the protein annotation results
        """
        try:
            # Lazy import modelseedpy only when needed
            import modelseedpy

            # Initialize RAST client
            rast_client = modelseedpy.RastClient()

            # Handle protein input
            sequences = []
            if "protein_sequences" in input_data:
                if isinstance(input_data["protein_sequences"], list):
                    sequences = input_data["protein_sequences"]
                elif isinstance(input_data["protein_sequences"], str):
                    # Assume it's a file path
                    protein_file = Path(input_data["protein_sequences"])
                    if protein_file.exists():
                        # Read FASTA file
                        from Bio import SeqIO

                        sequences = [
                            str(record.seq)
                            for record in SeqIO.parse(protein_file, "fasta")
                        ]
                    else:
                        # Assume it's a single sequence
                        sequences = [input_data["protein_sequences"]]
            else:
                raise ValueError("protein_sequences must be provided")

            # Annotate protein sequences
            if len(sequences) == 1:
                annotation_result = rast_client.annotate_protein_sequence(sequences[0])
                annotations = [annotation_result]
            else:
                annotations = rast_client.annotate_protein_sequences(sequences)

            # Process results
            processed_annotations = []
            for i, annotation in enumerate(annotations):
                processed_annotations.append(
                    {
                        "sequence_index": i,
                        "function": annotation.get("function", "Unknown"),
                        "confidence": annotation.get("confidence", 0),
                        "subsystem": annotation.get("subsystem", ""),
                        "ec_numbers": annotation.get("ec_numbers", []),
                    }
                )

            # Save results if output path provided
            output_file = None
            if "output_path" in input_data:
                output_file = Path(input_data["output_path"])
                output_file.parent.mkdir(parents=True, exist_ok=True)

                import json

                with open(output_file, "w") as f:
                    json.dump(processed_annotations, f, indent=2)

            # Gather statistics
            stats = {
                "num_sequences": len(sequences),
                "num_annotated": len(
                    [a for a in processed_annotations if a["function"] != "Unknown"]
                ),
                "num_with_ec": len(
                    [a for a in processed_annotations if a["ec_numbers"]]
                ),
                "num_with_subsystem": len(
                    [a for a in processed_annotations if a["subsystem"]]
                ),
            }

            return ToolResult(
                success=True,
                message=f"Annotated {stats['num_annotated']}/{stats['num_sequences']} protein sequences",
                data={
                    "annotations": processed_annotations,
                    "statistics": stats,
                    "output_path": str(output_file) if output_file else None,
                },
                metadata={
                    "tool_type": "protein_annotation",
                    "annotation_source": "RAST",
                    "num_sequences": stats["num_sequences"],
                    "success_rate": stats["num_annotated"] / stats["num_sequences"],
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error during protein annotation: {str(e)}",
                error=str(e),
            )
