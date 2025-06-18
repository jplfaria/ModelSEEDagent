from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..base import BaseTool, ToolRegistry, ToolResult


class RastAnnotationConfig(BaseModel):
    """Configuration for RAST protein annotation"""

    genome_domain: str = "Bacteria"
    genetic_code: int = 11
    include_quality_control: bool = True
    generate_clean_fasta: bool = False
    additional_config: Dict[str, Any] = Field(default_factory=dict)


@ToolRegistry.register
class RastAnnotationTool(BaseTool):
    """Modern tool for protein annotation using RAST with MSGenome"""

    tool_name = "annotate_proteins_rast"
    tool_description = """Annotate protein sequences using RAST (Rapid Annotation using Subsystem Technology) service.
    Uses MSGenome.from_fasta() for clean protein annotation workflow. Returns annotated genome object for model building."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Use private attribute to avoid Pydantic field conflicts
        self._annotation_config = RastAnnotationConfig(
            **config.get("annotation_config", {})
        )

    def _clean_function_string(self, function_str: str) -> str:
        """Clean function string by removing problematic characters"""
        if not function_str:
            return ""
            
        # Clean the function string - remove leading semicolons/spaces
        function_clean = function_str.strip()
        if function_clean.startswith(';'):
            function_clean = function_clean[1:].strip()
            
        # Remove brackets and other problematic characters
        function_clean = function_clean.replace('[', '').replace(']', '')
        
        return function_clean

    def _generate_clean_fasta(self, genome, output_path: str) -> str:
        """Generate clean annotated FASTA file without brackets/semicolons"""
        clean_fasta_path = output_path
        
        with open(clean_fasta_path, 'w') as f:
            for feature in genome.features:
                # Create a header with ID and function
                header = f">{feature.id}"
                
                if hasattr(feature, 'function') and feature.function:
                    # Clean the function string
                    function_clean = self._clean_function_string(feature.function)
                    if function_clean:
                        header += f" {function_clean}"
                
                # Only add ontology terms if they exist AND no function was found
                elif hasattr(feature, 'ontology_terms') and 'RAST' in feature.ontology_terms:
                    rast_terms = ' '.join(feature.ontology_terms['RAST'])  # Space instead of semicolon
                    header += f" {rast_terms}"
                
                # Write header and sequence
                f.write(header + '\n')
                if hasattr(feature, 'seq') and feature.seq:
                    # Write sequence in 60-character lines
                    seq = str(feature.seq)
                    for i in range(0, len(seq), 60):
                        f.write(seq[i:i+60] + '\n')
                        
        return clean_fasta_path

    def _run_tool(self, input_data: Dict[str, Any]) -> ToolResult:
        """
        Annotate protein sequences using RAST with modern MSGenome approach.

        Args:
            input_data: Dictionary containing:
                - protein_file: Path to protein FASTA file (.faa)
                - genome_name: Name for the genome (optional)
                - output_fasta: Path to save clean annotated FASTA (optional)
                - generate_clean_fasta: Whether to generate clean FASTA output

        Returns:
            ToolResult containing the annotation results and annotated genome object
        """
        try:
            # Validate inputs
            protein_file = Path(input_data["protein_file"])
            if not protein_file.exists():
                raise FileNotFoundError(f"Protein file not found: {protein_file}")

            genome_name = input_data.get("genome_name", protein_file.stem)
            
            # Lazy import modelseedpy only when needed
            import modelseedpy
            from modelseedpy import MSGenome

            # Create MSGenome from protein FASTA - modern approach
            genome = MSGenome.from_fasta(str(protein_file))
            
            # Initialize RAST client and annotate
            rast_client = modelseedpy.RastClient()
            annotation_result = rast_client.annotate_genome(genome)
            
            # Generate clean FASTA output if requested
            output_fasta_path = None
            if (input_data.get("generate_clean_fasta", self._annotation_config.generate_clean_fasta) or 
                "output_fasta" in input_data):
                output_path = input_data.get("output_fasta", f"{genome_name}_annotated.faa")
                output_fasta_path = self._generate_clean_fasta(genome, output_path)

            # Gather annotation statistics
            stats = {
                "genome_name": genome_name,
                "num_features": len(genome.features) if hasattr(genome, "features") else 0,
                "num_annotated": len([f for f in genome.features if hasattr(f, 'function') and f.function]) if hasattr(genome, "features") else 0,
                "annotation_source": "RAST",
                "annotation_jobs": annotation_result if isinstance(annotation_result, list) else [],
            }
            
            # Calculate annotation success rate
            success_rate = (stats["num_annotated"] / stats["num_features"]) if stats["num_features"] > 0 else 0

            return ToolResult(
                success=True,
                message=f"Protein annotation completed: {stats['num_annotated']}/{stats['num_features']} proteins annotated ({success_rate:.1%} success rate)",
                data={
                    "genome_name": genome_name,
                    "annotation_statistics": stats,
                    "output_fasta_path": output_fasta_path,
                    "genome_object": genome,  # Key output for downstream model building
                    "annotation_jobs": annotation_result,
                    "success_rate": success_rate,
                },
                metadata={
                    "tool_type": "protein_annotation",
                    "annotation_source": "RAST",
                    "num_features": stats["num_features"],
                    "num_annotated": stats["num_annotated"],
                    "success_rate": success_rate,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error during protein annotation: {str(e)}",
                error=str(e),
            )
