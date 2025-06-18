# `annotate_genome_rast`

Tool for genome and protein FASTA annotation using RAST

## Import

```python
from src.tools.modelseed.annotation import RastAnnotationTool
````

## Parameters

| Name | Type | Description |
|-----|------|-------------|
| input_data | typing.Dict[str, typing.Any] | Input data containing either genome_object or protein_fasta_path |

## Usage

The RastAnnotationTool now supports both genome annotation and protein FASTA annotation:

### Genome Annotation
```python
result = tool.execute({
    "genome_object": genome_obj
})
```

### Protein FASTA Annotation
```python
result = tool.execute({
    "protein_fasta_path": "/path/to/proteins.fasta"
})
```

The tool automatically detects the input type and uses the appropriate RAST annotation method.
