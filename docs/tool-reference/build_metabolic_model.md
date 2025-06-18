# `build_metabolic_model`

Tool for building metabolic models using ModelSEED with MSGenome support

## Import

```python
from src.tools.modelseed.builder import ModelBuildTool
````

## Parameters

| Name | Type | Description |
|-----|------|-------------|
| input_data | typing.Dict[str, typing.Any] | Input data containing genome object or MSGenome |

## Usage

The ModelBuildTool now supports enhanced model building with MSGenome:

### Standard Model Building
```python
result = tool.execute({
    "genome_object": genome_obj,
    "media": "complete"  # optional
})
```

### MSGenome-based Model Building
The tool automatically handles MSGenome objects when passed as the genome_object, providing improved model construction with better gene-protein-reaction associations.

## Features

- Supports both standard genome objects and MSGenome objects
- Improved gene-protein-reaction (GPR) associations
- Better handling of protein complexes and metabolic pathways
- Automatic template selection based on organism type
