# `gapfill_model`

Tool for gap filling metabolic models using ModelSEED with improved API

## Import

```python
from src.tools.modelseed.gapfill import GapFillTool
````

## Parameters

| Name | Type | Description |
|-----|------|-------------|
| input_data | typing.Dict[str, typing.Any] | Input data containing model and optional media/target |

## Usage

The GapFillTool provides gap filling functionality with an improved API:

```python
result = tool.execute({
    "model": model_obj,
    "media": "complete",  # optional, defaults to "complete"
    "target": None  # optional target reaction to enable
})
```

## Features

- Automated gap filling to enable biomass production
- Support for custom media conditions
- Target reaction specification for specific gap filling goals
- Returns both the gap-filled model and list of added reactions
- Improved error handling and validation
