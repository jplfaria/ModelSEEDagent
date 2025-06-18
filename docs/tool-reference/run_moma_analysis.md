# `run_moma_analysis`

Tool for MOMA (Minimization of Metabolic Adjustment) analysis to predict metabolic adjustments after genetic perturbations.

## Import

```python
from src.tools.cobra.moma import MOMATool
```

## Parameters

| Name | Type | Description |
|-----|------|-------------|
| model_path | str | Path to SBML model file |
| knockout_genes | List[str] | List of gene IDs to knock out |
| knockout_reactions | List[str] | List of reaction IDs to knock out |
| linear | bool | Use linear MOMA (True) vs quadratic MOMA (False) |
| compare_to_fba | bool | Compare MOMA results to standard FBA |
| solver | str | Solver to use for optimization (default: glpk) |

## Purpose

MOMA (Minimization of Metabolic Adjustment) finds flux distributions that minimize metabolic changes compared to wild-type after genetic perturbations. This provides more realistic predictions of metabolic responses than standard FBA.

## Key Features

- Gene and reaction knockout simulations
- Linear (faster) and quadratic MOMA variants
- Comparison with standard FBA predictions
- Detailed metabolic adjustment metrics
- Identification of most affected reactions

## Use Cases

- Predicting realistic metabolic responses to genetic modifications
- Understanding metabolic adaptation strategies
- Comparing different knockout strategies
- Identifying key reactions affected by perturbations

## Output

- Growth rates (wild-type vs MOMA prediction)
- Growth fraction and viability assessment
- Metabolic adjustment metrics (total flux adjustment, reactions changed)
- Most affected reactions with flux changes
- Optional comparison with FBA results
- Flux distributions for small models

## Example Usage

```python
tool = MOMATool(config)
result = tool._run({
    "model_path": "path/to/model.xml",
    "knockout_genes": ["b0008", "b0116"],
    "linear": True,
    "compare_to_fba": True
})
```
