# `run_flux_variability_analysis`

Tool for running Flux Variability Analysis (FVA) on metabolic models to determine the minimum and maximum possible flux through each reaction while maintaining optimal growth.

## Import

```python
from src.tools.cobra.flux_variability import FluxVariabilityTool
```

## Parameters

| Name | Type | Description |
|-----|------|-------------|
| model_path | str | Path to the metabolic model file (SBML, JSON, MAT) |
| reaction_list | List[str] | Optional. List of reaction IDs to analyze. If None, analyzes all reactions |
| loopless | bool | Optional. If True, performs loopless FVA (default: False) |
| fraction_of_optimum | float | Optional. Fraction of maximum objective value to maintain (default: 1.0) |

## Smart Summarization Output

This tool implements the Smart Summarization Framework, providing three levels of information:

### 1. Key Findings (≤2KB)
Critical insights for immediate understanding:
- Reaction variability statistics
- Network flexibility assessment
- Critical blocked reactions
- Top variable reactions

### 2. Summary Dict (≤5KB)
Structured data for analysis:
- Reaction counts by category
- Top 10 reactions per category
- Flux statistics
- Smart bucketing results

### 3. Full Data Path
Complete FVA results stored as JSON file containing all reaction min/max values.

## Example with iML1515

```python
# Run FVA on E. coli iML1515 model
result = metabolic_agent.run_tool(
    "run_flux_variability_analysis",
    {"model_path": "models/iML1515.xml"}
)

# Example output structure (with Smart Summarization enabled):
{
    "success": true,
    "message": "FVA analysis summarized: 2712 reactions analyzed",
    "key_findings": [
        "FVA analysis of iML1515: 2712 reactions analyzed",
        "Blocked reactions: 678 (25.0%) - cannot carry flux",
        "Variable reactions: 542 (20.0%) - flux can vary",
        "Fixed reactions: 1492 (55.0%) - carry fixed flux",
        "Essential reactions: 389 (14.3%) - required for growth",
        "✓ High flux variability - significant metabolic flexibility",
        "Top variable reactions: ACALD, ACALDt, ACKr",
        "High optimization potential: 282 highly variable reactions"
    ],
    "summary_dict": {
        "statistics": {
            "total_reactions": 2712,
            "blocked_count": 678,
            "variable_count": 542,
            "fixed_count": 1492,
            "essential_count": 389,
            "network_flexibility_score": 0.200
        },
        "reaction_categories": {
            "blocked_reactions": ["R_DBTS", "R_PPPGO", "R_DMATT", ...],  // Top 10
            "variable_reactions": ["R_ACALD", "R_ACALDt", "R_ACKr", ...],  // Top 10
            "essential_reactions": ["R_ATPS4r", "R_CYTBD", "R_ENO", ...]  // Top 10
        },
        "flux_statistics": {
            "min_flux_observed": -1000.0,
            "max_flux_observed": 1000.0,
            "avg_flux_range": 23.45
        },
        "smart_flux_buckets": {
            "bucketing_thresholds": {
                "high_variability": 70.35,
                "medium_variability": 11.73,
                "low_variability": 1.17,
                "max_range_observed": 234.5,
                "mean_range": 23.45
            },
            "variability_categories": {
                "high_variability": {
                    "count": 45,
                    "reactions": [
                        {"reaction_id": "R_ACALD", "flux_range": 234.5},
                        {"reaction_id": "R_ACALDt", "flux_range": 187.3},
                        {"reaction_id": "R_ACKr", "flux_range": 156.8},
                        {"reaction_id": "R_PTAr", "flux_range": 145.2},
                        {"reaction_id": "R_PFL", "flux_range": 132.7}
                    ],
                    "description": "Major flux alternatives - optimization targets"
                },
                "medium_variability": {
                    "count": 237,
                    "reactions": [
                        {"reaction_id": "R_SUCD1i", "flux_range": 45.2},
                        {"reaction_id": "R_FRD7", "flux_range": 38.7},
                        {"reaction_id": "R_NADH16", "flux_range": 32.1}
                    ],
                    "description": "Moderate flux flexibility - adaptation pathways"
                }
            },
            "insights": {
                "total_variable_reactions": 542,
                "optimization_potential": 282,
                "flexibility_distribution": {
                    "high_flex_pct": 8.3,
                    "medium_flex_pct": 43.7,
                    "low_flex_pct": 48.0
                }
            }
        },
        "model_context": {
            "model_id": "iML1515",
            "num_reactions": 2712,
            "num_genes": 1516,
            "num_metabolites": 1877
        },
        "analysis_metadata": {
            "fva_method": "flux_variability_analysis",
            "categorization_threshold": 1e-6,
            "framework_version": "1.0"
        }
    },
    "full_data_path": "/tmp/modelseed_artifacts/run_flux_variability_analysis_iML1515_20250117_143022_a7b3c9d1.json",
    "tool_name": "run_flux_variability_analysis",
    "schema_version": "1.0"
}
```

## Size Reduction Achieved

- **Original FVA output**: ~170 KB (for iML1515)
- **Summarized output**: ~2.4 KB (key_findings + summary_dict)
- **Reduction**: 98.6% while preserving all critical insights
- **Full data**: Available at `full_data_path` for detailed analysis

## Accessing Full Data

```python
# To access the complete FVA results:
import json

with open(result["full_data_path"], 'r') as f:
    full_fva_data = json.load(f)
    
# Full data structure:
{
    "minimum": {
        "R_ACALD": -234.5,
        "R_ACALDt": -187.3,
        // ... all 2712 reactions
    },
    "maximum": {
        "R_ACALD": 0.0,
        "R_ACALDt": 0.0,
        // ... all 2712 reactions
    }
}
```
