# `run_flux_sampling`

Tool for statistical flux sampling to explore the metabolic solution space and understand flux distributions across the network.

## Import

```python
from src.tools.cobra.flux_sampling import FluxSamplingTool
```

## Parameters

| Name | Type | Description |
|-----|------|-------------|
| model_path | str | Path to the metabolic model file (SBML, JSON, MAT) |
| n_samples | int | Number of flux samples to generate (default: 1000) |
| method | str | Sampling method: "optgp" or "achr" (default: "optgp") |
| thinning | int | Thinning parameter for sampling (default: 100) |
| seed | int | Optional. Random seed for reproducibility |

## Smart Summarization Output

This tool implements the Smart Summarization Framework with extreme compression (99.9% reduction) due to massive sampling data:

### 1. Key Findings (≤2KB)
Critical insights from flux distributions:
- Sample statistics and coverage
- Flux pattern categorization
- High correlation discoveries
- Most active subsystems
- Growth rate variability

### 2. Summary Dict (≤5KB)
Compressed statistical analysis:
- Flux pattern counts and examples
- Top correlated reaction pairs
- Subsystem activity summary
- Distribution statistics
- Optimization potential

### 3. Full Data Path
Complete sampling DataFrame stored as JSON containing all samples for all reactions.

## Example with iML1515

```python
# Run flux sampling on E. coli iML1515 model
result = metabolic_agent.run_tool(
    "run_flux_sampling",
    {
        "model_path": "models/iML1515.xml",
        "n_samples": 5000,
        "method": "optgp"
    }
)

# Example output structure (with Smart Summarization enabled):
{
    "success": true,
    "message": "Flux sampling summarized: 5000 samples across 2712 reactions",
    "key_findings": [
        "Flux sampling of iML1515: 5000 samples across 2712 reactions",
        "Sampling method explored metabolic solution space distribution",
        "Always active reactions: 187 (6.9%) - consistently carry flux",
        "Variable reactions: 823 (30.3%) - flux varies significantly",
        "Rarely active reactions: 298 (11.0%) - infrequent flux",
        "Most variable reactions: ACALD, ACALDt, ACKr",
        "High flux correlations found: 47 reaction pairs",
        "Strongest correlation: 0.987 between key reactions",
        "Most active subsystem: Glycolysis/Gluconeogenesis",
        "Growth variability: mean=0.877, CV=0.045"
    ],
    "summary_dict": {
        "sampling_statistics": {
            "total_samples": 5000,
            "total_reactions": 2712,
            "data_reduction_achieved": "99.9%",
            "sampling_coverage": 1.0
        },
        "flux_pattern_summary": {
            "always_active_count": 187,
            "variable_reactions_count": 823,
            "rarely_active_count": 298,
            "top_variable_reactions": [
                {"reaction_id": "R_ACALD", "std_dev": 45.23, "cv": 2.34},
                {"reaction_id": "R_ACALDt", "std_dev": 38.91, "cv": 1.87},
                {"reaction_id": "R_ACKr", "std_dev": 32.45, "cv": 1.56},
                {"reaction_id": "R_PTAr", "std_dev": 28.12, "cv": 1.43},
                {"reaction_id": "R_PFL", "std_dev": 24.78, "cv": 1.21}
            ]
        },
        "correlation_summary": {
            "high_correlation_pairs": 47,
            "strongest_correlations": [
                {
                    "reaction_pair": "R_ACALD <-> R_ACALDt",
                    "correlation": 0.987
                },
                {
                    "reaction_pair": "R_ACKr <-> R_PTAr", 
                    "correlation": 0.965
                },
                {
                    "reaction_pair": "R_SUCDi <-> R_SUCD4",
                    "correlation": -0.943
                }
            ]
        },
        "subsystem_summary": {
            "total_subsystems": 89,
            "top_active_subsystems": [
                {
                    "subsystem": "Glycolysis/Gluconeogenesis",
                    "num_reactions": 26,
                    "avg_flux": 12.34
                },
                {
                    "subsystem": "Citric Acid Cycle",
                    "num_reactions": 20,
                    "avg_flux": 8.91
                },
                {
                    "subsystem": "Pentose Phosphate Pathway",
                    "num_reactions": 21,
                    "avg_flux": 6.78
                },
                {
                    "subsystem": "Oxidative Phosphorylation",
                    "num_reactions": 45,
                    "avg_flux": 5.23
                },
                {
                    "subsystem": "Amino Acid Metabolism",
                    "num_reactions": 189,
                    "avg_flux": 3.45
                }
            ]
        },
        "distribution_summary": {
            "objective_function": {
                "mean": 0.8765,
                "std": 0.0394,
                "range": [0.7234, 0.9876]
            },
            "sample_coverage": {
                "reactions_sampled": 2712,
                "total_samples": 5000
            }
        },
        "model_context": {
            "model_id": "iML1515",
            "num_reactions": 2712,
            "num_genes": 1516,
            "num_metabolites": 1877
        },
        "analysis_metadata": {
            "sampling_method": "statistical_flux_sampling",
            "framework_version": "1.0",
            "artifact_size_mb": 138.5
        }
    },
    "full_data_path": "/tmp/modelseed_artifacts/run_flux_sampling_iML1515_20250117_143855_b8c4d0e2.json",
    "tool_name": "run_flux_sampling",
    "schema_version": "1.0"
}
```

## Size Reduction Achieved

- **Original sampling output**: ~138.5 MB (for iML1515 with 5000 samples)
- **Summarized output**: ~2.2 KB (key_findings + summary_dict)
- **Reduction**: 99.998% while preserving statistical insights
- **Full data**: Available at `full_data_path` for detailed analysis

## Accessing Full Data

```python
# To access the complete sampling results:
import json
import pandas as pd

with open(result["full_data_path"], 'r') as f:
    full_sampling_data = json.load(f)
    
# Convert to DataFrame for analysis
samples_df = pd.DataFrame(full_sampling_data)

# Full data structure:
{
    "R_ACALD": [0.0, -12.3, -45.6, ...],  # 5000 samples
    "R_ACALDt": [0.0, -11.2, -42.1, ...], # 5000 samples
    // ... all 2712 reactions with 5000 samples each
}

# Perform detailed analysis on full data:
flux_correlations = samples_df.corr()
flux_distributions = samples_df.describe()
```

## Notes

- Flux sampling generates massive datasets (25MB+ for typical models)
- Smart summarization achieves 99.9%+ compression while preserving key insights
- Full sampling data enables detailed statistical analysis when needed
- Consider using fewer samples (100-500) for quick exploration
