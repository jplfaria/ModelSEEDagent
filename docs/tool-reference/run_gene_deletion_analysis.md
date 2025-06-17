# `run_gene_deletion_analysis`

Tool for running single and double gene deletion analysis to identify essential genes and predict the effect of gene knockouts on model growth.

## Import

```python
from src.tools.cobra.gene_deletion import GeneDeletionTool
```

## Parameters

| Name | Type | Description |
|-----|------|-------------|
| model_path | str | Path to the metabolic model file (SBML, JSON, MAT) |
| gene_list | List[str] | Optional. List of gene IDs to test. If None, tests all genes |
| deletion_type | str | Type of deletion: "single" or "double" (default: "single") |
| method | str | FBA method: "fba", "moma", or "room" (default: "fba") |

## Smart Summarization Output

This tool implements the Smart Summarization Framework with focus on essential genes and growth impacts:

### 1. Key Findings (≤2KB)
Critical gene deletion insights:
- Essential gene counts and percentages
- Growth impact categorization
- Essentiality rate assessment
- Key essential gene examples
- Beneficial deletions identified

### 2. Summary Dict (≤5KB)
Structured gene categorization:
- Gene counts by impact category
- Essential gene examples (top 10)
- Criticality analysis
- Growth distribution statistics
- Robustness metrics

### 3. Full Data Path
Complete deletion results stored as JSON with growth rates for all tested genes.

## Example with iML1515

```python
# Run single gene deletion on E. coli iML1515 model
result = metabolic_agent.run_tool(
    "run_gene_deletion_analysis",
    {
        "model_path": "models/iML1515.xml",
        "deletion_type": "single"
    }
)

# Example output structure (with Smart Summarization enabled):
{
    "success": true,
    "message": "Gene deletion analysis summarized: 1516 genes tested",
    "key_findings": [
        "Gene deletion analysis of iML1515: 1516 genes tested",
        "Wild-type growth rate: 0.874",
        "Essential genes: 148 (9.8%) - lethal when deleted",
        "Growth-impaired genes: 289 (19.1%) - reduce growth",
        "Non-essential genes: 1067 (70.4%) - minimal impact",
        "✓ Normal essentiality rate for metabolic model",
        "Growth-improving deletions: 12 (0.8%)",
        "Key essential genes: b0025, b0114, b0115, b0116, b0118",
        "Severely impaired genes: 67 (growth 1-10%)",
        "Moderately impaired genes: 98 (growth 10-50%)",
        "Mildly impaired genes: 124 (growth 50-90%)"
    ],
    "summary_dict": {
        "deletion_statistics": {
            "total_genes_tested": 1516,
            "wild_type_growth": 0.8739,
            "model_coverage": 1.0
        },
        "gene_categories": {
            "essential": {
                "count": 148,
                "percentage": 9.76,
                "examples": [
                    "b0025", "b0114", "b0115", "b0116", "b0118",
                    "b0142", "b0170", "b0171", "b0323", "b0351"
                ]
            },
            "severely_impaired": {
                "count": 67,
                "percentage": 4.42,
                "examples": ["b0356", "b0474", "b0485", "b0726", "b0727"]
            },
            "moderately_impaired": {
                "count": 98,
                "percentage": 6.46
            },
            "mildly_impaired": {
                "count": 124,
                "percentage": 8.18
            },
            "no_effect": {
                "count": 1067,
                "percentage": 70.38
            },
            "improved_growth": {
                "count": 12,
                "percentage": 0.79,
                "examples": ["b1241", "b1380", "b2029", "b3236", "b3919"]
            }
        },
        "essentiality_analysis": {
            "essentiality_rate": 0.0976,
            "essentiality_category": "normal",
            "total_critical_genes": 215,
            "robustness_score": 0.7038
        },
        "growth_impact_distribution": {
            "mean_growth_retention": 0.8234,
            "min_growth_retention": 0.0,
            "max_growth_retention": 1.0523,
            "lethal_deletions": 148,
            "beneficial_deletions": 12
        },
        "critical_genes": {
            "total_critical": 215,
            "critical_gene_details": [
                {"gene_id": "b0025", "growth_ratio": 0.0, "category": "essential"},
                {"gene_id": "b0114", "growth_ratio": 0.0, "category": "essential"},
                {"gene_id": "b0115", "growth_ratio": 0.0, "category": "essential"},
                {"gene_id": "b0116", "growth_ratio": 0.0, "category": "essential"},
                {"gene_id": "b0118", "growth_ratio": 0.0, "category": "essential"},
                {"gene_id": "b0356", "growth_ratio": 0.0234, "category": "severely_impaired"},
                {"gene_id": "b0474", "growth_ratio": 0.0345, "category": "severely_impaired"},
                {"gene_id": "b0485", "growth_ratio": 0.0456, "category": "severely_impaired"},
                {"gene_id": "b0726", "growth_ratio": 0.0567, "category": "severely_impaired"},
                {"gene_id": "b0727", "growth_ratio": 0.0678, "category": "severely_impaired"},
                {"gene_id": "b0142", "growth_ratio": 0.0, "category": "essential"},
                {"gene_id": "b0170", "growth_ratio": 0.0, "category": "essential"},
                {"gene_id": "b0171", "growth_ratio": 0.0, "category": "essential"},
                {"gene_id": "b0323", "growth_ratio": 0.0, "category": "essential"},
                {"gene_id": "b0351", "growth_ratio": 0.0, "category": "essential"}
            ]
        },
        "model_context": {
            "model_id": "iML1515",
            "num_reactions": 2712,
            "num_genes": 1516,
            "num_metabolites": 1877
        },
        "analysis_metadata": {
            "deletion_method": "systematic_gene_knockout",
            "framework_version": "1.0",
            "growth_thresholds": {
                "essential": "< 1% wild-type",
                "severe": "1-10% wild-type",
                "moderate": "10-50% wild-type",
                "mild": "50-90% wild-type"
            }
        }
    },
    "full_data_path": "/tmp/modelseed_artifacts/run_gene_deletion_analysis_iML1515_20250117_144523_c9d5e1f3.json",
    "tool_name": "run_gene_deletion_analysis",
    "schema_version": "1.0"
}
```

## Size Reduction Achieved

- **Original gene deletion output**: ~130 KB (for iML1515)
- **Summarized output**: ~3.1 KB (key_findings + summary_dict)
- **Reduction**: 97.6% while preserving essential gene information
- **Full data**: Available at `full_data_path` for detailed analysis

## Accessing Full Data

```python
# To access the complete gene deletion results:
import json

with open(result["full_data_path"], 'r') as f:
    full_deletion_data = json.load(f)
    
# Full data structure:
{
    "b0001": {
        "growth": 0.8739,
        "growth_ratio": 1.0
    },
    "b0002": {
        "growth": 0.8739,
        "growth_ratio": 1.0
    },
    "b0025": {
        "growth": 0.0,
        "growth_ratio": 0.0
    },
    // ... all 1516 genes with growth data
}

# Find all essential genes:
essential_genes = [
    gene_id for gene_id, data in full_deletion_data.items()
    if data["growth_ratio"] < 0.01
]
```

## Notes

- Essential genes are critical for organism survival
- The essentiality rate (~10% for E. coli) indicates model quality
- Growth-improving deletions suggest regulatory constraints
- Full deletion data enables pathway-specific analysis
