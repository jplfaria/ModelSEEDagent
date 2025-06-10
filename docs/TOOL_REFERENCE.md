# ModelSEED Agent - Comprehensive Tool Reference Documentation

## Overview

This document provides a comprehensive reference for all 19 metabolic modeling tools available in the ModelSEED Agent platform. This documentation details each tool's capabilities, parameters, numerical precision improvements, and practical usage examples.

## Table of Contents

1. [Infrastructure and Framework](#infrastructure-and-framework)
2. [Core COBRA Tools](#core-cobra-tools)
3. [Biochemistry Tools](#biochemistry-tools)
4. [ModelSEED Tools](#modelseed-tools)
5. [RAST Tools](#rast-tools)
6. [Numerical Precision Improvements](#numerical-precision-improvements)
7. [Error Handling Enhancements](#error-handling-enhancements)
8. [Usage Examples](#usage-examples)

---

## Infrastructure and Framework

### Precision Configuration System

The ModelSEED Agent implements a unified numerical precision framework that ensures consistent and reliable calculations across all metabolic modeling tools.

#### PrecisionConfig Class (`src/tools/cobra/precision_config.py`)

**Purpose**: Provides standardized numerical thresholds and safe mathematical operations

**Key Parameters**:
- `model_tolerance`: 1e-9 (solver numerical accuracy)
- `flux_threshold`: 1e-6 (minimum significant flux magnitude)
- `growth_threshold`: 1e-6 (minimum significant growth rate)
- `essentiality_growth_fraction`: 0.01 (1% of wild-type for essentiality)
- `correlation_threshold`: 0.7 (statistical correlation significance)

**Gene Deletion Effect Categories**:
- **Essential**: < 1% of wild-type growth
- **Severe**: 1-10% of wild-type growth
- **Moderate**: 10-50% of wild-type growth
- **Mild**: 50-90% of wild-type growth
- **No Effect**: > 90% of wild-type growth

**Utility Functions**:
- `safe_divide()`: Protected division with numerical tolerance
- `is_significant_flux()`: Flux significance testing
- `calculate_growth_fraction()`: Safe growth ratio calculation

### Error Handling Framework

**Enhanced Error System** (`src/tools/cobra/error_handling.py`):
- Model path validation with intelligent file suggestions
- Tool-specific parameter validation and guidance
- Optimization failure diagnosis with actionable recommendations
- Progress logging for long-running operations
- Standardized error result formatting

---

## Core COBRA Tools

### 1. FBATool - Flux Balance Analysis
**Tool Name**: `run_metabolic_fba`

**Purpose**: Performs standard and advanced Flux Balance Analysis to predict metabolic flux distributions and growth rates.

**Key Features**:
- Multiple simulation methods (FBA, pFBA, geometric FBA, slim optimization)
- Precision-aware flux filtering and reporting
- Comprehensive subsystem flux analysis
- Results export in JSON and CSV formats
- Enhanced biomass flux extraction

**Parameters**:
```python
{
    "model_path": str,              # Path to SBML model file
    "simulation_method": str,       # "fba", "pfba", "geometric", "slim"
    "solver": str,                  # "glpk", "cplex", "gurobi"
    "output_dir": str,              # Optional results export directory
    "flux_threshold": float         # Minimum significant flux (default: 1e-6)
}
```

**Numerical Precision Improvements**:
- **Before**: Used raw objective value (518.422) which was optimization scaling artifact
- **After**: Extracts actual biomass flux (0.8739 h⁻¹) from flux distribution
- **Impact**: Provides biologically meaningful growth rates instead of solver artifacts
- **Filtering**: Only reports fluxes above significance threshold to reduce noise

**Example Usage**:
```python
result = fba_tool.run({
    "model_path": "data/models/e_coli_core.xml",
    "simulation_method": "fba",
    "flux_threshold": 1e-6
})
```

### 2. ModelAnalysisTool - Structural Analysis
**Tool Name**: `analyze_metabolic_model`

**Purpose**: Performs comprehensive structural analysis of metabolic networks including connectivity, dead-ends, and network statistics.

**Key Features**:
- Network connectivity analysis and statistics
- Hub metabolite identification (highly connected compounds)
- Choke point detection (reactions with unique products/substrates)
- Subsystem coverage and completeness assessment
- Dead-end metabolite identification

**Parameters**:
```python
{
    "model_path": str,              # Path to SBML model file
    "min_hub_connections": int,     # Minimum connections for hub metabolites (default: 5)
    "include_cofactors": bool,      # Include cofactors in analysis (default: True)
    "subsystem_analysis": bool      # Perform subsystem statistics (default: True)
}
```

**Numerical Precision Improvements**:
- **Connection Counting**: Uses integer precision for network topology
- **Statistical Calculations**: Safe division for connectivity ratios
- **Threshold Handling**: Configurable minimum connection thresholds

### 3. PathwayAnalysisTool - Pathway Analysis
**Tool Name**: `analyze_pathway`

**Purpose**: Analyzes specific metabolic pathways within the network, identifying pathway components, completeness, and connectivity.

**Key Features**:
- Subsystem-based pathway identification
- Reaction pattern matching for pathway discovery
- Gene association analysis for pathway control
- Input/output metabolite tracking
- Pathway connectivity assessment

**Parameters**:
```python
{
    "model_path": str,              # Path to SBML model file
    "pathway_id": str,              # Pathway/subsystem identifier
    "include_genes": bool,          # Include gene associations (default: True)
    "connectivity_analysis": bool   # Analyze pathway connectivity (default: True)
}
```

### 4. FluxVariabilityTool - Flux Variability Analysis
**Tool Name**: `run_flux_variability_analysis`

**Purpose**: Determines the minimum and maximum possible flux values for each reaction under optimal growth conditions.

**Key Features**:
- Minimum/maximum flux range calculation
- Reaction categorization (fixed, variable, blocked, essential)
- Precision-aware flux range classification
- Loopless FVA support for thermodynamically consistent solutions
- Statistical analysis of flux variability

**Parameters**:
```python
{
    "model_path": str,              # Path to SBML model file
    "reaction_list": list,          # Specific reactions to analyze (optional)
    "fraction_of_optimum": float,   # Fraction of optimal growth (default: 1.0)
    "loopless": bool,              # Use loopless FVA (default: False)
    "processes": int               # Parallel processes (default: 1)
}
```

**Numerical Precision Improvements**:
- **Before**: Fixed 1e-6 tolerance could miss biologically relevant small fluxes
- **After**: Configurable `flux_threshold` based on model characteristics
- **Classification**: Enhanced reaction categorization using precision-aware comparisons
- **Impact**: More accurate identification of truly variable vs. numerically variable reactions

### 5. GeneDeletionTool - Gene Deletion Analysis
**Tool Name**: `run_gene_deletion_analysis`

**Purpose**: Performs systematic gene knockout analysis to assess gene essentiality and growth impact.

**Key Features**:
- Essential gene identification with configurable thresholds
- Growth impact categorization (essential, severe, moderate, mild, no effect)
- Single and double gene deletion support
- Detailed deletion effect statistics
- Subsystem impact analysis

**Parameters**:
```python
{
    "model_path": str,              # Path to SBML model file
    "gene_list": list,              # Specific genes to analyze (optional)
    "deletion_type": str,           # "single" or "double"
    "essential_threshold": float,   # Growth fraction for essentiality (default: 0.01)
    "processes": int               # Parallel processes (default: 1)
}
```

**Numerical Precision Improvements**:
- **Before**: Used hard-coded 0.01 threshold inconsistent with other tools
- **After**: Unified `essential_growth_fraction` from PrecisionConfig
- **Effect Categories**: Standardized growth fraction thresholds across all tools
- **Impact**: Consistent essentiality classification regardless of which tool is used

### 6. EssentialityAnalysisTool - Comprehensive Essentiality
**Tool Name**: `analyze_essentiality`

**Purpose**: Comprehensive essentiality analysis for both genes and reactions with functional categorization.

**Key Features**:
- Gene essentiality with functional pathway categorization
- Reaction essentiality with network impact analysis
- Subsystem essentiality statistics
- Essential component identification across metabolic categories
- Cross-validation between gene and reaction essentiality

**Parameters**:
```python
{
    "model_path": str,              # Path to SBML model file
    "analysis_type": str,           # "genes", "reactions", or "both"
    "essential_threshold": float,   # Growth fraction threshold (default: 0.01)
    "categorize_functions": bool    # Functional categorization (default: True)
}
```

**Numerical Precision Improvements**:
- **Unified Thresholds**: Uses PrecisionConfig for consistent essentiality classification
- **Safe Calculations**: Protected growth fraction calculations
- **Statistical Robustness**: Enhanced numerical stability in essentiality scoring

### 7. FluxSamplingTool - Statistical Flux Analysis
**Tool Name**: `run_flux_sampling`

**Purpose**: Statistical exploration of the flux solution space to understand flux distributions and correlations.

**Key Features**:
- Multiple sampling methods (OptGP, ACHR)
- Flux variability pattern identification
- Statistical correlation analysis between reactions
- Subsystem activity assessment
- Distribution statistics and trend analysis

**Parameters**:
```python
{
    "model_path": str,              # Path to SBML model file
    "n_samples": int,               # Number of samples (default: 1000)
    "method": str,                  # "optgp" or "achr"
    "thinning": int,               # Thinning parameter (default: 100)
    "seed": int,                   # Random seed (optional)
    "processes": int               # Parallel processes (optional)
}
```

**Numerical Precision Improvements**:
- **Variable.id Compatibility**: Fixed COBRApy API compatibility issue
- **Statistical Robustness**: Enhanced correlation calculations with safe division
- **Significance Testing**: Configurable flux significance thresholds
- **Error Handling**: Comprehensive validation and helpful error messages

### 8. ProductionEnvelopeTool - Production Analysis
**Tool Name**: `run_production_envelope`

**Purpose**: Analyzes growth vs. production trade-offs for metabolic engineering design.

**Key Features**:
- Multi-dimensional production envelope calculation
- Trade-off severity assessment (positive, negative, neutral)
- Design point identification for metabolic engineering
- Pareto-optimal point finding
- Carbon source constraint configuration

**Parameters**:
```python
{
    "model_path": str,              # Path to SBML model file
    "reactions": list,              # Reactions to analyze for production
    "points": int,                  # Number of envelope points (default: 20)
    "objective": str,               # Objective reaction (optional)
    "c_source": str,               # Carbon source (optional)
    "c_uptake_rates": float        # Carbon uptake rate (optional)
}
```

**Numerical Precision Improvements**:
- **Enhanced Validation**: Comprehensive reaction existence checking
- **Smart Suggestions**: Similar reaction name suggestions for missing reactions
- **Error Context**: Detailed error information for troubleshooting

### 9. AuxotrophyTool - Nutrient Dependency Analysis
**Tool Name**: `identify_auxotrophies`

**Purpose**: Identifies nutrient dependencies by systematically testing metabolite requirements.

**Key Features**:
- Candidate metabolite dependency testing
- Growth impact assessment for each tested metabolite
- Auxotrophy classification and severity scoring
- Results export with detailed dependency information

**Parameters**:
```python
{
    "model_path": str,              # Path to SBML model file
    "candidate_metabolites": list,  # Metabolites to test (optional)
    "growth_threshold": float,      # Minimum growth for viability (default: 1e-6)
    "supplement_amount": float      # Amount to supplement (default: 10.0)
}
```

### 10. MinimalMediaTool - Essential Nutrients
**Tool Name**: `find_minimal_media`

**Purpose**: Identifies the minimal set of nutrients required for growth.

**Key Features**:
- Essential vs. non-essential nutrient classification
- Systematic nutrient removal testing
- Growth threshold validation with precision awareness
- Media composition optimization

**Parameters**:
```python
{
    "model_path": str,              # Path to SBML model file
    "growth_threshold": float,      # Minimum growth threshold (default: 1e-6)
    "test_all_exchanges": bool,     # Test all exchange reactions (default: True)
    "export_results": bool         # Export minimal media (default: False)
}
```

**Numerical Precision Improvements**:
- **Algorithm Rewrite**: Complete rewrite to fix state restoration issues
- **Precision-Aware Testing**: Uses configurable growth thresholds
- **Robust State Management**: Proper bound restoration between tests

### 11. MissingMediaTool - Media Gap Analysis
**Tool Name**: `check_missing_media`

**Purpose**: Identifies missing essential nutrients when growth is suboptimal.

**Key Features**:
- Essential metabolite supplementation testing
- Growth recovery assessment
- Media gap identification and reporting
- Progress logging for systematic testing

**Parameters**:
```python
{
    "model_path": str,              # Path to SBML model file
    "growth_threshold": float,      # Minimum growth threshold (default: 1e-6)
    "essential_metabolites": list,  # Metabolites to test (default list provided)
    "supplementation_amount": float # Supplement amount (default: 10.0)
}
```

**Numerical Precision Improvements**:
- **Enhanced Error Handling**: Comprehensive input validation and progress tracking
- **Safe Optimization**: Protected FBA calls with better error diagnosis
- **Configurable Thresholds**: Precision-aware growth significance testing

### 12. ReactionExpressionTool - Activity Analysis
**Tool Name**: `analyze_reaction_expression`

**Purpose**: Analyzes reaction activity levels using parsimonious FBA for minimal flux distributions.

**Key Features**:
- Parsimonious flux distribution calculation
- Active reaction identification and scoring
- Media condition support for different environments
- Expression level quantification across the network

**Parameters**:
```python
{
    "model_path": str,              # Path to SBML model file
    "medium": dict,                 # Medium composition (optional)
    "objective": str,               # Objective reaction (optional)
    "expression_threshold": float   # Minimum expression level (default: 1e-6)
}
```

---

## Biochemistry Tools

### 13. BiochemResolver - Metabolite Resolution
**Tool Name**: `resolve_biochemistry`

**Purpose**: Resolves metabolite identifiers and chemical information using biochemistry databases.

**Key Features**:
- Multi-database metabolite lookup
- Chemical structure and property resolution
- Identifier standardization and cross-referencing
- Biochemical pathway context integration

### 14. StandaloneBiochemResolver - Independent Resolution
**Tool Name**: `standalone_resolve_biochemistry`

**Purpose**: Independent biochemistry resolution without database dependencies.

**Key Features**:
- Local biochemistry database resolution
- Offline metabolite identification
- Chemical formula and structure processing
- Independent operation mode

---

## ModelSEED Tools

### 15. AnnotationTool - Metabolic Annotation
**Tool Name**: `annotate_with_modelseed`

**Purpose**: Annotates metabolic models with ModelSEED database information.

### 16. BuilderTool - Model Construction
**Tool Name**: `build_modelseed_model`

**Purpose**: Constructs metabolic models using ModelSEED framework.

### 17. CompatibilityTool - Format Conversion
**Tool Name**: `check_modelseed_compatibility`

**Purpose**: Checks and converts models for ModelSEED compatibility.

### 18. GapfillTool - Network Completion
**Tool Name**: `gapfill_modelseed_model`

**Purpose**: Performs automated gap-filling for incomplete metabolic networks.

---

## RAST Tools

### 19. RASTAnnotationTool - Genome Annotation
**Tool Name**: `annotate_with_rast`

**Purpose**: Integrates RAST genome annotation with metabolic modeling.

---

## Numerical Precision Improvements

### Problem Statement
Before the precision improvements, the ModelSEED Agent suffered from several numerical inconsistencies that affected analysis reliability:

1. **Inconsistent Thresholds**: Different tools used different cutoffs (FBA: 1e-6, Essentiality: 0.01)
2. **Solver Artifacts**: FBA returned optimization scaling values instead of biological rates
3. **Numerical Instability**: No protection against division by zero or floating-point errors
4. **Threshold Confusion**: Growth vs. flux thresholds mixed inappropriately

### Solution: Unified PrecisionConfig Framework

#### Key Improvements by Tool Category

**FBA and Optimization Tools**:
- **Before**: `solution.objective_value` returned 518.422 (optimization artifact)
- **After**: Extract actual biomass flux 0.8739 h⁻¹ from flux distribution
- **Impact**: Users get biologically meaningful growth rates

**Gene Deletion Analysis**:
- **Before**: Hard-coded 0.01 threshold only in essentiality tool
- **After**: Unified `essential_growth_fraction` across all deletion tools
- **Impact**: Consistent essentiality classification regardless of analysis method

**Flux Variability Analysis**:
- **Before**: Fixed 1e-6 tolerance could miss relevant small fluxes
- **After**: Configurable `flux_threshold` based on model and analysis needs
- **Impact**: More accurate identification of truly variable reactions

**Statistical Analysis (Flux Sampling)**:
- **Before**: Division by zero errors in correlation calculations
- **After**: `safe_divide()` function with numerical tolerance
- **Impact**: Robust statistical calculations even with zero-flux reactions

#### Quantitative Impact Analysis

| Tool Category | Before Precision Fix | After Precision Fix | Impact |
|---------------|---------------------|---------------------|---------|
| **FBA Growth Rate** | 518.422 (artifact) | 0.8739 h⁻¹ (biological) | ✅ Meaningful results |
| **Essentiality Threshold** | 0.01 (essentiality only) | 0.01 (all tools) | ✅ Consistency |
| **Flux Significance** | 1e-6 (fixed) | 1e-6 (configurable) | ✅ Flexibility |
| **Gene Categories** | Binary (essential/not) | 5 categories | ✅ Granular analysis |
| **Statistical Calculations** | Division errors | Safe operations | ✅ Numerical stability |

### Configuration Examples

**Default Configuration** (Recommended):
```python
precision_config = PrecisionConfig(
    flux_threshold=1e-6,        # Standard flux significance
    growth_threshold=1e-6,      # Standard growth significance
    essentiality_growth_fraction=0.01  # 1% for essentiality
)
```

**High Precision Configuration** (Research):
```python
precision_config = PrecisionConfig(
    flux_threshold=1e-9,        # Ultra-sensitive flux detection
    growth_threshold=1e-9,      # Ultra-sensitive growth detection
    essentiality_growth_fraction=0.001  # 0.1% for strict essentiality
)
```

**Lenient Configuration** (Noisy data):
```python
precision_config = PrecisionConfig(
    flux_threshold=1e-3,        # Less sensitive to noise
    growth_threshold=1e-3,      # Less sensitive to noise
    essentiality_growth_fraction=0.05  # 5% for lenient essentiality
)
```

---

## Error Handling Enhancements

### Intelligent Error Diagnosis

The enhanced error handling system provides context-aware error messages and actionable suggestions:

**Example Error Scenarios**:

1. **Missing Model File**:
   ```
   Error: Model file not found: e_coli_cor.xml
   Suggestions:
   - Did you mean: e_coli_core.xml?
   - Available model files: [e_coli_core.xml, iML1515.xml]
   - Check file permissions and accessibility
   ```

2. **Optimization Failure**:
   ```
   Error: FBA failed. Detected issues: No uptake reactions allowed
   Suggestions:
   - Open essential nutrient uptake: model.reactions.get_by_id('EX_glc__D_e').lower_bound = -1000
   - Check medium composition and exchange reaction bounds
   - Use model.medium to set up growth medium
   ```

3. **Parameter Validation**:
   ```
   Error: Growth threshold 2.0 > 1.0 is unusual
   Suggestions:
   - Typical values are 1e-6 to 1e-3 for absolute thresholds
   - Use fraction values: 0.01 = 1%, 0.1 = 10%
   ```

### Progress Logging

Long-running operations now provide real-time feedback:

```
Starting Gene deletion analysis for 1515 genes
Gene deletion progress: 151/1515 (10.0%) Testing gene b0001
Gene deletion progress: 303/1515 (20.0%) Testing gene b0002
...
```

---

## Usage Examples

### Basic FBA Analysis
```python
from src.tools.cobra import FBATool

# Initialize tool
fba_tool = FBATool({})

# Run analysis
result = fba_tool.run({
    "model_path": "data/models/e_coli_core.xml",
    "simulation_method": "fba"
})

print(f"Growth rate: {result.data['growth_rate']:.4f} h⁻¹")
print(f"Active reactions: {len(result.data['active_fluxes'])}")
```

### Gene Essentiality Analysis
```python
from src.tools.cobra import GeneDeletionTool

# Initialize with custom precision
gene_tool = GeneDeletionTool({
    "precision_config": {
        "essential_growth_fraction": 0.01  # 1% threshold
    }
})

# Run analysis
result = gene_tool.run({
    "model_path": "data/models/e_coli_core.xml",
    "deletion_type": "single"
})

print(f"Essential genes: {len(result.data['essential_genes'])}")
print(f"Severe impact genes: {len(result.data['severe_effect_genes'])}")
```

### Flux Sampling with Statistical Analysis
```python
from src.tools.cobra import FluxSamplingTool

# Initialize tool
sampling_tool = FluxSamplingTool({
    "sampling_config": {
        "n_samples": 1000,
        "method": "optgp",
        "seed": 42
    }
})

# Run sampling
result = sampling_tool.run({
    "model_path": "data/models/e_coli_core.xml"
})

print(f"Samples generated: {result.data['analysis']['distribution_analysis']['total_samples']}")
print(f"Variable reactions: {len(result.data['analysis']['flux_patterns']['variable_reactions'])}")
```

### Production Envelope Analysis
```python
from src.tools.cobra import ProductionEnvelopeTool

# Initialize tool
envelope_tool = ProductionEnvelopeTool({})

# Analyze ethanol vs. acetate production
result = envelope_tool.run({
    "model_path": "data/models/e_coli_core.xml",
    "reactions": ["EX_etoh_e", "EX_ac_e"],
    "points": 20,
    "c_source": "glc__D",
    "c_uptake_rates": 10.0
})

print(f"Trade-off analysis: {result.data['analysis']['trade_offs']}")
```

---

## Conclusion

The ModelSEED Agent provides a comprehensive suite of 19 metabolic modeling tools with sophisticated numerical precision handling, robust error management, and extensive analytical capabilities. The unified precision framework ensures consistent and reliable results across all analyses, while the enhanced error handling system provides users with actionable guidance for resolving issues.

The numerical precision improvements represent a significant advancement in metabolic modeling reliability, transforming solver artifacts into biologically meaningful results and ensuring consistent analysis standards across all tools.
