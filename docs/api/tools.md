# Tool Implementation Reference

This document provides comprehensive technical implementation details for all ModelSEEDagent tools, including parameters, configurations, and advanced usage patterns.

## Tool Architecture

### Base Tool Foundation

All tools inherit from the `BaseTool` class and implement consistent patterns:

```python
from src.tools.base import BaseTool, ToolResult
from pydantic import BaseModel

class MyTool(BaseTool):
    """Custom tool implementation"""

    def _run_tool(self, input_data: Any) -> ToolResult:
        # Tool implementation
        return ToolResult(
            success=True,
            message="Human-readable description",
            data={...},  # Core results
            metadata={...},  # Execution metadata
            error=None
        )
```

### Tool Registration

Tools are automatically registered using the decorator pattern:

```python
from src.tools.base import ToolRegistry

@ToolRegistry.register
class MyCustomTool(BaseTool):
    """Automatically registered tool"""
    pass
```

### Result Structure

All tools return standardized `ToolResult` objects:

```python
class ToolResult:
    success: bool           # Execution success/failure
    message: str           # Human-readable description
    data: Dict[str, Any]   # Core analysis results
    metadata: Dict         # Tool execution metadata
    error: Optional[str]   # Error details if failed
```

## COBRApy Tools (12 tools)

### Core Analysis Tools

#### `analyze_metabolic_model`

**Purpose**: Comprehensive model structure analysis

**Configuration**:
```python
class ModelAnalysisConfig(BaseModel):
    include_subsystems: bool = True
    include_genes: bool = True
    include_compartments: bool = True
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
```

**Input Parameters**:
- `model_path` (str): Path to SBML model file
- `analysis_type` (str, optional): "basic" | "detailed" | "comprehensive"
- `config` (dict, optional): Tool configuration overrides

**Output Structure**:
```python
{
    "reactions": {
        "total": int,
        "reversible": int,
        "irreversible": int,
        "exchange": int,
        "transport": int
    },
    "metabolites": {
        "total": int,
        "by_compartment": Dict[str, int]
    },
    "genes": {
        "total": int,
        "orphaned": List[str]
    },
    "subsystems": {
        "total": int,
        "distribution": Dict[str, int]
    },
    "network_properties": {
        "connectivity": float,
        "clustering_coefficient": float
    }
}
```

**Advanced Usage**:
```python
from src.tools.cobra.analysis import MetabolicAnalysisTool

tool = MetabolicAnalysisTool({
    "tool_config": {
        "include_subsystems": True,
        "precision": {"flux_threshold": 1e-8}
    }
})

result = tool.execute({
    "model_path": "path/to/model.xml",
    "analysis_type": "comprehensive"
})

# Access specific data
reaction_count = result.data["reactions"]["total"]
subsystems = result.data["subsystems"]["distribution"]
```

#### `run_metabolic_fba`

**Purpose**: Flux Balance Analysis with advanced simulation control

**Configuration**:
```python
class FBAConfig(BaseModel):
    objective: Optional[str] = None  # Auto-detect if None
    media: Optional[str] = "complete"
    simulation_method: str = "fba"  # "fba" | "pfba" | "robust"
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
    export_fluxes: bool = False
    flux_threshold: float = 1e-6
```

**Input Parameters**:
- `model_path` (str): Path to SBML model file
- `media_file` (str, optional): Media composition file
- `objective` (str, optional): Objective reaction ID
- `simulation_method` (str): FBA variant to use
- `export_fluxes` (bool): Export detailed flux data

**Output Structure**:
```python
{
    "growth_rate": float,
    "objective_value": float,
    "status": str,  # "optimal" | "infeasible" | "unbounded"
    "active_fluxes": Dict[str, float],  # Non-zero fluxes
    "exchange_fluxes": Dict[str, float],  # Media uptake/secretion
    "simulation_info": {
        "method": str,
        "solver": str,
        "solve_time": float
    },
    "flux_summary": {
        "active_reactions": int,
        "max_flux": float,
        "flux_distribution": Dict[str, float]
    }
}
```

**Advanced Configuration**:
```python
from src.tools.cobra.fba import FBATool

# High-precision FBA with flux export
tool = FBATool({
    "tool_config": {
        "simulation_method": "pfba",  # Parsimonious FBA
        "precision": {
            "tolerance": 1e-9,
            "flux_threshold": 1e-8
        },
        "export_fluxes": True
    }
})

result = tool.execute({
    "model_path": "model.xml",
    "media_file": "minimal_media.json",
    "objective": "BIOMASS_reaction"
})
```

#### `run_flux_variability_analysis`

**Purpose**: Determine flux ranges for model reactions

**Configuration**:
```python
class FVAConfig(BaseModel):
    reaction_list: Optional[List[str]] = None  # All reactions if None
    fraction_of_optimum: float = 1.0
    loopless: bool = False
    processes: int = 1  # Parallel processes
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
```

**Input Parameters**:
- `model_path` (str): Path to SBML model file
- `reactions` (List[str], optional): Specific reactions to analyze
- `fraction_of_optimum` (float): Growth rate fraction (0.0-1.0)
- `loopless` (bool): Use loopless FVA

**Output Structure**:
```python
{
    "variability_ranges": {
        "reaction_id": {
            "minimum": float,
            "maximum": float,
            "range": float
        }
    },
    "fixed_reactions": List[str],  # Zero variability
    "variable_reactions": List[str],  # Non-zero variability
    "analysis_summary": {
        "total_reactions": int,
        "constrained_reactions": int,
        "flexible_reactions": int,
        "growth_rate_constraint": float
    }
}
```

#### `run_gene_deletion_analysis`

**Purpose**: Single and double gene deletion studies

**Configuration**:
```python
class GeneDeletionConfig(BaseModel):
    deletion_type: str = "single"  # "single" | "double" | "both"
    gene_list: Optional[List[str]] = None
    growth_threshold: float = 0.01
    method: str = "fba"  # "fba" | "moma"
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
```

**Output Structure**:
```python
{
    "single_deletions": {
        "gene_id": {
            "growth_rate": float,
            "growth_ratio": float,  # Relative to wild-type
            "essential": bool,
            "affected_reactions": List[str]
        }
    },
    "double_deletions": {
        "gene1,gene2": {
            "growth_rate": float,
            "synthetic_lethal": bool,
            "interaction_score": float
        }
    },
    "summary": {
        "essential_genes": List[str],
        "synthetic_lethal_pairs": List[Tuple[str, str]],
        "wild_type_growth": float
    }
}
```

### AI Media Management Tools (6 tools)

#### `select_optimal_media`

**Purpose**: AI-driven optimal media selection for models

**Configuration**:
```python
class OptimalMediaConfig(BaseModel):
    optimization_target: str = "growth"  # "growth" | "production" | "biomass"
    target_compound: Optional[str] = None
    media_library: str = "default"
    ai_strategy: str = "comprehensive"  # "fast" | "comprehensive"
```

**Input Parameters**:
- `model_path` (str): Path to model file
- `target` (str): Optimization objective
- `constraints` (dict, optional): Additional constraints

**Output Structure**:
```python
{
    "optimal_media": {
        "compounds": List[str],
        "concentrations": Dict[str, float],
        "media_name": str
    },
    "performance_metrics": {
        "growth_rate": float,
        "production_rate": float,
        "efficiency_score": float
    },
    "ai_reasoning": {
        "selection_rationale": str,
        "alternative_options": List[dict],
        "confidence_score": float
    }
}
```

#### `manipulate_media_composition`

**Purpose**: Natural language media modification

**Input Parameters**:
- `model_path` (str): Path to model file
- `current_media` (str/dict): Current media composition
- `modification_request` (str): Natural language modification request

**Example Requests**:
- "Add glucose and remove lactose"
- "Increase nitrogen sources by 50%"
- "Switch to anaerobic conditions"
- "Make this a minimal media"

**Output Structure**:
```python
{
    "modified_media": {
        "compounds": Dict[str, float],
        "changes_made": List[str]
    },
    "predicted_effects": {
        "growth_change": str,
        "metabolic_changes": List[str]
    },
    "modification_summary": str
}
```

### Precision Configuration

All tools support advanced precision control:

```python
class PrecisionConfig(BaseModel):
    tolerance: float = 1e-6          # Solver tolerance
    flux_threshold: float = 1e-6     # Minimum flux significance
    growth_threshold: float = 1e-3   # Minimum growth rate
    feasibility_tolerance: float = 1e-9  # Feasibility check tolerance
    numerical_precision: int = 6      # Decimal places for output
```

## ModelSEED Tools (5 tools)

### Model Building Tools

#### `build_metabolic_model`

**Purpose**: Construct metabolic models from genome annotations

**Configuration**:
```python
class ModelBuilderConfig(BaseModel):
    template: str = "GramNegativeV5"  # Model template
    gapfill: bool = True
    media: str = "complete"
    namespace: str = "ModelSEED"
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
```

**Input Parameters**:
- `genome_file` (str): Genome annotation file (GBK/FASTA)
- `template` (str): ModelSEED template to use
- `organism_name` (str): Organism identifier

**Output Structure**:
```python
{
    "model_file": str,  # Path to generated model
    "model_statistics": {
        "reactions": int,
        "metabolites": int,
        "genes": int,
        "compartments": List[str]
    },
    "gapfilling_results": {
        "added_reactions": List[str],
        "filled_pathways": List[str],
        "growth_enabled": bool
    },
    "quality_metrics": {
        "completeness_score": float,
        "consistency_score": float
    }
}
```

#### `gapfill_metabolic_model`

**Purpose**: Automated model gap-filling for growth

**Configuration**:
```python
class GapfillConfig(BaseModel):
    media: str = "complete"
    target_reaction: Optional[str] = None  # Growth reaction
    gapfill_method: str = "comprehensive"  # "fast" | "comprehensive"
    max_gapfill_reactions: int = 50
    objective_fraction: float = 0.01
```

**Output Structure**:
```python
{
    "gapfilled_model": str,  # Path to gapfilled model
    "added_reactions": List[Dict[str, Any]],
    "pathways_completed": List[str],
    "growth_analysis": {
        "original_growth": float,
        "gapfilled_growth": float,
        "improvement": float
    },
    "gapfill_summary": {
        "total_added": int,
        "essential_additions": List[str],
        "confidence_scores": Dict[str, float]
    }
}
```

## Biochemistry Database Tools (2 tools)

#### `resolve_biochem_entity`

**Purpose**: Universal biochemistry ID and name resolution

**Input Parameters**:
- `query` (str): Entity ID or name to resolve
- `entity_type` (str): "compound" | "reaction" | "auto"
- `namespace` (str, optional): "ModelSEED" | "BIGG" | "KEGG"

**Output Structure**:
```python
{
    "resolved_entity": {
        "id": str,
        "name": str,
        "formula": str,
        "aliases": List[str],
        "database_refs": Dict[str, str]
    },
    "resolution_confidence": float,
    "alternative_matches": List[Dict[str, Any]]
}
```

#### `search_biochem`

**Purpose**: Natural language biochemistry database search

**Input Parameters**:
- `query` (str): Search query (natural language or specific terms)
- `search_type` (str): "compound" | "reaction" | "pathway" | "all"
- `limit` (int): Maximum results to return

**Output Structure**:
```python
{
    "search_results": List[{
        "id": str,
        "name": str,
        "type": str,
        "relevance_score": float,
        "description": str
    }],
    "search_metadata": {
        "total_matches": int,
        "search_time": float,
        "query_interpretation": str
    }
}
```

## Error Handling and Validation

### Common Error Types

```python
# Tool-specific exceptions
class ModelLoadError(Exception):
    """Model file loading failed"""
    pass

class ValidationError(Exception):
    """Input validation failed"""
    pass

class SimulationError(Exception):
    """Simulation execution failed"""
    pass

class PrecisionError(Exception):
    """Numerical precision issues"""
    pass
```

### Error Handling Pattern

```python
def robust_tool_execution(tool, inputs):
    """Example of robust tool execution with error handling"""
    try:
        result = tool.execute(inputs)
        if not result.success:
            print(f"Tool execution failed: {result.error}")
            return None
        return result.data

    except ValidationError as e:
        print(f"Input validation error: {e}")
    except ModelLoadError as e:
        print(f"Model loading error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None
```

## Performance Optimization

### Caching Configuration

```python
# Enable result caching for expensive operations
from src.tools.base import ToolCache

tool = FBATool({
    "cache_config": {
        "enabled": True,
        "ttl": 3600,  # 1 hour cache
        "max_size": 100
    }
})
```

### Parallel Execution

```python
# Configure parallel processing for FVA
tool = FluxVariabilityTool({
    "tool_config": {
        "processes": 4,  # Use 4 CPU cores
        "batch_size": 100
    }
})
```

### Memory Management

```python
# Configure memory-efficient execution
tool = GeneDeleteionTool({
    "tool_config": {
        "memory_efficient": True,
        "batch_deletions": True,
        "cleanup_intermediate": True
    }
})
```

## Integration Examples

### Complete Analysis Pipeline

```python
async def comprehensive_model_analysis(model_path: str):
    """Complete model analysis using multiple tools"""

    # Initialize tools
    analysis_tool = MetabolicAnalysisTool()
    fba_tool = FBATool()
    fva_tool = FluxVariabilityTool()
    deletion_tool = GeneDeletionTool()

    results = {}

    # Step 1: Model structure analysis
    structure = analysis_tool.execute({"model_path": model_path})
    results["structure"] = structure.data

    # Step 2: Growth analysis
    growth = fba_tool.execute({
        "model_path": model_path,
        "simulation_method": "pfba"
    })
    results["growth"] = growth.data

    # Step 3: Flux variability
    if growth.data["growth_rate"] > 0:
        fva = fva_tool.execute({
            "model_path": model_path,
            "fraction_of_optimum": 0.9
        })
        results["variability"] = fva.data

    # Step 4: Gene essentiality
    essentiality = deletion_tool.execute({
        "model_path": model_path,
        "deletion_type": "single"
    })
    results["essentiality"] = essentiality.data

    return results
```

### Custom Tool Development

```python
from src.tools.base import BaseTool, ToolResult
from pydantic import BaseModel

class CustomMetaboliteAnalysisTool(BaseTool):
    """Custom tool for specialized metabolite analysis"""

    class Config(BaseModel):
        metabolite_filter: List[str] = []
        include_cofactors: bool = False
        analysis_depth: str = "standard"

    def _run_tool(self, input_data: Any) -> ToolResult:
        try:
            model_path = self._extract_model_path(input_data)

            # Load model
            model = self._load_model(model_path)

            # Custom analysis logic
            results = self._analyze_metabolites(model)

            return ToolResult(
                success=True,
                message=f"Analyzed {len(results)} metabolites",
                data=results,
                metadata={
                    "tool_version": "1.0.0",
                    "model_reactions": len(model.reactions),
                    "analysis_time": self._get_execution_time()
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Metabolite analysis failed: {str(e)}",
                error=str(e)
            )

    def _analyze_metabolites(self, model):
        """Custom metabolite analysis implementation"""
        # Your analysis logic here
        return {"metabolite_data": {}}

# Register the custom tool
@ToolRegistry.register
class RegisteredCustomTool(CustomMetaboliteAnalysisTool):
    pass
```

This comprehensive tool implementation reference provides the technical details needed for advanced usage, custom tool development, and integration with the ModelSEEDagent platform.
