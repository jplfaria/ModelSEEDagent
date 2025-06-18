from typing import Any, Dict, List, Optional

import cobra
from cobra.flux_analysis import double_gene_deletion, single_gene_deletion
from pydantic import BaseModel, Field, PrivateAttr

from ..base import BaseTool, ToolRegistry, ToolResult
from .precision_config import (
    PrecisionConfig,
    calculate_growth_fraction,
    is_significant_growth,
)
from .utils import get_process_count_from_env, should_disable_auditing
from .utils_optimized import OptimizedModelUtils


class GeneDeletionConfig(BaseModel):
    """Configuration for Gene Deletion Analysis with enhanced numerical precision"""

    model_config = {"protected_namespaces": ()}
    gene_list: Optional[List[str]] = None
    deletion_type: str = "single"  # "single" or "double"
    method: str = "fba"  # "fba" or "moma" or "room"
    solver: str = "glpk"
    processes: Optional[int] = (
        1  # Default to 1 to prevent multiprocessing connection pool issues
    )
    return_solution: bool = False

    # Numerical precision settings
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)


@ToolRegistry.register
class GeneDeletionTool(BaseTool):
    """Tool for running single and double gene deletion analysis"""

    tool_name = "run_gene_deletion_analysis"
    tool_description = """Run gene deletion analysis to predict the effect of gene knockouts
    on model growth and identify essential genes."""

    _deletion_config: GeneDeletionConfig = PrivateAttr()
    _utils: OptimizedModelUtils = PrivateAttr()

    def __init__(self, config: Dict[str, Any]):
        # Disable auditing in subprocess to prevent connection pool issues
        if should_disable_auditing():
            config = config.copy()
            config["audit_enabled"] = False
        super().__init__(config)
        deletion_config_dict = config.get("deletion_config", {})
        if isinstance(deletion_config_dict, dict):
            self._deletion_config = GeneDeletionConfig(**deletion_config_dict)
        else:
            self._deletion_config = GeneDeletionConfig(
                gene_list=getattr(deletion_config_dict, "gene_list", None),
                deletion_type=getattr(deletion_config_dict, "deletion_type", "single"),
                method=getattr(deletion_config_dict, "method", "fba"),
                solver=getattr(deletion_config_dict, "solver", "glpk"),
                processes=getattr(deletion_config_dict, "processes", None),
                return_solution=getattr(deletion_config_dict, "return_solution", False),
                precision=PrecisionConfig(),
            )
        self._utils = OptimizedModelUtils(use_cache=True)

    @property
    def deletion_config(self) -> GeneDeletionConfig:
        """Get the gene deletion configuration"""
        return self._deletion_config

    def _run_tool(self, input_data: Any) -> ToolResult:
        try:
            # Support both dict and string inputs
            if isinstance(input_data, dict):
                model_path = input_data.get("model_path")
                gene_list = input_data.get("gene_list", self.deletion_config.gene_list)
                deletion_type = input_data.get(
                    "deletion_type", self.deletion_config.deletion_type
                )
                method = input_data.get("method", self.deletion_config.method)
            else:
                model_path = input_data
                gene_list = self.deletion_config.gene_list
                deletion_type = self.deletion_config.deletion_type
                method = self.deletion_config.method

            if not isinstance(model_path, str):
                raise ValueError("Model path must be a string")

            # Load model
            model = self._utils.load_model(model_path)
            model.solver = self.deletion_config.solver

            # Get wild-type growth rate
            wild_type_solution = model.optimize()
            if wild_type_solution.status != "optimal":
                return ToolResult(
                    success=False,
                    message="Wild-type model optimization failed",
                    error=f"Solution status: {wild_type_solution.status}",
                )

            wild_type_growth = wild_type_solution.objective_value

            # Set up gene list
            if gene_list is None:
                gene_list = [gene.id for gene in model.genes]

            # Get process count with environment variable override
            processes = get_process_count_from_env(
                self.deletion_config.processes, "COBRA_GENE_DELETION_PROCESSES"
            )

            # Run deletion analysis
            if deletion_type == "single":
                deletion_results = single_gene_deletion(
                    model=model,
                    gene_list=gene_list,
                    method=method,
                    solution=(
                        wild_type_solution
                        if self.deletion_config.return_solution
                        else None
                    ),
                    processes=processes,
                )
            elif deletion_type == "double":
                deletion_results = double_gene_deletion(
                    model=model,
                    gene_list1=gene_list,
                    gene_list2=None,  # Use same list for both
                    method=method,
                    solution=(
                        wild_type_solution
                        if self.deletion_config.return_solution
                        else None
                    ),
                    processes=processes,
                )
            else:
                raise ValueError(f"Unsupported deletion type: {deletion_type}")

            # Analyze results
            analysis = self._analyze_deletion_results(
                deletion_results, wild_type_growth, deletion_type
            )

            # Create lightweight results without redundant data
            simplified_results = {}
            for idx, row in deletion_results.iterrows():
                growth = row["growth"]
                gene_set = row["ids"]

                # Extract clean gene IDs from set
                if deletion_type == "single":
                    gene_id = (
                        list(gene_set)[0]
                        if hasattr(gene_set, "__iter__")
                        and not isinstance(gene_set, str)
                        else str(gene_set)
                    )
                else:
                    gene_id = (
                        "-".join(sorted(gene_set))
                        if hasattr(gene_set, "__iter__")
                        else str(gene_set)
                    )

                simplified_results[gene_id] = {
                    "growth": float(growth) if growth is not None else 0.0,
                    "growth_ratio": (
                        float(growth / wild_type_growth)
                        if growth is not None and wild_type_growth > 0
                        else 0.0
                    ),
                }

            return ToolResult(
                success=True,
                message=f"{deletion_type.title()} gene deletion analysis completed for {len(gene_list)} genes",
                data={
                    "deletion_results": simplified_results,
                    "analysis": analysis,
                    "wild_type_growth": float(wild_type_growth),
                },
                metadata={
                    "model_id": model.id,
                    "deletion_type": deletion_type,
                    "method": method,
                    "genes_tested": len(gene_list),
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message="Error running gene deletion analysis",
                error=str(e),
            )

    def _analyze_deletion_results(
        self, results, wild_type_growth: float, deletion_type: str
    ) -> Dict[str, Any]:
        """Analyze gene deletion results to categorize genes"""

        analysis = {
            "essential_genes": [],  # growth < 1% of wild-type
            "severely_impaired": [],  # growth 1-10% of wild-type
            "moderately_impaired": [],  # growth 10-50% of wild-type
            "mildly_impaired": [],  # growth 50-90% of wild-type
            "no_effect": [],  # growth >= 90% of wild-type
            "improved_growth": [],  # growth > wild-type
        }

        # Use configurable precision thresholds
        precision = self.deletion_config.precision
        tolerance = precision.growth_threshold
        essential_threshold = precision.essential_growth_fraction * wild_type_growth
        severe_threshold = precision.severe_effect_fraction * wild_type_growth
        moderate_threshold = precision.moderate_effect_fraction * wild_type_growth
        mild_threshold = precision.mild_effect_fraction * wild_type_growth

        for idx, row in results.iterrows():
            growth = row["growth"]
            gene_set = row["ids"]

            # Extract clean gene ID
            if deletion_type == "single":
                gene_id = (
                    list(gene_set)[0]
                    if hasattr(gene_set, "__iter__") and not isinstance(gene_set, str)
                    else str(gene_set)
                )
            else:
                gene_id = (
                    "-".join(sorted(gene_set))
                    if hasattr(gene_set, "__iter__")
                    else str(gene_set)
                )

            # Categorize based on growth impact (store only gene IDs to avoid duplication)
            if growth is None or growth < essential_threshold:
                analysis["essential_genes"].append(gene_id)
            elif growth < severe_threshold:
                analysis["severely_impaired"].append(gene_id)
            elif growth < moderate_threshold:
                analysis["moderately_impaired"].append(gene_id)
            elif growth < mild_threshold:
                analysis["mildly_impaired"].append(gene_id)
            elif growth > wild_type_growth + tolerance:
                analysis["improved_growth"].append(gene_id)
            else:
                analysis["no_effect"].append(gene_id)

        # Categories are now lists of gene IDs, no need to sort by growth

        # Add summary statistics
        analysis["summary"] = {
            "total_genes_tested": len(results),
            "essential_count": len(analysis["essential_genes"]),
            "impaired_count": len(analysis["severely_impaired"])
            + len(analysis["moderately_impaired"])
            + len(analysis["mildly_impaired"]),
            "no_effect_count": len(analysis["no_effect"]),
            "improved_count": len(analysis["improved_growth"]),
            "essentiality_rate": (
                len(analysis["essential_genes"]) / len(results)
                if len(results) > 0
                else 0.0
            ),
        }

        return analysis
