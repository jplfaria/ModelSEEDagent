from typing import Any, Dict, List, Optional

import cobra
from cobra.flux_analysis import double_gene_deletion, single_gene_deletion
from pydantic import BaseModel, Field, PrivateAttr

from ..base import BaseTool, ToolRegistry, ToolResult
from .utils import ModelUtils


class GeneDeletionConfig(BaseModel):
    """Configuration for Gene Deletion Analysis"""

    model_config = {"protected_namespaces": ()}
    gene_list: Optional[List[str]] = None
    deletion_type: str = "single"  # "single" or "double"
    method: str = "fba"  # "fba" or "moma" or "room"
    solver: str = "glpk"
    processes: Optional[int] = None
    return_solution: bool = False


@ToolRegistry.register
class GeneDeletionTool(BaseTool):
    """Tool for running single and double gene deletion analysis"""

    tool_name = "run_gene_deletion_analysis"
    tool_description = """Run gene deletion analysis to predict the effect of gene knockouts
    on model growth and identify essential genes."""

    _deletion_config: GeneDeletionConfig = PrivateAttr()
    _utils: ModelUtils = PrivateAttr()

    def __init__(self, config: Dict[str, Any]):
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
            )
        self._utils = ModelUtils()

    @property
    def deletion_config(self) -> GeneDeletionConfig:
        """Get the gene deletion configuration"""
        return self._deletion_config

    def _run(self, input_data: Any) -> ToolResult:
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
                    processes=self.deletion_config.processes,
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
                    processes=self.deletion_config.processes,
                )
            else:
                raise ValueError(f"Unsupported deletion type: {deletion_type}")

            # Analyze results
            analysis = self._analyze_deletion_results(
                deletion_results, wild_type_growth, deletion_type
            )

            return ToolResult(
                success=True,
                message=f"{deletion_type.title()} gene deletion analysis completed for {len(gene_list)} genes",
                data={
                    "deletion_results": deletion_results.to_dict(),
                    "analysis": analysis,
                    "wild_type_growth": float(wild_type_growth),
                },
                metadata={
                    "model_id": model.id,
                    "deletion_type": deletion_type,
                    "method": method,
                    "genes_tested": len(gene_list),
                    "wild_type_growth": float(wild_type_growth),
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

        tolerance = 1e-6
        essential_threshold = 0.01 * wild_type_growth
        severe_threshold = 0.10 * wild_type_growth
        moderate_threshold = 0.50 * wild_type_growth
        mild_threshold = 0.90 * wild_type_growth

        for index, row in results.iterrows():
            growth = row["growth"]

            # Handle single vs double gene deletion indexing
            if deletion_type == "single":
                genes = [index] if isinstance(index, str) else index
            else:
                genes = list(index) if hasattr(index, "__iter__") else [str(index)]

            gene_data = {
                "genes": genes,
                "growth": float(growth) if growth is not None else 0.0,
                "growth_rate_ratio": (
                    float(growth / wild_type_growth)
                    if growth is not None and wild_type_growth > tolerance
                    else 0.0
                ),
            }

            # Categorize based on growth impact
            if growth is None or growth < essential_threshold:
                analysis["essential_genes"].append(gene_data)
            elif growth < severe_threshold:
                analysis["severely_impaired"].append(gene_data)
            elif growth < moderate_threshold:
                analysis["moderately_impaired"].append(gene_data)
            elif growth < mild_threshold:
                analysis["mildly_impaired"].append(gene_data)
            elif growth > wild_type_growth + tolerance:
                analysis["improved_growth"].append(gene_data)
            else:
                analysis["no_effect"].append(gene_data)

        # Sort categories by growth impact
        for category in analysis.values():
            if isinstance(category, list):
                category.sort(key=lambda x: x["growth"])

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
