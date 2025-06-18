from typing import Any, Dict, List, Optional

import cobra
from cobra.flux_analysis import (
    find_essential_genes,
    find_essential_reactions,
    single_gene_deletion,
    single_reaction_deletion,
)
from pydantic import BaseModel, Field, PrivateAttr

from ..base import BaseTool, ToolRegistry, ToolResult
from .precision_config import (
    PrecisionConfig,
    calculate_growth_fraction,
    is_significant_growth,
)
from .utils import get_process_count_from_env, should_disable_auditing
from .utils_optimized import OptimizedModelUtils


class EssentialityConfig(BaseModel):
    """Configuration for Essentiality Analysis with enhanced numerical precision"""

    model_config = {"protected_namespaces": ()}
    include_genes: bool = True
    include_reactions: bool = True
    solver: str = "glpk"
    processes: Optional[int] = (
        1  # Default to 1 to prevent multiprocessing connection pool issues
    )

    # Numerical precision settings
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)


@ToolRegistry.register
class EssentialityAnalysisTool(BaseTool):
    """Tool for comprehensive essentiality analysis of genes and reactions"""

    tool_name = "analyze_essentiality"
    tool_description = """Analyze gene and reaction essentiality to identify critical components
    required for model growth and viability."""

    _essentiality_config: EssentialityConfig = PrivateAttr()
    _utils: OptimizedModelUtils = PrivateAttr()

    def __init__(self, config: Dict[str, Any]):
        # Disable auditing in subprocess to prevent connection pool issues
        if should_disable_auditing():
            config = config.copy()
            config["audit_enabled"] = False
        super().__init__(config)
        essentiality_config_dict = config.get("essentiality_config", {})
        if isinstance(essentiality_config_dict, dict):
            self._essentiality_config = EssentialityConfig(**essentiality_config_dict)
        else:
            self._essentiality_config = EssentialityConfig(
                precision=PrecisionConfig(),
                include_genes=getattr(essentiality_config_dict, "include_genes", True),
                include_reactions=getattr(
                    essentiality_config_dict, "include_reactions", True
                ),
                solver=getattr(essentiality_config_dict, "solver", "glpk"),
                processes=getattr(essentiality_config_dict, "processes", None),
            )
        self._utils = OptimizedModelUtils(use_cache=True)

    @property
    def essentiality_config(self) -> EssentialityConfig:
        """Get the essentiality configuration"""
        return self._essentiality_config

    def _run_tool(self, input_data: Any) -> ToolResult:
        try:
            # Support both dict and string inputs
            if isinstance(input_data, dict):
                model_path = input_data.get("model_path")
                # Use essentiality-specific threshold (fraction of wild-type growth)
                threshold = input_data.get(
                    "essentiality_threshold",
                    self.essentiality_config.precision.essentiality_growth_fraction,
                )
                include_genes = input_data.get(
                    "include_genes", self.essentiality_config.include_genes
                )
                include_reactions = input_data.get(
                    "include_reactions", self.essentiality_config.include_reactions
                )
            else:
                model_path = input_data
                threshold = (
                    self.essentiality_config.precision.essentiality_growth_fraction
                )
                include_genes = self.essentiality_config.include_genes
                include_reactions = self.essentiality_config.include_reactions

            if not isinstance(model_path, str):
                raise ValueError("Model path must be a string")

            # Load model
            model = self._utils.load_model(model_path)
            model.solver = self.essentiality_config.solver

            # Get wild-type growth rate
            wild_type_solution = model.optimize()
            if wild_type_solution.status != "optimal":
                return ToolResult(
                    success=False,
                    message="Wild-type model optimization failed",
                    error=f"Solution status: {wild_type_solution.status}",
                )

            wild_type_growth = wild_type_solution.objective_value

            analysis_results = {
                "wild_type_growth": float(wild_type_growth),
                "threshold": threshold,
                "essential_genes": None,
                "essential_reactions": None,
                "gene_analysis": None,
                "reaction_analysis": None,
            }

            # Get process count with environment variable override
            processes = get_process_count_from_env(
                self.essentiality_config.processes, "COBRA_ESSENTIALITY_PROCESSES"
            )

            # Analyze gene essentiality
            if include_genes:
                essential_genes = find_essential_genes(
                    model, threshold=threshold, processes=processes
                )
                gene_analysis = self._detailed_gene_analysis(
                    model, essential_genes, wild_type_growth, threshold
                )

                analysis_results["essential_genes"] = [
                    gene.id for gene in essential_genes
                ]
                analysis_results["gene_analysis"] = gene_analysis

            # Analyze reaction essentiality
            if include_reactions:
                essential_reactions = find_essential_reactions(
                    model, threshold=threshold, processes=processes
                )
                reaction_analysis = self._detailed_reaction_analysis(
                    model, essential_reactions, wild_type_growth, threshold
                )

                analysis_results["essential_reactions"] = [
                    rxn.id for rxn in essential_reactions
                ]
                analysis_results["reaction_analysis"] = reaction_analysis

            # Generate summary
            summary = self._generate_summary(analysis_results, model)
            analysis_results["summary"] = summary

            return ToolResult(
                success=True,
                message=f"Essentiality analysis completed. Found {len(analysis_results.get('essential_genes', []))} essential genes and {len(analysis_results.get('essential_reactions', []))} essential reactions",
                data=analysis_results,
                metadata={
                    "model_id": model.id,
                    "wild_type_growth": float(wild_type_growth),
                    "threshold": threshold,
                    "total_genes": len(model.genes),
                    "total_reactions": len(model.reactions),
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message="Error running essentiality analysis",
                error=str(e),
            )

    def _detailed_gene_analysis(
        self,
        model: cobra.Model,
        essential_genes: List,
        wild_type_growth: float,
        threshold: float,
    ) -> Dict[str, Any]:
        """Streamlined analysis of essential genes (no data duplication)"""

        functional_categories = {}
        subsystem_counts = {}
        reaction_counts = []

        # Single pass analysis to avoid data duplication
        for gene in essential_genes:
            # Get associated reactions
            associated_reactions = [rxn for rxn in model.reactions if gene in rxn.genes]
            num_reactions = len(associated_reactions)
            reaction_counts.append(num_reactions)

            # Categorize by subsystems (count only, no gene lists)
            subsystems = set()
            for rxn in associated_reactions:
                if rxn.subsystem:
                    subsystems.add(rxn.subsystem)

            # Count functional categories (no gene details stored)
            func_cat = self._categorize_gene_function(list(subsystems))
            functional_categories[func_cat] = functional_categories.get(func_cat, 0) + 1

            # Count subsystems (no gene lists stored)
            for subsystem in subsystems:
                subsystem_counts[subsystem] = subsystem_counts.get(subsystem, 0) + 1

        return {
            "summary": {
                "total_essential_genes": len(essential_genes),
                "functional_categories": functional_categories,
                "subsystems_affected": len(subsystem_counts),
                "top_subsystems": sorted(
                    subsystem_counts.items(), key=lambda x: x[1], reverse=True
                )[:5],
            },
            "connectivity": {
                "single_reaction_genes": sum(
                    1 for count in reaction_counts if count == 1
                ),
                "multi_reaction_genes": sum(
                    1 for count in reaction_counts if count > 1
                ),
                "max_reactions_per_gene": (
                    max(reaction_counts) if reaction_counts else 0
                ),
                "avg_reactions_per_gene": (
                    sum(reaction_counts) / len(reaction_counts)
                    if reaction_counts
                    else 0
                ),
            },
        }

    def _detailed_reaction_analysis(
        self,
        model: cobra.Model,
        essential_reactions: List,
        wild_type_growth: float,
        threshold: float,
    ) -> Dict[str, Any]:
        """Streamlined analysis of essential reactions (no data duplication)"""

        subsystem_counts = {}
        network_counts = {
            "exchange": 0,
            "transport": 0,
            "metabolic": 0,
            "gene_associated": 0,
            "spontaneous": 0,
        }
        gene_counts = []

        # Single pass analysis to avoid data duplication
        for rxn in essential_reactions:
            # Network classification
            is_exchange = rxn.id.startswith(("EX_", "DM_", "SK_"))
            is_transport = "transport" in rxn.name.lower() if rxn.name else False
            num_genes = len(rxn.genes)

            # Count network types
            if is_exchange:
                network_counts["exchange"] += 1
            elif is_transport:
                network_counts["transport"] += 1
            else:
                network_counts["metabolic"] += 1

            if num_genes > 0:
                network_counts["gene_associated"] += 1
            else:
                network_counts["spontaneous"] += 1

            gene_counts.append(num_genes)

            # Count subsystems (no reaction lists stored)
            subsystem = rxn.subsystem or "Unknown"
            subsystem_counts[subsystem] = subsystem_counts.get(subsystem, 0) + 1

        return {
            "summary": {
                "total_essential_reactions": len(essential_reactions),
                "subsystems_affected": len(subsystem_counts),
                "top_subsystems": sorted(
                    subsystem_counts.items(), key=lambda x: x[1], reverse=True
                )[:5],
            },
            "network_analysis": network_counts,
            "gene_association": {
                "max_genes_per_reaction": max(gene_counts) if gene_counts else 0,
                "avg_genes_per_reaction": (
                    sum(gene_counts) / len(gene_counts) if gene_counts else 0
                ),
                "reactions_with_genes": sum(1 for count in gene_counts if count > 0),
            },
        }

    def _categorize_gene_function(self, subsystems: List[str]) -> str:
        """Categorize gene function based on associated subsystems"""
        if not subsystems:
            return "Unknown"

        # Map subsystems to functional categories
        category_map = {
            "energy": [
                "Glycolysis",
                "TCA",
                "Citric acid cycle",
                "Oxidative phosphorylation",
                "ATP synthesis",
            ],
            "amino_acid": ["Amino acid", "Protein", "Peptide"],
            "nucleotide": ["Nucleotide", "Purine", "Pyrimidine", "DNA", "RNA"],
            "lipid": ["Lipid", "Fatty acid", "Membrane"],
            "carbohydrate": ["Carbohydrate", "Sugar", "Starch"],
            "cofactor": ["Cofactor", "Vitamin", "Coenzyme"],
            "transport": ["Transport", "ABC", "Permease"],
            "central_metabolism": ["Central", "Core", "Essential"],
        }

        for category, keywords in category_map.items():
            for subsystem in subsystems:
                for keyword in keywords:
                    if keyword.lower() in subsystem.lower():
                        return category

        return "Other"

    def _generate_summary(
        self, results: Dict[str, Any], model: cobra.Model
    ) -> Dict[str, Any]:
        """Generate a comprehensive summary of essentiality analysis"""

        summary = {
            "model_overview": {
                "total_genes": len(model.genes),
                "total_reactions": len(model.reactions),
                "total_metabolites": len(model.metabolites),
            }
        }

        if results["essential_genes"] is not None:
            gene_analysis = results.get("gene_analysis", {})
            summary["gene_essentiality"] = {
                "essential_count": len(results["essential_genes"]),
                "essentiality_rate": (
                    len(results["essential_genes"]) / len(model.genes)
                    if len(model.genes) > 0
                    else 0
                ),
                "top_functional_categories": (
                    sorted(
                        gene_analysis.get("summary", {})
                        .get("functional_categories", {})
                        .items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5]
                ),
                "connectivity_stats": gene_analysis.get("connectivity", {}),
            }

        if results["essential_reactions"] is not None:
            reaction_analysis = results.get("reaction_analysis", {})
            summary["reaction_essentiality"] = {
                "essential_count": len(results["essential_reactions"]),
                "essentiality_rate": (
                    len(results["essential_reactions"]) / len(model.reactions)
                    if len(model.reactions) > 0
                    else 0
                ),
                "network_breakdown": reaction_analysis.get("network_analysis", {}),
                "gene_association_stats": reaction_analysis.get("gene_association", {}),
            }

        return summary
