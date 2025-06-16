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
from .utils_optimized import OptimizedModelUtils


class EssentialityConfig(BaseModel):
    """Configuration for Essentiality Analysis with enhanced numerical precision"""

    model_config = {"protected_namespaces": ()}
    include_genes: bool = True
    include_reactions: bool = True
    solver: str = "glpk"
    processes: Optional[int] = None

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

            # Analyze gene essentiality
            if include_genes:
                essential_genes = find_essential_genes(model, threshold=threshold)
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
                    model, threshold=threshold
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
        """Perform detailed analysis of essential genes"""

        gene_analysis = {
            "essential_gene_details": [],
            "functional_categories": {},
            "subsystem_analysis": {},
            "statistics": {},
        }

        # Analyze each essential gene
        for gene in essential_genes:
            # Get associated reactions
            associated_reactions = [rxn for rxn in model.reactions if gene in rxn.genes]

            # Categorize by subsystems
            subsystems = set()
            for rxn in associated_reactions:
                if rxn.subsystem:
                    subsystems.add(rxn.subsystem)

            gene_details = {
                "gene_id": gene.id,
                "gene_name": gene.name if gene.name else gene.id,
                "associated_reactions": [rxn.id for rxn in associated_reactions],
                "num_reactions": len(associated_reactions),
                "subsystems": list(subsystems),
                "functional_category": self._categorize_gene_function(list(subsystems)),
            }

            gene_analysis["essential_gene_details"].append(gene_details)

            # Count by functional category
            func_cat = gene_details["functional_category"]
            if func_cat not in gene_analysis["functional_categories"]:
                gene_analysis["functional_categories"][func_cat] = 0
            gene_analysis["functional_categories"][func_cat] += 1

            # Count by subsystem
            for subsystem in subsystems:
                if subsystem not in gene_analysis["subsystem_analysis"]:
                    gene_analysis["subsystem_analysis"][subsystem] = []
                gene_analysis["subsystem_analysis"][subsystem].append(gene.id)

        # Generate statistics
        gene_analysis["statistics"] = {
            "total_essential_genes": len(essential_genes),
            "genes_with_single_reaction": len(
                [
                    g
                    for g in gene_analysis["essential_gene_details"]
                    if g["num_reactions"] == 1
                ]
            ),
            "genes_with_multiple_reactions": len(
                [
                    g
                    for g in gene_analysis["essential_gene_details"]
                    if g["num_reactions"] > 1
                ]
            ),
            "most_connected_gene": (
                max(
                    gene_analysis["essential_gene_details"],
                    key=lambda x: x["num_reactions"],
                )
                if gene_analysis["essential_gene_details"]
                else None
            ),
            "subsystems_affected": len(gene_analysis["subsystem_analysis"]),
        }

        return gene_analysis

    def _detailed_reaction_analysis(
        self,
        model: cobra.Model,
        essential_reactions: List,
        wild_type_growth: float,
        threshold: float,
    ) -> Dict[str, Any]:
        """Perform detailed analysis of essential reactions"""

        reaction_analysis = {
            "essential_reaction_details": [],
            "subsystem_analysis": {},
            "network_analysis": {},
            "statistics": {},
        }

        # Analyze each essential reaction
        for rxn in essential_reactions:
            # Analyze network position
            is_exchange = rxn.id.startswith(("EX_", "DM_", "SK_"))
            is_transport = "transport" in rxn.name.lower() if rxn.name else False

            reaction_details = {
                "reaction_id": rxn.id,
                "reaction_name": rxn.name if rxn.name else rxn.id,
                "equation": rxn.build_reaction_string(),
                "subsystem": rxn.subsystem or "Unknown",
                "genes": [gene.id for gene in rxn.genes],
                "num_genes": len(rxn.genes),
                "is_reversible": rxn.reversibility,
                "is_exchange": is_exchange,
                "is_transport": is_transport,
                "bounds": rxn.bounds,
            }

            reaction_analysis["essential_reaction_details"].append(reaction_details)

            # Count by subsystem
            subsystem = rxn.subsystem or "Unknown"
            if subsystem not in reaction_analysis["subsystem_analysis"]:
                reaction_analysis["subsystem_analysis"][subsystem] = []
            reaction_analysis["subsystem_analysis"][subsystem].append(rxn.id)

        # Network analysis
        reaction_analysis["network_analysis"] = {
            "exchange_reactions": len(
                [
                    r
                    for r in reaction_analysis["essential_reaction_details"]
                    if r["is_exchange"]
                ]
            ),
            "transport_reactions": len(
                [
                    r
                    for r in reaction_analysis["essential_reaction_details"]
                    if r["is_transport"]
                ]
            ),
            "metabolic_reactions": len(
                [
                    r
                    for r in reaction_analysis["essential_reaction_details"]
                    if not r["is_exchange"] and not r["is_transport"]
                ]
            ),
            "gene_associated": len(
                [
                    r
                    for r in reaction_analysis["essential_reaction_details"]
                    if r["num_genes"] > 0
                ]
            ),
            "spontaneous": len(
                [
                    r
                    for r in reaction_analysis["essential_reaction_details"]
                    if r["num_genes"] == 0
                ]
            ),
        }

        # Generate statistics
        reaction_analysis["statistics"] = {
            "total_essential_reactions": len(essential_reactions),
            "subsystems_affected": len(reaction_analysis["subsystem_analysis"]),
            "most_gene_associated": (
                max(
                    reaction_analysis["essential_reaction_details"],
                    key=lambda x: x["num_genes"],
                )
                if reaction_analysis["essential_reaction_details"]
                else None
            ),
        }

        return reaction_analysis

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
            summary["gene_essentiality"] = {
                "essential_count": len(results["essential_genes"]),
                "essentiality_rate": (
                    len(results["essential_genes"]) / len(model.genes)
                    if len(model.genes) > 0
                    else 0
                ),
                "top_functional_categories": (
                    sorted(
                        results["gene_analysis"]["functional_categories"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5]
                    if results["gene_analysis"]["functional_categories"]
                    else []
                ),
            }

        if results["essential_reactions"] is not None:
            summary["reaction_essentiality"] = {
                "essential_count": len(results["essential_reactions"]),
                "essentiality_rate": (
                    len(results["essential_reactions"]) / len(model.reactions)
                    if len(model.reactions) > 0
                    else 0
                ),
                "network_breakdown": results["reaction_analysis"]["network_analysis"],
            }

        return summary
