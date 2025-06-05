from typing import Any, Dict, List, Optional

import cobra
from pydantic import BaseModel, Field, PrivateAttr

from ..base import BaseTool, ToolRegistry, ToolResult
from .utils import ModelUtils


class ModelAnalysisConfig(BaseModel):
    """Configuration for model analysis tool"""

    model_config = {"protected_namespaces": ()}
    flux_threshold: float = 1e-6
    include_reactions: Optional[List[str]] = None
    include_subsystems: Optional[bool] = True
    track_metabolites: Optional[bool] = True


@ToolRegistry.register
class ModelAnalysisTool(BaseTool):
    """Tool for analyzing metabolic model properties"""

    tool_name = "analyze_metabolic_model"
    tool_description = """Analyze structural properties of a metabolic model including
    reaction connectivity, pathway completeness, and potential gaps."""

    _analysis_config: ModelAnalysisConfig = PrivateAttr()
    _utils: ModelUtils = PrivateAttr()

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Extract analysis_config from tool_config
        analysis_config_dict = config.get("analysis_config", {})
        if isinstance(analysis_config_dict, dict):
            self._analysis_config = ModelAnalysisConfig(**analysis_config_dict)
        else:
            # If analysis_config is already a Pydantic model or similar
            self._analysis_config = ModelAnalysisConfig(
                flux_threshold=getattr(analysis_config_dict, "flux_threshold", 1e-6),
                include_reactions=getattr(
                    analysis_config_dict, "include_reactions", None
                ),
                include_subsystems=getattr(
                    analysis_config_dict, "include_subsystems", True
                ),
                track_metabolites=getattr(
                    analysis_config_dict, "track_metabolites", True
                ),
            )
        self._utils = ModelUtils()

    @property
    def analysis_config(self) -> ModelAnalysisConfig:
        return self._analysis_config

    def _run_tool(self, model_path: str) -> ToolResult:
        try:
            # Load model
            model = self._utils.load_model(model_path)

            # Analyze model structure
            analysis_results = {
                "model_statistics": self._get_basic_statistics(model),
                "network_properties": self._analyze_network_properties(model),
                "subsystem_analysis": self._analyze_subsystems(model),
                "potential_issues": self._identify_model_issues(model),
            }

            return ToolResult(
                success=True,
                message="Model analysis completed successfully",
                data=analysis_results,
            )

        except Exception as e:
            return ToolResult(
                success=False, message="Error analyzing model", error=str(e)
            )

    def _get_basic_statistics(self, model: cobra.Model) -> Dict[str, Any]:
        """Get basic model statistics"""
        return {
            "num_reactions": len(model.reactions),
            "num_metabolites": len(model.metabolites),
            "num_genes": len(model.genes),
            "num_subsystems": len(
                set(rxn.subsystem for rxn in model.reactions if rxn.subsystem)
            ),
        }

    def _analyze_network_properties(self, model: cobra.Model) -> Dict[str, Any]:
        """Analyze network properties with improved metrics"""
        # Initialize statistics
        network_stats = {
            "connectivity_summary": {
                "total_connections": 0,
                "avg_connections_per_metabolite": 0,
                "max_connections": 0,
                "min_connections": float("inf"),
            },
            "highly_connected_metabolites": [],
            "isolated_metabolites": [],
            "choke_points": [],  # Metabolites that are sole producers/consumers
        }

        # Analyze each metabolite
        for metabolite in model.metabolites:
            num_reactions = len(metabolite.reactions)
            producing = [r for r in metabolite.reactions if metabolite in r.products]
            consuming = [r for r in metabolite.reactions if metabolite in r.reactants]

            # Update summary statistics
            network_stats["connectivity_summary"]["total_connections"] += num_reactions
            network_stats["connectivity_summary"]["max_connections"] = max(
                network_stats["connectivity_summary"]["max_connections"], num_reactions
            )
            network_stats["connectivity_summary"]["min_connections"] = min(
                network_stats["connectivity_summary"]["min_connections"], num_reactions
            )

            # Track highly connected metabolites (hub metabolites)
            if num_reactions > 10:
                network_stats["highly_connected_metabolites"].append(
                    {
                        "id": metabolite.id,
                        "name": metabolite.name,
                        "num_reactions": num_reactions,
                        "num_producing": len(producing),
                        "num_consuming": len(consuming),
                    }
                )

            # Track isolated metabolites
            if num_reactions <= 1:
                network_stats["isolated_metabolites"].append(
                    {
                        "id": metabolite.id,
                        "name": metabolite.name,
                        "num_reactions": num_reactions,
                    }
                )

            # Identify choke points
            if len(producing) == 1 or len(consuming) == 1:
                network_stats["choke_points"].append(
                    {
                        "id": metabolite.id,
                        "name": metabolite.name,
                        "single_producer": len(producing) == 1,
                        "single_consumer": len(consuming) == 1,
                    }
                )

        # Calculate average connectivity
        num_metabolites = len(model.metabolites)
        if num_metabolites > 0:
            network_stats["connectivity_summary"]["avg_connections_per_metabolite"] = (
                network_stats["connectivity_summary"]["total_connections"]
                / num_metabolites
            )

        # Sort and limit lists
        network_stats["highly_connected_metabolites"].sort(
            key=lambda x: x["num_reactions"], reverse=True
        )
        network_stats["highly_connected_metabolites"] = network_stats[
            "highly_connected_metabolites"
        ][
            :10
        ]  # Top 10 most connected

        return network_stats

    def _analyze_subsystems(self, model: cobra.Model) -> Dict[str, Any]:
        """Analyze subsystem properties"""
        subsystems = {}
        orphan_reactions = []

        for reaction in model.reactions:
            subsystem = reaction.subsystem or "Unknown"

            if subsystem not in subsystems:
                subsystems[subsystem] = {
                    "reaction_count": 0,
                    "gene_associated": 0,
                    "spontaneous": 0,
                    "reversible": 0,
                }

            stats = subsystems[subsystem]
            stats["reaction_count"] += 1

            if reaction.genes:
                stats["gene_associated"] += 1
            else:
                stats["spontaneous"] += 1

            if reaction.reversibility:
                stats["reversible"] += 1

            if not subsystem or subsystem == "Unknown":
                orphan_reactions.append(
                    {
                        "id": reaction.id,
                        "name": reaction.name,
                        "reversible": reaction.reversibility,
                    }
                )

        return {
            "subsystem_statistics": subsystems,
            "orphan_reactions": orphan_reactions,
            "num_subsystems": len(subsystems),
            "largest_subsystems": sorted(
                [(k, v["reaction_count"]) for k, v in subsystems.items()],
                key=lambda x: x[1],
                reverse=True,
            )[
                :5
            ],  # Top 5 largest subsystems
        }

    def _identify_model_issues(self, model: cobra.Model) -> Dict[str, Any]:
        """Identify potential issues in the model"""
        issues = {
            "dead_end_metabolites": [],
            "disconnected_reactions": [],
            "missing_genes": [],
            "boundary_issues": [],
        }

        # Find dead-end metabolites
        for metabolite in model.metabolites:
            if len(metabolite.reactions) <= 1:
                issues["dead_end_metabolites"].append(
                    {
                        "id": metabolite.id,
                        "name": metabolite.name,
                        "connected_reactions": [r.id for r in metabolite.reactions],
                    }
                )

        # Find disconnected reactions
        for reaction in model.reactions:
            if len(reaction.metabolites) == 0:
                issues["disconnected_reactions"].append(reaction.id)

        # Check for reactions without genes (excluding exchanges/demands)
        for reaction in model.reactions:
            if not reaction.genes and not (
                reaction.id.startswith("EX_")
                or reaction.id.startswith("DM_")
                or reaction.id.startswith("SK_")
            ):
                issues["missing_genes"].append(reaction.id)

        # Check boundary conditions
        boundary_metabolites = set()
        for reaction in model.reactions:
            if len(reaction.metabolites) == 1:
                boundary_metabolites.update(reaction.metabolites)

        issues["boundary_issues"] = [
            {"metabolite": met.id, "name": met.name} for met in boundary_metabolites
        ]

        return issues


@ToolRegistry.register
class PathwayAnalysisTool(BaseTool):
    """Tool for analyzing specific metabolic pathways"""

    tool_name = "analyze_pathway"
    tool_description = """Analyze specific metabolic pathways including flux distributions,
    gene associations, and regulatory features."""

    _utils: ModelUtils = PrivateAttr()

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._utils = ModelUtils()

    def _run_tool(self, input_data: Dict[str, Any]) -> ToolResult:
        try:
            model_path = input_data.get("model_path")
            pathway = input_data.get("pathway")
            if not model_path or not pathway:
                raise ValueError("Both model_path and pathway must be provided")

            model = self._utils.load_model(model_path)
            pathway_reactions = [
                rxn
                for rxn in model.reactions
                if rxn.subsystem and pathway.lower() in rxn.subsystem.lower()
            ]

            if not pathway_reactions:
                return ToolResult(
                    success=False,
                    message=f"No reactions found for pathway: {pathway}",
                    error="Pathway not found",
                )

            pathway_analysis = {
                "summary": {
                    "reaction_count": len(pathway_reactions),
                    "gene_coverage": len(
                        set(gene for rxn in pathway_reactions for gene in rxn.genes)
                    ),
                    "metabolite_count": len(
                        set(met for rxn in pathway_reactions for met in rxn.metabolites)
                    ),
                    "reversible_reactions": sum(
                        1 for rxn in pathway_reactions if rxn.reversibility
                    ),
                },
                "reactions": [
                    {
                        "id": rxn.id,
                        "name": rxn.name,
                        "reaction": rxn.build_reaction_string(),
                        "genes": [gene.id for gene in rxn.genes],
                        "metabolites": [met.id for met in rxn.metabolites],
                        "bounds": rxn.bounds,
                    }
                    for rxn in pathway_reactions
                ],
                "connectivity": {
                    "input_metabolites": list(
                        set(
                            met.id
                            for rxn in pathway_reactions
                            for met in rxn.reactants
                            if met
                            not in set(m for r in pathway_reactions for m in r.products)
                        )
                    ),
                    "output_metabolites": list(
                        set(
                            met.id
                            for rxn in pathway_reactions
                            for met in rxn.products
                            if met
                            not in set(
                                m for r in pathway_reactions for m in r.reactants
                            )
                        )
                    ),
                },
            }

            return ToolResult(
                success=True,
                message=f"Pathway analysis completed for: {pathway}",
                data=pathway_analysis,
            )

        except Exception as e:
            return ToolResult(
                success=False, message="Error analyzing pathway", error=str(e)
            )
