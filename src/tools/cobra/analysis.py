from typing import Any, Dict, List, Optional

import cobra
from pydantic import BaseModel, Field, PrivateAttr

from ..base import BaseTool, ToolRegistry, ToolResult
from .utils_optimized import OptimizedModelUtils


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
    _utils: OptimizedModelUtils = PrivateAttr()

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
        self._utils = OptimizedModelUtils(use_cache=True)

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
        # Lightweight network analysis - counts only, no full objects
        connectivity_counts = []
        highly_connected_count = 0
        isolated_count = 0
        choke_point_count = 0
        total_connections = 0

        # Single pass analysis to avoid storing full metabolite objects
        for metabolite in model.metabolites:
            num_reactions = len(metabolite.reactions)
            producing = [r for r in metabolite.reactions if metabolite in r.products]
            consuming = [r for r in metabolite.reactions if metabolite in r.reactants]

            connectivity_counts.append(num_reactions)
            total_connections += num_reactions

            # Count categories without storing objects
            if num_reactions > 10:
                highly_connected_count += 1
            if num_reactions <= 1:
                isolated_count += 1
            if len(producing) == 1 or len(consuming) == 1:
                choke_point_count += 1

        network_stats = {
            "connectivity_summary": {
                "total_metabolites": len(model.metabolites),
                "total_connections": total_connections,
                "avg_connections_per_metabolite": (
                    total_connections / len(model.metabolites)
                    if model.metabolites
                    else 0
                ),
                "max_connections": (
                    max(connectivity_counts) if connectivity_counts else 0
                ),
                "min_connections": (
                    min(connectivity_counts) if connectivity_counts else 0
                ),
            },
            "network_categories": {
                "highly_connected": highly_connected_count,  # >10 reactions
                "isolated": isolated_count,  # ≤1 reaction
                "choke_points": choke_point_count,  # single producer/consumer
                "well_connected": len(model.metabolites)
                - highly_connected_count
                - isolated_count,
            },
        }

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
        # Lightweight issue analysis - counts only
        dead_end_count = 0
        disconnected_reactions = []
        missing_genes = []
        boundary_count = 0

        # Count dead-end metabolites without storing objects
        for metabolite in model.metabolites:
            if len(metabolite.reactions) <= 1:
                dead_end_count += 1

        # Find disconnected reactions
        for reaction in model.reactions:
            if len(reaction.metabolites) == 0:
                disconnected_reactions.append(reaction.id)

        # Check for reactions without genes (excluding exchanges/demands)
        for reaction in model.reactions:
            if not reaction.genes and not (
                reaction.id.startswith("EX_")
                or reaction.id.startswith("DM_")
                or reaction.id.startswith("SK_")
            ):
                missing_genes.append(reaction.id)

        # Count boundary metabolites
        boundary_metabolites = set()
        for reaction in model.reactions:
            if len(reaction.metabolites) == 1:
                boundary_metabolites.update(reaction.metabolites)
        boundary_count = len(boundary_metabolites)

        return {
            "issue_summary": {
                "dead_end_metabolites": dead_end_count,
                "disconnected_reactions": len(disconnected_reactions),
                "missing_genes": len(missing_genes),
                "boundary_metabolites": boundary_count,
            },
            "critical_issues": {
                "disconnected_reactions": disconnected_reactions[:5],  # Top 5 only
                "reactions_missing_genes": missing_genes[:10],  # Top 10 only
            },
        }


@ToolRegistry.register
class PathwayAnalysisTool(BaseTool):
    """Tool for analyzing specific metabolic pathways"""

    tool_name = "analyze_pathway"
    tool_description = """Analyze specific metabolic pathways including flux distributions,
    gene associations, and regulatory features."""

    _utils: OptimizedModelUtils = PrivateAttr()

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._utils = OptimizedModelUtils(use_cache=True)

    def _run_tool(self, input_data: Any) -> ToolResult:
        try:
            # Support both dict and string inputs (for consistency with other tools)
            if isinstance(input_data, dict):
                model_path = input_data.get("model_path")
                model_object = input_data.get("model_object")
                pathway = input_data.get("pathway")

                # Load model from path or use provided object
                if model_object is not None:
                    model = model_object
                    # For pathway analysis with model objects, use default pathway analysis
                    pathway = pathway or "all_subsystems"
                elif model_path:
                    model = self._utils.load_model(model_path)
                    if not pathway:
                        raise ValueError(
                            "pathway parameter required when using model_path"
                        )
                else:
                    raise ValueError(
                        "Either model_path or model_object must be provided"
                    )
            else:
                # For string input, we can't determine pathway, so raise an error
                raise ValueError(
                    "PathwayAnalysisTool requires dictionary input with model_path/model_object and pathway keys"
                )

            # Handle models with empty subsystems by using alternative grouping methods
            pathway_reactions = []

            # Try subsystem-based search first
            if any(rxn.subsystem for rxn in model.reactions):
                pathway_reactions = [
                    rxn
                    for rxn in model.reactions
                    if rxn.subsystem and pathway.lower() in rxn.subsystem.lower()
                ]
            else:
                # Fallback: Use reaction ID patterns for pathway analysis
                pathway_mapping = {
                    "glycolysis": [
                        "PGI",
                        "PFK",
                        "FBA",
                        "TPI",
                        "GAPD",
                        "PGK",
                        "PGM",
                        "ENO",
                        "PYK",
                    ],
                    "tca": [
                        "CS",
                        "ACONTa",
                        "ACONTb",
                        "ICDHyr",
                        "AKGDH",
                        "SUCOAS",
                        "SUCDi",
                        "FUM",
                        "MDH",
                    ],
                    "pentose": [
                        "G6PDH2r",
                        "PGL",
                        "GND",
                        "RPI",
                        "RPE",
                        "TKT1",
                        "TALA",
                        "TKT2",
                    ],
                    "transport": ["EX_", "t", "abc", "upt"],
                    "exchange": ["EX_"],
                    "central": ["PGI", "PFK", "FBA", "CS", "ACONTa", "G6PDH2r"],
                    "energy": ["ATPS4r", "NADH", "CYTBD", "ATP"],
                }

                # Get reaction patterns for the requested pathway
                patterns = []
                for pathway_key, reaction_patterns in pathway_mapping.items():
                    if pathway_key in pathway.lower():
                        patterns = reaction_patterns
                        break

                # If no specific patterns found, search in reaction IDs and names
                if not patterns:
                    pathway_reactions = [
                        rxn
                        for rxn in model.reactions
                        if (
                            pathway.lower() in rxn.id.lower()
                            or (rxn.name and pathway.lower() in rxn.name.lower())
                        )
                    ]
                else:
                    # Search for reactions matching the patterns
                    pathway_reactions = []
                    for rxn in model.reactions:
                        for pattern in patterns:
                            if pattern.upper() in rxn.id.upper():
                                pathway_reactions.append(rxn)
                                break

            if not pathway_reactions:
                # Provide helpful error message with suggestions
                available_subsystems = set(
                    rxn.subsystem for rxn in model.reactions if rxn.subsystem
                )
                if available_subsystems:
                    suggestions = list(available_subsystems)[:5]
                    suggestion_text = f"Available subsystems: {', '.join(suggestions)}"
                else:
                    # For models without subsystems, suggest reaction ID patterns
                    suggestion_text = "Try: 'glycolysis', 'tca', 'pentose', 'transport', 'exchange', or 'central'"

                return ToolResult(
                    success=False,
                    message=f"No reactions found for pathway: {pathway}. {suggestion_text}",
                    error="Pathway not found",
                )

            # Streamlined pathway analysis - avoid redundant data storage
            total_genes = len(
                set(gene for rxn in pathway_reactions for gene in rxn.genes)
            )
            total_metabolites = len(
                set(met for rxn in pathway_reactions for met in rxn.metabolites)
            )
            reversible_count = sum(1 for rxn in pathway_reactions if rxn.reversibility)

            # Calculate connectivity counts without storing full lists
            all_reactants = set()
            all_products = set()
            for rxn in pathway_reactions:
                all_reactants.update(met.id for met in rxn.reactants)
                all_products.update(met.id for met in rxn.products)

            pathway_analysis = {
                "summary": {
                    "pathway_name": pathway,
                    "reaction_count": len(pathway_reactions),
                    "gene_coverage": total_genes,
                    "metabolite_count": total_metabolites,
                    "reversible_reactions": reversible_count,
                    "irreversible_reactions": len(pathway_reactions) - reversible_count,
                },
                "reaction_list": [
                    rxn.id for rxn in pathway_reactions
                ],  # IDs only, no full details
                "connectivity": {
                    "input_metabolites": len(all_reactants - all_products),
                    "output_metabolites": len(all_products - all_reactants),
                    "internal_metabolites": len(all_reactants & all_products),
                    "total_unique_metabolites": len(all_reactants | all_products),
                },
                "top_reactions": [
                    {
                        "id": rxn.id,
                        "name": rxn.name or rxn.id,
                        "gene_count": len(rxn.genes),
                        "metabolite_count": len(rxn.metabolites),
                        "reversible": rxn.reversibility,
                    }
                    for rxn in sorted(
                        pathway_reactions, key=lambda x: len(x.genes), reverse=True
                    )[
                        :5
                    ]  # Top 5 by gene count
                ],
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
