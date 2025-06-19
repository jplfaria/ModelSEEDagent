"""
Biochemical Context Enhancer for ModelSEEDagent

Automatically enriches tool outputs with biochemical context to enable
intelligent AI reasoning about metabolic processes and biochemical data.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class BiochemicalContext:
    """Container for biochemical context information"""

    def __init__(self, entity_id: str, entity_type: str):
        self.entity_id = entity_id
        self.entity_type = entity_type  # 'compound', 'reaction', 'gene'
        self.name: Optional[str] = None
        self.formula: Optional[str] = None
        self.aliases: List[str] = []
        self.pathways: List[str] = []
        self.properties: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.entity_id,
            "type": self.entity_type,
            "name": self.name,
            "formula": self.formula,
            "aliases": self.aliases,
            "pathways": self.pathways,
            "properties": self.properties,
        }


class BiochemContextEnhancer:
    """
    Automatic context injection for tool outputs

    Enriches biochemical tool results with human-readable names, formulas,
    pathway information, and cross-database aliases to enable intelligent
    AI reasoning about metabolic processes.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".modelseed-agent" / "context_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Context caches
        self._compound_cache: Dict[str, BiochemicalContext] = {}
        self._reaction_cache: Dict[str, BiochemicalContext] = {}
        self._gene_cache: Dict[str, BiochemicalContext] = {}
        self._pathway_cache: Dict[str, List[str]] = {}

        # Load cached context data
        self._load_context_cache()

        # Pattern matching for ID recognition
        self._compound_pattern = re.compile(r"cpd\d{5}")
        self._reaction_pattern = re.compile(r"rxn\d{5}")
        self._bigg_pattern = re.compile(r"[A-Za-z0-9_]+(_[cep])?$")

        logger.info("BiochemContextEnhancer initialized")

    def enhance_context(
        self, query: str, context: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """
        Enhance analysis context with biochemical knowledge for Intelligence Framework

        Args:
            query: User query
            context: Analysis context
            prompt: Generated prompt

        Returns:
            Enhanced context with biochemical enrichment
        """
        enhanced_context = context.copy()

        # Add biochemical knowledge based on query content
        query_lower = query.lower()

        # Detect biochemical entities mentioned in query
        compounds_mentioned = self._compound_pattern.findall(query)
        reactions_mentioned = self._reaction_pattern.findall(query)

        # Add relevant biochemical context
        if "glucose" in query_lower or "growth" in query_lower:
            enhanced_context["biochemical_focus"] = (
                "Central carbon metabolism and energy production"
            )
            enhanced_context["key_pathways"] = [
                "Glycolysis",
                "TCA cycle",
                "Respiration",
            ]
        elif "flux" in query_lower or "variability" in query_lower:
            enhanced_context["biochemical_focus"] = (
                "Metabolic flux analysis and network flexibility"
            )
            enhanced_context["key_pathways"] = ["Core metabolism", "Exchange reactions"]
        elif "media" in query_lower or "nutrient" in query_lower:
            enhanced_context["biochemical_focus"] = (
                "Nutritional requirements and media optimization"
            )
            enhanced_context["key_pathways"] = ["Transport systems", "Biosynthesis"]
        else:
            enhanced_context["biochemical_focus"] = "General metabolic analysis"
            enhanced_context["key_pathways"] = ["Core metabolism"]

        # Add entity-specific context if found
        if compounds_mentioned:
            enhanced_context["compounds_of_interest"] = compounds_mentioned
        if reactions_mentioned:
            enhanced_context["reactions_of_interest"] = reactions_mentioned

        # Add reasoning guidance
        enhanced_context["reasoning_guidance"] = self._generate_reasoning_guidance(
            query_lower
        )

        return enhanced_context

    def _generate_reasoning_guidance(self, query_lower: str) -> List[str]:
        """Generate reasoning guidance based on query type"""
        guidance = []

        if "analyze" in query_lower:
            guidance.append(
                "Focus on quantitative analysis and biological interpretation"
            )
            guidance.append("Identify key metabolic patterns and mechanisms")
        if "growth" in query_lower:
            guidance.append("Consider energy efficiency and resource utilization")
            guidance.append("Evaluate growth limitations and bottlenecks")
        if "flux" in query_lower:
            guidance.append("Examine flux distributions and network topology")
            guidance.append("Assess metabolic flexibility and robustness")
        if "optimization" in query_lower or "engineering" in query_lower:
            guidance.append("Identify targets for metabolic engineering")
            guidance.append("Evaluate trade-offs and constraints")

        if not guidance:
            guidance.append("Apply biochemical knowledge to interpret results")
            guidance.append("Connect findings to biological mechanisms")

        return guidance

    def enhance_tool_result(
        self,
        tool_name: str,
        result_data: Dict[str, Any],
        session_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add biochemical context to any tool result

        Args:
            tool_name: Name of the tool that generated the result
            result_data: Raw tool output data
            session_context: Additional context from current session

        Returns:
            Enhanced result with biochemical context
        """
        try:
            if not result_data or not isinstance(result_data, dict):
                return result_data

            enhanced_data = result_data.copy()

            # Tool-specific enhancement strategies
            if tool_name in ["run_metabolic_fba", "run_cobra_fba"]:
                enhanced_data = self._enhance_fba_result(enhanced_data)
            elif tool_name in ["run_flux_variability_analysis"]:
                enhanced_data = self._enhance_fva_result(enhanced_data)
            elif tool_name in ["run_flux_sampling"]:
                enhanced_data = self._enhance_flux_sampling_result(enhanced_data)
            elif "media" in tool_name.lower():
                enhanced_data = self._enhance_media_result(enhanced_data)
            elif tool_name in ["analyze_essentiality", "run_gene_deletion_analysis"]:
                enhanced_data = self._enhance_essentiality_result(enhanced_data)
            elif tool_name in ["run_moma"]:
                enhanced_data = self._enhance_moma_result(enhanced_data)
            else:
                # Generic enhancement for any tool with biochemical IDs
                enhanced_data = self._generic_enhancement(enhanced_data)

            # Add context summary for AI reasoning
            enhanced_data = self._add_context_summary(enhanced_data, tool_name)

            logger.debug(f"Enhanced {tool_name} result with biochemical context")
            return enhanced_data

        except Exception as e:
            logger.error(f"Failed to enhance {tool_name} result: {e}")
            return result_data

    def _enhance_fba_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance FBA results with reaction and compound context"""
        enhanced = data.copy()

        # Enhance flux data
        if "fluxes" in enhanced:
            enhanced["fluxes"] = self._enhance_flux_dict(enhanced["fluxes"])

        if "significant_fluxes" in enhanced:
            enhanced["significant_fluxes"] = self._enhance_flux_dict(
                enhanced["significant_fluxes"]
            )

        # Enhance exchange fluxes specifically
        if "exchange_fluxes" in enhanced:
            enhanced["exchange_fluxes"] = self._enhance_exchange_fluxes(
                enhanced["exchange_fluxes"]
            )

        # Add pathway analysis
        if "fluxes" in enhanced or "significant_fluxes" in enhanced:
            enhanced["pathway_analysis"] = self._analyze_pathway_activity(enhanced)

        return enhanced

    def _enhance_fva_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance FVA results with reaction context and variability interpretation"""
        enhanced = data.copy()

        if "variability" in enhanced:
            enhanced_variability = {}
            for reaction_id, bounds in enhanced["variability"].items():
                context = self._get_reaction_context(reaction_id)
                enhanced_variability[reaction_id] = {
                    "min_flux": (
                        bounds[0]
                        if isinstance(bounds, (list, tuple))
                        else bounds.get("min", 0)
                    ),
                    "max_flux": (
                        bounds[1]
                        if isinstance(bounds, (list, tuple))
                        else bounds.get("max", 0)
                    ),
                    "context": context.to_dict() if context else None,
                    "flexibility": self._calculate_flexibility(bounds),
                }
            enhanced["variability"] = enhanced_variability

        # Add flexibility analysis
        if "variability" in enhanced:
            enhanced["flexibility_analysis"] = self._analyze_network_flexibility(
                enhanced["variability"]
            )

        return enhanced

    def _enhance_flux_sampling_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance flux sampling with statistical context"""
        enhanced = data.copy()

        if "flux_samples" in enhanced:
            # Add reaction context to samples
            enhanced["reaction_statistics"] = {}
            for reaction_id in enhanced["flux_samples"].keys():
                context = self._get_reaction_context(reaction_id)
                samples = enhanced["flux_samples"][reaction_id]

                enhanced["reaction_statistics"][reaction_id] = {
                    "context": context.to_dict() if context else None,
                    "mean_flux": sum(samples) / len(samples) if samples else 0,
                    "flux_std": self._calculate_std(samples) if len(samples) > 1 else 0,
                    "active_fraction": (
                        len([s for s in samples if abs(s) > 1e-6]) / len(samples)
                        if samples
                        else 0
                    ),
                }

        return enhanced

    def _enhance_media_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance media composition with compound context"""
        enhanced = data.copy()

        # Enhance media components
        if "media_components" in enhanced:
            enhanced_components = []
            for component in enhanced["media_components"]:
                if isinstance(component, str):
                    context = self._get_compound_context(component)
                    enhanced_components.append(
                        {
                            "id": component,
                            "context": context.to_dict() if context else None,
                            "role": self._infer_nutrient_role(component, context),
                        }
                    )
                else:
                    enhanced_components.append(component)
            enhanced["media_components"] = enhanced_components

        # Enhance concentrations
        if "concentrations" in enhanced:
            enhanced_concentrations = {}
            for compound_id, conc in enhanced["concentrations"].items():
                context = self._get_compound_context(compound_id)
                enhanced_concentrations[compound_id] = {
                    "concentration": conc,
                    "context": context.to_dict() if context else None,
                    "role": self._infer_nutrient_role(compound_id, context),
                }
            enhanced["concentrations"] = enhanced_concentrations

        return enhanced

    def _enhance_essentiality_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance gene essentiality with functional context"""
        enhanced = data.copy()

        if "essential_genes" in enhanced:
            enhanced_genes = {}
            for gene_id, essentiality_data in enhanced["essential_genes"].items():
                gene_context = self._get_gene_context(gene_id)

                if isinstance(essentiality_data, dict):
                    enhanced_genes[gene_id] = essentiality_data.copy()
                    enhanced_genes[gene_id]["context"] = (
                        gene_context.to_dict() if gene_context else None
                    )
                else:
                    enhanced_genes[gene_id] = {
                        "essential": essentiality_data,
                        "context": gene_context.to_dict() if gene_context else None,
                    }
            enhanced["essential_genes"] = enhanced_genes

        return enhanced

    def _enhance_moma_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance MOMA results with perturbation context"""
        enhanced = data.copy()

        # Enhance flux changes
        if "flux_changes" in enhanced:
            enhanced["flux_changes"] = self._enhance_flux_dict(enhanced["flux_changes"])

        # Add perturbation impact analysis
        if "flux_changes" in enhanced:
            enhanced["perturbation_analysis"] = self._analyze_perturbation_impact(
                enhanced["flux_changes"]
            )

        return enhanced

    def _enhance_flux_dict(self, flux_dict: Dict[str, float]) -> Dict[str, Any]:
        """Enhance a dictionary of reaction fluxes with context"""
        enhanced = {}

        for reaction_id, flux_value in flux_dict.items():
            context = self._get_reaction_context(reaction_id)
            enhanced[reaction_id] = {
                "flux": flux_value,
                "context": context.to_dict() if context else None,
                "direction": (
                    "forward"
                    if flux_value > 0
                    else "reverse" if flux_value < 0 else "inactive"
                ),
                "magnitude": abs(flux_value),
            }

        return enhanced

    def _enhance_exchange_fluxes(
        self, exchange_dict: Dict[str, float]
    ) -> Dict[str, Any]:
        """Specifically enhance exchange reaction fluxes"""
        enhanced = {}

        for reaction_id, flux_value in exchange_dict.items():
            # Extract compound from exchange reaction
            compound_id = self._extract_compound_from_exchange(reaction_id)

            reaction_context = self._get_reaction_context(reaction_id)
            compound_context = (
                self._get_compound_context(compound_id) if compound_id else None
            )

            enhanced[reaction_id] = {
                "flux": flux_value,
                "reaction_context": (
                    reaction_context.to_dict() if reaction_context else None
                ),
                "compound_context": (
                    compound_context.to_dict() if compound_context else None
                ),
                "direction": (
                    "uptake"
                    if flux_value < 0
                    else "secretion" if flux_value > 0 else "inactive"
                ),
                "rate": abs(flux_value),
            }

        return enhanced

    def _get_compound_context(self, compound_id: str) -> Optional[BiochemicalContext]:
        """Get or create biochemical context for a compound"""
        if compound_id in self._compound_cache:
            return self._compound_cache[compound_id]

        # Try to resolve using biochemical tools
        context = self._resolve_compound_context(compound_id)
        if context:
            self._compound_cache[compound_id] = context

        return context

    def _get_reaction_context(self, reaction_id: str) -> Optional[BiochemicalContext]:
        """Get or create biochemical context for a reaction"""
        if reaction_id in self._reaction_cache:
            return self._reaction_cache[reaction_id]

        # Try to resolve using biochemical tools
        context = self._resolve_reaction_context(reaction_id)
        if context:
            self._reaction_cache[reaction_id] = context

        return context

    def _get_gene_context(self, gene_id: str) -> Optional[BiochemicalContext]:
        """Get or create biochemical context for a gene"""
        if gene_id in self._gene_cache:
            return self._gene_cache[gene_id]

        # Try to resolve using available data
        context = self._resolve_gene_context(gene_id)
        if context:
            self._gene_cache[gene_id] = context

        return context

    def _resolve_compound_context(
        self, compound_id: str
    ) -> Optional[BiochemicalContext]:
        """Resolve compound context using biochemical resolver"""
        try:
            # Import here to avoid circular imports
            try:
                from ..tools.biochem.resolver import resolve_biochem_entity
            except ImportError:
                from tools.biochem.resolver import resolve_biochem_entity

            result = resolve_biochem_entity(compound_id, entity_type="compound")
            if result and result.get("success"):
                data = result.get("data", {})
                context = BiochemicalContext(compound_id, "compound")
                context.name = data.get("name")
                context.formula = data.get("formula")
                context.aliases = data.get("aliases", [])
                context.properties = {
                    "mass": data.get("mass"),
                    "charge": data.get("charge"),
                    "inchikey": data.get("inchikey"),
                }
                return context
        except Exception as e:
            logger.debug(f"Could not resolve compound {compound_id}: {e}")

        return None

    def _resolve_reaction_context(
        self, reaction_id: str
    ) -> Optional[BiochemicalContext]:
        """Resolve reaction context using biochemical resolver"""
        try:
            # Import here to avoid circular imports
            try:
                from ..tools.biochem.resolver import resolve_biochem_entity
            except ImportError:
                from tools.biochem.resolver import resolve_biochem_entity

            result = resolve_biochem_entity(reaction_id, entity_type="reaction")
            if result and result.get("success"):
                data = result.get("data", {})
                context = BiochemicalContext(reaction_id, "reaction")
                context.name = data.get("name")
                context.formula = data.get("equation")
                context.aliases = data.get("aliases", [])
                context.pathways = data.get("pathways", [])
                context.properties = {
                    "ec_number": data.get("ec_number"),
                    "reversible": data.get("reversible"),
                    "subsystem": data.get("subsystem"),
                }
                return context
        except Exception as e:
            logger.debug(f"Could not resolve reaction {reaction_id}: {e}")

        return None

    def _resolve_gene_context(self, gene_id: str) -> Optional[BiochemicalContext]:
        """Resolve gene context using available data"""
        # Basic gene context - can be enhanced with genome annotation data
        context = BiochemicalContext(gene_id, "gene")
        context.name = gene_id  # Default to ID as name

        # Could be enhanced with:
        # - Gene product information
        # - Functional annotations
        # - Pathway associations

        return context

    def _analyze_pathway_activity(self, fba_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pathway-level activity from FBA results"""
        pathway_activity = {}

        flux_source = fba_data.get("significant_fluxes", fba_data.get("fluxes", {}))

        for reaction_id, flux_info in flux_source.items():
            flux_value = (
                flux_info.get("flux", flux_info)
                if isinstance(flux_info, dict)
                else flux_info
            )

            if abs(flux_value) > 1e-6:  # Active reaction
                context = self._get_reaction_context(reaction_id)
                if context and context.pathways:
                    for pathway in context.pathways:
                        if pathway not in pathway_activity:
                            pathway_activity[pathway] = {
                                "active_reactions": 0,
                                "total_flux": 0.0,
                                "reactions": [],
                            }

                        pathway_activity[pathway]["active_reactions"] += 1
                        pathway_activity[pathway]["total_flux"] += abs(flux_value)
                        pathway_activity[pathway]["reactions"].append(
                            {
                                "id": reaction_id,
                                "flux": flux_value,
                                "name": context.name,
                            }
                        )

        return pathway_activity

    def _analyze_network_flexibility(
        self, variability_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze network flexibility from FVA results"""
        flexibility_analysis = {
            "highly_flexible": [],
            "moderately_flexible": [],
            "rigid": [],
            "blocked": [],
        }

        for reaction_id, var_info in variability_data.items():
            flexibility = var_info.get("flexibility", 0)

            if flexibility > 10:
                flexibility_analysis["highly_flexible"].append(reaction_id)
            elif flexibility > 1:
                flexibility_analysis["moderately_flexible"].append(reaction_id)
            elif flexibility > 1e-6:
                flexibility_analysis["rigid"].append(reaction_id)
            else:
                flexibility_analysis["blocked"].append(reaction_id)

        return flexibility_analysis

    def _analyze_perturbation_impact(
        self, flux_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze perturbation impact from MOMA results"""
        impact_analysis = {
            "major_changes": [],
            "moderate_changes": [],
            "minor_changes": [],
            "pathway_impacts": {},
        }

        for reaction_id, change_info in flux_changes.items():
            flux_change = (
                change_info.get("flux", change_info)
                if isinstance(change_info, dict)
                else change_info
            )

            if abs(flux_change) > 5:
                impact_analysis["major_changes"].append(reaction_id)
            elif abs(flux_change) > 1:
                impact_analysis["moderate_changes"].append(reaction_id)
            else:
                impact_analysis["minor_changes"].append(reaction_id)

        return impact_analysis

    def _infer_nutrient_role(
        self, compound_id: str, context: Optional[BiochemicalContext]
    ) -> str:
        """Infer the nutritional role of a compound"""
        if not context:
            return "unknown"

        name = (context.name or "").lower()
        formula = context.formula or ""

        # Carbon sources
        if "glucose" in name or "glc" in compound_id or "C6H12O6" in formula:
            return "carbon_source"
        elif "pyruvate" in name or "acetate" in name:
            return "carbon_source"

        # Nitrogen sources
        elif "ammonia" in name or "nh3" in name or "glutamate" in name:
            return "nitrogen_source"
        elif "nitrate" in name or "nitrite" in name:
            return "nitrogen_source"

        # Phosphorus sources
        elif "phosphate" in name or "pi" in name:
            return "phosphorus_source"

        # Sulfur sources
        elif "sulfate" in name or "so4" in name:
            return "sulfur_source"

        # Trace elements
        elif any(metal in name for metal in ["mg", "ca", "fe", "zn", "mn", "co", "ni"]):
            return "trace_element"

        # Vitamins/cofactors
        elif any(
            vitamin in name for vitamin in ["biotin", "thiamine", "folate", "cobalamin"]
        ):
            return "vitamin_cofactor"

        return "other"

    def _extract_compound_from_exchange(
        self, exchange_reaction_id: str
    ) -> Optional[str]:
        """Extract compound ID from exchange reaction ID"""
        # Common patterns: EX_cpd00027_e, EX_glc__D_e
        if "EX_" in exchange_reaction_id:
            # Remove EX_ prefix and compartment suffix
            compound_part = exchange_reaction_id.replace("EX_", "").split("_")[0]

            # Check if it's already a ModelSEED ID
            if self._compound_pattern.match(compound_part):
                return compound_part

            # Try to translate from BiGG or other format
            try:
                try:
                    from ..tools.biochem.id_translator import translate_database_ids
                except ImportError:
                    from tools.biochem.id_translator import translate_database_ids
                result = translate_database_ids([compound_part], target_db="modelseed")
                if result and result.get("translations"):
                    return result["translations"].get(compound_part)
            except Exception:
                pass

        return None

    def _calculate_flexibility(self, bounds: Union[List, Tuple, Dict]) -> float:
        """Calculate reaction flexibility from FVA bounds"""
        if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
            return abs(bounds[1] - bounds[0])
        elif isinstance(bounds, dict):
            min_flux = bounds.get("min", 0)
            max_flux = bounds.get("max", 0)
            return abs(max_flux - min_flux)
        return 0.0

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def _add_context_summary(
        self, enhanced_data: Dict[str, Any], tool_name: str
    ) -> Dict[str, Any]:
        """Add AI-readable context summary"""
        summary = {
            "context_type": "biochemical_enhancement",
            "tool_name": tool_name,
            "enhancement_applied": True,
            "context_notes": [],
        }

        # Tool-specific context notes
        if tool_name in ["run_metabolic_fba", "run_cobra_fba"]:
            summary["context_notes"].append(
                "Flux data enhanced with reaction names and pathway information"
            )
            if "pathway_analysis" in enhanced_data:
                summary["context_notes"].append(
                    "Pathway-level activity analysis included"
                )

        elif tool_name in ["run_flux_variability_analysis"]:
            summary["context_notes"].append(
                "Reaction variability enhanced with flexibility analysis"
            )

        elif "media" in tool_name.lower():
            summary["context_notes"].append(
                "Media components enhanced with compound names and nutritional roles"
            )

        enhanced_data["_context_summary"] = summary
        return enhanced_data

    def _generic_enhancement(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generic enhancement for any data containing biochemical IDs"""
        enhanced = data.copy()

        # Look for compound and reaction IDs in the data
        for key, value in data.items():
            if isinstance(value, str):
                # Check if it's a biochemical ID
                if self._compound_pattern.match(value) or self._reaction_pattern.match(
                    value
                ):
                    context = (
                        self._get_compound_context(value)
                        if value.startswith("cpd")
                        else self._get_reaction_context(value)
                    )
                    if context:
                        enhanced[f"{key}_context"] = context.to_dict()

            elif isinstance(value, dict):
                enhanced[key] = self._generic_enhancement(value)

        return enhanced

    def _load_context_cache(self):
        """Load cached biochemical context data"""
        cache_file = self.cache_dir / "biochem_context_cache.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)

                # Load compounds
                for comp_id, comp_data in cache_data.get("compounds", {}).items():
                    context = BiochemicalContext(comp_id, "compound")
                    context.name = comp_data.get("name")
                    context.formula = comp_data.get("formula")
                    context.aliases = comp_data.get("aliases", [])
                    context.properties = comp_data.get("properties", {})
                    self._compound_cache[comp_id] = context

                # Load reactions
                for rxn_id, rxn_data in cache_data.get("reactions", {}).items():
                    context = BiochemicalContext(rxn_id, "reaction")
                    context.name = rxn_data.get("name")
                    context.formula = rxn_data.get("formula")
                    context.aliases = rxn_data.get("aliases", [])
                    context.pathways = rxn_data.get("pathways", [])
                    context.properties = rxn_data.get("properties", {})
                    self._reaction_cache[rxn_id] = context

                logger.info(
                    f"Loaded {len(self._compound_cache)} compounds and {len(self._reaction_cache)} reactions from cache"
                )

            except Exception as e:
                logger.warning(f"Failed to load context cache: {e}")

    def save_context_cache(self):
        """Save biochemical context cache to disk"""
        cache_file = self.cache_dir / "biochem_context_cache.json"

        try:
            cache_data = {
                "compounds": {
                    cid: ctx.to_dict() for cid, ctx in self._compound_cache.items()
                },
                "reactions": {
                    rid: ctx.to_dict() for rid, ctx in self._reaction_cache.items()
                },
                "genes": {gid: ctx.to_dict() for gid, ctx in self._gene_cache.items()},
            }

            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

            logger.info(
                f"Saved context cache with {len(self._compound_cache)} compounds and {len(self._reaction_cache)} reactions"
            )

        except Exception as e:
            logger.error(f"Failed to save context cache: {e}")


class ContextMemory:
    """
    Maintain biochemical context across tool executions in a session

    Enables progressive context building and cross-tool context propagation
    """

    def __init__(self):
        self.session_compounds: Dict[str, BiochemicalContext] = {}
        self.session_reactions: Dict[str, BiochemicalContext] = {}
        self.session_pathways: Dict[str, List[str]] = {}
        self.important_entities: Dict[str, float] = {}  # Entity -> importance score

    def remember_entities(
        self, entities: Dict[str, BiochemicalContext], importance: float = 1.0
    ):
        """Remember important entities from tool execution"""
        for entity_id, context in entities.items():
            if context.entity_type == "compound":
                self.session_compounds[entity_id] = context
            elif context.entity_type == "reaction":
                self.session_reactions[entity_id] = context

            self.important_entities[entity_id] = max(
                self.important_entities.get(entity_id, 0), importance
            )

    def get_session_context(self) -> Dict[str, Any]:
        """Get comprehensive session context for AI reasoning"""
        return {
            "important_compounds": [
                {
                    "id": cid,
                    "context": ctx.to_dict(),
                    "importance": self.important_entities.get(cid, 0),
                }
                for cid, ctx in self.session_compounds.items()
                if self.important_entities.get(cid, 0) > 0.5
            ],
            "important_reactions": [
                {
                    "id": rid,
                    "context": ctx.to_dict(),
                    "importance": self.important_entities.get(rid, 0),
                }
                for rid, ctx in self.session_reactions.items()
                if self.important_entities.get(rid, 0) > 0.5
            ],
            "active_pathways": self.session_pathways,
        }

    def get_context_for_reasoning(self) -> str:
        """Get context summary for AI reasoning prompts"""
        context_parts = []

        if self.session_compounds:
            key_compounds = [
                f"{ctx.name or cid} ({cid})"
                for cid, ctx in self.session_compounds.items()
                if self.important_entities.get(cid, 0) > 0.7
            ][:5]
            if key_compounds:
                context_parts.append(
                    f"Key compounds identified: {', '.join(key_compounds)}"
                )

        if self.session_reactions:
            key_reactions = [
                f"{ctx.name or rid} ({rid})"
                for rid, ctx in self.session_reactions.items()
                if self.important_entities.get(rid, 0) > 0.7
            ][:5]
            if key_reactions:
                context_parts.append(
                    f"Key reactions identified: {', '.join(key_reactions)}"
                )

        if self.session_pathways:
            active_pathways = list(self.session_pathways.keys())[:3]
            if active_pathways:
                context_parts.append(f"Active pathways: {', '.join(active_pathways)}")

        return (
            " ".join(context_parts)
            if context_parts
            else "No significant biochemical context identified yet."
        )


# Global context enhancer instance
_context_enhancer = None
_context_memory = None


def get_context_enhancer() -> BiochemContextEnhancer:
    """Get global context enhancer instance"""
    global _context_enhancer
    if _context_enhancer is None:
        _context_enhancer = BiochemContextEnhancer()
    return _context_enhancer


def get_context_memory() -> ContextMemory:
    """Get global context memory instance"""
    global _context_memory
    if _context_memory is None:
        _context_memory = ContextMemory()
    return _context_memory
