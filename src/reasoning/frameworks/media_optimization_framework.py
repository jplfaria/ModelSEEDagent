"""
Media Optimization Framework for ModelSEEDagent

Specialized reasoning framework for media composition analysis and optimization,
focusing on nutrient requirements and metabolic constraints.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .biochemical_reasoning import BiochemicalReasoningFramework, ReasoningDepth

logger = logging.getLogger(__name__)


class MediaOptimizationFramework(BiochemicalReasoningFramework):
    """
    Specialized framework for media optimization reasoning

    Focuses on nutrient requirements, metabolic constraints,
    and optimization strategies for growth media.
    """

    def __init__(self):
        super().__init__()
        self.nutrient_categories = self._initialize_nutrient_categories()
        self.optimization_strategies = self._initialize_optimization_strategies()

    def analyze_media_requirements(
        self,
        media_data: Dict[str, Any],
        growth_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze media requirements and optimization opportunities

        Args:
            media_data: Media composition and analysis results
            growth_context: Additional growth performance context

        Returns:
            Comprehensive media analysis with optimization guidance
        """
        analysis = {
            "nutrient_analysis": self._analyze_nutrient_composition(media_data),
            "essentiality_assessment": self._assess_nutrient_essentiality(media_data),
            "optimization_opportunities": self._identify_optimization_opportunities(
                media_data, growth_context
            ),
            "cost_efficiency": self._assess_cost_efficiency(media_data),
            "reasoning_questions": self._generate_media_questions(
                media_data, growth_context
            ),
            "optimization_strategies": self._recommend_optimization_strategies(
                media_data, growth_context
            ),
        }

        return analysis

    def get_media_reasoning_prompts(
        self, media_data: Dict[str, Any], optimization_goal: str = "balanced"
    ) -> Dict[str, str]:
        """
        Get media-specific reasoning prompts

        Args:
            media_data: Media analysis results
            optimization_goal: Optimization objective ('growth', 'cost', 'minimal', 'balanced')

        Returns:
            Dictionary of reasoning prompts
        """
        prompts = {}

        prompts.update(self._get_composition_analysis_prompts(media_data))
        prompts.update(self._get_optimization_prompts(media_data, optimization_goal))
        prompts.update(self._get_metabolic_constraint_prompts(media_data))

        return prompts

    def _initialize_nutrient_categories(self) -> Dict[str, Any]:
        """Initialize nutrient categorization and analysis patterns"""
        return {
            "macronutrients": {
                "carbon_sources": {
                    "primary": ["glucose", "glycerol", "acetate", "succinate"],
                    "alternative": ["fructose", "xylose", "lactate", "pyruvate"],
                    "analysis_focus": [
                        "utilization_efficiency",
                        "growth_rate_impact",
                        "overflow_metabolism",
                        "catabolite_repression",
                    ],
                },
                "nitrogen_sources": {
                    "preferred": ["ammonia", "glutamine", "glutamate"],
                    "alternative": ["nitrate", "nitrite", "amino_acids"],
                    "analysis_focus": [
                        "assimilation_efficiency",
                        "energy_cost",
                        "regulatory_effects",
                        "toxicity_levels",
                    ],
                },
                "phosphorus_sources": {
                    "primary": ["phosphate", "phosphoenolpyruvate"],
                    "alternative": ["glucose-6-phosphate", "glycerol-phosphate"],
                    "analysis_focus": [
                        "availability_limitation",
                        "uptake_kinetics",
                        "storage_mechanisms",
                        "growth_limitation",
                    ],
                },
                "sulfur_sources": {
                    "primary": ["sulfate", "cysteine", "methionine"],
                    "alternative": ["sulfite", "thiosulfate"],
                    "analysis_focus": [
                        "reduction_pathways",
                        "biosynthetic_demands",
                        "oxidative_stress",
                        "metal_cofactor_interactions",
                    ],
                },
            },
            "micronutrients": {
                "trace_elements": {
                    "essential": ["Fe", "Mg", "Mn", "Zn", "Co", "Ni", "Mo"],
                    "beneficial": ["Ca", "K", "Na", "Cu"],
                    "analysis_focus": [
                        "cofactor_requirements",
                        "enzyme_activation",
                        "metal_homeostasis",
                        "toxicity_thresholds",
                    ],
                },
                "vitamins_cofactors": {
                    "essential": ["biotin", "thiamine", "cobalamin"],
                    "conditional": ["folate", "riboflavin", "niacin"],
                    "analysis_focus": [
                        "biosynthetic_capabilities",
                        "salvage_pathways",
                        "growth_factor_dependencies",
                        "metabolic_burden",
                    ],
                },
            },
            "environmental_factors": {
                "physical_conditions": {
                    "parameters": ["pH", "temperature", "osmolarity", "oxygen"],
                    "analysis_focus": [
                        "enzyme_stability",
                        "membrane_integrity",
                        "metabolic_efficiency",
                        "stress_responses",
                    ],
                }
            },
        }

    def _initialize_optimization_strategies(self) -> Dict[str, Any]:
        """Initialize media optimization strategy patterns"""
        return {
            "growth_maximization": {
                "priorities": [
                    "carbon_source_quality",
                    "nitrogen_availability",
                    "trace_elements",
                ],
                "strategies": [
                    "Optimize carbon source concentration",
                    "Ensure adequate nitrogen supply",
                    "Balance trace element cocktail",
                    "Minimize growth inhibitors",
                ],
                "trade_offs": ["Higher cost", "Complex composition"],
            },
            "cost_minimization": {
                "priorities": [
                    "essential_nutrients_only",
                    "cheap_carbon_sources",
                    "minimal_complexity",
                ],
                "strategies": [
                    "Use cheapest adequate carbon source",
                    "Minimize trace element complexity",
                    "Eliminate non-essential components",
                    "Optimize concentrations for efficiency",
                ],
                "trade_offs": ["Potentially slower growth", "Less robust performance"],
            },
            "minimal_media": {
                "priorities": [
                    "essential_nutrients_only",
                    "defined_composition",
                    "reproducibility",
                ],
                "strategies": [
                    "Include only demonstrated essential nutrients",
                    "Use simple inorganic salts",
                    "Minimize undefined components",
                    "Optimize for reproducibility",
                ],
                "trade_offs": ["May limit growth rate", "Requires precise composition"],
            },
            "robustness_optimization": {
                "priorities": [
                    "stable_performance",
                    "environmental_tolerance",
                    "batch_consistency",
                ],
                "strategies": [
                    "Include protective compounds",
                    "Buffer against pH changes",
                    "Add antioxidants if needed",
                    "Ensure adequate reserves",
                ],
                "trade_offs": ["Higher complexity", "Increased cost"],
            },
        }

    def _analyze_nutrient_composition(
        self, media_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze nutrient composition of media"""
        composition_analysis = {
            "carbon_sources": self._identify_carbon_sources(media_data),
            "nitrogen_sources": self._identify_nitrogen_sources(media_data),
            "essential_elements": self._identify_essential_elements(media_data),
            "trace_elements": self._identify_trace_elements(media_data),
            "vitamins_cofactors": self._identify_vitamins_cofactors(media_data),
            "complexity_score": self._calculate_complexity_score(media_data),
            "completeness_score": self._assess_nutritional_completeness(media_data),
        }

        return composition_analysis

    def _assess_nutrient_essentiality(
        self, media_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess essentiality of media components"""
        essentiality_assessment = {}

        media_components = media_data.get("media_components", [])

        for component in media_components:
            if isinstance(component, dict):
                compound_id = component.get("id", "")
                context = component.get("context", {})
            else:
                compound_id = component
                context = {}

            essentiality = self._assess_component_essentiality(compound_id, context)
            essentiality_assessment[compound_id] = essentiality

        return essentiality_assessment

    def _identify_optimization_opportunities(
        self,
        media_data: Dict[str, Any],
        growth_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Identify media optimization opportunities"""
        opportunities = []

        # Check for over-supplementation
        concentrations = media_data.get("concentrations", {})
        for compound_id, conc_info in concentrations.items():
            concentration = (
                conc_info.get("concentration", conc_info)
                if isinstance(conc_info, dict)
                else conc_info
            )

            if concentration > self._get_typical_concentration(compound_id) * 2:
                opportunities.append(
                    {
                        "type": "reduce_concentration",
                        "component": compound_id,
                        "current_concentration": concentration,
                        "suggested_concentration": self._get_optimal_concentration(
                            compound_id
                        ),
                        "benefit": "Cost reduction without performance loss",
                    }
                )

        # Check for missing essential nutrients
        essential_nutrients = self._get_essential_nutrients()
        present_nutrients = set(media_data.get("media_components", []))

        for nutrient in essential_nutrients:
            if nutrient not in present_nutrients:
                opportunities.append(
                    {
                        "type": "add_essential_nutrient",
                        "component": nutrient,
                        "benefit": "Eliminate growth limitation",
                        "risk": "Essential for growth",
                    }
                )

        # Check for growth performance correlation
        if growth_context:
            growth_rate = growth_context.get("growth_rate", 0.0)
            if growth_rate < 0.5:
                opportunities.extend(
                    self._identify_growth_limiting_nutrients(media_data, growth_context)
                )

        return opportunities

    def _assess_cost_efficiency(self, media_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess cost efficiency of media composition"""
        cost_assessment = {
            "relative_cost_score": self._calculate_relative_cost(media_data),
            "cost_drivers": self._identify_cost_drivers(media_data),
            "cost_reduction_potential": self._assess_cost_reduction_potential(
                media_data
            ),
            "cost_vs_performance_ratio": self._calculate_cost_performance_ratio(
                media_data
            ),
        }

        return cost_assessment

    def _generate_media_questions(
        self,
        media_data: Dict[str, Any],
        growth_context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Generate media-specific reasoning questions"""
        questions = []

        # Basic composition questions
        questions.extend(
            [
                "What is the nutritional strategy of this media composition?",
                "Are all essential nutrients adequately supplied?",
                "What is the carbon/nitrogen/phosphorus balance?",
            ]
        )

        # Optimization questions
        nutrient_analysis = self._analyze_nutrient_composition(media_data)

        if nutrient_analysis["complexity_score"] > 0.7:
            questions.append(
                "Is this complex media composition necessary, or could it be simplified?"
            )

        if nutrient_analysis["completeness_score"] < 0.8:
            questions.append(
                "What essential nutrients might be missing or insufficient?"
            )

        # Growth context questions
        if growth_context:
            growth_rate = growth_context.get("growth_rate", 0.0)
            if growth_rate < 0.5:
                questions.extend(
                    [
                        "Which nutrients might be limiting growth in this media?",
                        "How could media composition be optimized to improve growth?",
                    ]
                )

        # Metabolic questions
        questions.extend(
            [
                "How does this media composition affect metabolic pathway usage?",
                "What metabolic constraints does this media impose?",
                "How might organisms adapt their metabolism to this nutrient environment?",
            ]
        )

        return questions

    def _recommend_optimization_strategies(
        self,
        media_data: Dict[str, Any],
        growth_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Recommend specific optimization strategies"""
        recommendations = []

        # Analyze current media
        composition_analysis = self._analyze_nutrient_composition(media_data)
        self._identify_optimization_opportunities(media_data, growth_context)

        # Growth-based recommendations
        if growth_context:
            growth_rate = growth_context.get("growth_rate", 0.0)

            if growth_rate < 0.3:
                recommendations.append(
                    {
                        "strategy": "growth_enhancement",
                        "priority": "high",
                        "actions": [
                            "Check for essential nutrient deficiencies",
                            "Optimize carbon source concentration",
                            "Ensure adequate trace element supply",
                        ],
                        "expected_benefit": "Significant growth rate improvement",
                    }
                )

        # Complexity-based recommendations
        if composition_analysis["complexity_score"] > 0.8:
            recommendations.append(
                {
                    "strategy": "simplification",
                    "priority": "medium",
                    "actions": [
                        "Eliminate redundant nutrients",
                        "Reduce to essential components only",
                        "Optimize concentrations",
                    ],
                    "expected_benefit": "Reduced cost and improved reproducibility",
                }
            )

        # Cost-based recommendations
        cost_assessment = self._assess_cost_efficiency(media_data)
        if cost_assessment["relative_cost_score"] > 0.7:
            recommendations.append(
                {
                    "strategy": "cost_reduction",
                    "priority": "medium",
                    "actions": [
                        "Replace expensive components with cheaper alternatives",
                        "Optimize concentrations to minimize waste",
                        "Consider bulk purchasing opportunities",
                    ],
                    "expected_benefit": "Significant cost reduction",
                }
            )

        return recommendations

    def _identify_carbon_sources(
        self, media_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify carbon sources in media"""
        carbon_sources = []

        carbon_compounds = ["glucose", "glycerol", "acetate", "succinate", "fructose"]
        media_components = media_data.get("media_components", [])

        for component in media_components:
            if isinstance(component, dict):
                context = component.get("context", {})
                compound_name = context.get("name", "").lower()
                role = component.get("role", "unknown")
            else:
                compound_name = str(component).lower()
                role = "unknown"

            if (
                any(carbon in compound_name for carbon in carbon_compounds)
                or role == "carbon_source"
            ):
                carbon_sources.append(
                    {
                        "compound": component,
                        "type": self._classify_carbon_source(compound_name),
                        "metabolic_pathway": self._infer_carbon_pathway(compound_name),
                    }
                )

        return carbon_sources

    def _identify_nitrogen_sources(
        self, media_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify nitrogen sources in media"""
        nitrogen_sources = []

        nitrogen_compounds = [
            "ammonia",
            "ammonium",
            "glutamate",
            "glutamine",
            "nitrate",
        ]
        media_components = media_data.get("media_components", [])

        for component in media_components:
            if isinstance(component, dict):
                context = component.get("context", {})
                compound_name = context.get("name", "").lower()
                role = component.get("role", "unknown")
            else:
                compound_name = str(component).lower()
                role = "unknown"

            if (
                any(nitrogen in compound_name for nitrogen in nitrogen_compounds)
                or role == "nitrogen_source"
            ):
                nitrogen_sources.append(
                    {
                        "compound": component,
                        "type": self._classify_nitrogen_source(compound_name),
                        "assimilation_pathway": self._infer_nitrogen_pathway(
                            compound_name
                        ),
                    }
                )

        return nitrogen_sources

    def _identify_essential_elements(self, media_data: Dict[str, Any]) -> List[str]:
        """Identify essential elements present in media"""
        essential_elements = ["C", "N", "P", "S", "K", "Mg"]
        present_elements = []

        # This would require more sophisticated compound analysis
        # For now, assume presence based on common media components
        present_elements = essential_elements  # Simplified

        return present_elements

    def _identify_trace_elements(self, media_data: Dict[str, Any]) -> List[str]:
        """Identify trace elements in media"""
        trace_elements = []

        trace_metals = ["Fe", "Mn", "Zn", "Co", "Ni", "Mo", "Cu"]
        media_components = media_data.get("media_components", [])

        for component in media_components:
            if isinstance(component, dict):
                context = component.get("context", {})
                compound_name = context.get("name", "").lower()
            else:
                compound_name = str(component).lower()

            for metal in trace_metals:
                if metal.lower() in compound_name:
                    trace_elements.append(metal)

        return list(set(trace_elements))

    def _identify_vitamins_cofactors(self, media_data: Dict[str, Any]) -> List[str]:
        """Identify vitamins and cofactors in media"""
        vitamins_cofactors = []

        vitamins = ["biotin", "thiamine", "cobalamin", "folate", "riboflavin"]
        media_components = media_data.get("media_components", [])

        for component in media_components:
            if isinstance(component, dict):
                context = component.get("context", {})
                compound_name = context.get("name", "").lower()
            else:
                compound_name = str(component).lower()

            for vitamin in vitamins:
                if vitamin in compound_name:
                    vitamins_cofactors.append(vitamin)

        return list(set(vitamins_cofactors))

    def _calculate_complexity_score(self, media_data: Dict[str, Any]) -> float:
        """Calculate media complexity score"""
        media_components = media_data.get("media_components", [])
        component_count = len(media_components)

        # Simple scoring based on component count
        complexity_score = min(1.0, component_count / 20.0)
        return complexity_score

    def _assess_nutritional_completeness(self, media_data: Dict[str, Any]) -> float:
        """Assess nutritional completeness of media"""
        essential_categories = [
            "carbon_sources",
            "nitrogen_sources",
            "phosphorus_sources",
            "trace_elements",
            "essential_cofactors",
        ]

        present_categories = 0

        composition_analysis = self._analyze_nutrient_composition(media_data)

        if composition_analysis.get("carbon_sources"):
            present_categories += 1
        if composition_analysis.get("nitrogen_sources"):
            present_categories += 1
        if "P" in composition_analysis.get("essential_elements", []):
            present_categories += 1
        if composition_analysis.get("trace_elements"):
            present_categories += 1
        if composition_analysis.get("vitamins_cofactors"):
            present_categories += 1

        completeness_score = present_categories / len(essential_categories)
        return completeness_score

    def _assess_component_essentiality(
        self, compound_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess essentiality of a media component"""
        # Simplified essentiality assessment
        essential_patterns = ["glucose", "ammonia", "phosphate", "sulfate", "mg", "fe"]

        compound_name = context.get("name", compound_id).lower()
        role = context.get("role", "unknown")

        is_essential = any(pattern in compound_name for pattern in essential_patterns)
        is_essential = is_essential or role in ["carbon_source", "nitrogen_source"]

        return {
            "essential": is_essential,
            "category": role,
            "alternatives_available": not is_essential,
            "growth_impact": "high" if is_essential else "low",
        }

    def _get_typical_concentration(self, compound_id: str) -> float:
        """Get typical concentration for a compound"""
        # Simplified - would use database in practice
        typical_concentrations = {"glucose": 10.0, "ammonia": 1.0, "phosphate": 1.0}

        return typical_concentrations.get(compound_id, 1.0)

    def _get_optimal_concentration(self, compound_id: str) -> float:
        """Get optimal concentration for a compound"""
        return self._get_typical_concentration(compound_id)

    def _get_essential_nutrients(self) -> List[str]:
        """Get list of essential nutrients"""
        return ["glucose", "ammonia", "phosphate", "sulfate", "mg_ion", "fe_ion"]

    def _identify_growth_limiting_nutrients(
        self, media_data: Dict[str, Any], growth_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify nutrients that might be limiting growth"""
        limiting_opportunities = []

        # Check for low concentrations of essential nutrients
        concentrations = media_data.get("concentrations", {})

        for compound_id, conc_info in concentrations.items():
            concentration = (
                conc_info.get("concentration", conc_info)
                if isinstance(conc_info, dict)
                else conc_info
            )
            typical_conc = self._get_typical_concentration(compound_id)

            if concentration < typical_conc * 0.5:
                limiting_opportunities.append(
                    {
                        "type": "increase_concentration",
                        "component": compound_id,
                        "current_concentration": concentration,
                        "suggested_concentration": typical_conc,
                        "benefit": "Potentially eliminate growth limitation",
                    }
                )

        return limiting_opportunities

    def _calculate_relative_cost(self, media_data: Dict[str, Any]) -> float:
        """Calculate relative cost score"""
        # Simplified cost calculation
        media_components = media_data.get("media_components", [])
        expensive_components = ["cobalamin", "biotin", "complex_nutrients"]

        expensive_count = 0
        for component in media_components:
            if isinstance(component, dict):
                context = component.get("context", {})
                compound_name = context.get("name", "").lower()
            else:
                compound_name = str(component).lower()

            if any(expensive in compound_name for expensive in expensive_components):
                expensive_count += 1

        cost_score = min(1.0, expensive_count / 5.0)
        return cost_score

    def _identify_cost_drivers(self, media_data: Dict[str, Any]) -> List[str]:
        """Identify main cost drivers in media"""
        # Simplified identification
        return [
            "complex_vitamins",
            "trace_element_cocktail",
            "expensive_carbon_sources",
        ]

    def _assess_cost_reduction_potential(self, media_data: Dict[str, Any]) -> float:
        """Assess potential for cost reduction"""
        complexity_score = self._calculate_complexity_score(media_data)
        return complexity_score  # High complexity = high reduction potential

    def _calculate_cost_performance_ratio(self, media_data: Dict[str, Any]) -> float:
        """Calculate cost vs performance ratio"""
        cost_score = self._calculate_relative_cost(media_data)
        completeness_score = self._assess_nutritional_completeness(media_data)

        if completeness_score > 0:
            return cost_score / completeness_score
        return 1.0

    def _classify_carbon_source(self, compound_name: str) -> str:
        """Classify type of carbon source"""
        if "glucose" in compound_name:
            return "preferred_sugar"
        elif "glycerol" in compound_name:
            return "alternative_carbon"
        elif "acetate" in compound_name:
            return "organic_acid"
        else:
            return "other"

    def _infer_carbon_pathway(self, compound_name: str) -> str:
        """Infer metabolic pathway for carbon source"""
        if "glucose" in compound_name:
            return "glycolysis"
        elif "acetate" in compound_name:
            return "acetate_metabolism"
        else:
            return "unknown"

    def _classify_nitrogen_source(self, compound_name: str) -> str:
        """Classify type of nitrogen source"""
        if "ammonia" in compound_name or "ammonium" in compound_name:
            return "inorganic_nitrogen"
        elif "glutamate" in compound_name or "glutamine" in compound_name:
            return "amino_acid"
        else:
            return "other"

    def _infer_nitrogen_pathway(self, compound_name: str) -> str:
        """Infer assimilation pathway for nitrogen source"""
        if "ammonia" in compound_name:
            return "glutamine_synthetase"
        elif "glutamate" in compound_name:
            return "direct_assimilation"
        else:
            return "unknown"

    def _get_composition_analysis_prompts(
        self, media_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Get prompts for media composition analysis"""
        return {
            "composition_assessment": """
            Analyze the media composition systematically:
            - What are the primary carbon, nitrogen, and phosphorus sources?
            - Are all essential nutrients adequately represented?
            - What is the overall nutritional strategy of this media?
            """,
            "nutrient_balance": """
            Evaluate the nutrient balance:
            - Is the C:N:P ratio appropriate for the target organism?
            - Are there potential nutrient interactions or antagonisms?
            - How might nutrient ratios affect metabolic pathway usage?
            """,
        }

    def _get_optimization_prompts(
        self, media_data: Dict[str, Any], optimization_goal: str
    ) -> Dict[str, str]:
        """Get optimization-specific prompts"""
        if optimization_goal == "growth":
            return {
                "growth_optimization": """
                Optimize for maximum growth:
                - Which nutrients might be limiting growth rate?
                - How could concentrations be adjusted to maximize growth?
                - What trace elements or cofactors might enhance performance?
                """
            }
        elif optimization_goal == "cost":
            return {
                "cost_optimization": """
                Optimize for cost efficiency:
                - Which components contribute most to media cost?
                - What cheaper alternatives exist for expensive components?
                - How could the formulation be simplified without losing performance?
                """
            }
        else:
            return {
                "balanced_optimization": """
                Balance multiple objectives:
                - How can growth performance and cost be balanced?
                - What trade-offs exist between simplicity and performance?
                - How robust is this formulation to component variations?
                """
            }

    def _get_metabolic_constraint_prompts(
        self, media_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Get prompts for metabolic constraint analysis"""
        return {
            "metabolic_constraints": """
            Analyze metabolic constraints imposed by this media:
            - How does nutrient availability shape metabolic pathway usage?
            - What metabolic adaptations would this environment select for?
            - How might organisms optimize their metabolism for this media?
            """,
            "pathway_implications": """
            Consider pathway-level implications:
            - Which biosynthetic pathways must be active due to nutrient limitations?
            - How do nutrient ratios affect central metabolic flux distribution?
            - What regulatory responses would this nutrient environment trigger?
            """,
        }
