"""
Growth Analysis Framework for ModelSEEDagent

Specialized reasoning framework for growth rate analysis and biomass production,
providing context-aware questions and mechanistic reasoning guidance.
"""

import logging
from typing import Any, Dict, List, Optional

from .biochemical_reasoning import BiochemicalReasoningFramework, ReasoningDepth

logger = logging.getLogger(__name__)


class GrowthAnalysisFramework(BiochemicalReasoningFramework):
    """
    Specialized framework for growth analysis reasoning

    Extends biochemical reasoning with growth-specific patterns,
    questions, and mechanistic insights.
    """

    def __init__(self):
        super().__init__()
        self.growth_specific_patterns = self._initialize_growth_patterns()
        self.growth_rate_thresholds = self._initialize_growth_thresholds()

    def analyze_growth_context(self, growth_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze growth context and generate reasoning guidance

        Args:
            growth_data: Growth analysis results

        Returns:
            Comprehensive growth analysis with reasoning questions
        """
        analysis = {
            "growth_characterization": self._characterize_growth(growth_data),
            "limiting_factors": self._identify_limiting_factors(growth_data),
            "metabolic_efficiency": self._assess_metabolic_efficiency(growth_data),
            "reasoning_questions": self._generate_growth_questions(growth_data),
            "mechanistic_hypotheses": self._generate_growth_hypotheses(growth_data),
        }

        return analysis

    def get_growth_reasoning_prompts(
        self,
        growth_data: Dict[str, Any],
        depth: ReasoningDepth = ReasoningDepth.INTERMEDIATE,
    ) -> Dict[str, str]:
        """
        Get growth-specific reasoning prompts

        Args:
            growth_data: Growth analysis results
            depth: Desired reasoning depth

        Returns:
            Dictionary of reasoning prompts
        """
        growth_rate = growth_data.get("growth_rate", 0.0)
        biomass_yield = growth_data.get("biomass_yield", 0.0)

        prompts = {}

        if depth == ReasoningDepth.SURFACE:
            prompts.update(self._get_surface_growth_prompts(growth_rate, biomass_yield))
        elif depth == ReasoningDepth.INTERMEDIATE:
            prompts.update(self._get_intermediate_growth_prompts(growth_data))
        elif depth == ReasoningDepth.DEEP:
            prompts.update(self._get_deep_growth_prompts(growth_data))
        elif depth == ReasoningDepth.MECHANISTIC:
            prompts.update(self._get_mechanistic_growth_prompts(growth_data))

        return prompts

    def _initialize_growth_patterns(self) -> Dict[str, Any]:
        """Initialize growth-specific reasoning patterns"""
        return {
            "growth_rate_analysis": {
                "high_growth": {
                    "questions": [
                        "What metabolic strategies enable this high growth rate?",
                        "Are there trade-offs with metabolic efficiency?",
                        "How does resource allocation support rapid growth?",
                    ],
                    "mechanisms": [
                        "Efficient central carbon metabolism",
                        "Optimized protein synthesis machinery",
                        "Streamlined regulatory networks",
                    ],
                },
                "moderate_growth": {
                    "questions": [
                        "What factors limit growth to this moderate rate?",
                        "How could growth be optimized?",
                        "What metabolic bottlenecks exist?",
                    ],
                    "mechanisms": [
                        "Metabolic bottlenecks in key pathways",
                        "Resource allocation trade-offs",
                        "Regulatory constraints",
                    ],
                },
                "slow_growth": {
                    "questions": [
                        "What severe limitations cause slow growth?",
                        "Are there missing essential capabilities?",
                        "How does the organism survive with limited growth?",
                    ],
                    "mechanisms": [
                        "Missing essential biosynthetic pathways",
                        "Inefficient central metabolism",
                        "Stress response activation",
                    ],
                },
            },
            "biomass_composition_analysis": {
                "questions": [
                    "How does biomass composition reflect growth strategy?",
                    "What macromolecular priorities drive resource allocation?",
                    "How do environmental conditions shape biomass composition?",
                ],
                "focus_areas": [
                    "protein_synthesis_machinery",
                    "energy_generation_systems",
                    "biosynthetic_capabilities",
                    "stress_response_systems",
                ],
            },
            "nutrient_utilization_patterns": {
                "carbon_metabolism": [
                    "How efficiently is the carbon source being utilized?",
                    "What carbon flux distribution supports growth?",
                    "Are there overflow metabolism patterns?",
                ],
                "nitrogen_metabolism": [
                    "How does nitrogen availability affect growth?",
                    "What amino acid biosynthesis pathways are active?",
                    "Is nitrogen assimilation optimized?",
                ],
                "phosphorus_metabolism": [
                    "How does phosphorus limitation affect nucleotide synthesis?",
                    "What energy metabolism adaptations support growth?",
                    "Are there phosphorus storage mechanisms?",
                ],
            },
        }

    def _initialize_growth_thresholds(self) -> Dict[str, float]:
        """Initialize growth rate thresholds for classification"""
        return {
            "very_slow": 0.1,
            "slow": 0.3,
            "moderate": 0.7,
            "fast": 1.2,
            "very_fast": 2.0,
        }

    def _characterize_growth(self, growth_data: Dict[str, Any]) -> Dict[str, Any]:
        """Characterize growth performance"""
        growth_rate = growth_data.get("growth_rate", 0.0)

        # Classify growth rate
        if growth_rate < self.growth_rate_thresholds["very_slow"]:
            category = "very_slow"
            description = (
                "Severely limited growth indicating major metabolic constraints"
            )
        elif growth_rate < self.growth_rate_thresholds["slow"]:
            category = "slow"
            description = "Slow growth suggesting significant limitations or stress"
        elif growth_rate < self.growth_rate_thresholds["moderate"]:
            category = "moderate"
            description = "Moderate growth with some metabolic limitations"
        elif growth_rate < self.growth_rate_thresholds["fast"]:
            category = "fast"
            description = "Good growth with efficient metabolic function"
        else:
            category = "very_fast"
            description = "Exceptionally fast growth indicating optimal conditions"

        return {
            "category": category,
            "rate": growth_rate,
            "description": description,
            "percentile": self._estimate_growth_percentile(growth_rate),
        }

    def _identify_limiting_factors(
        self, growth_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify potential growth limiting factors"""
        limiting_factors = []

        growth_rate = growth_data.get("growth_rate", 0.0)

        # Check for obvious limitations
        if growth_rate < 0.1:
            limiting_factors.append(
                {
                    "factor": "severe_metabolic_constraint",
                    "evidence": f"Growth rate {growth_rate:.3f} indicates major limitations",
                    "investigation": "Check for missing essential pathways or toxic conditions",
                }
            )

        # Check exchange fluxes for nutrient limitations
        exchange_fluxes = growth_data.get("exchange_fluxes", {})
        for reaction_id, flux_value in exchange_fluxes.items():
            if abs(flux_value) > 20:  # Very high uptake
                limiting_factors.append(
                    {
                        "factor": "high_nutrient_demand",
                        "evidence": f"High uptake rate for {reaction_id}: {flux_value:.2f}",
                        "investigation": "Investigate if this nutrient becomes limiting",
                    }
                )

        # Check biomass yield
        biomass_yield = growth_data.get("biomass_yield", 0.0)
        if biomass_yield < 0.3:
            limiting_factors.append(
                {
                    "factor": "low_metabolic_efficiency",
                    "evidence": f"Low biomass yield {biomass_yield:.3f}",
                    "investigation": "Analyze energy dissipation and metabolic inefficiencies",
                }
            )

        return limiting_factors

    def _assess_metabolic_efficiency(
        self, growth_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess metabolic efficiency from growth data"""
        efficiency_metrics = {}

        # Calculate basic efficiency metrics
        growth_rate = growth_data.get("growth_rate", 0.0)
        glucose_uptake = self._extract_glucose_uptake(growth_data)

        if glucose_uptake > 0:
            efficiency_metrics["carbon_efficiency"] = growth_rate / glucose_uptake
            efficiency_metrics["carbon_efficiency_description"] = (
                self._describe_carbon_efficiency(
                    efficiency_metrics["carbon_efficiency"]
                )
            )

        # ATP efficiency
        atp_production = self._estimate_atp_production(growth_data)
        if atp_production > 0:
            efficiency_metrics["atp_efficiency"] = growth_rate / atp_production

        # Overall assessment
        efficiency_metrics["overall_assessment"] = self._assess_overall_efficiency(
            growth_data
        )

        return efficiency_metrics

    def _generate_growth_questions(self, growth_data: Dict[str, Any]) -> List[str]:
        """Generate context-specific growth analysis questions"""
        questions = []

        growth_characterization = self._characterize_growth(growth_data)
        growth_category = growth_characterization["category"]

        # Add category-specific questions
        pattern_data = self.growth_specific_patterns["growth_rate_analysis"].get(
            growth_category, {}
        )
        questions.extend(pattern_data.get("questions", []))

        # Add nutrient-specific questions
        exchange_fluxes = growth_data.get("exchange_fluxes", {})
        if exchange_fluxes:
            questions.extend(self._generate_nutrient_questions(exchange_fluxes))

        # Add efficiency questions
        if "biomass_yield" in growth_data:
            questions.extend(
                [
                    "How does the biomass yield compare to theoretical maximum?",
                    "What metabolic processes account for energy dissipation?",
                ]
            )

        return questions

    def _generate_growth_hypotheses(
        self, growth_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate mechanistic hypotheses about growth"""
        hypotheses = []

        growth_rate = growth_data.get("growth_rate", 0.0)

        # Growth rate hypotheses
        if growth_rate > 1.5:
            hypotheses.append(
                {
                    "hypothesis": "Optimized central carbon metabolism enables high growth rate",
                    "prediction": "High flux through glycolysis and TCA cycle",
                    "test": "Analyze flux through key central metabolic reactions",
                }
            )
        elif growth_rate < 0.3:
            hypotheses.append(
                {
                    "hypothesis": "Metabolic bottleneck limits growth rate",
                    "prediction": "Specific pathway shows very low flux or essentiality",
                    "test": "Perform gene deletion analysis to identify critical steps",
                }
            )

        # Efficiency hypotheses
        biomass_yield = growth_data.get("biomass_yield", 0.0)
        if biomass_yield < 0.4:
            hypotheses.append(
                {
                    "hypothesis": "Energy is dissipated through overflow metabolism",
                    "prediction": "High secretion of organic acids or other metabolites",
                    "test": "Analyze secretion fluxes for overflow products",
                }
            )

        return hypotheses

    def _generate_nutrient_questions(
        self, exchange_fluxes: Dict[str, float]
    ) -> List[str]:
        """Generate nutrient-specific questions"""
        questions = []

        # Find highest uptake rates
        uptake_fluxes = {k: abs(v) for k, v in exchange_fluxes.items() if v < 0}
        if uptake_fluxes:
            max_uptake = max(uptake_fluxes.values())
            dominant_nutrients = [
                k for k, v in uptake_fluxes.items() if v > max_uptake * 0.5
            ]

            for nutrient in dominant_nutrients[:2]:  # Top 2 nutrients
                questions.append(f"Why does {nutrient} show such high uptake rate?")
                questions.append(
                    f"How does {nutrient} utilization efficiency compare to other organisms?"
                )

        return questions

    def _get_surface_growth_prompts(
        self, growth_rate: float, biomass_yield: float
    ) -> Dict[str, str]:
        """Get surface-level growth reasoning prompts"""
        return {
            "growth_observation": f"""
            Analyze the growth performance:
            - Growth rate: {growth_rate:.3f} h⁻¹
            - Biomass yield: {biomass_yield:.3f} g/g

            What do these numbers tell you about the organism's performance?
            How do they compare to typical values for similar organisms?
            """,
            "basic_assessment": """
            Provide a basic assessment:
            - Is this good, moderate, or poor growth?
            - What might be limiting growth?
            - Are there obvious inefficiencies?
            """,
        }

    def _get_intermediate_growth_prompts(
        self, growth_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Get intermediate-level growth reasoning prompts"""
        return {
            "pathway_analysis": """
            Analyze the metabolic pathways supporting growth:
            - Which central metabolic pathways are most active?
            - How do nutrient uptake patterns relate to growth?
            - What biosynthetic demands drive resource allocation?
            """,
            "efficiency_analysis": """
            Evaluate metabolic efficiency:
            - How efficiently is the carbon source being converted to biomass?
            - Where might energy be lost in the metabolic network?
            - What trade-offs between growth rate and yield are evident?
            """,
            "limitation_analysis": """
            Identify potential limitations:
            - What factors might be constraining growth rate?
            - Are there metabolic bottlenecks in key pathways?
            - How do environmental conditions affect growth performance?
            """,
        }

    def _get_deep_growth_prompts(self, growth_data: Dict[str, Any]) -> Dict[str, str]:
        """Get deep-level growth reasoning prompts"""
        return {
            "mechanistic_analysis": """
            Explain the biochemical mechanisms controlling growth:
            - How do enzyme kinetics shape metabolic flux distribution?
            - What regulatory networks coordinate growth-related processes?
            - How do thermodynamic constraints limit pathway efficiency?
            """,
            "systems_integration": """
            Analyze growth as a systems-level phenomenon:
            - How do different metabolic subsystems integrate for growth?
            - What feedback mechanisms coordinate resource allocation?
            - How does the organism balance competing metabolic demands?
            """,
            "evolutionary_perspective": """
            Consider evolutionary aspects of growth strategy:
            - How might this growth pattern reflect evolutionary optimization?
            - What environmental pressures shaped this metabolic strategy?
            - How do growth rate and efficiency trade-offs reflect ecology?
            """,
        }

    def _get_mechanistic_growth_prompts(
        self, growth_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Get mechanistic-level growth reasoning prompts"""
        return {
            "molecular_mechanisms": """
            Explain growth at the molecular level:
            - How do ribosome kinetics limit protein synthesis rate?
            - What role do metabolite concentrations play in flux control?
            - How do allosteric regulations coordinate metabolic flow?
            """,
            "energetic_analysis": """
            Analyze the energetics of growth:
            - How does ATP/ADP ratio affect central metabolism?
            - What is the thermodynamic efficiency of biomass synthesis?
            - How do redox balances constrain metabolic pathways?
            """,
            "protein_synthesis": """
            Focus on protein synthesis machinery:
            - How does ribosome allocation affect growth rate?
            - What amino acid availability limits protein synthesis?
            - How does translation efficiency impact overall growth?
            """,
        }

    def _extract_glucose_uptake(self, growth_data: Dict[str, Any]) -> float:
        """Extract glucose uptake rate from growth data"""
        exchange_fluxes = growth_data.get("exchange_fluxes", {})

        # Look for glucose exchange reactions
        glucose_patterns = ["glc", "glucose", "EX_glc", "cpd00027"]

        for reaction_id, flux_value in exchange_fluxes.items():
            if any(pattern in reaction_id.lower() for pattern in glucose_patterns):
                if flux_value < 0:  # Uptake
                    return abs(flux_value)

        return 0.0

    def _estimate_atp_production(self, growth_data: Dict[str, Any]) -> float:
        """Estimate ATP production rate"""
        # Simple estimation based on glucose uptake
        glucose_uptake = self._extract_glucose_uptake(growth_data)

        # Assume ~30 ATP per glucose (rough estimate)
        return glucose_uptake * 30

    def _describe_carbon_efficiency(self, efficiency: float) -> str:
        """Describe carbon utilization efficiency"""
        if efficiency > 0.15:
            return "High carbon efficiency - excellent conversion to biomass"
        elif efficiency > 0.10:
            return "Good carbon efficiency - reasonable biomass yield"
        elif efficiency > 0.05:
            return "Moderate carbon efficiency - some carbon loss apparent"
        else:
            return "Low carbon efficiency - significant carbon waste"

    def _assess_overall_efficiency(self, growth_data: Dict[str, Any]) -> str:
        """Assess overall metabolic efficiency"""
        growth_rate = growth_data.get("growth_rate", 0.0)
        biomass_yield = growth_data.get("biomass_yield", 0.0)

        # Combined efficiency score
        efficiency_score = (growth_rate * 0.6) + (biomass_yield * 0.4)

        if efficiency_score > 0.8:
            return "Highly efficient metabolism with good growth rate and yield"
        elif efficiency_score > 0.6:
            return "Reasonably efficient metabolism with moderate performance"
        elif efficiency_score > 0.4:
            return "Suboptimal metabolism with notable inefficiencies"
        else:
            return "Poor metabolic efficiency requiring investigation"

    def _estimate_growth_percentile(self, growth_rate: float) -> float:
        """Estimate growth rate percentile (rough approximation)"""
        # Based on typical E. coli growth rates
        if growth_rate > 1.5:
            return 95.0
        elif growth_rate > 1.0:
            return 80.0
        elif growth_rate > 0.7:
            return 60.0
        elif growth_rate > 0.4:
            return 40.0
        elif growth_rate > 0.2:
            return 20.0
        else:
            return 5.0
