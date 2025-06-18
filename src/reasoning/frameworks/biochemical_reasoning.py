"""
Biochemical Reasoning Framework for ModelSEEDagent

Provides question-driven reasoning guidance for biochemical analysis,
integrating multimodal context with structured scientific inquiry.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ReasoningDepth(Enum):
    """Levels of reasoning depth for analysis"""

    SURFACE = "surface"
    INTERMEDIATE = "intermediate"
    DEEP = "deep"
    MECHANISTIC = "mechanistic"


class BiochemicalQuestionType(Enum):
    """Types of biochemical questions for reasoning guidance"""

    WHAT_IS_HAPPENING = "what_is_happening"
    WHY_IS_IT_HAPPENING = "why_is_it_happening"
    HOW_DOES_IT_WORK = "how_does_it_work"
    WHAT_IF_CHANGED = "what_if_changed"
    WHAT_DOES_IT_MEAN = "what_does_it_mean"


class BiochemicalReasoningFramework:
    """
    Core framework for biochemical reasoning guidance

    Provides structured questions and reasoning patterns to guide AI
    analysis from surface-level observations to mechanistic insights.
    """

    def __init__(self):
        self.reasoning_patterns = self._initialize_reasoning_patterns()
        self.context_triggers = self._initialize_context_triggers()
        self.depth_escalation_rules = self._initialize_depth_rules()

    def generate_reasoning_questions(
        self,
        analysis_context: Dict[str, Any],
        current_depth: ReasoningDepth = ReasoningDepth.SURFACE,
    ) -> List[str]:
        """
        Generate context-appropriate reasoning questions

        Args:
            analysis_context: Current analysis state and results
            current_depth: Current reasoning depth level

        Returns:
            List of structured questions to guide deeper analysis
        """
        questions = []

        # Identify the type of analysis
        analysis_type = self._identify_analysis_type(analysis_context)

        # Generate questions based on depth and context
        base_questions = self._get_base_questions(analysis_type, current_depth)
        questions.extend(base_questions)

        # Add context-specific questions
        context_questions = self._generate_context_questions(
            analysis_context, current_depth
        )
        questions.extend(context_questions)

        # Add depth escalation questions if appropriate
        if self._should_escalate_depth(analysis_context, current_depth):
            escalation_questions = self._get_escalation_questions(
                analysis_type, current_depth
            )
            questions.extend(escalation_questions)

        return questions

    def get_reasoning_prompts(
        self, question_type: BiochemicalQuestionType, context: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Get reasoning prompts for specific question types

        Args:
            question_type: Type of biochemical question
            context: Analysis context and data

        Returns:
            Dictionary of reasoning prompts and guidance
        """
        prompts = {}

        if question_type == BiochemicalQuestionType.WHAT_IS_HAPPENING:
            prompts.update(self._get_observation_prompts(context))
        elif question_type == BiochemicalQuestionType.WHY_IS_IT_HAPPENING:
            prompts.update(self._get_causal_prompts(context))
        elif question_type == BiochemicalQuestionType.HOW_DOES_IT_WORK:
            prompts.update(self._get_mechanistic_prompts(context))
        elif question_type == BiochemicalQuestionType.WHAT_IF_CHANGED:
            prompts.update(self._get_perturbation_prompts(context))
        elif question_type == BiochemicalQuestionType.WHAT_DOES_IT_MEAN:
            prompts.update(self._get_interpretation_prompts(context))

        return prompts

    def assess_reasoning_quality(
        self, reasoning_text: str, expected_depth: ReasoningDepth
    ) -> Dict[str, Any]:
        """
        Assess the quality of biochemical reasoning

        Args:
            reasoning_text: Text to analyze
            expected_depth: Expected depth of reasoning

        Returns:
            Quality assessment metrics
        """
        assessment = {
            "depth_achieved": self._assess_depth(reasoning_text),
            "mechanistic_content": self._assess_mechanistic_content(reasoning_text),
            "biochemical_accuracy": self._assess_biochemical_terms(reasoning_text),
            "cross_pathway_connections": self._assess_pathway_connections(
                reasoning_text
            ),
            "quantitative_reasoning": self._assess_quantitative_content(reasoning_text),
            "overall_score": 0.0,
        }

        # Calculate overall score
        scores = [v for v in assessment.values() if isinstance(v, (int, float))]
        if scores:
            assessment["overall_score"] = sum(scores) / len(scores)

        return assessment

    def _initialize_reasoning_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize reasoning patterns for different analysis types"""
        return {
            "growth_analysis": {
                "surface_questions": [
                    "What is the predicted growth rate?",
                    "Which nutrients are being consumed?",
                    "What is the biomass composition?",
                ],
                "intermediate_questions": [
                    "Which metabolic pathways are most active during growth?",
                    "What are the rate-limiting steps in biomass production?",
                    "How does nutrient uptake relate to growth yield?",
                ],
                "deep_questions": [
                    "What is the biochemical basis for the observed growth rate?",
                    "How do enzyme kinetics limit pathway flux?",
                    "What regulatory mechanisms control metabolic state?",
                ],
                "mechanistic_questions": [
                    "Explain the electron transport chain efficiency in this growth condition",
                    "How does ATP/NADH ratio affect central carbon metabolism?",
                    "What allosteric regulations govern flux distribution?",
                ],
            },
            "flux_analysis": {
                "surface_questions": [
                    "Which reactions carry the highest flux?",
                    "What is the flux distribution pattern?",
                    "Are there any blocked reactions?",
                ],
                "intermediate_questions": [
                    "How does flux distribution reflect metabolic priorities?",
                    "Which pathways show coordinated flux patterns?",
                    "What does flux variability indicate about network flexibility?",
                ],
                "deep_questions": [
                    "What biochemical constraints shape the flux distribution?",
                    "How do thermodynamic limitations affect pathway usage?",
                    "What does the flux pattern reveal about enzyme regulation?",
                ],
                "mechanistic_questions": [
                    "Explain how cofactor availability shapes flux patterns",
                    "How do metabolite concentrations drive flux directions?",
                    "What role does protein expression play in flux control?",
                ],
            },
            "essentiality_analysis": {
                "surface_questions": [
                    "Which genes are essential for growth?",
                    "What fraction of genes can be deleted?",
                    "Are there conditionally essential genes?",
                ],
                "intermediate_questions": [
                    "What biological functions do essential genes serve?",
                    "How do essential genes cluster by pathway?",
                    "What backup mechanisms exist for non-essential functions?",
                ],
                "deep_questions": [
                    "Why are certain metabolic steps uniquely essential?",
                    "How does network topology determine gene essentiality?",
                    "What evolutionary pressures maintain essential pathways?",
                ],
                "mechanistic_questions": [
                    "Explain the biochemical consequences of deleting each essential gene",
                    "How do epistatic interactions affect essentiality patterns?",
                    "What compensatory mutations could rescue essentiality?",
                ],
            },
        }

    def _initialize_context_triggers(self) -> Dict[str, List[str]]:
        """Initialize triggers for deeper context analysis"""
        return {
            "high_flux_variability": [
                "What biological mechanisms could explain this flux flexibility?",
                "How might the organism benefit from maintaining multiple pathway states?",
                "What environmental conditions would favor different flux distributions?",
            ],
            "multiple_optimal_solutions": [
                "Why does the network have multiple optimal states?",
                "What biological significance does solution degeneracy have?",
                "How might the organism switch between optimal states?",
            ],
            "unexpected_essentiality": [
                "Why is this gene essential when alternatives exist?",
                "What unique biochemical function does this gene provide?",
                "How does this essentiality pattern reflect network structure?",
            ],
            "low_growth_yield": [
                "What biochemical inefficiencies limit growth yield?",
                "How could metabolic engineering improve efficiency?",
                "What evolutionary tradeoffs might explain low yield?",
            ],
            "substrate_specialization": [
                "What biochemical properties make this substrate preferred?",
                "How do transport and metabolic capabilities shape specialization?",
                "What evolutionary advantages does substrate specialization provide?",
            ],
        }

    def _initialize_depth_rules(self) -> Dict[ReasoningDepth, Dict[str, Any]]:
        """Initialize rules for reasoning depth escalation"""
        return {
            ReasoningDepth.SURFACE: {
                "escalation_triggers": [
                    "unusual_patterns_detected",
                    "user_requests_explanation",
                    "contradictory_results",
                ],
                "required_elements": ["basic_observations", "numerical_results"],
            },
            ReasoningDepth.INTERMEDIATE: {
                "escalation_triggers": [
                    "pathway_interactions_complex",
                    "regulatory_effects_apparent",
                    "systems_level_behavior",
                ],
                "required_elements": ["pathway_context", "functional_relationships"],
            },
            ReasoningDepth.DEEP: {
                "escalation_triggers": [
                    "mechanistic_questions_raised",
                    "molecular_details_needed",
                    "quantitative_predictions_required",
                ],
                "required_elements": [
                    "biochemical_mechanisms",
                    "quantitative_analysis",
                ],
            },
            ReasoningDepth.MECHANISTIC: {
                "escalation_triggers": [],  # Deepest level
                "required_elements": [
                    "molecular_mechanisms",
                    "physical_chemistry",
                    "evolutionary_context",
                ],
            },
        }

    def _identify_analysis_type(self, context: Dict[str, Any]) -> str:
        """Identify the type of analysis from context"""
        if "growth_rate" in context or "biomass" in context:
            return "growth_analysis"
        elif "fluxes" in context or "flux_samples" in context:
            return "flux_analysis"
        elif "essential_genes" in context or "gene_deletion" in context:
            return "essentiality_analysis"
        elif "media" in context:
            return "media_analysis"
        else:
            return "general_analysis"

    def _get_base_questions(
        self, analysis_type: str, depth: ReasoningDepth
    ) -> List[str]:
        """Get base questions for analysis type and depth"""
        patterns = self.reasoning_patterns.get(analysis_type, {})

        if depth == ReasoningDepth.SURFACE:
            return patterns.get("surface_questions", [])
        elif depth == ReasoningDepth.INTERMEDIATE:
            return patterns.get("intermediate_questions", [])
        elif depth == ReasoningDepth.DEEP:
            return patterns.get("deep_questions", [])
        elif depth == ReasoningDepth.MECHANISTIC:
            return patterns.get("mechanistic_questions", [])

        return []

    def _generate_context_questions(
        self, context: Dict[str, Any], depth: ReasoningDepth
    ) -> List[str]:
        """Generate questions based on specific context"""
        questions = []

        # Check for context triggers
        for trigger, trigger_questions in self.context_triggers.items():
            if self._check_trigger_condition(trigger, context):
                questions.extend(
                    trigger_questions[:2]
                )  # Limit to top 2 questions per trigger

        # Add data-specific questions
        if "fluxes" in context:
            questions.extend(self._generate_flux_specific_questions(context, depth))

        if "essential_genes" in context:
            questions.extend(
                self._generate_essentiality_specific_questions(context, depth)
            )

        return questions

    def _check_trigger_condition(self, trigger: str, context: Dict[str, Any]) -> bool:
        """Check if a context trigger condition is met"""
        if trigger == "high_flux_variability":
            # Check if flux variability is high
            variability_data = context.get("variability", {})
            if variability_data:
                high_var_count = sum(
                    1
                    for var in variability_data.values()
                    if isinstance(var, dict) and var.get("flexibility", 0) > 5
                )
                return high_var_count > len(variability_data) * 0.1

        elif trigger == "unexpected_essentiality":
            # Check for unexpected essential genes
            essential_genes = context.get("essential_genes", {})
            return (
                len(essential_genes) > 100
            )  # Many essential genes might indicate something unusual

        elif trigger == "low_growth_yield":
            # Check for low growth rate
            growth_rate = context.get("growth_rate", 1.0)
            return growth_rate < 0.5

        return False

    def _generate_flux_specific_questions(
        self, context: Dict[str, Any], depth: ReasoningDepth
    ) -> List[str]:
        """Generate flux-specific questions"""
        questions = []

        fluxes = context.get("fluxes", {})
        if not fluxes:
            return questions

        if depth == ReasoningDepth.SURFACE:
            questions.append("Which metabolic pathways show the highest activity?")
            questions.append("Are there any surprising flux patterns?")

        elif depth == ReasoningDepth.INTERMEDIATE:
            questions.append(
                "How do the flux patterns reflect the organism's metabolic strategy?"
            )
            questions.append(
                "What do the flux ratios tell us about pathway regulation?"
            )

        elif depth == ReasoningDepth.DEEP:
            questions.append(
                "What biochemical constraints determine these specific flux values?"
            )
            questions.append(
                "How do cofactor balances influence the flux distribution?"
            )

        return questions

    def _generate_essentiality_specific_questions(
        self, context: Dict[str, Any], depth: ReasoningDepth
    ) -> List[str]:
        """Generate essentiality-specific questions"""
        questions = []

        essential_genes = context.get("essential_genes", {})
        if not essential_genes:
            return questions

        if depth == ReasoningDepth.SURFACE:
            questions.append(
                "What types of biological functions are represented among essential genes?"
            )

        elif depth == ReasoningDepth.INTERMEDIATE:
            questions.append("How do essential genes cluster by metabolic pathway?")
            questions.append(
                "Are there essential genes that could be targeted for intervention?"
            )

        elif depth == ReasoningDepth.DEEP:
            questions.append(
                "What makes certain metabolic steps irreplaceable in this organism?"
            )
            questions.append(
                "How does the essentiality pattern reflect evolutionary constraints?"
            )

        return questions

    def _should_escalate_depth(
        self, context: Dict[str, Any], current_depth: ReasoningDepth
    ) -> bool:
        """Determine if reasoning depth should be escalated"""
        if current_depth == ReasoningDepth.MECHANISTIC:
            return False  # Already at deepest level

        depth_rules = self.depth_escalation_rules.get(current_depth, {})
        triggers = depth_rules.get("escalation_triggers", [])

        # Check for escalation triggers in context
        for trigger in triggers:
            if trigger == "unusual_patterns_detected":
                if self._detect_unusual_patterns(context):
                    return True
            elif trigger == "pathway_interactions_complex":
                if self._detect_complex_interactions(context):
                    return True
            elif trigger == "systems_level_behavior":
                if self._detect_systems_behavior(context):
                    return True

        return False

    def _get_escalation_questions(
        self, analysis_type: str, current_depth: ReasoningDepth
    ) -> List[str]:
        """Get questions for depth escalation"""
        next_depth = self._get_next_depth(current_depth)
        if next_depth:
            return self._get_base_questions(analysis_type, next_depth)
        return []

    def _get_next_depth(
        self, current_depth: ReasoningDepth
    ) -> Optional[ReasoningDepth]:
        """Get the next reasoning depth level"""
        if current_depth == ReasoningDepth.SURFACE:
            return ReasoningDepth.INTERMEDIATE
        elif current_depth == ReasoningDepth.INTERMEDIATE:
            return ReasoningDepth.DEEP
        elif current_depth == ReasoningDepth.DEEP:
            return ReasoningDepth.MECHANISTIC
        return None

    def _get_observation_prompts(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Get prompts for observational reasoning"""
        return {
            "observation_prompt": """
            Analyze the data systematically and describe what you observe:
            - What are the key numerical results?
            - What patterns are apparent in the data?
            - What stands out as unusual or expected?
            Focus on factual observations before interpretation.
            """,
            "pattern_recognition": """
            Look for patterns in the data:
            - Are there correlations between different measurements?
            - Do you see groupings or clusters in the results?
            - What trends or relationships are evident?
            """,
        }

    def _get_causal_prompts(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Get prompts for causal reasoning"""
        return {
            "causal_analysis": """
            Explain the biological causes behind your observations:
            - What biological processes could produce these results?
            - How do the underlying mechanisms explain the patterns?
            - What biochemical principles drive these outcomes?
            """,
            "mechanism_identification": """
            Identify the mechanisms at work:
            - Which enzymes and pathways are involved?
            - How do regulatory networks control these processes?
            - What molecular interactions drive the observed behavior?
            """,
        }

    def _get_mechanistic_prompts(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Get prompts for mechanistic reasoning"""
        return {
            "mechanistic_detail": """
            Explain the detailed biochemical mechanisms:
            - How do enzyme kinetics affect pathway flux?
            - What role do cofactors and allosteric regulation play?
            - How do thermodynamic constraints shape the system?
            """,
            "molecular_basis": """
            Describe the molecular basis of the phenomena:
            - What protein structures and binding sites are critical?
            - How do metabolite concentrations drive reactions?
            - What physical chemistry principles apply?
            """,
        }

    def _get_perturbation_prompts(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Get prompts for perturbation analysis"""
        return {
            "perturbation_analysis": """
            Analyze how the system would respond to changes:
            - What would happen if key enzymes were inhibited?
            - How would the system adapt to different nutrients?
            - What compensatory mechanisms would activate?
            """,
            "prediction_generation": """
            Generate testable predictions:
            - What experiments would test your hypotheses?
            - What outcomes would you expect from perturbations?
            - How could you validate your mechanistic understanding?
            """,
        }

    def _get_interpretation_prompts(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Get prompts for biological interpretation"""
        return {
            "biological_significance": """
            Interpret the biological significance:
            - What does this tell us about the organism's ecology?
            - How do these results inform biotechnology applications?
            - What evolutionary insights can be drawn?
            """,
            "broader_implications": """
            Consider broader implications:
            - How do these findings connect to known biology?
            - What questions do they raise for future research?
            - How might this knowledge be applied?
            """,
        }

    def _detect_unusual_patterns(self, context: Dict[str, Any]) -> bool:
        """Detect unusual patterns that warrant deeper analysis"""
        # Check for extreme values
        if "growth_rate" in context:
            growth_rate = context["growth_rate"]
            if growth_rate > 2.0 or growth_rate < 0.1:
                return True

        # Check for highly variable fluxes
        if "variability" in context:
            high_var_count = sum(
                1
                for var in context["variability"].values()
                if isinstance(var, dict) and var.get("flexibility", 0) > 10
            )
            if high_var_count > 5:
                return True

        return False

    def _detect_complex_interactions(self, context: Dict[str, Any]) -> bool:
        """Detect complex pathway interactions"""
        # Multiple active pathways
        if "pathway_analysis" in context:
            active_pathways = len(context["pathway_analysis"])
            return active_pathways > 5

        return False

    def _detect_systems_behavior(self, context: Dict[str, Any]) -> bool:
        """Detect systems-level behavior"""
        # Cross-pathway flux coordination
        if "fluxes" in context:
            return len(context["fluxes"]) > 50  # Large flux networks

        return False

    def _assess_depth(self, reasoning_text: str) -> float:
        """Assess the depth of reasoning in text"""
        depth_indicators = {
            "surface": ["result", "shows", "indicates", "value"],
            "intermediate": ["because", "pathway", "function", "process"],
            "deep": ["mechanism", "regulation", "kinetics", "thermodynamics"],
            "mechanistic": ["allosteric", "cofactor", "structure", "binding"],
        }

        text_lower = reasoning_text.lower()
        scores = {}

        for depth, indicators in depth_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            scores[depth] = score / len(indicators)

        # Weight deeper levels more heavily
        weights = {
            "surface": 0.25,
            "intermediate": 0.5,
            "deep": 0.75,
            "mechanistic": 1.0,
        }
        weighted_score = sum(scores[depth] * weights[depth] for depth in scores)

        return min(1.0, weighted_score)

    def _assess_mechanistic_content(self, reasoning_text: str) -> float:
        """Assess mechanistic content in reasoning"""
        mechanistic_terms = [
            "enzyme",
            "substrate",
            "product",
            "kinetics",
            "binding",
            "allosteric",
            "cofactor",
            "regulation",
            "inhibition",
            "activation",
        ]

        text_lower = reasoning_text.lower()
        term_count = sum(1 for term in mechanistic_terms if term in text_lower)

        return min(1.0, term_count / 5.0)  # Normalize to 0-1

    def _assess_biochemical_terms(self, reasoning_text: str) -> float:
        """Assess usage of biochemical terminology"""
        biochemical_terms = [
            "glucose",
            "pyruvate",
            "atp",
            "nadh",
            "glycolysis",
            "tca",
            "phosphorylation",
            "oxidation",
            "reduction",
            "metabolism",
        ]

        text_lower = reasoning_text.lower()
        term_count = sum(1 for term in biochemical_terms if term in text_lower)

        return min(1.0, term_count / 3.0)  # Normalize to 0-1

    def _assess_pathway_connections(self, reasoning_text: str) -> float:
        """Assess cross-pathway reasoning"""
        connection_terms = [
            "connects to",
            "linked with",
            "affects",
            "influences",
            "upstream",
            "downstream",
            "feedback",
            "coordinated",
        ]

        text_lower = reasoning_text.lower()
        connection_count = sum(1 for term in connection_terms if term in text_lower)

        return min(1.0, connection_count / 2.0)  # Normalize to 0-1

    def _assess_quantitative_content(self, reasoning_text: str) -> float:
        """Assess quantitative reasoning"""
        # Look for numbers and quantitative terms
        import re

        numbers = len(re.findall(r"\d+\.?\d*", reasoning_text))
        quantitative_terms = ["ratio", "rate", "concentration", "fold", "percent"]

        text_lower = reasoning_text.lower()
        quant_term_count = sum(1 for term in quantitative_terms if term in text_lower)

        score = numbers * 0.1 + quant_term_count * 0.2
        return min(1.0, score)  # Normalize to 0-1
