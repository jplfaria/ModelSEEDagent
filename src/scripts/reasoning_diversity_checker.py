"""
Reasoning Diversity Checker for ModelSEEDagent Phase 3

Anti-bias validation system for ensuring diverse, unbiased reasoning patterns
and preventing over-constrained or templated responses in biochemical analysis.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DiversityMetrics:
    """Metrics for reasoning diversity assessment"""

    vocabulary_diversity: float  # Unique words / total words
    structural_diversity: float  # Variation in reasoning patterns
    tool_usage_diversity: float  # Diversity in tool selection
    approach_diversity: float  # Variation in analytical approaches
    hypothesis_diversity: float  # Diversity in hypothesis generation
    overall_diversity_score: float
    bias_risk_level: str  # "low", "medium", "high"
    diversity_grade: str


@dataclass
class BiasDetectionResult:
    """Result of bias detection analysis"""

    bias_type: str
    severity: str  # "low", "medium", "high", "critical"
    confidence: float  # 0.0 to 1.0
    evidence: List[str]
    affected_dimensions: List[str]
    recommendation: str
    risk_mitigation: List[str]


class ReasoningDiversityChecker:
    """
    Advanced anti-bias validation and diversity checking system

    Detects and prevents various forms of bias in biochemical reasoning:
    - Tool selection bias
    - Confirmation bias
    - Anchoring bias
    - Availability heuristic bias
    - Template over-reliance
    - Vocabulary limitations
    - Structural rigidity
    """

    def __init__(self):
        self.bias_detectors = {
            "tool_selection_bias": self._detect_tool_selection_bias,
            "confirmation_bias": self._detect_confirmation_bias,
            "anchoring_bias": self._detect_anchoring_bias,
            "availability_bias": self._detect_availability_bias,
            "template_bias": self._detect_template_bias,
            "vocabulary_bias": self._detect_vocabulary_bias,
            "approach_rigidity": self._detect_approach_rigidity,
            "hypothesis_narrowing": self._detect_hypothesis_narrowing,
        }

        self.diversity_thresholds = {
            "vocabulary_diversity": {"excellent": 0.8, "good": 0.6, "acceptable": 0.4},
            "structural_diversity": {"excellent": 0.9, "good": 0.7, "acceptable": 0.5},
            "tool_usage_diversity": {"excellent": 0.8, "good": 0.6, "acceptable": 0.4},
            "approach_diversity": {"excellent": 0.85, "good": 0.65, "acceptable": 0.45},
        }

        # Initialize bias detection patterns
        self._initialize_bias_patterns()

        logger.info(
            "ReasoningDiversityChecker initialized with 8 bias detection methods"
        )

    def assess_reasoning_diversity(
        self,
        reasoning_traces: List[Dict[str, Any]],
        session_context: Optional[Dict[str, Any]] = None,
    ) -> DiversityMetrics:
        """
        Assess diversity across multiple reasoning traces

        Args:
            reasoning_traces: List of reasoning execution traces
            session_context: Optional session context for analysis

        Returns:
            Comprehensive diversity assessment metrics
        """
        try:
            if not reasoning_traces:
                return self._create_empty_diversity_metrics()

            # Calculate individual diversity dimensions
            vocab_diversity = self._calculate_vocabulary_diversity(reasoning_traces)
            structural_diversity = self._calculate_structural_diversity(
                reasoning_traces
            )
            tool_diversity = self._calculate_tool_usage_diversity(reasoning_traces)
            approach_diversity = self._calculate_approach_diversity(reasoning_traces)
            hypothesis_diversity = self._calculate_hypothesis_diversity(
                reasoning_traces
            )

            # Calculate overall diversity score
            overall_score = (
                vocab_diversity * 0.2
                + structural_diversity * 0.25
                + tool_diversity * 0.2
                + approach_diversity * 0.2
                + hypothesis_diversity * 0.15
            )

            # Determine bias risk level
            bias_risk = self._assess_bias_risk_level(overall_score, reasoning_traces)

            # Assign diversity grade
            diversity_grade = self._assign_diversity_grade(overall_score)

            metrics = DiversityMetrics(
                vocabulary_diversity=vocab_diversity,
                structural_diversity=structural_diversity,
                tool_usage_diversity=tool_diversity,
                approach_diversity=approach_diversity,
                hypothesis_diversity=hypothesis_diversity,
                overall_diversity_score=overall_score,
                bias_risk_level=bias_risk,
                diversity_grade=diversity_grade,
            )

            logger.info(
                f"Diversity assessment completed: {diversity_grade} "
                f"(score: {overall_score:.3f}, bias risk: {bias_risk})"
            )

            return metrics

        except Exception as e:
            logger.error(f"Diversity assessment failed: {e}")
            return self._create_error_diversity_metrics()

    def detect_bias_patterns(
        self,
        reasoning_traces: List[Dict[str, Any]],
        historical_context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[BiasDetectionResult]:
        """
        Detect various bias patterns in reasoning traces

        Args:
            reasoning_traces: Recent reasoning traces to analyze
            historical_context: Historical traces for comparison

        Returns:
            List of detected bias patterns with severity and recommendations
        """
        try:
            detected_biases = []

            # Run all bias detection methods
            for bias_type, detector_method in self.bias_detectors.items():
                bias_result = detector_method(reasoning_traces, historical_context)
                if bias_result:
                    detected_biases.append(bias_result)

            # Sort by severity (critical first)
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            detected_biases.sort(key=lambda x: severity_order.get(x.severity, 4))

            logger.info(f"Detected {len(detected_biases)} bias patterns")

            return detected_biases

        except Exception as e:
            logger.error(f"Bias detection failed: {e}")
            return []

    def generate_diversity_recommendations(
        self,
        diversity_metrics: DiversityMetrics,
        bias_results: List[BiasDetectionResult],
    ) -> List[str]:
        """
        Generate specific recommendations for improving reasoning diversity

        Args:
            diversity_metrics: Current diversity assessment
            bias_results: Detected bias patterns

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        # Dimension-specific recommendations
        if diversity_metrics.vocabulary_diversity < 0.6:
            recommendations.append(
                "Expand biochemical vocabulary usage - incorporate more domain-specific terminology"
            )

        if diversity_metrics.structural_diversity < 0.7:
            recommendations.append(
                "Vary reasoning structure - avoid repetitive analytical patterns"
            )

        if diversity_metrics.tool_usage_diversity < 0.6:
            recommendations.append(
                "Diversify tool selection - explore alternative analytical approaches"
            )

        if diversity_metrics.approach_diversity < 0.65:
            recommendations.append(
                "Adopt multiple analytical perspectives - consider alternative methodologies"
            )

        if diversity_metrics.hypothesis_diversity < 0.5:
            recommendations.append(
                "Generate more diverse hypotheses - explore competing explanations"
            )

        # Bias-specific recommendations
        for bias_result in bias_results:
            if bias_result.severity in ["critical", "high"]:
                recommendations.extend(bias_result.risk_mitigation)

        # Overall recommendations based on bias risk
        if diversity_metrics.bias_risk_level == "high":
            recommendations.append(
                "Critical: Implement systematic bias mitigation protocols"
            )
        elif diversity_metrics.bias_risk_level == "medium":
            recommendations.append(
                "Moderate: Regular bias monitoring and diverse approach validation"
            )

        return recommendations

    def _initialize_bias_patterns(self):
        """Initialize patterns for bias detection"""

        # Confirmation bias indicators
        self.confirmation_patterns = [
            r"as expected",
            r"confirms?",
            r"supports? the",
            r"validates? our",
            r"consistent with",
            r"in line with",
            r"agrees? with",
        ]

        # Anchoring bias indicators
        self.anchoring_patterns = [
            r"initially",
            r"first impression",
            r"starting point",
            r"baseline assumption",
        ]

        # Template language patterns
        self.template_patterns = [
            r"the analysis shows",
            r"results indicate",
            r"data suggests",
            r"we can conclude",
            r"it appears that",
            r"the findings reveal",
        ]

        # Vocabulary limitation indicators
        self.limited_vocab_patterns = [
            r"\bthing\b",
            r"\bstuff\b",
            r"\bit\b",
            r"\bsomething\b",
            r"\bsomehow\b",
        ]

    def _calculate_vocabulary_diversity(self, traces: List[Dict[str, Any]]) -> float:
        """Calculate vocabulary diversity across reasoning traces"""

        all_words = []
        for trace in traces:
            text = self._extract_text_from_trace(trace)
            words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
            all_words.extend(words)

        if not all_words:
            return 0.0

        unique_words = len(set(all_words))
        total_words = len(all_words)

        # Vocabulary diversity = unique words / total words
        diversity = unique_words / total_words

        # Bonus for biochemical domain vocabulary
        domain_vocab_bonus = self._calculate_domain_vocabulary_bonus(all_words)

        return min(1.0, diversity + domain_vocab_bonus)

    def _calculate_structural_diversity(self, traces: List[Dict[str, Any]]) -> float:
        """Calculate structural diversity in reasoning patterns"""

        structural_patterns = []

        for trace in traces:
            # Analyze reasoning structure patterns
            steps = trace.get("steps", [])
            pattern = self._extract_structural_pattern(steps)
            structural_patterns.append(pattern)

        if not structural_patterns:
            return 0.0

        unique_patterns = len(set(structural_patterns))
        total_patterns = len(structural_patterns)

        return unique_patterns / total_patterns if total_patterns > 0 else 0.0

    def _calculate_tool_usage_diversity(self, traces: List[Dict[str, Any]]) -> float:
        """Calculate diversity in tool usage patterns"""

        all_tools = []
        tool_sequences = []

        for trace in traces:
            tools_used = trace.get("tools_used", [])
            all_tools.extend(tools_used)

            # Create tool sequence signature
            sequence = "_".join(tools_used[:5])  # First 5 tools
            tool_sequences.append(sequence)

        if not all_tools:
            return 0.0

        # Tool type diversity
        unique_tools = len(set(all_tools))
        total_tool_uses = len(all_tools)
        tool_diversity = unique_tools / max(1, total_tool_uses)

        # Sequence diversity
        unique_sequences = len(set(tool_sequences))
        total_sequences = len(tool_sequences)
        sequence_diversity = unique_sequences / max(1, total_sequences)

        return (tool_diversity + sequence_diversity) / 2

    def _calculate_approach_diversity(self, traces: List[Dict[str, Any]]) -> float:
        """Calculate diversity in analytical approaches"""

        approach_signatures = []

        for trace in traces:
            # Extract approach signature from reasoning patterns
            text = self._extract_text_from_trace(trace)
            tools = trace.get("tools_used", [])

            signature = self._create_approach_signature(text, tools)
            approach_signatures.append(signature)

        if not approach_signatures:
            return 0.0

        unique_approaches = len(set(approach_signatures))
        total_approaches = len(approach_signatures)

        return unique_approaches / total_approaches

    def _calculate_hypothesis_diversity(self, traces: List[Dict[str, Any]]) -> float:
        """Calculate diversity in hypothesis generation"""

        hypotheses = []

        for trace in traces:
            text = self._extract_text_from_trace(trace)

            # Extract hypothesis-related statements
            hypothesis_patterns = [
                r"hypothesis",
                r"propose",
                r"suggest",
                r"predict",
                r"expect",
                r"likely",
                r"possible",
                r"potential",
                r"might",
                r"could",
            ]

            for pattern in hypothesis_patterns:
                matches = re.findall(f"{pattern}[^.]*", text, re.IGNORECASE)
                hypotheses.extend(matches)

        if not hypotheses:
            return 0.0

        # Calculate semantic diversity of hypotheses
        unique_hypotheses = len(set(h.lower().strip() for h in hypotheses))
        total_hypotheses = len(hypotheses)

        return unique_hypotheses / total_hypotheses

    def _assess_bias_risk_level(
        self, diversity_score: float, traces: List[Dict[str, Any]]
    ) -> str:
        """Assess overall bias risk level"""

        # Check for critical bias indicators
        critical_indicators = 0

        for trace in traces:
            text = self._extract_text_from_trace(trace)

            # Check for template over-reliance
            template_count = sum(
                len(re.findall(pattern, text, re.IGNORECASE))
                for pattern in self.template_patterns
            )
            if template_count > 5:
                critical_indicators += 1

            # Check for confirmation bias language
            confirmation_count = sum(
                len(re.findall(pattern, text, re.IGNORECASE))
                for pattern in self.confirmation_patterns
            )
            if confirmation_count > 3:
                critical_indicators += 1

        # Determine risk level
        if diversity_score < 0.4 or critical_indicators >= 2:
            return "high"
        elif diversity_score < 0.6 or critical_indicators >= 1:
            return "medium"
        else:
            return "low"

    def _assign_diversity_grade(self, score: float) -> str:
        """Assign letter grade based on diversity score"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C+"
        elif score >= 0.4:
            return "C"
        elif score >= 0.3:
            return "D"
        else:
            return "F"

    # Bias Detection Methods

    def _detect_tool_selection_bias(
        self, traces: List[Dict[str, Any]], historical: Optional[List[Dict[str, Any]]]
    ) -> Optional[BiasDetectionResult]:
        """Detect tool selection bias patterns"""

        tool_usage = Counter()
        total_traces = 0

        for trace in traces:
            tools_used = trace.get("tools_used", [])
            tool_usage.update(tools_used)
            total_traces += 1

        if not tool_usage or total_traces == 0:
            return None

        # Check for over-reliance on single tool
        most_common_tool, usage_count = tool_usage.most_common(1)[0]
        usage_rate = usage_count / total_traces

        if usage_rate > 0.7:  # Used in >70% of traces
            return BiasDetectionResult(
                bias_type="tool_selection_bias",
                severity="medium" if usage_rate < 0.85 else "high",
                confidence=min(0.9, usage_rate),
                evidence=[f"{most_common_tool} used in {usage_rate:.1%} of analyses"],
                affected_dimensions=["methodological_rigor", "synthesis_effectiveness"],
                recommendation=f"Diversify tool usage - {most_common_tool} is over-used",
                risk_mitigation=[
                    "Explicitly consider alternative analytical tools",
                    "Implement tool rotation strategies",
                    "Review tool selection criteria",
                ],
            )

        return None

    def _detect_confirmation_bias(
        self, traces: List[Dict[str, Any]], historical: Optional[List[Dict[str, Any]]]
    ) -> Optional[BiasDetectionResult]:
        """Detect confirmation bias in reasoning"""

        confirmation_indicators = 0
        total_statements = 0
        evidence = []

        for trace in traces:
            text = self._extract_text_from_trace(trace)

            for pattern in self.confirmation_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                confirmation_indicators += len(matches)
                if matches:
                    evidence.extend(matches[:2])  # Add first 2 matches as evidence

            # Count total statements for normalization
            sentences = re.split(r"[.!?]+", text)
            total_statements += len(sentences)

        if total_statements == 0:
            return None

        confirmation_rate = confirmation_indicators / total_statements

        if confirmation_rate > 0.15:  # >15% of statements show confirmation bias
            return BiasDetectionResult(
                bias_type="confirmation_bias",
                severity="medium" if confirmation_rate < 0.25 else "high",
                confidence=min(0.85, confirmation_rate * 3),
                evidence=evidence[:5],  # Top 5 examples
                affected_dimensions=["biological_accuracy", "reasoning_transparency"],
                recommendation="Include alternative explanations and contradictory evidence",
                risk_mitigation=[
                    "Actively seek disconfirming evidence",
                    "Consider multiple competing hypotheses",
                    "Include uncertainty acknowledgment",
                ],
            )

        return None

    def _detect_anchoring_bias(
        self, traces: List[Dict[str, Any]], historical: Optional[List[Dict[str, Any]]]
    ) -> Optional[BiasDetectionResult]:
        """Detect anchoring bias patterns"""

        anchoring_indicators = 0
        evidence = []

        for trace in traces:
            text = self._extract_text_from_trace(trace)
            steps = trace.get("steps", [])

            # Check for anchoring language
            for pattern in self.anchoring_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                anchoring_indicators += len(matches)
                if matches:
                    evidence.extend(matches[:2])

            # Check for limited exploration (few analysis steps)
            if len(steps) < 3:
                anchoring_indicators += 1
                evidence.append("Limited analytical exploration")

        if anchoring_indicators > 2:
            return BiasDetectionResult(
                bias_type="anchoring_bias",
                severity="low" if anchoring_indicators < 4 else "medium",
                confidence=min(0.8, anchoring_indicators / 5.0),
                evidence=evidence[:3],
                affected_dimensions=["synthesis_effectiveness", "methodological_rigor"],
                recommendation="Explore multiple starting points and approaches",
                risk_mitigation=[
                    "Delay initial judgments",
                    "Consider multiple entry points",
                    "Validate preliminary conclusions",
                ],
            )

        return None

    def _detect_availability_bias(
        self, traces: List[Dict[str, Any]], historical: Optional[List[Dict[str, Any]]]
    ) -> Optional[BiasDetectionResult]:
        """Detect availability heuristic bias"""

        # Look for over-reliance on easily recalled examples
        repeated_examples = Counter()

        for trace in traces:
            text = self._extract_text_from_trace(trace)

            # Extract example references (simplified)
            example_patterns = [
                r"for example",
                r"such as",
                r"including",
                r"like",
                r"similar to",
            ]

            for pattern in example_patterns:
                matches = re.findall(f"{pattern}[^.]*", text, re.IGNORECASE)
                for match in matches:
                    repeated_examples[match.lower().strip()] += 1

        # Check for overused examples
        overused_examples = [ex for ex, count in repeated_examples.items() if count > 2]

        if overused_examples:
            return BiasDetectionResult(
                bias_type="availability_bias",
                severity="low",
                confidence=0.6,
                evidence=overused_examples[:3],
                affected_dimensions=["biological_accuracy", "synthesis_effectiveness"],
                recommendation="Diversify examples and case studies",
                risk_mitigation=[
                    "Source examples from multiple contexts",
                    "Validate example representativeness",
                    "Include less obvious examples",
                ],
            )

        return None

    def _detect_template_bias(
        self, traces: List[Dict[str, Any]], historical: Optional[List[Dict[str, Any]]]
    ) -> Optional[BiasDetectionResult]:
        """Detect over-reliance on template language"""

        template_usage = 0
        total_statements = 0
        evidence = []

        for trace in traces:
            text = self._extract_text_from_trace(trace)
            sentences = re.split(r"[.!?]+", text)
            total_statements += len(sentences)

            for pattern in self.template_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                template_usage += len(matches)
                if matches:
                    evidence.extend(matches[:2])

        if total_statements == 0:
            return None

        template_rate = template_usage / total_statements

        if template_rate > 0.2:  # >20% template language
            return BiasDetectionResult(
                bias_type="template_bias",
                severity="medium" if template_rate < 0.3 else "high",
                confidence=min(0.9, template_rate * 2),
                evidence=evidence[:5],
                affected_dimensions=[
                    "reasoning_transparency",
                    "synthesis_effectiveness",
                ],
                recommendation="Reduce template language usage, increase natural expression",
                risk_mitigation=[
                    "Vary expression patterns",
                    "Use domain-specific language",
                    "Personalize reasoning style",
                ],
            )

        return None

    def _detect_vocabulary_bias(
        self, traces: List[Dict[str, Any]], historical: Optional[List[Dict[str, Any]]]
    ) -> Optional[BiasDetectionResult]:
        """Detect vocabulary limitations and bias"""

        limited_vocab_count = 0
        total_words = 0
        evidence = []

        for trace in traces:
            text = self._extract_text_from_trace(trace)
            words = re.findall(r"\b\w+\b", text.lower())
            total_words += len(words)

            for pattern in self.limited_vocab_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                limited_vocab_count += len(matches)
                if matches:
                    evidence.extend(matches[:2])

        if total_words == 0:
            return None

        limitation_rate = limited_vocab_count / total_words

        if limitation_rate > 0.05:  # >5% limited vocabulary
            return BiasDetectionResult(
                bias_type="vocabulary_bias",
                severity="low" if limitation_rate < 0.1 else "medium",
                confidence=min(0.8, limitation_rate * 10),
                evidence=evidence[:3],
                affected_dimensions=["reasoning_transparency", "biological_accuracy"],
                recommendation="Expand biochemical vocabulary usage",
                risk_mitigation=[
                    "Use specific domain terminology",
                    "Avoid vague language",
                    "Define technical concepts",
                ],
            )

        return None

    def _detect_approach_rigidity(
        self, traces: List[Dict[str, Any]], historical: Optional[List[Dict[str, Any]]]
    ) -> Optional[BiasDetectionResult]:
        """Detect rigidity in analytical approaches"""

        approach_signatures = []

        for trace in traces:
            tools = trace.get("tools_used", [])
            steps = trace.get("steps", [])

            # Create approach signature
            signature = f"{len(tools)}:{len(steps)}:{':'.join(tools[:3])}"
            approach_signatures.append(signature)

        if not approach_signatures:
            return None

        unique_approaches = len(set(approach_signatures))
        total_approaches = len(approach_signatures)

        flexibility_ratio = unique_approaches / total_approaches

        if flexibility_ratio < 0.5:  # <50% unique approaches
            return BiasDetectionResult(
                bias_type="approach_rigidity",
                severity="medium" if flexibility_ratio > 0.3 else "high",
                confidence=0.8,
                evidence=[
                    f"Only {unique_approaches} unique approaches in {total_approaches} traces"
                ],
                affected_dimensions=["methodological_rigor", "synthesis_effectiveness"],
                recommendation="Diversify analytical approaches and methodologies",
                risk_mitigation=[
                    "Experiment with different tool combinations",
                    "Vary reasoning sequences",
                    "Consider alternative methodologies",
                ],
            )

        return None

    def _detect_hypothesis_narrowing(
        self, traces: List[Dict[str, Any]], historical: Optional[List[Dict[str, Any]]]
    ) -> Optional[BiasDetectionResult]:
        """Detect narrowing of hypothesis space"""

        hypothesis_diversity = self._calculate_hypothesis_diversity(traces)

        if hypothesis_diversity < 0.4:  # Low hypothesis diversity
            return BiasDetectionResult(
                bias_type="hypothesis_narrowing",
                severity="medium",
                confidence=0.7,
                evidence=[f"Hypothesis diversity score: {hypothesis_diversity:.2f}"],
                affected_dimensions=["biological_accuracy", "synthesis_effectiveness"],
                recommendation="Generate more diverse and competing hypotheses",
                risk_mitigation=[
                    "Consider alternative explanations",
                    "Challenge initial assumptions",
                    "Explore competing theories",
                ],
            )

        return None

    # Helper Methods

    def _extract_text_from_trace(self, trace: Dict[str, Any]) -> str:
        """Extract all text content from a reasoning trace"""
        text_parts = []

        if "final_conclusion" in trace:
            text_parts.append(str(trace["final_conclusion"]))

        if "steps" in trace:
            for step in trace["steps"]:
                if isinstance(step, dict):
                    for value in step.values():
                        if isinstance(value, str):
                            text_parts.append(value)
                elif isinstance(step, str):
                    text_parts.append(step)

        return " ".join(text_parts)

    def _extract_structural_pattern(self, steps: List[Dict[str, Any]]) -> str:
        """Extract structural pattern from reasoning steps"""
        if not steps:
            return "empty"

        # Simplified pattern extraction
        pattern_elements = []
        for step in steps:
            if isinstance(step, dict):
                if "tool" in step:
                    pattern_elements.append("tool")
                elif "analysis" in step:
                    pattern_elements.append("analysis")
                elif "conclusion" in step:
                    pattern_elements.append("conclusion")
                else:
                    pattern_elements.append("other")

        return "_".join(pattern_elements)

    def _calculate_domain_vocabulary_bonus(self, words: List[str]) -> float:
        """Calculate bonus for domain-specific vocabulary usage"""
        domain_terms = {
            "metabolism",
            "glycolysis",
            "tca",
            "enzyme",
            "substrate",
            "flux",
            "biomass",
            "growth",
            "essential",
            "pathway",
            "reaction",
            "compound",
            "atp",
            "nadh",
            "cofactor",
            "metabolite",
            "stoichiometry",
        }

        domain_count = sum(1 for word in words if word in domain_terms)
        total_count = len(words)

        if total_count == 0:
            return 0.0

        domain_ratio = domain_count / total_count
        return min(0.2, domain_ratio)  # Max 20% bonus

    def _create_approach_signature(self, text: str, tools: List[str]) -> str:
        """Create signature for analytical approach"""
        # Simplified approach signature
        tool_signature = "_".join(sorted(set(tools))[:3])

        # Text-based approach indicators
        approach_words = ["systematic", "comprehensive", "detailed", "rapid", "focused"]
        text_lower = text.lower()
        approach_indicators = [word for word in approach_words if word in text_lower]

        text_signature = "_".join(sorted(approach_indicators))

        return f"{tool_signature}:{text_signature}"

    def _create_empty_diversity_metrics(self) -> DiversityMetrics:
        """Create empty diversity metrics for error cases"""
        return DiversityMetrics(
            vocabulary_diversity=0.0,
            structural_diversity=0.0,
            tool_usage_diversity=0.0,
            approach_diversity=0.0,
            hypothesis_diversity=0.0,
            overall_diversity_score=0.0,
            bias_risk_level="high",
            diversity_grade="F",
        )

    def _create_error_diversity_metrics(self) -> DiversityMetrics:
        """Create error diversity metrics"""
        return DiversityMetrics(
            vocabulary_diversity=0.0,
            structural_diversity=0.0,
            tool_usage_diversity=0.0,
            approach_diversity=0.0,
            hypothesis_diversity=0.0,
            overall_diversity_score=0.0,
            bias_risk_level="high",
            diversity_grade="F",
        )
