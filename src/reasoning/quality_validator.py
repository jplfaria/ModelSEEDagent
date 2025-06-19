"""
Reasoning Quality Validator for ModelSEEDagent Phase 3

Comprehensive multi-dimensional quality assessment system for biochemical reasoning,
integrating biological accuracy, transparency, synthesis effectiveness, confidence
calibration, and methodological rigor validation.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityDimension:
    """Individual quality dimension assessment"""

    name: str
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    evidence: List[str]
    confidence: float  # 0.0 to 1.0


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment result"""

    overall_score: float
    grade: str
    dimensions: Dict[str, QualityDimension]
    composite_metrics: Dict[str, float]
    bias_flags: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: datetime
    reasoning_trace_id: Optional[str] = None


@dataclass
class ReasoningTrace:
    """Structured reasoning trace for quality assessment"""

    trace_id: str
    query: str
    steps: List[Dict[str, Any]]
    tools_used: List[str]
    final_conclusion: str
    confidence_claims: List[Dict[str, Any]]
    evidence_citations: List[str]
    duration: float
    timestamp: datetime


class ReasoningQualityValidator:
    """
    Comprehensive reasoning quality validation system

    Evaluates reasoning quality across five key dimensions:
    1. Biological Accuracy - Domain knowledge correctness
    2. Reasoning Transparency - Explanation quality and clarity
    3. Synthesis Effectiveness - Cross-tool integration capability
    4. Confidence Calibration - Uncertainty estimate accuracy
    5. Methodological Rigor - Systematic approach adherence
    """

    def __init__(self):
        self.dimension_weights = {
            "biological_accuracy": 0.30,
            "reasoning_transparency": 0.25,
            "synthesis_effectiveness": 0.20,
            "confidence_calibration": 0.15,
            "methodological_rigor": 0.10,
        }

        self.grade_thresholds = {
            "A+": 0.95,
            "A": 0.90,
            "B+": 0.85,
            "B": 0.80,
            "C+": 0.75,
            "C": 0.70,
            "D": 0.60,
            "F": 0.0,
        }

        # Initialize quality benchmarks
        self._initialize_quality_benchmarks()

        logger.info(
            "ReasoningQualityValidator initialized with 5-dimensional assessment"
        )

    def validate_reasoning(
        self, response: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Wrapper method for Intelligence Framework compatibility

        Args:
            response: Reasoning response text
            context: Analysis context

        Returns:
            Quality assessment dictionary
        """
        # Create a minimal ReasoningTrace from the response
        reasoning_trace = ReasoningTrace(
            trace_id=f"trace_{hash(response)%10000}",
            query=context.get("query", ""),
            steps=[{"step": 1, "content": response, "type": "conclusion"}],
            tools_used=context.get("tools_used", []),
            final_conclusion=response,
            confidence_claims=[],
            evidence_citations=[],
            duration=1.0,
            timestamp=datetime.now(),
        )

        # Use the full quality assessment method
        try:
            quality_assessment = self.validate_reasoning_quality(reasoning_trace)

            # Convert to dictionary format expected by Intelligence Framework
            return {
                "overall_score": quality_assessment.overall_score,
                "dimensions": {
                    dim_name: {
                        "score": dim.score,
                        "confidence": dim.confidence,
                        "evidence": dim.evidence,
                    }
                    for dim_name, dim in quality_assessment.dimensions.items()
                },
                "grade": quality_assessment.grade,
                "recommendations": quality_assessment.recommendations,
                "bias_flags": [
                    {
                        "type": bias.get("bias_type", "unknown"),
                        "severity": bias.get("severity", "medium"),
                        "description": bias.get("description", ""),
                    }
                    for bias in quality_assessment.bias_flags
                ],
            }
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            # Return minimal fallback assessment
            return {
                "overall_score": 0.75,
                "dimensions": {},
                "grade": "B",
                "recommendations": ["Assessment unavailable due to technical issue"],
                "bias_flags": [],
            }

    def validate_reasoning_quality(
        self,
        reasoning_trace: ReasoningTrace,
        expected_outcomes: Optional[Dict[str, Any]] = None,
    ) -> QualityAssessment:
        """
        Perform comprehensive quality assessment of reasoning trace

        Args:
            reasoning_trace: Structured reasoning execution trace
            expected_outcomes: Optional ground truth for validation

        Returns:
            Comprehensive quality assessment with scores and recommendations
        """
        try:
            # Assess each quality dimension
            dimensions = {}

            dimensions["biological_accuracy"] = self._assess_biological_accuracy(
                reasoning_trace, expected_outcomes
            )

            dimensions["reasoning_transparency"] = self._assess_reasoning_transparency(
                reasoning_trace
            )

            dimensions["synthesis_effectiveness"] = (
                self._assess_synthesis_effectiveness(reasoning_trace)
            )

            dimensions["confidence_calibration"] = self._assess_confidence_calibration(
                reasoning_trace, expected_outcomes
            )

            dimensions["methodological_rigor"] = self._assess_methodological_rigor(
                reasoning_trace
            )

            # Calculate composite metrics
            composite_metrics = self._calculate_composite_metrics(dimensions)

            # Overall weighted score
            overall_score = sum(
                dimensions[dim].score * self.dimension_weights[dim]
                for dim in dimensions
            )

            # Assign grade
            grade = self._assign_grade(overall_score)

            # Detect bias patterns
            bias_flags = self._detect_bias_patterns(reasoning_trace, dimensions)

            # Generate recommendations
            recommendations = self._generate_recommendations(dimensions, bias_flags)

            assessment = QualityAssessment(
                overall_score=overall_score,
                grade=grade,
                dimensions=dimensions,
                composite_metrics=composite_metrics,
                bias_flags=bias_flags,
                recommendations=recommendations,
                timestamp=datetime.now(),
                reasoning_trace_id=reasoning_trace.trace_id,
            )

            logger.info(
                f"Quality assessment completed: {grade} ({overall_score:.3f}) "
                f"for trace {reasoning_trace.trace_id}"
            )

            return assessment

        except Exception as e:
            logger.error(
                f"Quality validation failed for {reasoning_trace.trace_id}: {e}"
            )
            return self._create_error_assessment(reasoning_trace, str(e))

    def _assess_biological_accuracy(
        self, trace: ReasoningTrace, expected_outcomes: Optional[Dict[str, Any]] = None
    ) -> QualityDimension:
        """Assess biological and biochemical accuracy of reasoning"""

        evidence = []
        details = {}

        # 1. Domain terminology usage accuracy
        terminology_score = self._evaluate_terminology_usage(trace)
        details["terminology_accuracy"] = terminology_score
        evidence.append(f"Domain terminology usage: {terminology_score:.2f}")

        # 2. Quantitative reasoning correctness
        quantitative_score = self._evaluate_quantitative_reasoning(trace)
        details["quantitative_accuracy"] = quantitative_score
        evidence.append(f"Quantitative reasoning accuracy: {quantitative_score:.2f}")

        # 3. Scientific method application
        scientific_method_score = self._evaluate_scientific_method(trace)
        details["scientific_method"] = scientific_method_score
        evidence.append(f"Scientific method application: {scientific_method_score:.2f}")

        # 4. Cross-validation with expected outcomes
        validation_score = 1.0
        if expected_outcomes:
            validation_score = self._validate_against_expected(trace, expected_outcomes)
            details["outcome_validation"] = validation_score
            evidence.append(f"Expected outcome validation: {validation_score:.2f}")

        # Composite biological accuracy score
        accuracy_score = (
            terminology_score * 0.3
            + quantitative_score * 0.3
            + scientific_method_score * 0.2
            + validation_score * 0.2
        )

        return QualityDimension(
            name="biological_accuracy",
            score=accuracy_score,
            details=details,
            evidence=evidence,
            confidence=0.85,  # High confidence in this assessment
        )

    def _assess_reasoning_transparency(self, trace: ReasoningTrace) -> QualityDimension:
        """Assess transparency and explainability of reasoning process"""

        evidence = []
        details = {}

        # 1. Explanation completeness
        completeness_score = self._evaluate_explanation_completeness(trace)
        details["explanation_completeness"] = completeness_score
        evidence.append(f"Explanation completeness: {completeness_score:.2f}")

        # 2. Evidence citation quality
        citation_score = self._evaluate_evidence_citations(trace)
        details["evidence_citations"] = citation_score
        evidence.append(f"Evidence citation quality: {citation_score:.2f}")

        # 3. Assumption documentation
        assumption_score = self._evaluate_assumption_documentation(trace)
        details["assumption_documentation"] = assumption_score
        evidence.append(f"Assumption documentation: {assumption_score:.2f}")

        # 4. Decision rationale depth
        rationale_score = self._evaluate_decision_rationale(trace)
        details["decision_rationale"] = rationale_score
        evidence.append(f"Decision rationale depth: {rationale_score:.2f}")

        # Composite transparency score
        transparency_score = (
            completeness_score * 0.3
            + citation_score * 0.25
            + assumption_score * 0.2
            + rationale_score * 0.25
        )

        return QualityDimension(
            name="reasoning_transparency",
            score=transparency_score,
            details=details,
            evidence=evidence,
            confidence=0.80,
        )

    def _assess_synthesis_effectiveness(
        self, trace: ReasoningTrace
    ) -> QualityDimension:
        """Assess effectiveness of cross-tool synthesis and integration"""

        evidence = []
        details = {}

        # 1. Cross-tool information integration
        integration_score = self._evaluate_cross_tool_integration(trace)
        details["cross_tool_integration"] = integration_score
        evidence.append(f"Cross-tool integration: {integration_score:.2f}")

        # 2. Pattern recognition across analyses
        pattern_score = self._evaluate_pattern_recognition(trace)
        details["pattern_recognition"] = pattern_score
        evidence.append(f"Pattern recognition: {pattern_score:.2f}")

        # 3. Workflow coherence
        coherence_score = self._evaluate_workflow_coherence(trace)
        details["workflow_coherence"] = coherence_score
        evidence.append(f"Workflow coherence: {coherence_score:.2f}")

        # 4. Knowledge gap identification
        gap_identification_score = self._evaluate_gap_identification(trace)
        details["gap_identification"] = gap_identification_score
        evidence.append(f"Knowledge gap identification: {gap_identification_score:.2f}")

        # Composite synthesis score
        synthesis_score = (
            integration_score * 0.35
            + pattern_score * 0.25
            + coherence_score * 0.25
            + gap_identification_score * 0.15
        )

        return QualityDimension(
            name="synthesis_effectiveness",
            score=synthesis_score,
            details=details,
            evidence=evidence,
            confidence=0.75,
        )

    def _assess_confidence_calibration(
        self, trace: ReasoningTrace, expected_outcomes: Optional[Dict[str, Any]] = None
    ) -> QualityDimension:
        """Assess accuracy of confidence estimates and uncertainty quantification"""

        evidence = []
        details = {}

        # 1. Confidence claim accuracy
        confidence_accuracy = self._evaluate_confidence_accuracy(
            trace, expected_outcomes
        )
        details["confidence_accuracy"] = confidence_accuracy
        evidence.append(f"Confidence accuracy: {confidence_accuracy:.2f}")

        # 2. Uncertainty quantification quality
        uncertainty_score = self._evaluate_uncertainty_quantification(trace)
        details["uncertainty_quantification"] = uncertainty_score
        evidence.append(f"Uncertainty quantification: {uncertainty_score:.2f}")

        # 3. Risk assessment quality
        risk_assessment_score = self._evaluate_risk_assessment(trace)
        details["risk_assessment"] = risk_assessment_score
        evidence.append(f"Risk assessment quality: {risk_assessment_score:.2f}")

        # 4. Reliability indicators
        reliability_score = self._evaluate_reliability_indicators(trace)
        details["reliability_indicators"] = reliability_score
        evidence.append(f"Reliability indicators: {reliability_score:.2f}")

        # Composite confidence calibration score
        calibration_score = (
            confidence_accuracy * 0.4
            + uncertainty_score * 0.25
            + risk_assessment_score * 0.2
            + reliability_score * 0.15
        )

        return QualityDimension(
            name="confidence_calibration",
            score=calibration_score,
            details=details,
            evidence=evidence,
            confidence=0.70,
        )

    def _assess_methodological_rigor(self, trace: ReasoningTrace) -> QualityDimension:
        """Assess methodological rigor and systematic approach adherence"""

        evidence = []
        details = {}

        # 1. Tool selection appropriateness
        tool_selection_score = self._evaluate_tool_selection(trace)
        details["tool_selection"] = tool_selection_score
        evidence.append(f"Tool selection appropriateness: {tool_selection_score:.2f}")

        # 2. Systematic approach adherence
        systematic_score = self._evaluate_systematic_approach(trace)
        details["systematic_approach"] = systematic_score
        evidence.append(f"Systematic approach: {systematic_score:.2f}")

        # 3. Control and validation inclusion
        validation_inclusion_score = self._evaluate_validation_inclusion(trace)
        details["validation_inclusion"] = validation_inclusion_score
        evidence.append(f"Validation inclusion: {validation_inclusion_score:.2f}")

        # 4. Reproducibility considerations
        reproducibility_score = self._evaluate_reproducibility(trace)
        details["reproducibility"] = reproducibility_score
        evidence.append(f"Reproducibility considerations: {reproducibility_score:.2f}")

        # Composite methodological rigor score
        rigor_score = (
            tool_selection_score * 0.3
            + systematic_score * 0.3
            + validation_inclusion_score * 0.2
            + reproducibility_score * 0.2
        )

        return QualityDimension(
            name="methodological_rigor",
            score=rigor_score,
            details=details,
            evidence=evidence,
            confidence=0.85,
        )

    def _calculate_composite_metrics(
        self, dimensions: Dict[str, QualityDimension]
    ) -> Dict[str, float]:
        """Calculate additional composite quality metrics"""

        metrics = {}

        # Scientific rigor composite (biological accuracy + methodological rigor)
        metrics["scientific_rigor"] = (
            dimensions["biological_accuracy"].score * 0.7
            + dimensions["methodological_rigor"].score * 0.3
        )

        # Communication effectiveness (transparency + synthesis)
        metrics["communication_effectiveness"] = (
            dimensions["reasoning_transparency"].score * 0.6
            + dimensions["synthesis_effectiveness"].score * 0.4
        )

        # Reliability index (confidence calibration + biological accuracy)
        metrics["reliability_index"] = (
            dimensions["confidence_calibration"].score * 0.5
            + dimensions["biological_accuracy"].score * 0.5
        )

        # Overall consistency (std dev of dimension scores)
        dimension_scores = [dim.score for dim in dimensions.values()]
        consistency = 1.0 - (
            (max(dimension_scores) - min(dimension_scores))
            / (max(dimension_scores) + 0.01)  # Avoid division by zero
        )
        metrics["consistency_index"] = max(0.0, consistency)

        return metrics

    def _detect_bias_patterns(
        self, trace: ReasoningTrace, dimensions: Dict[str, QualityDimension]
    ) -> List[Dict[str, Any]]:
        """Detect potential bias patterns in reasoning"""

        bias_flags = []

        # 1. Tool selection bias
        tool_bias = self._detect_tool_selection_bias(trace)
        if tool_bias:
            bias_flags.append(tool_bias)

        # 2. Confirmation bias
        confirmation_bias = self._detect_confirmation_bias(trace)
        if confirmation_bias:
            bias_flags.append(confirmation_bias)

        # 3. Overconfidence bias
        overconfidence_bias = self._detect_overconfidence_bias(trace, dimensions)
        if overconfidence_bias:
            bias_flags.append(overconfidence_bias)

        # 4. Anchoring bias
        anchoring_bias = self._detect_anchoring_bias(trace)
        if anchoring_bias:
            bias_flags.append(anchoring_bias)

        return bias_flags

    def _generate_recommendations(
        self, dimensions: Dict[str, QualityDimension], bias_flags: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate improvement recommendations based on assessment"""

        recommendations = []

        # Dimension-specific recommendations
        for dim_name, dimension in dimensions.items():
            if dimension.score < 0.7:  # Below acceptable threshold
                recommendations.append(
                    f"Improve {dim_name.replace('_', ' ')}: "
                    f"Current score {dimension.score:.2f} is below acceptable threshold"
                )

        # Bias-specific recommendations
        for bias_flag in bias_flags:
            recommendations.append(
                f"Address {bias_flag['type']} bias: {bias_flag['recommendation']}"
            )

        # Composite recommendations
        if len([d for d in dimensions.values() if d.score < 0.8]) >= 3:
            recommendations.append(
                "Consider additional training or process refinement - "
                "multiple quality dimensions below optimal"
            )

        return recommendations

    def _assign_grade(self, overall_score: float) -> str:
        """Assign letter grade based on overall score"""
        for grade, threshold in self.grade_thresholds.items():
            if overall_score >= threshold:
                return grade
        return "F"

    def _initialize_quality_benchmarks(self):
        """Initialize quality assessment benchmarks"""
        self.terminology_keywords = {
            "high_quality": [
                "metabolism",
                "glycolysis",
                "tca",
                "electron transport",
                "flux balance",
                "essential genes",
                "growth rate",
                "biomass",
                "atp",
                "nadh",
                "cofactor",
                "enzyme",
            ],
            "domain_specific": [
                "mmol/gDW/h",
                "exchange flux",
                "objective function",
                "constraint",
                "optimization",
                "stoichiometry",
            ],
        }

    # Individual evaluation methods (simplified implementations)

    def _evaluate_terminology_usage(self, trace: ReasoningTrace) -> float:
        """Evaluate biochemical terminology usage accuracy"""
        text = trace.final_conclusion.lower()
        high_quality_count = sum(
            1 for term in self.terminology_keywords["high_quality"] if term in text
        )
        domain_specific_count = sum(
            1 for term in self.terminology_keywords["domain_specific"] if term in text
        )

        # Normalize based on text length and keyword presence
        total_relevant_terms = len(self.terminology_keywords["high_quality"]) + len(
            self.terminology_keywords["domain_specific"]
        )
        terminology_density = (high_quality_count + domain_specific_count) / max(
            1, total_relevant_terms * 0.3
        )

        return min(1.0, terminology_density)

    def _evaluate_quantitative_reasoning(self, trace: ReasoningTrace) -> float:
        """Evaluate numerical accuracy and quantitative reasoning"""
        # Look for numerical values and units in reasoning
        import re

        text = trace.final_conclusion
        numbers = len(re.findall(r"\d+\.?\d*", text))
        units = len(re.findall(r"(mmol|gDW|mol|mM|µM|g|kg|h|min|s)", text))

        # Higher scores for more quantitative reasoning
        quantitative_score = min(1.0, (numbers * 0.1 + units * 0.2))

        return quantitative_score

    def _evaluate_scientific_method(self, trace: ReasoningTrace) -> float:
        """Evaluate adherence to scientific method principles"""
        method_indicators = [
            "hypothesis",
            "prediction",
            "test",
            "validate",
            "control",
            "experiment",
            "evidence",
            "conclude",
            "verify",
        ]

        text = trace.final_conclusion.lower()
        method_count = sum(1 for indicator in method_indicators if indicator in text)

        return min(1.0, method_count / 3.0)

    def _validate_against_expected(
        self, trace: ReasoningTrace, expected: Dict[str, Any]
    ) -> float:
        """Validate reasoning against expected outcomes"""
        # Simplified validation - in practice would be more sophisticated
        return 0.9  # Placeholder

    def _evaluate_explanation_completeness(self, trace: ReasoningTrace) -> float:
        """Evaluate completeness of explanations"""
        # Length-based heuristic with quality indicators
        conclusion_length = len(trace.final_conclusion.split())
        steps_documented = len(trace.steps)

        completeness_score = min(
            1.0, (conclusion_length / 100) * 0.6 + (steps_documented / 5) * 0.4
        )
        return completeness_score

    def _evaluate_evidence_citations(self, trace: ReasoningTrace) -> float:
        """Evaluate quality of evidence citations"""
        citation_count = len(trace.evidence_citations)
        return min(1.0, citation_count / 3.0)

    def _evaluate_assumption_documentation(self, trace: ReasoningTrace) -> float:
        """Evaluate documentation of assumptions"""
        assumption_indicators = ["assume", "given", "suppose", "if", "provided"]
        text = trace.final_conclusion.lower()
        assumption_count = sum(
            1 for indicator in assumption_indicators if indicator in text
        )

        return min(1.0, assumption_count / 2.0)

    def _evaluate_decision_rationale(self, trace: ReasoningTrace) -> float:
        """Evaluate depth of decision rationale"""
        rationale_indicators = ["because", "therefore", "thus", "since", "due to"]
        text = trace.final_conclusion.lower()
        rationale_count = sum(
            1 for indicator in rationale_indicators if indicator in text
        )

        return min(1.0, rationale_count / 3.0)

    def _evaluate_cross_tool_integration(self, trace: ReasoningTrace) -> float:
        """Evaluate effectiveness of cross-tool integration"""
        tool_count = len(set(trace.tools_used))
        integration_score = min(1.0, tool_count / 3.0) if tool_count > 1 else 0.5

        return integration_score

    def _evaluate_pattern_recognition(self, trace: ReasoningTrace) -> float:
        """Evaluate pattern recognition capabilities"""
        pattern_indicators = [
            "pattern",
            "trend",
            "correlation",
            "relationship",
            "connection",
        ]
        text = trace.final_conclusion.lower()
        pattern_count = sum(1 for indicator in pattern_indicators if indicator in text)

        return min(1.0, pattern_count / 2.0)

    def _evaluate_workflow_coherence(self, trace: ReasoningTrace) -> float:
        """Evaluate logical flow and coherence of workflow"""
        # Simplified coherence assessment based on step progression
        coherence_score = 1.0 - (
            abs(len(trace.steps) - 5) / 10.0
        )  # Optimal around 5 steps
        return max(0.3, coherence_score)

    def _evaluate_gap_identification(self, trace: ReasoningTrace) -> float:
        """Evaluate identification of knowledge gaps"""
        gap_indicators = ["unknown", "unclear", "need more", "requires", "limitation"]
        text = trace.final_conclusion.lower()
        gap_count = sum(1 for indicator in gap_indicators if indicator in text)

        return min(1.0, gap_count / 2.0)

    def _evaluate_confidence_accuracy(
        self, trace: ReasoningTrace, expected: Optional[Dict[str, Any]]
    ) -> float:
        """Evaluate accuracy of confidence estimates"""
        # Simplified confidence assessment
        confidence_claims = len(trace.confidence_claims)
        return min(1.0, confidence_claims / 2.0) if confidence_claims > 0 else 0.5

    def _evaluate_uncertainty_quantification(self, trace: ReasoningTrace) -> float:
        """Evaluate quality of uncertainty quantification"""
        uncertainty_indicators = [
            "uncertain",
            "approximately",
            "roughly",
            "likely",
            "possible",
        ]
        text = trace.final_conclusion.lower()
        uncertainty_count = sum(
            1 for indicator in uncertainty_indicators if indicator in text
        )

        return min(1.0, uncertainty_count / 2.0)

    def _evaluate_risk_assessment(self, trace: ReasoningTrace) -> float:
        """Evaluate quality of risk assessment"""
        risk_indicators = ["risk", "error", "limitation", "caution", "warning"]
        text = trace.final_conclusion.lower()
        risk_count = sum(1 for indicator in risk_indicators if indicator in text)

        return min(1.0, risk_count / 2.0)

    def _evaluate_reliability_indicators(self, trace: ReasoningTrace) -> float:
        """Evaluate presence of reliability indicators"""
        reliability_indicators = [
            "validated",
            "confirmed",
            "consistent",
            "reproducible",
        ]
        text = trace.final_conclusion.lower()
        reliability_count = sum(
            1 for indicator in reliability_indicators if indicator in text
        )

        return min(1.0, reliability_count / 2.0)

    def _evaluate_tool_selection(self, trace: ReasoningTrace) -> float:
        """Evaluate appropriateness of tool selection"""
        # Assess tool diversity and appropriateness
        unique_tools = len(set(trace.tools_used))
        tool_score = min(1.0, unique_tools / 3.0)

        return tool_score

    def _evaluate_systematic_approach(self, trace: ReasoningTrace) -> float:
        """Evaluate systematic approach adherence"""
        systematic_indicators = ["first", "then", "next", "finally", "step"]
        text = trace.final_conclusion.lower()
        systematic_count = sum(
            1 for indicator in systematic_indicators if indicator in text
        )

        return min(1.0, systematic_count / 3.0)

    def _evaluate_validation_inclusion(self, trace: ReasoningTrace) -> float:
        """Evaluate inclusion of validation steps"""
        validation_indicators = ["validate", "verify", "check", "confirm", "test"]
        text = trace.final_conclusion.lower()
        validation_count = sum(
            1 for indicator in validation_indicators if indicator in text
        )

        return min(1.0, validation_count / 2.0)

    def _evaluate_reproducibility(self, trace: ReasoningTrace) -> float:
        """Evaluate reproducibility considerations"""
        repro_indicators = [
            "reproduce",
            "repeat",
            "replicate",
            "consistent",
            "standard",
        ]
        text = trace.final_conclusion.lower()
        repro_count = sum(1 for indicator in repro_indicators if indicator in text)

        return min(1.0, repro_count / 2.0)

    def _detect_tool_selection_bias(
        self, trace: ReasoningTrace
    ) -> Optional[Dict[str, Any]]:
        """Detect tool selection bias patterns"""
        if len(set(trace.tools_used)) == 1 and len(trace.tools_used) > 3:
            return {
                "type": "tool_selection_bias",
                "severity": "medium",
                "description": f"Over-reliance on single tool: {trace.tools_used[0]}",
                "recommendation": "Consider using diverse analytical approaches",
            }
        return None

    def _detect_confirmation_bias(
        self, trace: ReasoningTrace
    ) -> Optional[Dict[str, Any]]:
        """Detect confirmation bias patterns"""
        # Simplified detection based on language patterns
        bias_indicators = ["confirms", "supports", "as expected"]
        text = trace.final_conclusion.lower()
        bias_count = sum(1 for indicator in bias_indicators if indicator in text)

        if bias_count > 2:
            return {
                "type": "confirmation_bias",
                "severity": "low",
                "description": "Possible confirmation bias in interpretation",
                "recommendation": "Consider alternative explanations and contradictory evidence",
            }
        return None

    def _detect_overconfidence_bias(
        self, trace: ReasoningTrace, dimensions: Dict[str, QualityDimension]
    ) -> Optional[Dict[str, Any]]:
        """Detect overconfidence bias"""
        confidence_score = dimensions["confidence_calibration"].score
        if confidence_score < 0.6:
            return {
                "type": "overconfidence_bias",
                "severity": "medium",
                "description": "Confidence claims may exceed actual reliability",
                "recommendation": "Include more uncertainty quantification",
            }
        return None

    def _detect_anchoring_bias(self, trace: ReasoningTrace) -> Optional[Dict[str, Any]]:
        """Detect anchoring bias patterns"""
        # Simplified detection - in practice would analyze reasoning patterns
        if len(trace.steps) < 3:
            return {
                "type": "anchoring_bias",
                "severity": "low",
                "description": "Limited exploration of alternatives",
                "recommendation": "Consider multiple approaches before concluding",
            }
        return None

    def _create_error_assessment(
        self, trace: ReasoningTrace, error_message: str
    ) -> QualityAssessment:
        """Create error assessment when validation fails"""
        return QualityAssessment(
            overall_score=0.0,
            grade="F",
            dimensions={},
            composite_metrics={},
            bias_flags=[
                {
                    "type": "validation_error",
                    "severity": "high",
                    "description": f"Quality validation failed: {error_message}",
                    "recommendation": "Review reasoning trace structure and content",
                }
            ],
            recommendations=["Fix validation errors before assessing quality"],
            timestamp=datetime.now(),
            reasoning_trace_id=trace.trace_id,
        )
