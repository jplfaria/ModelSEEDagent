#!/usr/bin/env python3
"""
Hypothesis-Driven Analysis System - Phase 8.2

Implements AI-powered hypothesis generation and testing for metabolic modeling.
Enables the AI to form scientific hypotheses about metabolic behavior and
systematically test them using available tools.

Key Features:
- Automatic hypothesis generation from observations
- Systematic hypothesis testing workflows
- Evidence evaluation and hypothesis validation
- Iterative hypothesis refinement
- Scientific reasoning documentation
"""

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class HypothesisType(Enum):
    """Types of metabolic hypotheses that can be generated"""

    GROWTH_LIMITATION = "growth_limitation"  # Growth rate limitations
    NUTRITIONAL_GAP = "nutritional_gap"  # Missing nutrients/auxotrophies
    METABOLIC_EFFICIENCY = "metabolic_efficiency"  # Pathway efficiency issues
    GENE_ESSENTIALITY = "gene_essentiality"  # Essential gene predictions
    PATHWAY_ACTIVITY = "pathway_activity"  # Pathway utilization patterns
    BIOMASS_COMPOSITION = "biomass_composition"  # Biomass synthesis issues
    REGULATORY_CONSTRAINT = "regulatory_constraint"  # Regulatory limitations


class HypothesisStatus(Enum):
    """Status of hypothesis testing"""

    GENERATED = "generated"  # Hypothesis formulated
    TESTING = "testing"  # Currently being tested
    SUPPORTED = "supported"  # Evidence supports hypothesis
    REFUTED = "refuted"  # Evidence contradicts hypothesis
    INCONCLUSIVE = "inconclusive"  # Mixed or insufficient evidence
    REFINED = "refined"  # Hypothesis modified based on evidence


class Evidence(BaseModel):
    """Evidence for or against a hypothesis"""

    evidence_id: str = Field(description="Unique evidence identifier")
    source_tool: str = Field(description="Tool that generated this evidence")
    tool_result: Dict[str, Any] = Field(description="Raw tool result data")
    interpretation: str = Field(description="AI interpretation of the evidence")

    # Evidence evaluation
    supports_hypothesis: bool = Field(
        description="Does this evidence support the hypothesis"
    )
    strength: float = Field(description="Strength of evidence (0-1)")
    confidence: float = Field(description="Confidence in interpretation (0-1)")

    # Context
    timestamp: str = Field(description="When evidence was collected")
    context: str = Field(description="Context in which evidence was collected")


class Hypothesis(BaseModel):
    """Scientific hypothesis about metabolic behavior"""

    hypothesis_id: str = Field(description="Unique hypothesis identifier")
    hypothesis_type: HypothesisType = Field(description="Category of hypothesis")
    statement: str = Field(description="Clear, testable hypothesis statement")

    # Scientific details
    rationale: str = Field(description="Why this hypothesis was generated")
    predictions: List[str] = Field(description="Testable predictions from hypothesis")
    testable_with_tools: List[str] = Field(
        description="Tools that can test this hypothesis"
    )

    # Testing and validation
    status: HypothesisStatus = Field(default=HypothesisStatus.GENERATED)
    evidence_for: List[Evidence] = Field(default_factory=list)
    evidence_against: List[Evidence] = Field(default_factory=list)

    # Evaluation
    confidence_score: float = Field(
        default=0.0, description="Overall confidence in hypothesis"
    )
    last_updated: str = Field(description="Last modification timestamp")

    # Relationships
    generated_from: Optional[str] = Field(
        default=None, description="What observation triggered this"
    )
    leads_to_hypotheses: List[str] = Field(
        default_factory=list, description="Child hypotheses"
    )
    refined_versions: List[str] = Field(
        default_factory=list, description="Refined versions"
    )


class HypothesisGenerator:
    """AI-powered system for generating scientific hypotheses"""

    def __init__(self, llm):
        """Initialize the hypothesis generator"""
        self.llm = llm
        self.hypothesis_templates = self._load_hypothesis_templates()

    def generate_hypotheses_from_observation(
        self, observation: str, context: Dict[str, Any], available_tools: List[str]
    ) -> List[Hypothesis]:
        """Generate hypotheses based on an observation"""

        # Use AI to generate hypotheses
        generation_prompt = f"""
        Based on this metabolic modeling observation, generate 2-4 testable scientific hypotheses:

        Observation: {observation}
        Analysis Context: {json.dumps(context, indent=2)}
        Available Testing Tools: {', '.join(available_tools)}

        For each hypothesis, provide:
        1. A clear, specific, testable statement
        2. The scientific rationale for why this hypothesis is plausible
        3. Specific predictions that can be tested
        4. Which tools could be used to test it
        5. The type of hypothesis (growth_limitation, nutritional_gap, etc.)

        Format as JSON array:
        [
            {{
                "statement": "Specific testable hypothesis",
                "rationale": "Scientific reasoning for this hypothesis",
                "predictions": ["prediction 1", "prediction 2"],
                "testable_with_tools": ["tool1", "tool2"],
                "hypothesis_type": "growth_limitation"
            }}
        ]

        Focus on hypotheses that are:
        - Specific and testable with available tools
        - Scientifically plausible given the observation
        - Likely to provide actionable insights
        """

        response = self.llm._generate_response(generation_prompt)

        try:
            hypotheses_data = json.loads(response.text)
            return self._create_hypothesis_objects(hypotheses_data, observation)

        except json.JSONDecodeError:
            # Fallback to simple hypothesis generation
            return self._generate_fallback_hypotheses(
                observation, context, available_tools
            )

    def generate_hypotheses_from_results(
        self,
        tool_results: Dict[str, Any],
        query_context: str,
        available_tools: List[str],
    ) -> List[Hypothesis]:
        """Generate hypotheses based on tool results"""

        # Analyze results to identify interesting patterns
        analysis_prompt = f"""
        Analyze these metabolic modeling results and identify patterns that suggest testable hypotheses:

        Original Query Context: {query_context}
        Tool Results: {json.dumps(tool_results, indent=2)}

        Look for patterns such as:
        - Unexpected growth rates (very high/low)
        - Unusual nutrient requirements
        - Essential gene patterns
        - Pathway utilization anomalies
        - Biomass composition issues

        For each interesting pattern, formulate a hypothesis that could explain it.
        Focus on hypotheses that would deepen understanding of the metabolic system.

        Generate 1-3 hypotheses as JSON:
        [
            {{
                "pattern_observed": "Description of the pattern that triggered this hypothesis",
                "statement": "Testable hypothesis statement",
                "rationale": "Why this hypothesis explains the pattern",
                "predictions": ["what we expect to find if true"],
                "testable_with_tools": ["tools to test with"],
                "hypothesis_type": "appropriate_category"
            }}
        ]
        """

        response = self.llm._generate_response(analysis_prompt)

        try:
            hypotheses_data = json.loads(response.text)
            return self._create_hypothesis_objects(
                hypotheses_data, f"Results analysis: {query_context}"
            )

        except json.JSONDecodeError:
            return []

    def _create_hypothesis_objects(
        self, hypotheses_data: List[Dict], source: str
    ) -> List[Hypothesis]:
        """Convert AI-generated hypothesis data to Hypothesis objects"""

        hypotheses = []

        for hyp_data in hypotheses_data:
            try:
                hypothesis = Hypothesis(
                    hypothesis_id=str(uuid.uuid4())[:8],
                    hypothesis_type=HypothesisType(
                        hyp_data.get("hypothesis_type", "growth_limitation")
                    ),
                    statement=hyp_data["statement"],
                    rationale=hyp_data["rationale"],
                    predictions=hyp_data.get("predictions", []),
                    testable_with_tools=hyp_data.get("testable_with_tools", []),
                    last_updated=datetime.now().isoformat(),
                    generated_from=source,
                )
                hypotheses.append(hypothesis)

            except (KeyError, ValueError):
                # Skip malformed hypotheses
                continue

        return hypotheses

    def _generate_fallback_hypotheses(
        self, observation: str, context: Dict, tools: List[str]
    ) -> List[Hypothesis]:
        """Generate basic fallback hypotheses if AI generation fails"""

        fallback_hypotheses = []

        # Growth limitation hypothesis
        if any(
            word in observation.lower()
            for word in ["growth", "slow", "rate", "biomass"]
        ):
            hypothesis = Hypothesis(
                hypothesis_id=str(uuid.uuid4())[:8],
                hypothesis_type=HypothesisType.GROWTH_LIMITATION,
                statement="The observed growth pattern indicates specific metabolic limitations",
                rationale="Growth anomalies often indicate underlying metabolic constraints",
                predictions=["Specific nutrients or pathways are limiting growth"],
                testable_with_tools=["find_minimal_media", "analyze_essentiality"],
                last_updated=datetime.now().isoformat(),
                generated_from=observation,
            )
            fallback_hypotheses.append(hypothesis)

        return fallback_hypotheses

    def _load_hypothesis_templates(self) -> Dict[str, Any]:
        """Load predefined hypothesis templates"""

        templates = {
            "growth_limitation": {
                "common_patterns": ["low growth rate", "biomass production issues"],
                "typical_tools": [
                    "run_metabolic_fba",
                    "find_minimal_media",
                    "analyze_essentiality",
                ],
            },
            "nutritional_gap": {
                "common_patterns": ["auxotrophy", "nutrient requirement"],
                "typical_tools": ["identify_auxotrophies", "find_missing_media"],
            },
            "metabolic_efficiency": {
                "common_patterns": ["pathway efficiency", "flux distribution"],
                "typical_tools": ["run_flux_variability_analysis", "run_flux_sampling"],
            },
        }

        return templates


class HypothesisTester:
    """System for systematically testing hypotheses"""

    def __init__(self, llm, tool_orchestrator):
        """Initialize the hypothesis tester"""
        self.llm = llm
        self.tool_orchestrator = tool_orchestrator

    async def test_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        """Execute tests for a specific hypothesis"""

        hypothesis.status = HypothesisStatus.TESTING
        hypothesis.last_updated = datetime.now().isoformat()

        # Plan testing strategy
        testing_plan = await self._plan_hypothesis_testing(hypothesis)

        # Execute tests and collect evidence
        for tool_name in testing_plan:
            try:
                # Determine tool inputs based on hypothesis
                tool_input = await self._determine_tool_input_for_hypothesis(
                    hypothesis, tool_name
                )

                # Execute tool
                tool_result = await self.tool_orchestrator.execute_tool(
                    tool_name, tool_input
                )

                # Interpret results as evidence
                evidence = await self._interpret_result_as_evidence(
                    tool_result, hypothesis, tool_name
                )

                # Add evidence to hypothesis
                if evidence.supports_hypothesis:
                    hypothesis.evidence_for.append(evidence)
                else:
                    hypothesis.evidence_against.append(evidence)

            except Exception as e:
                # Log testing failure but continue with other tests
                error_evidence = Evidence(
                    evidence_id=str(uuid.uuid4())[:8],
                    source_tool=tool_name,
                    tool_result={"error": str(e)},
                    interpretation=f"Tool execution failed: {str(e)}",
                    supports_hypothesis=False,
                    strength=0.0,
                    confidence=0.1,
                    timestamp=datetime.now().isoformat(),
                    context=f"Testing hypothesis: {hypothesis.statement}",
                )
                hypothesis.evidence_against.append(error_evidence)

        # Evaluate hypothesis based on collected evidence
        hypothesis = await self._evaluate_hypothesis(hypothesis)

        return hypothesis

    async def _plan_hypothesis_testing(self, hypothesis: Hypothesis) -> List[str]:
        """Plan which tools to use for testing a hypothesis"""

        planning_prompt = f"""
        Plan how to test this scientific hypothesis using available tools:

        Hypothesis: {hypothesis.statement}
        Rationale: {hypothesis.rationale}
        Predictions: {hypothesis.predictions}
        Available Tools: {hypothesis.testable_with_tools}

        Select 1-3 tools that would provide the most relevant evidence for or against this hypothesis.
        Consider:
        - Which tools directly test the predictions
        - What evidence would be most convincing
        - Logical order of testing

        Return as JSON array of tool names:
        ["tool1", "tool2", "tool3"]
        """

        response = self.llm._generate_response(planning_prompt)

        try:
            tools = json.loads(response.text)
            return (
                tools if isinstance(tools, list) else hypothesis.testable_with_tools[:2]
            )
        except json.JSONDecodeError:
            return hypothesis.testable_with_tools[:2]  # Use first 2 available tools

    async def _determine_tool_input_for_hypothesis(
        self, hypothesis: Hypothesis, tool_name: str
    ) -> Dict[str, Any]:
        """Determine appropriate tool inputs for testing a hypothesis"""

        input_prompt = f"""
        Determine the appropriate input parameters for testing this hypothesis:

        Hypothesis: {hypothesis.statement}
        Tool to Use: {tool_name}

        What parameters should be passed to this tool to best test the hypothesis?
        Consider the specific predictions and what evidence would be most relevant.

        Return as JSON object with parameter names and values:
        {{"parameter1": "value1", "parameter2": "value2"}}

        If no special parameters are needed, return an empty object: {{}}
        """

        response = self.llm._generate_response(input_prompt)

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return {}  # Use default parameters

    async def _interpret_result_as_evidence(
        self, tool_result: Any, hypothesis: Hypothesis, tool_name: str
    ) -> Evidence:
        """Interpret tool results as evidence for or against a hypothesis"""

        result_data = tool_result.data if hasattr(tool_result, "data") else tool_result

        interpretation_prompt = f"""
        Interpret this tool result as evidence for or against the hypothesis:

        Hypothesis: {hypothesis.statement}
        Predictions: {hypothesis.predictions}
        Tool Used: {tool_name}
        Tool Result: {json.dumps(result_data, indent=2)}

        Analyze:
        1. Does this result support or contradict the hypothesis?
        2. How strong is this evidence (0.0 = no relevance, 1.0 = definitive)?
        3. How confident are you in this interpretation (0.0 = very uncertain, 1.0 = very certain)?
        4. What specific aspects of the result are most relevant?

        Format as JSON:
        {{
            "supports_hypothesis": true/false,
            "strength": 0.8,
            "confidence": 0.9,
            "interpretation": "Detailed explanation of how the result relates to the hypothesis"
        }}
        """

        response = self.llm._generate_response(interpretation_prompt)

        try:
            interpretation_data = json.loads(response.text)

            evidence = Evidence(
                evidence_id=str(uuid.uuid4())[:8],
                source_tool=tool_name,
                tool_result=result_data,
                interpretation=interpretation_data["interpretation"],
                supports_hypothesis=interpretation_data["supports_hypothesis"],
                strength=interpretation_data["strength"],
                confidence=interpretation_data["confidence"],
                timestamp=datetime.now().isoformat(),
                context=f"Testing: {hypothesis.statement}",
            )

            return evidence

        except (json.JSONDecodeError, KeyError):
            # Fallback evidence interpretation
            return Evidence(
                evidence_id=str(uuid.uuid4())[:8],
                source_tool=tool_name,
                tool_result=result_data,
                interpretation=response.text,
                supports_hypothesis=True,  # Default assumption
                strength=0.5,
                confidence=0.5,
                timestamp=datetime.now().isoformat(),
                context=f"Testing: {hypothesis.statement}",
            )

    async def _evaluate_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        """Evaluate hypothesis based on all collected evidence"""

        # Calculate evidence scores
        supporting_evidence = len(hypothesis.evidence_for)
        contradicting_evidence = len(hypothesis.evidence_against)

        # Weight evidence by strength and confidence
        support_score = sum(e.strength * e.confidence for e in hypothesis.evidence_for)
        contradiction_score = sum(
            e.strength * e.confidence for e in hypothesis.evidence_against
        )

        # Determine status
        if support_score > contradiction_score and supporting_evidence > 0:
            if support_score > 2.0:  # Strong evidence
                hypothesis.status = HypothesisStatus.SUPPORTED
            else:
                hypothesis.status = HypothesisStatus.INCONCLUSIVE
        elif contradiction_score > support_score and contradicting_evidence > 0:
            hypothesis.status = HypothesisStatus.REFUTED
        else:
            hypothesis.status = HypothesisStatus.INCONCLUSIVE

        # Calculate overall confidence
        total_evidence = supporting_evidence + contradicting_evidence
        if total_evidence > 0:
            hypothesis.confidence_score = abs(support_score - contradiction_score) / (
                support_score + contradiction_score + 1
            )
        else:
            hypothesis.confidence_score = 0.0

        hypothesis.last_updated = datetime.now().isoformat()

        return hypothesis


class HypothesisManager:
    """Manages collections of hypotheses and their testing workflows"""

    def __init__(self, llm, tool_orchestrator):
        """Initialize the hypothesis manager"""
        self.generator = HypothesisGenerator(llm)
        self.tester = HypothesisTester(llm, tool_orchestrator)
        self.active_hypotheses = {}
        self.hypothesis_history = []

    async def process_observation(
        self, observation: str, context: Dict[str, Any], available_tools: List[str]
    ) -> List[Hypothesis]:
        """Process an observation to generate and test hypotheses"""

        # Generate hypotheses from observation
        new_hypotheses = self.generator.generate_hypotheses_from_observation(
            observation, context, available_tools
        )

        # Test each hypothesis
        tested_hypotheses = []
        for hypothesis in new_hypotheses:
            self.active_hypotheses[hypothesis.hypothesis_id] = hypothesis

            # Test the hypothesis
            tested_hypothesis = await self.tester.test_hypothesis(hypothesis)
            tested_hypotheses.append(tested_hypothesis)

            # Update management records
            self.active_hypotheses[hypothesis.hypothesis_id] = tested_hypothesis
            self.hypothesis_history.append(tested_hypothesis)

        return tested_hypotheses

    def get_supported_hypotheses(self) -> List[Hypothesis]:
        """Get all currently supported hypotheses"""
        return [
            h
            for h in self.active_hypotheses.values()
            if h.status == HypothesisStatus.SUPPORTED
        ]

    def get_hypothesis_summary(self) -> Dict[str, Any]:
        """Get summary of all hypothesis testing"""

        status_counts = {}
        for hypothesis in self.active_hypotheses.values():
            status = hypothesis.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_hypotheses": len(self.active_hypotheses),
            "status_breakdown": status_counts,
            "supported_hypotheses": len(self.get_supported_hypotheses()),
            "average_confidence": (
                sum(h.confidence_score for h in self.active_hypotheses.values())
                / len(self.active_hypotheses)
                if self.active_hypotheses
                else 0.0
            ),
        }


def create_hypothesis_system(llm, tool_orchestrator):
    """Factory function to create a complete hypothesis testing system"""

    return HypothesisManager(llm, tool_orchestrator)
