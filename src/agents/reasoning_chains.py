#!/usr/bin/env python3
"""
Advanced Multi-Step Reasoning Chains - Phase 8.1

Implements sophisticated AI reasoning capabilities for complex metabolic modeling
analysis workflows. Enables the AI to plan, execute, and adapt multi-step
analysis sequences based on discovered results.

Key Features:
- Multi-step reasoning chain planning (5-10 steps)
- Dynamic adaptation based on intermediate results
- Hypothesis generation and testing workflows
- Memory of entire reasoning chain for context
- Collaborative decision points with user interaction
"""

import json
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class ReasoningStepType(Enum):
    """Types of reasoning steps in analysis chains"""

    ANALYSIS = "analysis"  # Direct tool execution
    HYPOTHESIS = "hypothesis"  # Hypothesis generation
    TESTING = "testing"  # Hypothesis testing
    EVALUATION = "evaluation"  # Result evaluation
    DECISION = "decision"  # Strategic decision point
    COLLABORATION = "collaboration"  # User input request
    SYNTHESIS = "synthesis"  # Final conclusion


class ReasoningStep(BaseModel):
    """Individual step in a reasoning chain"""

    step_id: str = Field(description="Unique identifier for this step")
    step_number: int = Field(description="Sequential number in the chain")
    step_type: ReasoningStepType = Field(description="Type of reasoning step")
    timestamp: str = Field(description="Step execution timestamp")

    # Step content
    reasoning: str = Field(description="AI's reasoning for this step")
    tool_selected: Optional[str] = Field(default=None, description="Tool to execute")
    tool_input: Optional[Dict[str, Any]] = Field(
        default=None, description="Tool input parameters"
    )
    hypothesis: Optional[str] = Field(
        default=None, description="Hypothesis being tested"
    )

    # Decision making
    confidence: float = Field(description="AI confidence in this step (0-1)")
    alternatives_considered: List[str] = Field(
        default_factory=list, description="Other options considered"
    )
    selection_rationale: str = Field(description="Why this step was chosen")

    # Results and outcomes
    tool_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Tool execution result"
    )
    insights_gained: List[str] = Field(
        default_factory=list, description="Key insights from this step"
    )
    questions_raised: List[str] = Field(
        default_factory=list, description="New questions from results"
    )

    # Chain management
    leads_to_steps: List[int] = Field(
        default_factory=list, description="Steps this enables"
    )
    requires_user_input: bool = Field(
        default=False, description="Needs user collaboration"
    )
    user_guidance: Optional[str] = Field(
        default=None, description="User input received"
    )


class ReasoningChain(BaseModel):
    """Complete multi-step reasoning chain for analysis"""

    chain_id: str = Field(description="Unique identifier for this chain")
    session_id: Optional[str] = Field(description="Session this chain belongs to")
    user_query: str = Field(description="Original user query")
    analysis_goal: str = Field(description="High-level analysis objective")

    # Chain metadata
    timestamp_start: str = Field(description="Chain start time")
    timestamp_end: Optional[str] = Field(
        default=None, description="Chain completion time"
    )
    status: str = Field(default="planning", description="Chain execution status")

    # Reasoning steps
    planned_steps: List[ReasoningStep] = Field(
        default_factory=list, description="Initially planned steps"
    )
    executed_steps: List[ReasoningStep] = Field(
        default_factory=list, description="Completed steps"
    )
    current_step: int = Field(default=0, description="Current step being executed")

    # Chain dynamics
    adaptations_made: List[str] = Field(
        default_factory=list, description="How chain was adapted"
    )
    hypotheses_tested: List[str] = Field(
        default_factory=list, description="Hypotheses explored"
    )
    key_discoveries: List[str] = Field(
        default_factory=list, description="Major findings"
    )

    # Final results
    final_conclusion: Optional[str] = Field(
        default=None, description="Chain conclusion"
    )
    confidence_score: float = Field(
        default=0.0, description="Overall confidence in results"
    )
    success: bool = Field(
        default=False, description="Whether chain completed successfully"
    )


class ReasoningChainPlanner:
    """AI-powered planning system for multi-step reasoning chains"""

    def __init__(self, llm, tools_registry):
        """Initialize the reasoning chain planner"""
        self.llm = llm
        self.tools_registry = tools_registry
        self.chain_templates = self._load_chain_templates()

    def plan_reasoning_chain(
        self, user_query: str, analysis_context: Dict[str, Any]
    ) -> ReasoningChain:
        """Plan a multi-step reasoning chain for the given query"""

        # Generate chain ID and metadata
        chain_id = str(uuid.uuid4())[:8]

        # Analyze query to determine analysis goal
        analysis_goal = self._determine_analysis_goal(user_query, analysis_context)

        # Generate initial reasoning plan
        planned_steps = self._generate_step_plan(
            user_query, analysis_goal, analysis_context
        )

        # Create reasoning chain
        chain = ReasoningChain(
            chain_id=chain_id,
            user_query=user_query,
            analysis_goal=analysis_goal,
            timestamp_start=datetime.now().isoformat(),
            planned_steps=planned_steps,
        )

        return chain

    def _determine_analysis_goal(self, user_query: str, context: Dict[str, Any]) -> str:
        """Use AI to determine the high-level analysis objective"""

        prompt = f"""
        Analyze this user query and determine the high-level analysis objective:

        User Query: "{user_query}"
        Context: {json.dumps(context, indent=2)}

        Provide a clear, specific analysis goal that captures what the user wants to achieve.
        Examples:
        - "Comprehensive characterization of metabolic capabilities"
        - "Investigation of growth limitations and bottlenecks"
        - "Optimization of metabolic efficiency and resource utilization"

        Analysis Goal:"""

        response = self.llm._generate_response(prompt)
        return response.text.strip()

    def _generate_step_plan(
        self, query: str, goal: str, context: Dict[str, Any]
    ) -> List[ReasoningStep]:
        """Generate initial multi-step plan using AI reasoning"""

        # Get available tools
        available_tools = list(self.tools_registry.keys())

        prompt = f"""
        Create a detailed multi-step reasoning plan for this metabolic modeling analysis:

        User Query: "{query}"
        Analysis Goal: "{goal}"
        Available Tools: {', '.join(available_tools)}

        Plan 5-8 steps that will systematically achieve the analysis goal. For each step:
        1. Specify the reasoning/purpose
        2. Select the most appropriate tool
        3. Explain why this step is necessary
        4. Describe what insights you expect to gain
        5. Note how this step enables subsequent steps

        Format as a JSON array with this structure for each step:
        {{
            "step_number": 1,
            "reasoning": "Why this step is needed",
            "tool_selected": "tool_name",
            "selection_rationale": "Why this tool was chosen",
            "expected_insights": ["insight1", "insight2"],
            "enables_steps": [2, 3]
        }}

        Focus on creating a logical flow where each step builds on previous results.
        """

        response = self.llm._generate_response(prompt)

        try:
            # Parse AI-generated plan
            plan_data = json.loads(response.text)

            # Convert to ReasoningStep objects
            planned_steps = []
            for step_data in plan_data:
                step = ReasoningStep(
                    step_id=f"step_{step_data['step_number']}",
                    step_number=step_data["step_number"],
                    step_type=ReasoningStepType.ANALYSIS,
                    timestamp=datetime.now().isoformat(),
                    reasoning=step_data["reasoning"],
                    tool_selected=step_data["tool_selected"],
                    selection_rationale=step_data["selection_rationale"],
                    confidence=0.8,  # Initial confidence
                    insights_gained=step_data.get("expected_insights", []),
                    leads_to_steps=step_data.get("enables_steps", []),
                )
                planned_steps.append(step)

            return planned_steps

        except (json.JSONDecodeError, KeyError):
            # Fallback to basic plan if AI response parsing fails
            return self._create_fallback_plan(query, goal)

    def _create_fallback_plan(self, query: str, goal: str) -> List[ReasoningStep]:
        """Create a basic fallback plan if AI planning fails"""

        # Basic metabolic analysis sequence
        basic_steps = [
            {
                "reasoning": "Establish baseline metabolic capabilities",
                "tool": "run_metabolic_fba",
                "rationale": "FBA provides fundamental growth and flux information",
            },
            {
                "reasoning": "Analyze structural properties of the model",
                "tool": "analyze_metabolic_model",
                "rationale": "Model structure informs interpretation of results",
            },
            {
                "reasoning": "Identify essential components for growth",
                "tool": "analyze_essentiality",
                "rationale": "Essential analysis reveals critical metabolic dependencies",
            },
            {
                "reasoning": "Determine minimal nutritional requirements",
                "tool": "find_minimal_media",
                "rationale": "Media requirements indicate metabolic autonomy",
            },
            {
                "reasoning": "Synthesize findings into comprehensive assessment",
                "tool": None,
                "rationale": "Integration step to combine all insights",
            },
        ]

        planned_steps = []
        for i, step_data in enumerate(basic_steps, 1):
            step = ReasoningStep(
                step_id=f"fallback_step_{i}",
                step_number=i,
                step_type=(
                    ReasoningStepType.ANALYSIS
                    if step_data["tool"]
                    else ReasoningStepType.SYNTHESIS
                ),
                timestamp=datetime.now().isoformat(),
                reasoning=step_data["reasoning"],
                tool_selected=step_data["tool"],
                selection_rationale=step_data["rationale"],
                confidence=0.7,
                leads_to_steps=[i + 1] if i < len(basic_steps) else [],
            )
            planned_steps.append(step)

        return planned_steps

    def _load_chain_templates(self) -> Dict[str, Any]:
        """Load predefined reasoning chain templates"""

        templates = {
            "comprehensive_analysis": {
                "description": "Complete metabolic model characterization",
                "steps": [
                    "baseline_analysis",
                    "structural_analysis",
                    "constraint_analysis",
                    "synthesis",
                ],
            },
            "growth_investigation": {
                "description": "Investigation of growth limitations",
                "steps": [
                    "growth_check",
                    "constraint_identification",
                    "gap_analysis",
                    "validation",
                ],
            },
            "optimization_workflow": {
                "description": "Metabolic efficiency optimization",
                "steps": [
                    "baseline_fba",
                    "variability_analysis",
                    "bottleneck_identification",
                    "optimization",
                ],
            },
        }

        return templates


class ReasoningChainExecutor:
    """Executes and adapts reasoning chains dynamically"""

    def __init__(self, llm, tool_orchestrator, audit_logger):
        """Initialize the chain executor"""
        self.llm = llm
        self.tool_orchestrator = tool_orchestrator
        self.audit_logger = audit_logger
        self.active_chains = {}

    async def execute_reasoning_chain(self, chain: ReasoningChain) -> ReasoningChain:
        """Execute a reasoning chain with dynamic adaptation"""

        self.active_chains[chain.chain_id] = chain
        chain.status = "executing"

        try:
            for step_num in range(len(chain.planned_steps)):
                current_step = chain.planned_steps[step_num]
                chain.current_step = step_num

                # Execute current step
                executed_step = await self._execute_step(current_step, chain)
                chain.executed_steps.append(executed_step)

                # Analyze results and adapt plan if needed
                if step_num < len(chain.planned_steps) - 1:
                    chain = await self._adapt_plan_based_on_results(
                        executed_step, chain
                    )

                # Log execution
                self.audit_logger.log_reasoning_step(
                    executed_step.reasoning,
                    executed_step.tool_result or {},
                    executed_step.confidence,
                )

            # Generate final synthesis
            chain = await self._synthesize_final_results(chain)
            chain.status = "completed"
            chain.success = True
            chain.timestamp_end = datetime.now().isoformat()

        except Exception as e:
            chain.status = f"failed: {str(e)}"
            chain.success = False
            chain.timestamp_end = datetime.now().isoformat()

        return chain

    async def _execute_step(
        self, step: ReasoningStep, chain: ReasoningChain
    ) -> ReasoningStep:
        """Execute an individual reasoning step"""

        step.timestamp = datetime.now().isoformat()

        if step.tool_selected and step.step_type == ReasoningStepType.ANALYSIS:
            # Execute tool
            try:
                tool_result = await self.tool_orchestrator.execute_tool(
                    step.tool_selected, step.tool_input or {}
                )
                step.tool_result = (
                    tool_result.data if hasattr(tool_result, "data") else tool_result
                )

                # Extract insights from results
                step.insights_gained = await self._extract_insights_from_result(
                    step.tool_result, step.reasoning, chain.analysis_goal
                )

                # Identify new questions raised
                step.questions_raised = await self._identify_new_questions(
                    step.tool_result, chain.user_query
                )

            except Exception as e:
                step.tool_result = {"error": str(e)}
                step.insights_gained = [f"Tool execution failed: {str(e)}"]

        return step

    async def _extract_insights_from_result(
        self, result: Dict[str, Any], reasoning: str, goal: str
    ) -> List[str]:
        """Use AI to extract key insights from tool results"""

        prompt = f"""
        Analyze this tool result and extract the key insights relevant to our analysis goal:

        Analysis Goal: {goal}
        Step Reasoning: {reasoning}
        Tool Result: {json.dumps(result, indent=2)}

        Extract 2-4 key insights that are most relevant to the analysis goal.
        Focus on actionable findings that inform next steps.

        Format as a JSON array of strings:
        ["insight 1", "insight 2", "insight 3"]
        """

        response = self.llm._generate_response(prompt)

        try:
            insights = json.loads(response.text)
            return insights if isinstance(insights, list) else [response.text]
        except json.JSONDecodeError:
            return [response.text]

    async def _identify_new_questions(
        self, result: Dict[str, Any], original_query: str
    ) -> List[str]:
        """Identify new questions raised by tool results"""

        prompt = f"""
        Based on this tool result, what new questions emerge that could guide further analysis?

        Original Query: {original_query}
        Tool Result: {json.dumps(result, indent=2)}

        Identify 1-3 specific questions that arise from these results.
        Focus on questions that could lead to deeper understanding.

        Format as a JSON array of strings:
        ["question 1", "question 2"]
        """

        response = self.llm._generate_response(prompt)

        try:
            questions = json.loads(response.text)
            return questions if isinstance(questions, list) else [response.text]
        except json.JSONDecodeError:
            return [response.text]

    async def _adapt_plan_based_on_results(
        self, completed_step: ReasoningStep, chain: ReasoningChain
    ) -> ReasoningChain:
        """Dynamically adapt the reasoning chain based on intermediate results"""

        # Analyze results to determine if plan adaptation is needed
        adaptation_prompt = f"""
        Analyze this completed step and determine if we should adapt our analysis plan:

        Original Goal: {chain.analysis_goal}
        Completed Step: {completed_step.reasoning}
        Tool Used: {completed_step.tool_selected}
        Key Insights: {completed_step.insights_gained}
        Questions Raised: {completed_step.questions_raised}

        Remaining Planned Steps: {[s.reasoning for s in chain.planned_steps[chain.current_step + 1:]]}

        Should we modify our plan based on what we discovered? Consider:
        1. Do the insights suggest a different analytical direction?
        2. Are there urgent questions that should be addressed immediately?
        3. Should we add, remove, or reorder remaining steps?

        Respond with:
        - "CONTINUE" if the current plan is still optimal
        - "ADAPT" followed by specific modifications needed

        Response:"""

        response = self.llm._generate_response(adaptation_prompt)

        if response.text.strip().startswith("ADAPT"):
            adaptation_description = response.text.replace("ADAPT", "").strip()
            chain.adaptations_made.append(
                f"Step {completed_step.step_number}: {adaptation_description}"
            )

            # Generate modified plan (simplified implementation)
            # In a full implementation, this would intelligently modify the remaining steps

        return chain

    async def _synthesize_final_results(self, chain: ReasoningChain) -> ReasoningChain:
        """Generate final synthesis and conclusions"""

        # Gather all insights and results
        all_insights = []
        all_results = {}

        for step in chain.executed_steps:
            all_insights.extend(step.insights_gained)
            if step.tool_result:
                all_results[step.tool_selected or f"step_{step.step_number}"] = (
                    step.tool_result
                )

        synthesis_prompt = f"""
        Synthesize the results of this multi-step analysis into a comprehensive conclusion:

        Original Query: {chain.user_query}
        Analysis Goal: {chain.analysis_goal}

        Key Insights Discovered:
        {json.dumps(all_insights, indent=2)}

        Tool Results Summary:
        {json.dumps({k: str(v)[:200] + "..." if len(str(v)) > 200 else v for k, v in all_results.items()}, indent=2)}

        Provide a comprehensive conclusion that:
        1. Directly answers the original user query
        2. Highlights the most important findings
        3. Explains how the different analysis steps connected
        4. Identifies any remaining questions or limitations

        Conclusion:"""

        response = self.llm._generate_response(synthesis_prompt)

        chain.final_conclusion = response.text.strip()
        chain.key_discoveries = all_insights
        chain.confidence_score = sum(
            step.confidence for step in chain.executed_steps
        ) / len(chain.executed_steps)

        return chain


def create_reasoning_chain_system(llm, tools_registry, tool_orchestrator, audit_logger):
    """Factory function to create a complete reasoning chain system"""

    planner = ReasoningChainPlanner(llm, tools_registry)
    executor = ReasoningChainExecutor(llm, tool_orchestrator, audit_logger)

    return planner, executor
