"""
Meta-Reasoning Engine for ModelSEEDagent Phase 4

Implements advanced meta-reasoning and self-assessment capabilities that enable
the system to reason about its own reasoning processes, evaluate cognitive
strategies, and continuously improve analytical approaches through deep
introspective analysis.
"""

import json
import logging
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class ReasoningLevel(Enum):
    """Levels of reasoning abstraction"""

    OBJECT_LEVEL = "object_level"  # Direct problem solving
    META_LEVEL = "meta_level"  # Reasoning about reasoning
    META_META_LEVEL = "meta_meta_level"  # Reasoning about meta-reasoning


class CognitiveStrategy(Enum):
    """Cognitive strategies for problem solving"""

    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"
    SYSTEMATIC = "systematic"
    CREATIVE = "creative"
    CONSERVATIVE = "conservative"
    EXPERIMENTAL = "experimental"


@dataclass
class MetaReasoningStep:
    """Individual step in meta-reasoning process"""

    step_id: str
    reasoning_level: ReasoningLevel
    cognitive_strategy: CognitiveStrategy
    step_type: str

    # Step content
    description: str
    rationale: str
    evidence: List[str]
    assumptions: List[str]

    # Assessment
    confidence: float
    validity_score: float
    novelty_score: float

    # Context
    timestamp: datetime
    parent_step: Optional[str] = None
    child_steps: List[str] = field(default_factory=list)

    # Outcomes
    success_indicators: List[str] = field(default_factory=list)
    failure_indicators: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)


@dataclass
class CognitiveProcess:
    """Representation of a cognitive reasoning process"""

    process_id: str
    process_name: str
    start_time: datetime
    end_time: Optional[datetime] = None

    # Process structure
    reasoning_steps: List[MetaReasoningStep] = field(default_factory=list)
    cognitive_flow: List[str] = field(default_factory=list)
    decision_points: List[Dict[str, Any]] = field(default_factory=list)

    # Process assessment
    overall_effectiveness: Optional[float] = None
    strategic_coherence: Optional[float] = None
    adaptive_capability: Optional[float] = None

    # Learning outcomes
    insights_generated: List[str] = field(default_factory=list)
    strategies_discovered: List[str] = field(default_factory=list)
    improvement_opportunities: List[str] = field(default_factory=list)


@dataclass
class SelfAssessmentResult:
    """Results from self-assessment analysis"""

    assessment_id: str
    assessment_timestamp: datetime
    assessment_scope: str

    # Cognitive performance assessment
    reasoning_effectiveness: Dict[str, float]
    strategy_utilization: Dict[str, float]
    adaptive_learning: Dict[str, float]

    # Meta-cognitive insights
    strength_areas: List[str]
    improvement_areas: List[str]
    cognitive_biases_detected: List[Dict[str, Any]]

    # Strategic recommendations
    strategy_recommendations: List[Dict[str, Any]]
    learning_priorities: List[str]
    adaptive_modifications: List[Dict[str, Any]]

    # Confidence and reliability
    assessment_confidence: float
    reliability_indicators: Dict[str, float]


class MetaReasoningEngine:
    """
    Advanced meta-reasoning engine for self-aware cognitive processing.

    Provides sophisticated meta-reasoning capabilities including strategy
    evaluation, cognitive process analysis, and adaptive learning from
    reasoning experiences.
    """

    def __init__(self, storage_path: str = "/tmp/modelseed_meta_reasoning"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Cognitive process tracking
        self.active_processes: Dict[str, CognitiveProcess] = {}
        self.completed_processes: Dict[str, CognitiveProcess] = {}
        self.meta_reasoning_history: List[MetaReasoningStep] = []

        # Strategy management
        self.cognitive_strategies: Dict[CognitiveStrategy, Dict[str, Any]] = {}
        self.strategy_performance: Dict[CognitiveStrategy, Dict[str, float]] = {}
        self.strategy_evolution: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Self-assessment components
        self.assessment_frameworks = {
            "reasoning_effectiveness": self._assess_reasoning_effectiveness,
            "strategy_coherence": self._assess_strategy_coherence,
            "adaptive_learning": self._assess_adaptive_learning,
            "cognitive_bias_detection": self._detect_cognitive_biases,
            "meta_cognitive_awareness": self._assess_meta_cognitive_awareness,
        }

        # Learning and adaptation
        self.learning_experiences: List[Dict[str, Any]] = []
        self.cognitive_models: Dict[str, Any] = {}
        self.adaptation_history: List[Dict[str, Any]] = []

        # Initialize meta-reasoning state
        self._initialize_meta_reasoning_state()

    def _initialize_meta_reasoning_state(self):
        """Initialize meta-reasoning state and cognitive strategies"""

        # Initialize cognitive strategies with baseline performance
        for strategy in CognitiveStrategy:
            self.cognitive_strategies[strategy] = {
                "description": self._get_strategy_description(strategy),
                "typical_use_cases": self._get_strategy_use_cases(strategy),
                "strengths": self._get_strategy_strengths(strategy),
                "limitations": self._get_strategy_limitations(strategy),
                "performance_baseline": 0.7,
            }

            self.strategy_performance[strategy] = {
                "success_rate": 0.7,
                "efficiency": 0.6,
                "quality": 0.7,
                "adaptability": 0.5,
                "total_usage": 0,
            }

    def initiate_meta_reasoning_process(
        self, context: Dict[str, Any], objective: str
    ) -> str:
        """
        Initiate a new meta-reasoning process for analyzing cognitive approaches.

        Args:
            context: Context information for the reasoning process
            objective: Objective or goal of the meta-reasoning

        Returns:
            Process ID for tracking the meta-reasoning process
        """
        process_id = f"meta_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        process = CognitiveProcess(
            process_id=process_id,
            process_name=f"Meta-reasoning: {objective}",
            start_time=datetime.now(),
        )

        # Initial meta-reasoning step - problem analysis
        initial_step = self._create_meta_reasoning_step(
            step_type="problem_analysis",
            reasoning_level=ReasoningLevel.META_LEVEL,
            cognitive_strategy=CognitiveStrategy.ANALYTICAL,
            description=f"Analyzing cognitive approach for: {objective}",
            rationale="Need to evaluate optimal reasoning strategy for this problem",
            context=context,
        )

        process.reasoning_steps.append(initial_step)
        process.cognitive_flow.append(initial_step.step_id)

        # Store active process
        self.active_processes[process_id] = process

        logger.info(
            f"Initiated meta-reasoning process {process_id} for objective: {objective}"
        )
        return process_id

    def execute_meta_reasoning_step(
        self,
        process_id: str,
        step_type: str,
        reasoning_level: ReasoningLevel,
        cognitive_strategy: CognitiveStrategy,
        step_context: Dict[str, Any],
    ) -> MetaReasoningStep:
        """
        Execute a specific meta-reasoning step within a process.

        Args:
            process_id: ID of the meta-reasoning process
            step_type: Type of reasoning step to execute
            reasoning_level: Level of reasoning abstraction
            cognitive_strategy: Cognitive strategy to employ
            step_context: Context for this specific step

        Returns:
            Executed meta-reasoning step
        """
        if process_id not in self.active_processes:
            raise ValueError(
                f"Meta-reasoning process {process_id} not found or not active"
            )

        process = self.active_processes[process_id]

        # Create and execute meta-reasoning step
        step = self._create_meta_reasoning_step(
            step_type=step_type,
            reasoning_level=reasoning_level,
            cognitive_strategy=cognitive_strategy,
            description=step_context.get("description", f"Executing {step_type}"),
            rationale=step_context.get("rationale", "Strategic reasoning step"),
            context=step_context,
        )

        # Execute step-specific logic
        step = self._execute_step_logic(step, step_context)

        # Add to process
        process.reasoning_steps.append(step)
        process.cognitive_flow.append(step.step_id)

        # Update strategy performance
        self._update_strategy_performance(cognitive_strategy, step)

        # Add to meta-reasoning history
        self.meta_reasoning_history.append(step)

        logger.info(
            f"Executed meta-reasoning step {step.step_id} in process {process_id}"
        )
        return step

    def perform_comprehensive_self_assessment(
        self, time_window_hours: int = 168
    ) -> SelfAssessmentResult:
        """
        Perform comprehensive self-assessment of reasoning capabilities.

        Args:
            time_window_hours: Time window for assessment analysis

        Returns:
            Comprehensive self-assessment results
        """
        assessment_id = f"self_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)

        # Filter recent meta-reasoning steps
        recent_steps = [
            step
            for step in self.meta_reasoning_history
            if step.timestamp >= cutoff_time
        ]

        # Perform multi-dimensional assessment
        assessment_results = {}

        for framework_name, framework_func in self.assessment_frameworks.items():
            assessment_results[framework_name] = framework_func(recent_steps)

        # Synthesize assessment results
        assessment = SelfAssessmentResult(
            assessment_id=assessment_id,
            assessment_timestamp=datetime.now(),
            assessment_scope=f"{time_window_hours} hours, {len(recent_steps)} steps",
            reasoning_effectiveness=assessment_results["reasoning_effectiveness"],
            strategy_utilization=assessment_results["strategy_coherence"],
            adaptive_learning=assessment_results["adaptive_learning"],
            strength_areas=self._identify_strength_areas(assessment_results),
            improvement_areas=self._identify_improvement_areas(assessment_results),
            cognitive_biases_detected=assessment_results["cognitive_bias_detection"],
            strategy_recommendations=self._generate_strategy_recommendations(
                assessment_results
            ),
            learning_priorities=self._identify_learning_priorities(assessment_results),
            adaptive_modifications=self._suggest_adaptive_modifications(
                assessment_results
            ),
            assessment_confidence=self._calculate_assessment_confidence(
                assessment_results
            ),
            reliability_indicators=self._calculate_reliability_indicators(recent_steps),
        )

        logger.info(f"Completed comprehensive self-assessment {assessment_id}")
        return assessment

    def analyze_cognitive_strategy_effectiveness(
        self,
        strategy: CognitiveStrategy,
        context_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze effectiveness of a specific cognitive strategy.

        Args:
            strategy: Cognitive strategy to analyze
            context_filter: Optional filter for specific contexts

        Returns:
            Detailed strategy effectiveness analysis
        """
        # Filter steps using this strategy
        strategy_steps = [
            step
            for step in self.meta_reasoning_history
            if step.cognitive_strategy == strategy
        ]

        if context_filter:
            # Apply additional filtering based on context
            strategy_steps = self._filter_steps_by_context(
                strategy_steps, context_filter
            )

        if not strategy_steps:
            return {"error": f"No data available for strategy {strategy.value}"}

        # Analyze strategy performance
        analysis = {
            "strategy": strategy.value,
            "analysis_timestamp": datetime.now().isoformat(),
            "sample_size": len(strategy_steps),
            "performance_metrics": self._calculate_strategy_performance_metrics(
                strategy_steps
            ),
            "effectiveness_patterns": self._identify_effectiveness_patterns(
                strategy_steps
            ),
            "optimal_contexts": self._identify_optimal_contexts(strategy_steps),
            "limitation_patterns": self._identify_limitation_patterns(strategy_steps),
            "improvement_opportunities": self._identify_strategy_improvement_opportunities(
                strategy_steps
            ),
            "comparative_analysis": self._compare_strategy_with_others(
                strategy, strategy_steps
            ),
        }

        logger.info(f"Analyzed cognitive strategy {strategy.value} effectiveness")
        return analysis

    def discover_meta_cognitive_patterns(
        self, pattern_type: str = "all"
    ) -> Dict[str, Any]:
        """
        Discover meta-cognitive patterns in reasoning processes.

        Args:
            pattern_type: Type of patterns to discover ("all", "success", "failure", "adaptation")

        Returns:
            Discovered meta-cognitive patterns and insights
        """
        pattern_discovery = {
            "discovery_id": f"meta_pattern_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "pattern_type": pattern_type,
            "discovery_timestamp": datetime.now().isoformat(),
            "patterns_discovered": {},
        }

        if pattern_type in ["all", "success"]:
            pattern_discovery["patterns_discovered"][
                "success_patterns"
            ] = self._discover_success_patterns()

        if pattern_type in ["all", "failure"]:
            pattern_discovery["patterns_discovered"][
                "failure_patterns"
            ] = self._discover_failure_patterns()

        if pattern_type in ["all", "adaptation"]:
            pattern_discovery["patterns_discovered"][
                "adaptation_patterns"
            ] = self._discover_adaptation_patterns()

        if pattern_type in ["all", "cognitive_flow"]:
            pattern_discovery["patterns_discovered"][
                "cognitive_flow_patterns"
            ] = self._discover_cognitive_flow_patterns()

        if pattern_type in ["all", "strategy_transition"]:
            pattern_discovery["patterns_discovered"][
                "strategy_transition_patterns"
            ] = self._discover_strategy_transition_patterns()

        # Generate insights from discovered patterns
        pattern_discovery["insights"] = self._generate_pattern_insights(
            pattern_discovery["patterns_discovered"]
        )
        pattern_discovery["recommendations"] = self._generate_pattern_recommendations(
            pattern_discovery["patterns_discovered"]
        )

        logger.info(f"Discovered meta-cognitive patterns of type: {pattern_type}")
        return pattern_discovery

    def evaluate_reasoning_coherence(self, process_id: str) -> Dict[str, Any]:
        """
        Evaluate coherence of reasoning within a cognitive process.

        Args:
            process_id: ID of the cognitive process to evaluate

        Returns:
            Detailed coherence evaluation results
        """
        if (
            process_id not in self.active_processes
            and process_id not in self.completed_processes
        ):
            raise ValueError(f"Cognitive process {process_id} not found")

        process = self.active_processes.get(process_id) or self.completed_processes.get(
            process_id
        )

        coherence_evaluation = {
            "process_id": process_id,
            "evaluation_timestamp": datetime.now().isoformat(),
            "logical_coherence": self._evaluate_logical_coherence(process),
            "strategic_coherence": self._evaluate_strategic_coherence(process),
            "temporal_coherence": self._evaluate_temporal_coherence(process),
            "causal_coherence": self._evaluate_causal_coherence(process),
            "overall_coherence_score": 0.0,
            "coherence_strengths": [],
            "coherence_weaknesses": [],
            "improvement_suggestions": [],
        }

        # Calculate overall coherence score
        coherence_scores = [
            coherence_evaluation["logical_coherence"]["score"],
            coherence_evaluation["strategic_coherence"]["score"],
            coherence_evaluation["temporal_coherence"]["score"],
            coherence_evaluation["causal_coherence"]["score"],
        ]
        coherence_evaluation["overall_coherence_score"] = statistics.mean(
            coherence_scores
        )

        # Identify strengths and weaknesses
        coherence_evaluation["coherence_strengths"] = [
            dimension
            for dimension, data in coherence_evaluation.items()
            if isinstance(data, dict) and data.get("score", 0) > 0.8
        ]
        coherence_evaluation["coherence_weaknesses"] = [
            dimension
            for dimension, data in coherence_evaluation.items()
            if isinstance(data, dict) and data.get("score", 0) < 0.6
        ]

        # Generate improvement suggestions
        coherence_evaluation["improvement_suggestions"] = (
            self._generate_coherence_improvement_suggestions(coherence_evaluation)
        )

        logger.info(f"Evaluated reasoning coherence for process {process_id}")
        return coherence_evaluation

    def adapt_cognitive_strategies(
        self, adaptation_trigger: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt cognitive strategies based on performance feedback and context.

        Args:
            adaptation_trigger: Reason for adaptation (e.g., "poor_performance", "new_context")
            context: Context information for adaptation

        Returns:
            Results of cognitive strategy adaptation
        """
        adaptation_id = f"adaptation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        adaptation_result = {
            "adaptation_id": adaptation_id,
            "trigger": adaptation_trigger,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "strategies_modified": [],
            "new_strategies_created": [],
            "performance_predictions": {},
            "adaptation_confidence": 0.0,
        }

        # Analyze current strategy performance
        strategy_analysis = self._analyze_current_strategy_performance()

        # Identify strategies needing adaptation
        strategies_to_adapt = self._identify_strategies_for_adaptation(
            strategy_analysis, context
        )

        for strategy in strategies_to_adapt:
            adaptation = self._adapt_strategy(strategy, adaptation_trigger, context)

            if adaptation["modified"]:
                adaptation_result["strategies_modified"].append(
                    {
                        "strategy": strategy.value,
                        "modifications": adaptation["modifications"],
                        "expected_improvement": adaptation["expected_improvement"],
                    }
                )

        # Consider creating new strategies if needed
        if adaptation_trigger == "new_context" or len(strategies_to_adapt) == 0:
            new_strategy = self._create_adaptive_strategy(context)
            if new_strategy:
                adaptation_result["new_strategies_created"].append(new_strategy)

        # Predict performance impact
        adaptation_result["performance_predictions"] = self._predict_adaptation_impact(
            adaptation_result
        )
        adaptation_result["adaptation_confidence"] = (
            self._calculate_adaptation_confidence(adaptation_result)
        )

        # Record adaptation in history
        self.adaptation_history.append(adaptation_result)

        logger.info(f"Completed cognitive strategy adaptation {adaptation_id}")
        return adaptation_result

    def complete_meta_reasoning_process(
        self, process_id: str, outcomes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Complete a meta-reasoning process and extract learning insights.

        Args:
            process_id: ID of the process to complete
            outcomes: Outcomes and results from the process

        Returns:
            Process completion analysis and learning insights
        """
        if process_id not in self.active_processes:
            raise ValueError(f"Active meta-reasoning process {process_id} not found")

        process = self.active_processes[process_id]
        process.end_time = datetime.now()

        # Analyze process effectiveness
        process_analysis = self._analyze_process_effectiveness(process, outcomes)

        # Extract learning insights
        learning_insights = self._extract_process_learning_insights(process, outcomes)

        # Update process with analysis results
        process.overall_effectiveness = process_analysis["effectiveness_score"]
        process.strategic_coherence = process_analysis["strategic_coherence"]
        process.adaptive_capability = process_analysis["adaptive_capability"]
        process.insights_generated = learning_insights["insights"]
        process.strategies_discovered = learning_insights["strategies"]
        process.improvement_opportunities = learning_insights["improvements"]

        # Move to completed processes
        self.completed_processes[process_id] = process
        del self.active_processes[process_id]

        # Record learning experience
        learning_experience = {
            "process_id": process_id,
            "completion_timestamp": datetime.now().isoformat(),
            "process_analysis": process_analysis,
            "learning_insights": learning_insights,
            "outcomes": outcomes,
        }
        self.learning_experiences.append(learning_experience)

        completion_result = {
            "process_id": process_id,
            "completion_status": "success",
            "process_duration": (process.end_time - process.start_time).total_seconds(),
            "steps_executed": len(process.reasoning_steps),
            "effectiveness_score": process.overall_effectiveness,
            "learning_insights": learning_insights,
            "recommendations": self._generate_process_recommendations(
                process, process_analysis
            ),
        }

        logger.info(
            f"Completed meta-reasoning process {process_id} with effectiveness {process.overall_effectiveness:.3f}"
        )
        return completion_result

    # Step creation and execution methods
    def _create_meta_reasoning_step(
        self,
        step_type: str,
        reasoning_level: ReasoningLevel,
        cognitive_strategy: CognitiveStrategy,
        description: str,
        rationale: str,
        context: Dict[str, Any],
    ) -> MetaReasoningStep:
        """Create a new meta-reasoning step"""

        step_id = f"step_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        step = MetaReasoningStep(
            step_id=step_id,
            reasoning_level=reasoning_level,
            cognitive_strategy=cognitive_strategy,
            step_type=step_type,
            description=description,
            rationale=rationale,
            evidence=context.get("evidence", []),
            assumptions=context.get("assumptions", []),
            confidence=context.get("confidence", 0.7),
            validity_score=context.get("validity_score", 0.7),
            novelty_score=context.get("novelty_score", 0.5),
            timestamp=datetime.now(),
        )

        return step

    def _execute_step_logic(
        self, step: MetaReasoningStep, context: Dict[str, Any]
    ) -> MetaReasoningStep:
        """Execute the logic for a specific meta-reasoning step"""

        if step.step_type == "problem_analysis":
            step = self._execute_problem_analysis(step, context)
        elif step.step_type == "strategy_selection":
            step = self._execute_strategy_selection(step, context)
        elif step.step_type == "approach_evaluation":
            step = self._execute_approach_evaluation(step, context)
        elif step.step_type == "outcome_assessment":
            step = self._execute_outcome_assessment(step, context)
        elif step.step_type == "learning_integration":
            step = self._execute_learning_integration(step, context)
        else:
            step = self._execute_generic_step(step, context)

        return step

    def _execute_problem_analysis(
        self, step: MetaReasoningStep, context: Dict[str, Any]
    ) -> MetaReasoningStep:
        """Execute problem analysis step"""

        problem_complexity = context.get("complexity", "medium")
        available_information = context.get("information_completeness", 0.7)
        time_constraints = context.get("time_constraints", "moderate")

        # Analyze problem characteristics
        if problem_complexity == "high" and available_information < 0.6:
            step.evidence.append(
                "High complexity with limited information requires systematic approach"
            )
            step.assumptions.append(
                "Additional information gathering will be beneficial"
            )
            step.confidence = 0.6
        elif problem_complexity == "low" and time_constraints == "tight":
            step.evidence.append(
                "Simple problem with time pressure favors direct approach"
            )
            step.assumptions.append("Efficiency is prioritized over thoroughness")
            step.confidence = 0.8
        else:
            step.evidence.append(
                "Balanced problem characteristics allow flexible approach"
            )
            step.confidence = 0.75

        step.success_indicators = [
            "Clear problem understanding",
            "Appropriate complexity assessment",
        ]
        step.lessons_learned = [
            f"Problem analysis for {problem_complexity} complexity completed"
        ]

        return step

    def _execute_strategy_selection(
        self, step: MetaReasoningStep, context: Dict[str, Any]
    ) -> MetaReasoningStep:
        """Execute strategy selection step"""

        available_strategies = context.get(
            "available_strategies", list(CognitiveStrategy)
        )
        problem_type = context.get("problem_type", "analytical")

        # Select optimal strategy based on context
        strategy_scores = {}
        for strategy in available_strategies:
            score = self._score_strategy_for_context(strategy, context)
            strategy_scores[strategy.value] = score

        best_strategy = max(strategy_scores, key=strategy_scores.get)

        step.evidence.append(f"Selected {best_strategy} based on context analysis")
        step.evidence.append(f"Strategy scores: {strategy_scores}")
        step.assumptions.append(
            f"Selected strategy is optimal for {problem_type} problems"
        )

        step.success_indicators = [
            "Strategy aligns with problem type",
            "Historical performance supports choice",
        ]
        step.lessons_learned = [f"Strategy selection for {problem_type} completed"]

        return step

    def _execute_approach_evaluation(
        self, step: MetaReasoningStep, context: Dict[str, Any]
    ) -> MetaReasoningStep:
        """Execute approach evaluation step"""

        intermediate_results = context.get("intermediate_results", {})

        # Evaluate approach effectiveness
        effectiveness_indicators = [
            "Progress toward goal",
            "Resource efficiency",
            "Quality of intermediate results",
            "Adaptability to new information",
        ]

        effectiveness_score = 0.0
        for indicator in effectiveness_indicators:
            indicator_score = intermediate_results.get(indicator, 0.7)
            effectiveness_score += indicator_score

        effectiveness_score /= len(effectiveness_indicators)

        step.evidence.append(f"Approach effectiveness score: {effectiveness_score:.3f}")
        step.evidence.append(
            f"Evaluated against {len(effectiveness_indicators)} indicators"
        )

        if effectiveness_score > 0.8:
            step.assumptions.append("Current approach is highly effective")
            step.success_indicators.append("High effectiveness maintained")
        elif effectiveness_score < 0.6:
            step.assumptions.append("Current approach may need modification")
            step.failure_indicators.append("Low effectiveness detected")

        step.validity_score = effectiveness_score
        step.lessons_learned = [
            f"Approach evaluation completed with score {effectiveness_score:.3f}"
        ]

        return step

    def _execute_outcome_assessment(
        self, step: MetaReasoningStep, context: Dict[str, Any]
    ) -> MetaReasoningStep:
        """Execute outcome assessment step"""

        expected_outcomes = context.get("expected_outcomes", [])
        actual_outcomes = context.get("actual_outcomes", [])

        # Assess outcome alignment
        outcome_alignment = 0.0
        if expected_outcomes and actual_outcomes:
            alignment_count = len(set(expected_outcomes) & set(actual_outcomes))
            outcome_alignment = alignment_count / max(
                len(expected_outcomes), len(actual_outcomes)
            )

        # Assess success criteria fulfillment
        criteria_fulfillment = context.get("criteria_fulfillment", 0.7)

        step.evidence.append(f"Outcome alignment: {outcome_alignment:.3f}")
        step.evidence.append(
            f"Success criteria fulfillment: {criteria_fulfillment:.3f}"
        )

        overall_success = (outcome_alignment + criteria_fulfillment) / 2

        if overall_success > 0.8:
            step.success_indicators.extend(
                ["High outcome alignment", "Success criteria well met"]
            )
        elif overall_success < 0.5:
            step.failure_indicators.extend(
                ["Poor outcome alignment", "Success criteria not met"]
            )

        step.validity_score = overall_success
        step.lessons_learned = [
            f"Outcome assessment completed with success score {overall_success:.3f}"
        ]

        return step

    def _execute_learning_integration(
        self, step: MetaReasoningStep, context: Dict[str, Any]
    ) -> MetaReasoningStep:
        """Execute learning integration step"""

        new_insights = context.get("new_insights", [])
        contradictory_evidence = context.get("contradictory_evidence", [])
        learning_opportunities = context.get("learning_opportunities", [])

        # Integrate new learning
        integration_success = len(new_insights) / max(1, len(learning_opportunities))

        step.evidence.append(f"Integrated {len(new_insights)} new insights")
        step.evidence.append(f"Learning integration success: {integration_success:.3f}")

        if contradictory_evidence:
            step.evidence.append(
                f"Resolved {len(contradictory_evidence)} contradictions"
            )
            step.assumptions.append(
                "Contradiction resolution strengthens understanding"
            )

        step.novelty_score = min(1.0, len(new_insights) * 0.2)
        step.validity_score = integration_success

        step.success_indicators = [
            "New insights integrated",
            "Learning opportunities identified",
        ]
        step.lessons_learned = new_insights

        return step

    def _execute_generic_step(
        self, step: MetaReasoningStep, context: Dict[str, Any]
    ) -> MetaReasoningStep:
        """Execute generic meta-reasoning step"""

        step.evidence.append(f"Executed {step.step_type} step")
        step.assumptions.append("Generic step execution completed successfully")
        step.success_indicators = ["Step completed"]
        step.lessons_learned = [f"Generic {step.step_type} step executed"]

        return step

    # Assessment framework methods
    def _assess_reasoning_effectiveness(
        self, steps: List[MetaReasoningStep]
    ) -> Dict[str, float]:
        """Assess overall reasoning effectiveness"""

        if not steps:
            return {
                "overall": 0.0,
                "logical_consistency": 0.0,
                "goal_achievement": 0.0,
                "efficiency": 0.0,
            }

        # Calculate logical consistency
        validity_scores = [step.validity_score for step in steps]
        logical_consistency = statistics.mean(validity_scores)

        # Calculate goal achievement
        success_rate = len([step for step in steps if step.success_indicators]) / len(
            steps
        )

        # Calculate efficiency (novelty and confidence balance)
        confidence_scores = [step.confidence for step in steps]
        novelty_scores = [step.novelty_score for step in steps]

        efficiency = (
            statistics.mean(confidence_scores) + statistics.mean(novelty_scores)
        ) / 2

        overall = (logical_consistency + success_rate + efficiency) / 3

        return {
            "overall": overall,
            "logical_consistency": logical_consistency,
            "goal_achievement": success_rate,
            "efficiency": efficiency,
        }

    def _assess_strategy_coherence(
        self, steps: List[MetaReasoningStep]
    ) -> Dict[str, float]:
        """Assess coherence of strategy utilization"""

        if not steps:
            return {"coherence": 0.0, "diversity": 0.0, "appropriateness": 0.0}

        # Analyze strategy distribution
        strategy_usage = Counter([step.cognitive_strategy for step in steps])

        # Calculate diversity (not too concentrated, not too scattered)
        strategy_diversity = len(strategy_usage) / len(CognitiveStrategy)
        optimal_diversity = 0.6  # Target around 60% of available strategies
        diversity_score = 1.0 - abs(strategy_diversity - optimal_diversity)

        # Calculate appropriateness (strategies match step types)
        appropriate_usage = 0
        for step in steps:
            if self._is_strategy_appropriate(step.cognitive_strategy, step.step_type):
                appropriate_usage += 1

        appropriateness = appropriate_usage / len(steps)

        # Calculate overall coherence
        coherence = (diversity_score + appropriateness) / 2

        return {
            "coherence": coherence,
            "diversity": diversity_score,
            "appropriateness": appropriateness,
        }

    def _assess_adaptive_learning(
        self, steps: List[MetaReasoningStep]
    ) -> Dict[str, float]:
        """Assess adaptive learning capability"""

        if not steps:
            return {"adaptability": 0.0, "learning_rate": 0.0, "improvement_trend": 0.0}

        # Calculate learning rate (lessons learned per step)
        total_lessons = sum(len(step.lessons_learned) for step in steps)
        learning_rate = total_lessons / len(steps)

        # Calculate improvement trend (validity scores over time)
        validity_trend = 0.0
        if len(steps) > 3:
            early_validity = statistics.mean(
                [step.validity_score for step in steps[: len(steps) // 2]]
            )
            late_validity = statistics.mean(
                [step.validity_score for step in steps[len(steps) // 2 :]]
            )
            validity_trend = max(0.0, late_validity - early_validity)

        # Calculate adaptability (strategy changes in response to step outcomes)
        strategy_adaptations = 0
        for i in range(1, len(steps)):
            if steps[i].cognitive_strategy != steps[i - 1].cognitive_strategy:
                if steps[i - 1].failure_indicators:  # Strategy changed after failure
                    strategy_adaptations += 1

        adaptability = strategy_adaptations / max(1, len(steps) - 1)

        return {
            "adaptability": adaptability,
            "learning_rate": min(1.0, learning_rate / 3.0),  # Normalize to [0,1]
            "improvement_trend": validity_trend,
        }

    def _detect_cognitive_biases(
        self, steps: List[MetaReasoningStep]
    ) -> List[Dict[str, Any]]:
        """Detect potential cognitive biases in reasoning"""

        biases_detected = []

        # Confirmation bias detection
        confirmation_patterns = 0
        for step in steps:
            evidence_text = " ".join(step.evidence).lower()
            if any(
                word in evidence_text
                for word in ["confirms", "supports", "validates", "as expected"]
            ):
                confirmation_patterns += 1

        if (
            confirmation_patterns > len(steps) * 0.6
        ):  # More than 60% confirmation language
            biases_detected.append(
                {
                    "bias_type": "confirmation_bias",
                    "severity": "medium",
                    "evidence": f"{confirmation_patterns} instances of confirmation language",
                    "recommendation": "Actively seek disconfirming evidence",
                }
            )

        # Anchoring bias detection
        strategy_usage = [step.cognitive_strategy for step in steps]
        if (
            len(set(strategy_usage)) == 1 and len(steps) > 5
        ):  # Same strategy for many steps
            biases_detected.append(
                {
                    "bias_type": "anchoring_bias",
                    "severity": "medium",
                    "evidence": f"Single strategy used for {len(steps)} consecutive steps",
                    "recommendation": "Consider alternative cognitive strategies",
                }
            )

        # Overconfidence bias detection
        high_confidence_steps = [step for step in steps if step.confidence > 0.9]
        if (
            len(high_confidence_steps) > len(steps) * 0.7
        ):  # More than 70% high confidence
            biases_detected.append(
                {
                    "bias_type": "overconfidence_bias",
                    "severity": "low",
                    "evidence": f"{len(high_confidence_steps)} high-confidence assessments",
                    "recommendation": "Implement systematic uncertainty quantification",
                }
            )

        return biases_detected

    def _assess_meta_cognitive_awareness(
        self, steps: List[MetaReasoningStep]
    ) -> Dict[str, float]:
        """Assess meta-cognitive awareness level"""

        if not steps:
            return {"awareness": 0.0, "self_monitoring": 0.0, "strategy_awareness": 0.0}

        # Calculate meta-level reasoning proportion
        meta_level_steps = len(
            [
                step
                for step in steps
                if step.reasoning_level == ReasoningLevel.META_LEVEL
            ]
        )
        meta_meta_level_steps = len(
            [
                step
                for step in steps
                if step.reasoning_level == ReasoningLevel.META_META_LEVEL
            ]
        )

        meta_awareness = (meta_level_steps + meta_meta_level_steps * 2) / len(steps)

        # Calculate self-monitoring (steps with assumptions and evidence)
        monitored_steps = len(
            [step for step in steps if step.assumptions and step.evidence]
        )
        self_monitoring = monitored_steps / len(steps)

        # Calculate strategy awareness (explicit strategy reasoning)
        strategy_aware_steps = len(
            [step for step in steps if "strategy" in step.description.lower()]
        )
        strategy_awareness = strategy_aware_steps / len(steps)

        overall_awareness = (meta_awareness + self_monitoring + strategy_awareness) / 3

        return {
            "awareness": overall_awareness,
            "self_monitoring": self_monitoring,
            "strategy_awareness": strategy_awareness,
        }

    # Pattern discovery methods
    def _discover_success_patterns(self) -> List[Dict[str, Any]]:
        """Discover patterns associated with successful reasoning"""

        successful_steps = [
            step
            for step in self.meta_reasoning_history
            if step.success_indicators and step.validity_score > 0.8
        ]

        if not successful_steps:
            return []

        patterns = []

        # Strategy success patterns
        strategy_success = Counter(
            [step.cognitive_strategy for step in successful_steps]
        )
        for strategy, count in strategy_success.most_common(3):
            patterns.append(
                {
                    "pattern_type": "strategy_success",
                    "pattern": f"{strategy.value}_high_success",
                    "evidence": f"Strategy used in {count} successful steps",
                    "confidence": min(1.0, count / len(successful_steps)),
                }
            )

        # Step type success patterns
        step_type_success = Counter([step.step_type for step in successful_steps])
        for step_type, count in step_type_success.most_common(3):
            patterns.append(
                {
                    "pattern_type": "step_type_success",
                    "pattern": f"{step_type}_effective",
                    "evidence": f"Step type successful in {count} instances",
                    "confidence": min(1.0, count / len(successful_steps)),
                }
            )

        return patterns

    def _discover_failure_patterns(self) -> List[Dict[str, Any]]:
        """Discover patterns associated with reasoning failures"""

        failed_steps = [
            step
            for step in self.meta_reasoning_history
            if step.failure_indicators and step.validity_score < 0.5
        ]

        if not failed_steps:
            return []

        patterns = []

        # Strategy failure patterns
        strategy_failures = Counter([step.cognitive_strategy for step in failed_steps])
        for strategy, count in strategy_failures.most_common(2):
            patterns.append(
                {
                    "pattern_type": "strategy_failure",
                    "pattern": f"{strategy.value}_low_success",
                    "evidence": f"Strategy associated with {count} failures",
                    "confidence": min(1.0, count / len(failed_steps)),
                }
            )

        return patterns

    def _discover_adaptation_patterns(self) -> List[Dict[str, Any]]:
        """Discover patterns in adaptive behavior"""

        patterns = []

        # Strategy switching patterns
        strategy_switches = []
        for i in range(1, len(self.meta_reasoning_history)):
            current_step = self.meta_reasoning_history[i]
            previous_step = self.meta_reasoning_history[i - 1]

            if current_step.cognitive_strategy != previous_step.cognitive_strategy:
                strategy_switches.append(
                    {
                        "from_strategy": previous_step.cognitive_strategy,
                        "to_strategy": current_step.cognitive_strategy,
                        "trigger": (
                            "failure"
                            if previous_step.failure_indicators
                            else "optimization"
                        ),
                    }
                )

        if strategy_switches:
            switch_patterns = Counter(
                [
                    f"{s['from_strategy'].value}_to_{s['to_strategy'].value}"
                    for s in strategy_switches
                ]
            )
            for pattern, count in switch_patterns.most_common(3):
                patterns.append(
                    {
                        "pattern_type": "adaptation_strategy_switch",
                        "pattern": pattern,
                        "evidence": f"Strategy switch occurred {count} times",
                        "confidence": min(1.0, count / len(strategy_switches)),
                    }
                )

        return patterns

    def _discover_cognitive_flow_patterns(self) -> List[Dict[str, Any]]:
        """Discover patterns in cognitive flow and reasoning sequences"""

        patterns = []

        # Reasoning level progression patterns
        level_sequences = []
        for i in range(len(self.meta_reasoning_history) - 2):
            sequence = [
                self.meta_reasoning_history[i].reasoning_level,
                self.meta_reasoning_history[i + 1].reasoning_level,
                self.meta_reasoning_history[i + 2].reasoning_level,
            ]
            level_sequences.append("_".join([level.value for level in sequence]))

        if level_sequences:
            sequence_patterns = Counter(level_sequences)
            for pattern, count in sequence_patterns.most_common(3):
                patterns.append(
                    {
                        "pattern_type": "cognitive_flow_reasoning_levels",
                        "pattern": pattern,
                        "evidence": f"Reasoning level sequence occurred {count} times",
                        "confidence": min(1.0, count / len(level_sequences)),
                    }
                )

        return patterns

    def _discover_strategy_transition_patterns(self) -> List[Dict[str, Any]]:
        """Discover patterns in strategy transitions"""

        patterns = []

        # Analyze strategy transition effectiveness
        transitions = []
        for i in range(1, len(self.meta_reasoning_history)):
            current_step = self.meta_reasoning_history[i]
            previous_step = self.meta_reasoning_history[i - 1]

            if current_step.cognitive_strategy != previous_step.cognitive_strategy:
                transitions.append(
                    {
                        "transition": f"{previous_step.cognitive_strategy.value}_to_{current_step.cognitive_strategy.value}",
                        "effectiveness": current_step.validity_score
                        - previous_step.validity_score,
                    }
                )

        if transitions:
            # Find most effective transitions
            transition_effectiveness = defaultdict(list)
            for transition in transitions:
                transition_effectiveness[transition["transition"]].append(
                    transition["effectiveness"]
                )

            for transition_type, effectiveness_list in transition_effectiveness.items():
                avg_effectiveness = statistics.mean(effectiveness_list)
                if avg_effectiveness > 0.1:  # Significant improvement
                    patterns.append(
                        {
                            "pattern_type": "effective_strategy_transition",
                            "pattern": transition_type,
                            "evidence": f"Average effectiveness improvement: {avg_effectiveness:.3f}",
                            "confidence": min(1.0, len(effectiveness_list) / 5),
                        }
                    )

        return patterns

    # Helper methods for cognitive strategies
    def _get_strategy_description(self, strategy: CognitiveStrategy) -> str:
        """Get description for a cognitive strategy"""
        descriptions = {
            CognitiveStrategy.ANALYTICAL: "Systematic, logical, step-by-step reasoning approach",
            CognitiveStrategy.INTUITIVE: "Intuition-based, pattern-recognition driven approach",
            CognitiveStrategy.SYSTEMATIC: "Methodical, comprehensive, thorough approach",
            CognitiveStrategy.CREATIVE: "Innovative, novel, out-of-the-box thinking approach",
            CognitiveStrategy.CONSERVATIVE: "Risk-averse, proven-method, reliable approach",
            CognitiveStrategy.EXPERIMENTAL: "Exploratory, trial-and-error, learning-oriented approach",
        }
        return descriptions.get(strategy, "Unknown strategy")

    def _get_strategy_use_cases(self, strategy: CognitiveStrategy) -> List[str]:
        """Get typical use cases for a cognitive strategy"""
        use_cases = {
            CognitiveStrategy.ANALYTICAL: [
                "Complex problem decomposition",
                "Logical reasoning",
                "Mathematical analysis",
            ],
            CognitiveStrategy.INTUITIVE: [
                "Pattern recognition",
                "Quick decisions",
                "Creative insights",
            ],
            CognitiveStrategy.SYSTEMATIC: [
                "Comprehensive analysis",
                "Quality assurance",
                "Process optimization",
            ],
            CognitiveStrategy.CREATIVE: [
                "Innovation challenges",
                "Problem reframing",
                "Novel solution generation",
            ],
            CognitiveStrategy.CONSERVATIVE: [
                "Risk management",
                "Reliability requirements",
                "Proven approaches",
            ],
            CognitiveStrategy.EXPERIMENTAL: [
                "Learning opportunities",
                "Exploration phases",
                "Hypothesis testing",
            ],
        }
        return use_cases.get(strategy, [])

    def _get_strategy_strengths(self, strategy: CognitiveStrategy) -> List[str]:
        """Get strengths of a cognitive strategy"""
        strengths = {
            CognitiveStrategy.ANALYTICAL: [
                "High precision",
                "Logical consistency",
                "Systematic coverage",
            ],
            CognitiveStrategy.INTUITIVE: [
                "Speed",
                "Pattern recognition",
                "Holistic understanding",
            ],
            CognitiveStrategy.SYSTEMATIC: [
                "Thoroughness",
                "Reliability",
                "Comprehensive coverage",
            ],
            CognitiveStrategy.CREATIVE: [
                "Innovation",
                "Novel perspectives",
                "Breakthrough potential",
            ],
            CognitiveStrategy.CONSERVATIVE: [
                "Reliability",
                "Risk mitigation",
                "Proven effectiveness",
            ],
            CognitiveStrategy.EXPERIMENTAL: [
                "Learning acceleration",
                "Adaptability",
                "Discovery potential",
            ],
        }
        return strengths.get(strategy, [])

    def _get_strategy_limitations(self, strategy: CognitiveStrategy) -> List[str]:
        """Get limitations of a cognitive strategy"""
        limitations = {
            CognitiveStrategy.ANALYTICAL: [
                "Time intensive",
                "May miss intuitive insights",
                "Can be rigid",
            ],
            CognitiveStrategy.INTUITIVE: [
                "May lack precision",
                "Hard to validate",
                "Inconsistent",
            ],
            CognitiveStrategy.SYSTEMATIC: [
                "Time consuming",
                "May be overkill",
                "Can be inflexible",
            ],
            CognitiveStrategy.CREATIVE: [
                "Unpredictable",
                "May lack practicality",
                "Hard to control",
            ],
            CognitiveStrategy.CONSERVATIVE: [
                "May miss opportunities",
                "Slow adaptation",
                "Limited innovation",
            ],
            CognitiveStrategy.EXPERIMENTAL: [
                "Uncertain outcomes",
                "Resource intensive",
                "May not converge",
            ],
        }
        return limitations.get(strategy, [])

    def _update_strategy_performance(
        self, strategy: CognitiveStrategy, step: MetaReasoningStep
    ):
        """Update performance metrics for a cognitive strategy"""

        performance = self.strategy_performance[strategy]
        performance["total_usage"] += 1

        # Update success rate
        if step.success_indicators and not step.failure_indicators:
            success_contribution = 1.0
        elif step.failure_indicators:
            success_contribution = 0.0
        else:
            success_contribution = 0.5

        # Running average update
        usage = performance["total_usage"]
        performance["success_rate"] = (
            (performance["success_rate"] * (usage - 1)) + success_contribution
        ) / usage

        # Update quality (validity score)
        performance["quality"] = (
            (performance["quality"] * (usage - 1)) + step.validity_score
        ) / usage

        # Update efficiency (inverse of reasoning complexity)
        complexity = (
            len(step.evidence) + len(step.assumptions) + len(step.lessons_learned)
        )
        efficiency_score = 1.0 / (1.0 + complexity * 0.1)  # Normalize complexity impact
        performance["efficiency"] = (
            (performance["efficiency"] * (usage - 1)) + efficiency_score
        ) / usage

        # Update adaptability (novelty score)
        performance["adaptability"] = (
            (performance["adaptability"] * (usage - 1)) + step.novelty_score
        ) / usage

    # Additional helper methods
    def _score_strategy_for_context(
        self, strategy: CognitiveStrategy, context: Dict[str, Any]
    ) -> float:
        """Score how well a strategy fits a given context"""

        base_score = self.strategy_performance[strategy]["success_rate"]

        # Adjust based on context factors
        problem_type = context.get("problem_type", "analytical")
        time_pressure = context.get("time_pressure", "moderate")
        complexity = context.get("complexity", "medium")

        # Strategy-specific adjustments
        if strategy == CognitiveStrategy.ANALYTICAL:
            if problem_type == "analytical":
                base_score += 0.2
            if time_pressure == "high":
                base_score -= 0.1
        elif strategy == CognitiveStrategy.INTUITIVE:
            if time_pressure == "high":
                base_score += 0.2
            if complexity == "high":
                base_score -= 0.1
        elif strategy == CognitiveStrategy.CREATIVE:
            if problem_type == "innovative":
                base_score += 0.3
            if time_pressure == "high":
                base_score -= 0.2

        return max(0.0, min(1.0, base_score))

    def _is_strategy_appropriate(
        self, strategy: CognitiveStrategy, step_type: str
    ) -> bool:
        """Check if a strategy is appropriate for a step type"""

        appropriate_combinations = {
            "problem_analysis": [
                CognitiveStrategy.ANALYTICAL,
                CognitiveStrategy.SYSTEMATIC,
            ],
            "strategy_selection": [
                CognitiveStrategy.ANALYTICAL,
                CognitiveStrategy.INTUITIVE,
            ],
            "approach_evaluation": [
                CognitiveStrategy.ANALYTICAL,
                CognitiveStrategy.SYSTEMATIC,
            ],
            "outcome_assessment": [
                CognitiveStrategy.ANALYTICAL,
                CognitiveStrategy.CONSERVATIVE,
            ],
            "learning_integration": [
                CognitiveStrategy.CREATIVE,
                CognitiveStrategy.EXPERIMENTAL,
            ],
        }

        return strategy in appropriate_combinations.get(
            step_type, list(CognitiveStrategy)
        )

    def _filter_steps_by_context(
        self, steps: List[MetaReasoningStep], context_filter: Dict[str, Any]
    ) -> List[MetaReasoningStep]:
        """Filter steps based on context criteria"""

        filtered_steps = []

        for step in steps:
            matches_filter = True

            # Filter by reasoning level
            if "reasoning_level" in context_filter:
                if step.reasoning_level.value != context_filter["reasoning_level"]:
                    matches_filter = False

            # Filter by step type
            if "step_type" in context_filter:
                if step.step_type != context_filter["step_type"]:
                    matches_filter = False

            # Filter by time range
            if "time_range" in context_filter:
                start_time, end_time = context_filter["time_range"]
                if not (start_time <= step.timestamp <= end_time):
                    matches_filter = False

            if matches_filter:
                filtered_steps.append(step)

        return filtered_steps

    def _calculate_strategy_performance_metrics(
        self, steps: List[MetaReasoningStep]
    ) -> Dict[str, float]:
        """Calculate performance metrics for strategy steps"""

        if not steps:
            return {}

        metrics = {
            "average_validity": statistics.mean(
                [step.validity_score for step in steps]
            ),
            "average_confidence": statistics.mean([step.confidence for step in steps]),
            "average_novelty": statistics.mean([step.novelty_score for step in steps]),
            "success_rate": len([step for step in steps if step.success_indicators])
            / len(steps),
            "failure_rate": len([step for step in steps if step.failure_indicators])
            / len(steps),
            "learning_rate": sum(len(step.lessons_learned) for step in steps)
            / len(steps),
        }

        return metrics

    def _identify_effectiveness_patterns(
        self, steps: List[MetaReasoningStep]
    ) -> List[str]:
        """Identify effectiveness patterns in strategy usage"""

        patterns = []

        # High effectiveness patterns
        high_validity_steps = [step for step in steps if step.validity_score > 0.8]
        if len(high_validity_steps) > len(steps) * 0.6:
            patterns.append("Consistently high validity scores")

        # Learning patterns
        learning_steps = [step for step in steps if len(step.lessons_learned) > 2]
        if len(learning_steps) > len(steps) * 0.4:
            patterns.append("High learning productivity")

        # Confidence patterns
        confidence_scores = [step.confidence for step in steps]
        if statistics.stdev(confidence_scores) < 0.2:
            patterns.append("Stable confidence levels")

        return patterns

    def _identify_optimal_contexts(self, steps: List[MetaReasoningStep]) -> List[str]:
        """Identify optimal contexts for strategy usage"""

        optimal_contexts = []

        # Analyze step types where strategy performs well
        step_type_performance = defaultdict(list)
        for step in steps:
            step_type_performance[step.step_type].append(step.validity_score)

        for step_type, scores in step_type_performance.items():
            if statistics.mean(scores) > 0.8:
                optimal_contexts.append(f"Effective for {step_type} steps")

        # Analyze reasoning levels where strategy performs well
        level_performance = defaultdict(list)
        for step in steps:
            level_performance[step.reasoning_level].append(step.validity_score)

        for level, scores in level_performance.items():
            if statistics.mean(scores) > 0.8:
                optimal_contexts.append(f"Effective at {level.value} reasoning")

        return optimal_contexts

    def _identify_limitation_patterns(
        self, steps: List[MetaReasoningStep]
    ) -> List[str]:
        """Identify limitation patterns in strategy usage"""

        limitations = []

        # Low performance patterns
        low_validity_steps = [step for step in steps if step.validity_score < 0.5]
        if len(low_validity_steps) > len(steps) * 0.3:
            limitations.append("Frequent low validity scores")

        # Failure patterns
        failure_steps = [step for step in steps if step.failure_indicators]
        if len(failure_steps) > len(steps) * 0.2:
            limitations.append("High failure rate")

        # Confidence issues
        low_confidence_steps = [step for step in steps if step.confidence < 0.5]
        if len(low_confidence_steps) > len(steps) * 0.3:
            limitations.append("Frequent low confidence")

        return limitations

    def _identify_strategy_improvement_opportunities(
        self, steps: List[MetaReasoningStep]
    ) -> List[str]:
        """Identify improvement opportunities for strategy"""

        opportunities = []

        # Analyze failure patterns for improvement
        failure_steps = [step for step in steps if step.failure_indicators]
        if failure_steps:
            common_failures = Counter()
            for step in failure_steps:
                for failure in step.failure_indicators:
                    common_failures[failure] += 1

            for failure, count in common_failures.most_common(3):
                opportunities.append(f"Address recurring failure: {failure}")

        # Analyze low-novelty patterns
        low_novelty_steps = [step for step in steps if step.novelty_score < 0.3]
        if len(low_novelty_steps) > len(steps) * 0.5:
            opportunities.append("Increase creative and innovative approaches")

        # Analyze learning gaps
        no_learning_steps = [step for step in steps if not step.lessons_learned]
        if len(no_learning_steps) > len(steps) * 0.4:
            opportunities.append("Improve learning extraction from experiences")

        return opportunities

    def _compare_strategy_with_others(
        self, target_strategy: CognitiveStrategy, target_steps: List[MetaReasoningStep]
    ) -> Dict[str, Any]:
        """Compare strategy performance with other strategies"""

        target_metrics = self._calculate_strategy_performance_metrics(target_steps)

        comparison = {
            "target_strategy": target_strategy.value,
            "target_performance": target_metrics,
            "comparison_with_others": {},
        }

        for strategy in CognitiveStrategy:
            if strategy == target_strategy:
                continue

            strategy_steps = [
                step
                for step in self.meta_reasoning_history
                if step.cognitive_strategy == strategy
            ]

            if strategy_steps:
                other_metrics = self._calculate_strategy_performance_metrics(
                    strategy_steps
                )

                performance_diff = {}
                for metric, target_value in target_metrics.items():
                    other_value = other_metrics.get(metric, 0.0)
                    performance_diff[metric] = target_value - other_value

                comparison["comparison_with_others"][strategy.value] = {
                    "performance_difference": performance_diff,
                    "relative_ranking": self._calculate_relative_ranking(
                        target_metrics, other_metrics
                    ),
                }

        return comparison

    def _calculate_relative_ranking(
        self, target_metrics: Dict[str, float], other_metrics: Dict[str, float]
    ) -> str:
        """Calculate relative ranking between two sets of metrics"""

        target_score = sum(target_metrics.values())
        other_score = sum(other_metrics.values())

        if target_score > other_score * 1.1:
            return "significantly_better"
        elif target_score > other_score * 1.05:
            return "better"
        elif target_score < other_score * 0.9:
            return "significantly_worse"
        elif target_score < other_score * 0.95:
            return "worse"
        else:
            return "comparable"

    # Additional analysis methods continue...
    # [The remaining methods would follow similar patterns for coherence evaluation,
    # adaptation, pattern insights, etc. - implementing the complete meta-reasoning framework]
