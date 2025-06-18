"""
Reasoning Trace Logger for ModelSEEDagent

Captures and logs step-by-step AI reasoning decisions with full transparency
and traceability for scientific analysis validation.
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of decisions that can be tracked"""

    TOOL_SELECTION = "tool_selection"
    ANALYSIS_APPROACH = "analysis_approach"
    RESULT_INTERPRETATION = "result_interpretation"
    HYPOTHESIS_FORMATION = "hypothesis_formation"
    WORKFLOW_ADAPTATION = "workflow_adaptation"
    SYNTHESIS = "synthesis"
    QUALITY_ASSESSMENT = "quality_assessment"
    TERMINATION = "termination"


class ConfidenceLevel(Enum):
    """Confidence levels for decisions"""

    VERY_LOW = 0.2
    LOW = 0.4
    MODERATE = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class DecisionPoint:
    """A single decision point in the reasoning trace"""

    decision_id: str
    decision_type: DecisionType
    timestamp: datetime
    context: Dict[str, Any]
    reasoning: str
    chosen_option: Any
    alternatives_considered: List[Any]
    confidence: float
    rationale: str
    evidence: List[str]
    assumptions: List[str]
    step_number: int
    parent_decision_id: Optional[str] = None
    outcome_validated: Optional[bool] = None
    validation_reasoning: Optional[str] = None


@dataclass
class HypothesisTrace:
    """Track hypothesis formation and testing"""

    hypothesis_id: str
    statement: str
    rationale: str
    predictions: List[str]
    evidence_for: List[str]
    evidence_against: List[str]
    confidence: float
    status: str  # "formed", "testing", "supported", "refuted", "inconclusive"
    related_decisions: List[str]


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a session"""

    session_id: str
    query: str
    start_time: datetime
    end_time: Optional[datetime]
    decision_points: List[DecisionPoint]
    hypotheses: List[HypothesisTrace]
    synthesis_reasoning: Optional[str]
    final_conclusions: List[str]
    confidence_in_conclusions: float
    metadata: Dict[str, Any]


class ReasoningTraceLogger:
    """Captures and logs step-by-step AI reasoning"""

    def __init__(
        self, session_id: Optional[str] = None, trace_dir: Optional[Path] = None
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.trace_dir = (
            trace_dir or Path.home() / ".modelseed-agent" / "reasoning_traces"
        )
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        # Initialize trace
        self.trace = ReasoningTrace(
            session_id=self.session_id,
            query="",
            start_time=datetime.now(),
            end_time=None,
            decision_points=[],
            hypotheses=[],
            synthesis_reasoning=None,
            final_conclusions=[],
            confidence_in_conclusions=0.0,
            metadata={},
        )

        self.current_step = 0
        self.decision_stack: List[str] = []  # Track decision hierarchy

        logger.info(f"Started reasoning trace session: {self.session_id}")

    def set_query(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Set the initial query and metadata for this reasoning session"""
        self.trace.query = query
        if metadata:
            self.trace.metadata.update(metadata)

        logger.info(f"Set query for session {self.session_id}: {query[:100]}...")

    def log_decision(
        self,
        decision_type: DecisionType,
        reasoning: str,
        chosen_option: Any,
        alternatives_considered: List[Any],
        confidence: Union[float, ConfidenceLevel],
        context: Optional[Dict[str, Any]] = None,
        evidence: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
        parent_decision_id: Optional[str] = None,
    ) -> str:
        """
        Log a reasoning decision with full context and rationale

        Returns:
            The decision_id for this decision point
        """
        decision_id = str(uuid.uuid4())
        self.current_step += 1

        # Handle confidence level conversion
        if isinstance(confidence, ConfidenceLevel):
            confidence_value = confidence.value
        else:
            confidence_value = max(0.0, min(1.0, confidence))

        # Validate reasoning quality
        if len(reasoning.strip()) < 20:
            logger.warning(
                f"Short reasoning provided for decision {decision_id}: {reasoning}"
            )

        decision = DecisionPoint(
            decision_id=decision_id,
            decision_type=decision_type,
            timestamp=datetime.now(),
            context=context or {},
            reasoning=reasoning,
            chosen_option=chosen_option,
            alternatives_considered=alternatives_considered,
            confidence=confidence_value,
            rationale=reasoning,  # Same as reasoning for now, could be different
            evidence=evidence or [],
            assumptions=assumptions or [],
            step_number=self.current_step,
            parent_decision_id=parent_decision_id,
        )

        self.trace.decision_points.append(decision)

        # Update decision stack
        if parent_decision_id:
            # This is a sub-decision
            pass
        else:
            # This is a top-level decision
            self.decision_stack.append(decision_id)

        logger.info(f"Logged {decision_type.value} decision: {chosen_option}")
        self._save_trace()

        return decision_id

    def log_tool_selection(
        self,
        query: str,
        selected_tool: str,
        reasoning: str,
        available_tools: List[str],
        confidence: Union[float, ConfidenceLevel],
        tool_rationale: Optional[str] = None,
        previous_results: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log tool selection reasoning with domain-specific context"""
        context = {
            "query": query,
            "available_tools": available_tools,
            "previous_results": previous_results or {},
            "selection_criteria": self._extract_selection_criteria(reasoning),
        }

        evidence = []
        if tool_rationale:
            evidence.append(f"Tool rationale: {tool_rationale}")
        if previous_results:
            evidence.append(
                f"Previous results inform selection: {len(previous_results)} prior tool executions"
            )

        return self.log_decision(
            decision_type=DecisionType.TOOL_SELECTION,
            reasoning=reasoning,
            chosen_option=selected_tool,
            alternatives_considered=available_tools,
            confidence=confidence,
            context=context,
            evidence=evidence,
        )

    def log_hypothesis_formation(
        self,
        statement: str,
        rationale: str,
        predictions: List[str],
        confidence: Union[float, ConfidenceLevel],
        evidence_basis: List[str],
        related_decision_id: Optional[str] = None,
    ) -> str:
        """Log formation of a scientific hypothesis"""
        hypothesis_id = str(uuid.uuid4())

        hypothesis = HypothesisTrace(
            hypothesis_id=hypothesis_id,
            statement=statement,
            rationale=rationale,
            predictions=predictions,
            evidence_for=evidence_basis,
            evidence_against=[],
            confidence=(
                confidence.value
                if isinstance(confidence, ConfidenceLevel)
                else confidence
            ),
            status="formed",
            related_decisions=[related_decision_id] if related_decision_id else [],
        )

        self.trace.hypotheses.append(hypothesis)

        # Also log as a decision
        self.log_decision(
            decision_type=DecisionType.HYPOTHESIS_FORMATION,
            reasoning=f"Hypothesis formation: {rationale}",
            chosen_option=statement,
            alternatives_considered=[],  # Could include alternative hypotheses
            confidence=confidence,
            context={"predictions": predictions, "hypothesis_id": hypothesis_id},
            evidence=evidence_basis,
        )

        logger.info(f"Formed hypothesis {hypothesis_id}: {statement[:100]}...")
        return hypothesis_id

    def update_hypothesis_evidence(
        self,
        hypothesis_id: str,
        new_evidence_for: Optional[List[str]] = None,
        new_evidence_against: Optional[List[str]] = None,
        updated_confidence: Optional[float] = None,
        status: Optional[str] = None,
    ) -> None:
        """Update evidence for a hypothesis as analysis progresses"""
        for hypothesis in self.trace.hypotheses:
            if hypothesis.hypothesis_id == hypothesis_id:
                if new_evidence_for:
                    hypothesis.evidence_for.extend(new_evidence_for)
                if new_evidence_against:
                    hypothesis.evidence_against.extend(new_evidence_against)
                if updated_confidence is not None:
                    hypothesis.confidence = updated_confidence
                if status:
                    hypothesis.status = status
                break

        self._save_trace()

    def log_result_interpretation(
        self,
        tool_name: str,
        raw_result: Dict[str, Any],
        extracted_insights: List[str],
        reasoning: str,
        confidence: Union[float, ConfidenceLevel],
        biological_significance: Optional[str] = None,
    ) -> str:
        """Log interpretation of tool results"""
        context = {
            "tool_name": tool_name,
            "result_summary": self._summarize_result(raw_result),
            "insights_count": len(extracted_insights),
            "biological_significance": biological_significance,
        }

        evidence = [
            f"Insight {i+1}: {insight}" for i, insight in enumerate(extracted_insights)
        ]

        return self.log_decision(
            decision_type=DecisionType.RESULT_INTERPRETATION,
            reasoning=reasoning,
            chosen_option=extracted_insights,
            alternatives_considered=[],  # Could include alternative interpretations
            confidence=confidence,
            context=context,
            evidence=evidence,
        )

    def log_workflow_adaptation(
        self,
        current_plan: List[str],
        adaptation_reasoning: str,
        new_plan: List[str],
        confidence: Union[float, ConfidenceLevel],
        trigger_event: str,
    ) -> str:
        """Log adaptation of analysis workflow based on results"""
        context = {
            "current_plan": current_plan,
            "new_plan": new_plan,
            "trigger_event": trigger_event,
            "plan_changes": self._compare_plans(current_plan, new_plan),
        }

        return self.log_decision(
            decision_type=DecisionType.WORKFLOW_ADAPTATION,
            reasoning=adaptation_reasoning,
            chosen_option=new_plan,
            alternatives_considered=[current_plan],
            confidence=confidence,
            context=context,
            evidence=[f"Triggered by: {trigger_event}"],
        )

    def log_synthesis_reasoning(
        self,
        synthesis_approach: str,
        integrated_findings: List[str],
        confidence: Union[float, ConfidenceLevel],
        cross_tool_connections: Dict[str, List[str]],
    ) -> str:
        """Log cross-tool synthesis and integration reasoning"""
        context = {
            "synthesis_approach": synthesis_approach,
            "findings_count": len(integrated_findings),
            "tools_integrated": list(cross_tool_connections.keys()),
            "connection_count": sum(
                len(conns) for conns in cross_tool_connections.values()
            ),
        }

        evidence = []
        for tool, connections in cross_tool_connections.items():
            evidence.extend([f"{tool} → {conn}" for conn in connections])

        decision_id = self.log_decision(
            decision_type=DecisionType.SYNTHESIS,
            reasoning=synthesis_approach,
            chosen_option=integrated_findings,
            alternatives_considered=[],
            confidence=confidence,
            context=context,
            evidence=evidence,
        )

        # Store synthesis reasoning in trace
        self.trace.synthesis_reasoning = synthesis_approach

        return decision_id

    def validate_decision_outcome(
        self, decision_id: str, outcome_valid: bool, validation_reasoning: str
    ) -> None:
        """Validate whether a previous decision led to expected outcomes"""
        for decision in self.trace.decision_points:
            if decision.decision_id == decision_id:
                decision.outcome_validated = outcome_valid
                decision.validation_reasoning = validation_reasoning

                if not outcome_valid:
                    logger.warning(
                        f"Decision {decision_id} outcome invalid: {validation_reasoning}"
                    )

                break

        self._save_trace()

    def finalize_trace(
        self,
        final_conclusions: List[str],
        confidence_in_conclusions: float,
        summary_reasoning: Optional[str] = None,
    ) -> str:
        """Finalize the reasoning trace with conclusions"""
        self.trace.end_time = datetime.now()
        self.trace.final_conclusions = final_conclusions
        self.trace.confidence_in_conclusions = confidence_in_conclusions

        if summary_reasoning:
            self.trace.synthesis_reasoning = summary_reasoning

        # Log final decision
        final_decision_id = self.log_decision(
            decision_type=DecisionType.TERMINATION,
            reasoning=summary_reasoning or "Analysis complete",
            chosen_option=final_conclusions,
            alternatives_considered=[],
            confidence=confidence_in_conclusions,
        )

        # Save final trace
        self._save_trace(final=True)

        duration = (self.trace.end_time - self.trace.start_time).total_seconds()
        logger.info(
            f"Finalized reasoning trace {self.session_id} - Duration: {duration:.1f}s, Decisions: {len(self.trace.decision_points)}"
        )

        return final_decision_id

    def get_decision_chain(self, decision_id: str) -> List[DecisionPoint]:
        """Get the chain of decisions leading to a specific decision"""
        chain = []
        current_id = decision_id

        while current_id:
            decision = next(
                (d for d in self.trace.decision_points if d.decision_id == current_id),
                None,
            )
            if decision:
                chain.insert(0, decision)
                current_id = decision.parent_decision_id
            else:
                break

        return chain

    def get_trace_summary(self) -> Dict[str, Any]:
        """Get a summary of the reasoning trace"""
        total_decisions = len(self.trace.decision_points)
        decision_types = {}

        for decision in self.trace.decision_points:
            dt = decision.decision_type.value
            decision_types[dt] = decision_types.get(dt, 0) + 1

        avg_confidence = 0.0
        if total_decisions > 0:
            avg_confidence = (
                sum(d.confidence for d in self.trace.decision_points) / total_decisions
            )

        duration = None
        if self.trace.end_time:
            duration = (self.trace.end_time - self.trace.start_time).total_seconds()

        return {
            "session_id": self.session_id,
            "query": self.trace.query,
            "total_decisions": total_decisions,
            "decision_types": decision_types,
            "hypotheses_formed": len(self.trace.hypotheses),
            "avg_confidence": avg_confidence,
            "duration_seconds": duration,
            "finalized": self.trace.end_time is not None,
            "final_conclusions_count": len(self.trace.final_conclusions),
        }

    def export_trace(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """Export the complete reasoning trace"""
        if format == "json":
            return json.dumps(asdict(self.trace), indent=2, default=str)
        elif format == "dict":
            return asdict(self.trace)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _extract_selection_criteria(self, reasoning: str) -> List[str]:
        """Extract selection criteria from reasoning text"""
        criteria = []

        # Simple pattern matching for common criteria
        if "growth" in reasoning.lower():
            criteria.append("growth_analysis")
        if "essential" in reasoning.lower():
            criteria.append("essentiality_analysis")
        if "media" in reasoning.lower():
            criteria.append("media_analysis")
        if "comprehensive" in reasoning.lower():
            criteria.append("comprehensive_coverage")

        return criteria

    def _summarize_result(self, result: Dict[str, Any]) -> str:
        """Create a brief summary of a tool result"""
        if not result:
            return "Empty result"

        summary_parts = []
        for key, value in list(result.items())[:3]:  # First 3 items
            if isinstance(value, (int, float)):
                summary_parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                summary_parts.append(f"{key}: {len(value)} items")
            elif isinstance(value, dict):
                summary_parts.append(f"{key}: {len(value)} entries")
            else:
                summary_parts.append(f"{key}: {str(value)[:50]}...")

        return "; ".join(summary_parts)

    def _compare_plans(
        self, old_plan: List[str], new_plan: List[str]
    ) -> Dict[str, Any]:
        """Compare two analysis plans and identify changes"""
        return {
            "added_steps": [step for step in new_plan if step not in old_plan],
            "removed_steps": [step for step in old_plan if step not in new_plan],
            "reordered": old_plan != new_plan,
            "length_change": len(new_plan) - len(old_plan),
        }

    def _save_trace(self, final: bool = False) -> None:
        """Save the current trace to disk"""
        try:
            suffix = "_final" if final else ""
            filename = f"trace_{self.session_id}{suffix}.json"
            filepath = self.trace_dir / filename

            with open(filepath, "w") as f:
                json.dump(asdict(self.trace), f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save reasoning trace: {e}")


def load_reasoning_trace(
    session_id: str, trace_dir: Optional[Path] = None
) -> Optional[ReasoningTrace]:
    """Load a reasoning trace from disk"""
    trace_dir = trace_dir or Path.home() / ".modelseed-agent" / "reasoning_traces"

    # Try final trace first, then regular trace
    for suffix in ["_final", ""]:
        filepath = trace_dir / f"trace_{session_id}{suffix}.json"
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                # Convert datetime strings back to datetime objects
                data["start_time"] = datetime.fromisoformat(data["start_time"])
                if data["end_time"]:
                    data["end_time"] = datetime.fromisoformat(data["end_time"])

                # Convert decision points
                decision_points = []
                for dp_data in data["decision_points"]:
                    # Handle enum conversion safely
                    if isinstance(dp_data["decision_type"], str):
                        try:
                            dp_data["decision_type"] = DecisionType(
                                dp_data["decision_type"]
                            )
                        except ValueError:
                            # Handle legacy enum values
                            dp_data["decision_type"] = DecisionType.TOOL_SELECTION
                    dp_data["timestamp"] = datetime.fromisoformat(dp_data["timestamp"])
                    decision_points.append(DecisionPoint(**dp_data))

                data["decision_points"] = decision_points

                # Convert hypotheses
                hypotheses = []
                for h_data in data.get("hypotheses", []):
                    hypotheses.append(HypothesisTrace(**h_data))
                data["hypotheses"] = hypotheses

                return ReasoningTrace(**data)

            except Exception as e:
                logger.error(f"Failed to load trace {session_id}: {e}")

    return None
