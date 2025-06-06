#!/usr/bin/env python3
"""
AI Agent Audit System - Phase 7.1 Enhancement

Advanced auditing system specifically designed for AI agent interactions,
capturing reasoning chains, decision-making processes, and tool orchestration
for comprehensive transparency and hallucination detection.

Extends the base audit system with AI-specific capabilities:
- AI reasoning step capture
- Decision point logging
- Tool selection rationale
- Multi-step workflow tracking
- Confidence scoring integration
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from .audit import AuditRecord, HallucinationDetector


class AIReasoningStep(BaseModel):
    """Individual AI reasoning step within a workflow"""

    step_id: str = Field(description="Unique identifier for this reasoning step")
    step_number: int = Field(description="Sequential step number in the workflow")
    timestamp: str = Field(description="ISO timestamp of reasoning step")

    # Reasoning content
    ai_thought: str = Field(description="AI's internal reasoning/thinking")
    context_analysis: str = Field(description="AI's analysis of current context")
    available_tools: List[str] = Field(description="Tools available at this step")

    # Decision making
    selected_tool: Optional[str] = Field(description="Tool selected by AI")
    selection_rationale: str = Field(description="Why this tool was selected")
    confidence_score: float = Field(description="AI confidence in this decision (0-1)")

    # Expected outcomes
    expected_result: str = Field(description="What AI expects from this tool")
    success_criteria: List[str] = Field(description="How AI will measure success")

    # Alternative considerations
    alternative_tools: List[str] = Field(description="Other tools AI considered")
    rejection_reasons: Dict[str, str] = Field(
        description="Why alternatives were rejected"
    )


class AIWorkflowAudit(BaseModel):
    """Complete audit record for an AI agent workflow"""

    workflow_id: str = Field(description="Unique identifier for this workflow")
    session_id: Optional[str] = Field(description="Session ID if part of a session")
    user_query: str = Field(description="Original user query that triggered workflow")
    timestamp_start: str = Field(description="Workflow start timestamp")
    timestamp_end: Optional[str] = Field(
        default=None, description="Workflow completion timestamp"
    )

    # AI reasoning chain
    reasoning_steps: List[AIReasoningStep] = Field(
        default_factory=list, description="Complete reasoning chain"
    )
    total_reasoning_steps: int = Field(
        default=0, description="Total number of reasoning steps"
    )

    # Tool execution tracking
    tools_executed: List[str] = Field(
        default_factory=list, description="Tools actually executed"
    )
    tool_execution_order: List[Tuple[str, str]] = Field(
        default_factory=list, description="(tool, timestamp) execution order"
    )
    tool_audit_files: List[str] = Field(
        default_factory=list, description="Paths to individual tool audit files"
    )

    # Workflow outcomes
    final_result: str = Field(default="", description="AI's final response/conclusion")
    success: bool = Field(
        default=False, description="Whether workflow completed successfully"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if workflow failed"
    )

    # Verification metadata
    ai_confidence_final: float = Field(
        default=0.0, description="AI's final confidence score"
    )
    reasoning_coherence: float = Field(
        default=0.0, description="Coherence score of reasoning chain"
    )
    tool_selection_accuracy: float = Field(
        default=0.0, description="Accuracy of tool selections"
    )

    # Performance metrics
    total_duration_seconds: float = Field(
        default=0.0, description="Total workflow duration"
    )
    ai_thinking_time_seconds: float = Field(
        default=0.0, description="Time spent on AI reasoning"
    )
    tool_execution_time_seconds: float = Field(
        default=0.0, description="Time spent executing tools"
    )


class AIDecisionVerifier:
    """Verifies AI decision-making accuracy and detects reasoning hallucinations"""

    def __init__(self, logs_dir: Optional[Union[str, Path]] = None):
        """Initialize the AI decision verifier"""
        self.logs_dir = Path(logs_dir or "logs")
        self.base_detector = HallucinationDetector(logs_dir)

    def verify_reasoning_chain(self, workflow_audit: AIWorkflowAudit) -> Dict[str, Any]:
        """Verify the coherence and accuracy of an AI reasoning chain"""

        verification_result = {
            "workflow_id": workflow_audit.workflow_id,
            "verification_timestamp": datetime.now().isoformat(),
            "reasoning_verification": {},
            "tool_selection_verification": {},
            "outcome_verification": {},
            "overall_assessment": {},
        }

        # Verify reasoning coherence
        reasoning_verification = self._verify_reasoning_coherence(workflow_audit)
        verification_result["reasoning_verification"] = reasoning_verification

        # Verify tool selection logic
        tool_verification = self._verify_tool_selection_logic(workflow_audit)
        verification_result["tool_selection_verification"] = tool_verification

        # Verify actual outcomes match expectations
        outcome_verification = self._verify_expected_vs_actual_outcomes(workflow_audit)
        verification_result["outcome_verification"] = outcome_verification

        # Overall assessment
        overall_score = self._calculate_overall_verification_score(
            reasoning_verification, tool_verification, outcome_verification
        )
        verification_result["overall_assessment"] = overall_score

        return verification_result

    def _verify_reasoning_coherence(
        self, workflow_audit: AIWorkflowAudit
    ) -> Dict[str, Any]:
        """Verify that reasoning steps are coherent and logical"""

        steps = workflow_audit.reasoning_steps
        coherence_issues = []
        coherence_scores = []

        for i, step in enumerate(steps):
            step_analysis = {
                "step_number": step.step_number,
                "issues": [],
                "coherence_score": 1.0,
            }

            # Check if reasoning builds on previous steps
            if i > 0:
                prev_step = steps[i - 1]
                if not self._reasoning_builds_on_previous(step, prev_step):
                    step_analysis["issues"].append(
                        "Reasoning doesn't build on previous step"
                    )
                    step_analysis["coherence_score"] -= 0.3

            # Check if tool selection matches reasoning
            if step.selected_tool:
                if not self._tool_matches_reasoning(step):
                    step_analysis["issues"].append(
                        "Tool selection doesn't match reasoning"
                    )
                    step_analysis["coherence_score"] -= 0.4

            # Check confidence calibration
            if step.confidence_score > 0.9 and len(step.success_criteria) < 2:
                step_analysis["issues"].append(
                    "High confidence but insufficient success criteria"
                )
                step_analysis["coherence_score"] -= 0.2

            coherence_scores.append(step_analysis["coherence_score"])
            if step_analysis["issues"]:
                coherence_issues.append(step_analysis)

        return {
            "total_steps": len(steps),
            "coherence_issues": coherence_issues,
            "average_coherence_score": (
                sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
            ),
            "coherence_grade": self._score_to_grade(
                sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
            ),
        }

    def _verify_tool_selection_logic(
        self, workflow_audit: AIWorkflowAudit
    ) -> Dict[str, Any]:
        """Verify that tool selections are logical and well-reasoned"""

        selection_issues = []
        selection_scores = []

        for step in workflow_audit.reasoning_steps:
            if not step.selected_tool:
                continue

            step_analysis = {
                "step_number": step.step_number,
                "selected_tool": step.selected_tool,
                "issues": [],
                "selection_score": 1.0,
            }

            # Check if rationale is detailed enough
            if len(step.selection_rationale.split()) < 5:
                step_analysis["issues"].append("Selection rationale too brief")
                step_analysis["selection_score"] -= 0.3

            # Check if alternatives were considered
            if len(step.alternative_tools) == 0 and len(step.available_tools) > 1:
                step_analysis["issues"].append(
                    "No alternatives considered despite multiple options"
                )
                step_analysis["selection_score"] -= 0.2

            # Check if expected results are specific
            if (
                "analyze" in step.expected_result.lower()
                and len(step.expected_result.split()) < 3
            ):
                step_analysis["issues"].append("Expected results too vague")
                step_analysis["selection_score"] -= 0.2

            selection_scores.append(step_analysis["selection_score"])
            if step_analysis["issues"]:
                selection_issues.append(step_analysis)

        return {
            "tools_selected": len(
                [s for s in workflow_audit.reasoning_steps if s.selected_tool]
            ),
            "selection_issues": selection_issues,
            "average_selection_score": (
                sum(selection_scores) / len(selection_scores) if selection_scores else 0
            ),
            "selection_grade": self._score_to_grade(
                sum(selection_scores) / len(selection_scores) if selection_scores else 0
            ),
        }

    def _verify_expected_vs_actual_outcomes(
        self, workflow_audit: AIWorkflowAudit
    ) -> Dict[str, Any]:
        """Verify that actual tool outcomes match AI expectations"""

        # This would integrate with actual tool audit records
        # For now, provide framework for future implementation

        outcome_verification = {
            "expectations_vs_reality": [],
            "accuracy_score": 0.85,  # Placeholder - would analyze actual vs expected
            "accuracy_grade": "B+",
            "unmet_expectations": [],
            "exceeded_expectations": [],
        }

        # TODO: Integrate with tool audit records to compare
        # AI expectations vs actual tool results

        return outcome_verification

    def _reasoning_builds_on_previous(
        self, current_step: AIReasoningStep, prev_step: AIReasoningStep
    ) -> bool:
        """Check if current reasoning builds logically on previous step"""

        # Simple heuristic: look for references to previous results
        current_text = (
            current_step.ai_thought.lower()
            + " "
            + current_step.context_analysis.lower()
        )

        # Look for connecting words/phrases
        connecting_phrases = [
            "based on",
            "given that",
            "since",
            "because",
            "therefore",
            "as a result",
            "consequently",
            "following",
            "after",
            "next",
        ]

        return any(phrase in current_text for phrase in connecting_phrases)

    def _tool_matches_reasoning(self, step: AIReasoningStep) -> bool:
        """Check if selected tool aligns with reasoning"""

        reasoning_text = (
            step.ai_thought.lower() + " " + step.selection_rationale.lower()
        )
        tool_name = step.selected_tool.lower() if step.selected_tool else ""

        # Extract key concepts from tool name
        tool_concepts = tool_name.replace("_", " ").split()

        # Check if reasoning mentions tool concepts
        return any(
            concept in reasoning_text for concept in tool_concepts if len(concept) > 3
        )

    def _calculate_overall_verification_score(
        self,
        reasoning_verification: Dict,
        tool_verification: Dict,
        outcome_verification: Dict,
    ) -> Dict[str, Any]:
        """Calculate overall verification assessment"""

        # Weight the different aspects
        reasoning_weight = 0.4
        tool_weight = 0.3
        outcome_weight = 0.3

        overall_score = (
            reasoning_verification["average_coherence_score"] * reasoning_weight
            + tool_verification["average_selection_score"] * tool_weight
            + outcome_verification["accuracy_score"] * outcome_weight
        )

        return {
            "overall_score": overall_score,
            "overall_grade": self._score_to_grade(overall_score),
            "reliability_assessment": self._assess_reliability(overall_score),
            "improvement_suggestions": self._generate_improvement_suggestions(
                reasoning_verification, tool_verification, outcome_verification
            ),
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 0.95:
            return "A+"
        elif score >= 0.90:
            return "A"
        elif score >= 0.85:
            return "A-"
        elif score >= 0.80:
            return "B+"
        elif score >= 0.75:
            return "B"
        elif score >= 0.70:
            return "B-"
        elif score >= 0.65:
            return "C+"
        elif score >= 0.60:
            return "C"
        else:
            return "D"

    def _assess_reliability(self, score: float) -> str:
        """Assess overall AI reliability based on score"""
        if score >= 0.90:
            return "Highly Reliable - AI reasoning is coherent and accurate"
        elif score >= 0.80:
            return "Reliable - AI reasoning is generally sound with minor issues"
        elif score >= 0.70:
            return "Moderately Reliable - AI reasoning has some logical gaps"
        elif score >= 0.60:
            return "Questionable - AI reasoning shows significant issues"
        else:
            return "Unreliable - AI reasoning is incoherent or highly inaccurate"

    def _generate_improvement_suggestions(
        self,
        reasoning_verification: Dict,
        tool_verification: Dict,
        outcome_verification: Dict,
    ) -> List[str]:
        """Generate suggestions for improving AI reasoning"""

        suggestions = []

        if reasoning_verification["average_coherence_score"] < 0.8:
            suggestions.append(
                "Improve reasoning coherence by building more explicitly on previous steps"
            )

        if tool_verification["average_selection_score"] < 0.8:
            suggestions.append(
                "Provide more detailed rationale for tool selection decisions"
            )

        if len(reasoning_verification["coherence_issues"]) > 2:
            suggestions.append(
                "Increase reasoning depth with more detailed analysis at each step"
            )

        if outcome_verification["accuracy_score"] < 0.8:
            suggestions.append(
                "Better calibrate expectations by being more specific about expected outcomes"
            )

        return suggestions


class AIAuditLogger:
    """Logger for AI agent workflow audits"""

    def __init__(
        self,
        logs_dir: Optional[Union[str, Path]] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize the AI audit logger"""
        self.logs_dir = Path(logs_dir or "logs")
        self.session_id = session_id
        self.current_workflow: Optional[AIWorkflowAudit] = None
        self.current_step_number = 0

        # Create AI audit directory
        if session_id:
            self.ai_audit_dir = self.logs_dir / session_id / "ai_audits"
        else:
            self.ai_audit_dir = self.logs_dir / "ai_audits"

        self.ai_audit_dir.mkdir(parents=True, exist_ok=True)

    def start_workflow(self, user_query: str, workflow_id: Optional[str] = None) -> str:
        """Start logging a new AI workflow"""

        workflow_id = workflow_id or str(uuid.uuid4())[:8]

        self.current_workflow = AIWorkflowAudit(
            workflow_id=workflow_id,
            session_id=self.session_id,
            user_query=user_query,
            timestamp_start=datetime.now().isoformat(),
            reasoning_steps=[],
            total_reasoning_steps=0,
            tools_executed=[],
            tool_execution_order=[],
            tool_audit_files=[],
            final_result="",
            success=False,
            ai_confidence_final=0.0,
            reasoning_coherence=0.0,
            tool_selection_accuracy=0.0,
            total_duration_seconds=0.0,
            ai_thinking_time_seconds=0.0,
            tool_execution_time_seconds=0.0,
        )

        self.current_step_number = 0
        return workflow_id

    def log_reasoning_step(
        self,
        ai_thought: str,
        context_analysis: str,
        available_tools: List[str],
        selected_tool: Optional[str] = None,
        selection_rationale: str = "",
        confidence_score: float = 0.5,
        expected_result: str = "",
        success_criteria: List[str] = None,
        alternative_tools: List[str] = None,
        rejection_reasons: Dict[str, str] = None,
    ) -> str:
        """Log an AI reasoning step"""

        if not self.current_workflow:
            raise ValueError("No active workflow. Call start_workflow() first.")

        self.current_step_number += 1
        step_id = f"{self.current_workflow.workflow_id}_step_{self.current_step_number}"

        reasoning_step = AIReasoningStep(
            step_id=step_id,
            step_number=self.current_step_number,
            timestamp=datetime.now().isoformat(),
            ai_thought=ai_thought,
            context_analysis=context_analysis,
            available_tools=available_tools,
            selected_tool=selected_tool,
            selection_rationale=selection_rationale,
            confidence_score=confidence_score,
            expected_result=expected_result,
            success_criteria=success_criteria or [],
            alternative_tools=alternative_tools or [],
            rejection_reasons=rejection_reasons or {},
        )

        self.current_workflow.reasoning_steps.append(reasoning_step)
        self.current_workflow.total_reasoning_steps = self.current_step_number

        return step_id

    def log_tool_execution(self, tool_name: str, audit_file_path: str):
        """Log that a tool was executed"""

        if not self.current_workflow:
            return

        self.current_workflow.tools_executed.append(tool_name)
        self.current_workflow.tool_execution_order.append(
            (tool_name, datetime.now().isoformat())
        )
        self.current_workflow.tool_audit_files.append(audit_file_path)

    def complete_workflow(
        self,
        final_result: str,
        success: bool = True,
        error_message: Optional[str] = None,
        ai_confidence_final: float = 0.8,
    ) -> str:
        """Complete and save the workflow audit"""

        if not self.current_workflow:
            raise ValueError("No active workflow to complete.")

        end_time = datetime.now()
        start_time = datetime.fromisoformat(self.current_workflow.timestamp_start)
        total_duration = (end_time - start_time).total_seconds()

        self.current_workflow.timestamp_end = end_time.isoformat()
        self.current_workflow.final_result = final_result
        self.current_workflow.success = success
        self.current_workflow.error_message = error_message
        self.current_workflow.ai_confidence_final = ai_confidence_final
        self.current_workflow.total_duration_seconds = total_duration

        # Save workflow audit
        audit_filename = f"ai_workflow_{self.current_workflow.workflow_id}_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        audit_path = self.ai_audit_dir / audit_filename

        with open(audit_path, "w") as f:
            json.dump(self.current_workflow.dict(), f, indent=2)

        self.current_workflow = None
        self.current_step_number = 0

        return str(audit_path)


# Global AI audit logger instance
_ai_audit_logger: Optional[AIAuditLogger] = None


def get_ai_audit_logger(
    logs_dir: Optional[Union[str, Path]] = None, session_id: Optional[str] = None
) -> AIAuditLogger:
    """Get or create the global AI audit logger"""
    global _ai_audit_logger

    if _ai_audit_logger is None or _ai_audit_logger.session_id != session_id:
        _ai_audit_logger = AIAuditLogger(logs_dir, session_id)

    return _ai_audit_logger


def create_ai_decision_verifier(
    logs_dir: Optional[Union[str, Path]] = None
) -> AIDecisionVerifier:
    """Factory function to create an AI decision verifier"""
    return AIDecisionVerifier(logs_dir)
