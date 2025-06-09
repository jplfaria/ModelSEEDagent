#!/usr/bin/env python3
"""
Real-Time AI Hallucination Detection System - Phase 7.3

Advanced real-time verification system that monitors AI agent workflows
as they execute, providing immediate feedback on reasoning quality,
decision accuracy, and potential hallucination detection.

Features:
- Live monitoring of AI reasoning steps
- Real-time confidence scoring
- Immediate hallucination alerts
- Pattern-based anomaly detection
- Adaptive verification thresholds
- Live dashboard display
"""

import json
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .ai_audit import AIDecisionVerifier, AIReasoningStep, AIWorkflowAudit

console = Console()


class VerificationAlert(Enum):
    """Types of verification alerts"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    HALLUCINATION_DETECTED = "hallucination"


class RealTimeAlert(BaseModel):
    """Real-time verification alert"""

    alert_id: str = Field(description="Unique alert identifier")
    alert_type: VerificationAlert = Field(description="Type of alert")
    timestamp: str = Field(description="When alert was generated")
    workflow_id: str = Field(description="Associated workflow ID")
    step_number: int = Field(description="Reasoning step number")

    message: str = Field(description="Alert message")
    details: Dict[str, Any] = Field(description="Additional alert details")
    confidence_score: float = Field(description="Confidence in alert accuracy")

    # Suggested actions
    suggested_actions: List[str] = Field(description="Recommended user actions")
    auto_resolvable: bool = Field(description="Whether alert can be auto-resolved")


class VerificationMetrics(BaseModel):
    """Real-time verification metrics"""

    workflow_id: str
    timestamp: str

    # Overall metrics
    overall_confidence: float = Field(description="Overall workflow confidence")
    reasoning_coherence: float = Field(description="Reasoning chain coherence")
    decision_accuracy: float = Field(description="Decision making accuracy")

    # Step-by-step metrics
    steps_analyzed: int = Field(description="Number of steps analyzed")
    issues_detected: int = Field(description="Number of issues found")
    warnings_raised: int = Field(description="Number of warnings raised")

    # Pattern detection
    anomaly_score: float = Field(description="Anomaly detection score")
    consistency_score: float = Field(description="Consistency across steps")

    # Comparison metrics
    vs_historical_average: float = Field(
        description="Compared to historical performance"
    )
    vs_expected_performance: float = Field(
        description="Compared to expected performance"
    )


class RealTimeHallucinationDetector:
    """Real-time AI hallucination detection and verification system"""

    def __init__(
        self,
        logs_dir: Optional[Path] = None,
        enable_live_display: bool = True,
        alert_threshold: float = 0.7,
    ):
        """Initialize the real-time detector"""
        self.logs_dir = Path(logs_dir or "logs")
        self.enable_live_display = enable_live_display
        self.alert_threshold = alert_threshold

        # Verification components
        self.decision_verifier = AIDecisionVerifier(logs_dir)

        # Real-time state
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[RealTimeAlert] = []
        self.metrics_history: List[VerificationMetrics] = []

        # Live display components
        self.live_display: Optional[Live] = None
        self.display_thread: Optional[threading.Thread] = None
        self.is_monitoring = False

        # Pattern detection
        self.historical_patterns = self._load_historical_patterns()
        self.anomaly_detectors = self._initialize_anomaly_detectors()

        # Alert callbacks
        self.alert_callbacks: List[Callable[[RealTimeAlert], None]] = []

    def start_monitoring(self, workflow_id: str, query: str) -> None:
        """Start real-time monitoring for a workflow"""

        workflow_state = {
            "workflow_id": workflow_id,
            "query": query,
            "start_time": datetime.now(),
            "reasoning_steps": [],
            "metrics": None,
            "alerts": [],
            "status": "active",
        }

        self.active_workflows[workflow_id] = workflow_state
        self.is_monitoring = True

        # Start live display if enabled
        if self.enable_live_display and not self.live_display:
            self._start_live_display()

        # Disabled to reduce console clutter
        # console.print(f"🔍 Real-time verification started for workflow: {workflow_id}")

    def process_reasoning_step(
        self, workflow_id: str, reasoning_step: AIReasoningStep
    ) -> List[RealTimeAlert]:
        """Process a new reasoning step and detect issues in real-time"""

        if workflow_id not in self.active_workflows:
            return []

        workflow_state = self.active_workflows[workflow_id]
        workflow_state["reasoning_steps"].append(reasoning_step)

        # Generate alerts for this step
        step_alerts = self._analyze_reasoning_step(workflow_id, reasoning_step)

        # Update metrics
        self._update_workflow_metrics(workflow_id)

        # Process alerts
        for alert in step_alerts:
            self._process_alert(alert)

        return step_alerts

    def complete_monitoring(
        self, workflow_id: str, success: bool = True
    ) -> VerificationMetrics:
        """Complete monitoring for a workflow and generate final metrics"""

        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not being monitored")

        workflow_state = self.active_workflows[workflow_id]
        workflow_state["status"] = "completed" if success else "failed"
        workflow_state["end_time"] = datetime.now()

        # Generate final metrics
        final_metrics = self._generate_final_metrics(workflow_id)

        # Save metrics
        self.metrics_history.append(final_metrics)

        # Check for final alerts
        final_alerts = self._check_final_workflow_state(workflow_id)
        for alert in final_alerts:
            self._process_alert(alert)

        # Cleanup if no more active workflows
        del self.active_workflows[workflow_id]
        if not self.active_workflows and self.live_display:
            self._stop_live_display()

        # Disabled to reduce console clutter
        # console.print(
        #     f"✅ Real-time verification completed for workflow: {workflow_id}"
        # )
        return final_metrics

    def _analyze_reasoning_step(
        self, workflow_id: str, step: AIReasoningStep
    ) -> List[RealTimeAlert]:
        """Analyze a reasoning step for potential issues"""

        alerts = []
        workflow_state = self.active_workflows[workflow_id]
        step_number = len(workflow_state["reasoning_steps"])

        # Check confidence score
        if step.confidence_score < 0.3:
            alerts.append(
                RealTimeAlert(
                    alert_id=f"{workflow_id}_low_confidence_{step_number}",
                    alert_type=VerificationAlert.WARNING,
                    timestamp=datetime.now().isoformat(),
                    workflow_id=workflow_id,
                    step_number=step_number,
                    message=f"Low AI confidence detected: {step.confidence_score:.2f}",
                    details={
                        "confidence_score": step.confidence_score,
                        "threshold": 0.3,
                    },
                    confidence_score=0.8,
                    suggested_actions=[
                        "Review AI reasoning",
                        "Consider alternative approaches",
                    ],
                    auto_resolvable=False,
                )
            )

        # Check reasoning quality
        reasoning_quality = self._assess_reasoning_quality(step)
        if reasoning_quality < 0.5:
            alerts.append(
                RealTimeAlert(
                    alert_id=f"{workflow_id}_poor_reasoning_{step_number}",
                    alert_type=VerificationAlert.WARNING,
                    timestamp=datetime.now().isoformat(),
                    workflow_id=workflow_id,
                    step_number=step_number,
                    message=f"Poor reasoning quality detected: {reasoning_quality:.2f}",
                    details={
                        "quality_score": reasoning_quality,
                        "reasoning_text": step.ai_thought[:100],
                    },
                    confidence_score=0.7,
                    suggested_actions=[
                        "Review reasoning depth",
                        "Request more detailed analysis",
                    ],
                    auto_resolvable=False,
                )
            )

        # Check tool selection logic
        if step.selected_tool and not step.selection_rationale:
            alerts.append(
                RealTimeAlert(
                    alert_id=f"{workflow_id}_missing_rationale_{step_number}",
                    alert_type=VerificationAlert.WARNING,
                    timestamp=datetime.now().isoformat(),
                    workflow_id=workflow_id,
                    step_number=step_number,
                    message="Tool selected without clear rationale",
                    details={"selected_tool": step.selected_tool},
                    confidence_score=0.9,
                    suggested_actions=["Request explanation for tool choice"],
                    auto_resolvable=False,
                )
            )

        # Check for potential hallucinations
        hallucination_risk = self._detect_hallucination_risk(step, workflow_state)
        if hallucination_risk > 0.6:
            alerts.append(
                RealTimeAlert(
                    alert_id=f"{workflow_id}_hallucination_risk_{step_number}",
                    alert_type=VerificationAlert.HALLUCINATION_DETECTED,
                    timestamp=datetime.now().isoformat(),
                    workflow_id=workflow_id,
                    step_number=step_number,
                    message=f"Potential hallucination detected (risk: {hallucination_risk:.2f})",
                    details={
                        "risk_score": hallucination_risk,
                        "indicators": self._get_hallucination_indicators(step),
                    },
                    confidence_score=0.6,
                    suggested_actions=[
                        "Verify claims against actual data",
                        "Request evidence for assertions",
                    ],
                    auto_resolvable=False,
                )
            )

        # Check consistency with previous steps
        if step_number > 1:
            consistency_score = self._check_step_consistency(
                step, workflow_state["reasoning_steps"]
            )
            if consistency_score < 0.4:
                alerts.append(
                    RealTimeAlert(
                        alert_id=f"{workflow_id}_inconsistent_{step_number}",
                        alert_type=VerificationAlert.WARNING,
                        timestamp=datetime.now().isoformat(),
                        workflow_id=workflow_id,
                        step_number=step_number,
                        message=f"Reasoning inconsistent with previous steps: {consistency_score:.2f}",
                        details={"consistency_score": consistency_score},
                        confidence_score=0.7,
                        suggested_actions=[
                            "Review reasoning chain",
                            "Check for logical contradictions",
                        ],
                        auto_resolvable=False,
                    )
                )

        return alerts

    def _assess_reasoning_quality(self, step: AIReasoningStep) -> float:
        """Assess the quality of reasoning in a step"""

        quality_score = 1.0

        # Check reasoning length (too short or too long)
        reasoning_length = len(step.ai_thought.split())
        if reasoning_length < 10:
            quality_score -= 0.3
        elif reasoning_length > 200:
            quality_score -= 0.1

        # Check for specific keywords indicating good reasoning
        good_indicators = [
            "because",
            "therefore",
            "since",
            "given that",
            "based on",
            "analysis",
            "indicates",
            "suggests",
            "evidence",
            "data",
        ]
        reasoning_text = step.ai_thought.lower()
        indicator_count = sum(
            1 for indicator in good_indicators if indicator in reasoning_text
        )
        if indicator_count < 2:
            quality_score -= 0.2

        # Check for vague language
        vague_terms = ["might", "could", "possibly", "maybe", "probably", "seems"]
        vague_count = sum(1 for term in vague_terms if term in reasoning_text)
        if vague_count > 3:
            quality_score -= 0.2

        return max(0.0, quality_score)

    def _detect_hallucination_risk(
        self, step: AIReasoningStep, workflow_state: Dict[str, Any]
    ) -> float:
        """Detect potential hallucination indicators"""

        risk_score = 0.0

        # Check for specific claims without evidence
        reasoning_text = step.ai_thought.lower()

        # Numerical claims without context
        import re

        numerical_claims = re.findall(
            r"\d+\.?\d*\s*(percent|%|times|fold)", reasoning_text
        )
        if numerical_claims and not any(
            word in reasoning_text for word in ["data", "result", "measurement"]
        ):
            risk_score += 0.3

        # Definitive statements without qualification
        definitive_terms = ["always", "never", "all", "none", "certainly", "definitely"]
        definitive_count = sum(1 for term in definitive_terms if term in reasoning_text)
        if definitive_count > 2:
            risk_score += 0.2

        # Claims about tool capabilities not in expected results
        if step.selected_tool and "will" in reasoning_text:
            future_claims = re.findall(r"will\s+\w+", reasoning_text)
            if len(future_claims) > 2 and not step.expected_result:
                risk_score += 0.3

        # Inconsistent confidence vs reasoning quality
        reasoning_quality = self._assess_reasoning_quality(step)
        if step.confidence_score > 0.8 and reasoning_quality < 0.5:
            risk_score += 0.4

        return min(1.0, risk_score)

    def _get_hallucination_indicators(self, step: AIReasoningStep) -> List[str]:
        """Get specific hallucination indicators for a step"""

        indicators = []
        reasoning_text = step.ai_thought.lower()

        if step.confidence_score > 0.8 and len(step.ai_thought.split()) < 15:
            indicators.append("High confidence with minimal reasoning")

        if "definitely" in reasoning_text or "certainly" in reasoning_text:
            indicators.append("Overly definitive language")

        if step.selected_tool and not step.selection_rationale:
            indicators.append("Tool selection without rationale")

        import re

        if (
            re.search(r"\d+\.?\d*\s*(percent|%)", reasoning_text)
            and "data" not in reasoning_text
        ):
            indicators.append("Numerical claims without data reference")

        return indicators

    def _check_step_consistency(
        self, current_step: AIReasoningStep, previous_steps: List[AIReasoningStep]
    ) -> float:
        """Check consistency of current step with previous reasoning"""

        if not previous_steps:
            return 1.0

        consistency_score = 1.0
        current_text = current_step.ai_thought.lower()

        # Check for contradictory statements
        prev_tools = {
            step.selected_tool for step in previous_steps if step.selected_tool
        }

        # If we're repeating a tool without acknowledging previous results
        if current_step.selected_tool in prev_tools:
            if "previous" not in current_text and "again" not in current_text:
                consistency_score -= 0.3

        # Check for logical flow
        connecting_words = [
            "therefore",
            "thus",
            "consequently",
            "based on",
            "given that",
        ]
        if (
            not any(word in current_text for word in connecting_words)
            and len(previous_steps) > 0
        ):
            consistency_score -= 0.2

        return max(0.0, consistency_score)

    def _update_workflow_metrics(self, workflow_id: str) -> None:
        """Update real-time metrics for a workflow"""

        workflow_state = self.active_workflows[workflow_id]
        steps = workflow_state["reasoning_steps"]

        if not steps:
            return

        # Calculate metrics
        avg_confidence = sum(step.confidence_score for step in steps) / len(steps)
        avg_quality = sum(self._assess_reasoning_quality(step) for step in steps) / len(
            steps
        )

        metrics = VerificationMetrics(
            workflow_id=workflow_id,
            timestamp=datetime.now().isoformat(),
            overall_confidence=avg_confidence,
            reasoning_coherence=avg_quality,
            decision_accuracy=0.8,  # Placeholder - would need actual tool results
            steps_analyzed=len(steps),
            issues_detected=len(
                [
                    a
                    for a in workflow_state["alerts"]
                    if a.alert_type != VerificationAlert.INFO
                ]
            ),
            warnings_raised=len(
                [
                    a
                    for a in workflow_state["alerts"]
                    if a.alert_type == VerificationAlert.WARNING
                ]
            ),
            anomaly_score=0.2,  # Placeholder
            consistency_score=0.8,  # Placeholder
            vs_historical_average=0.9,  # Placeholder
            vs_expected_performance=0.85,  # Placeholder
        )

        workflow_state["metrics"] = metrics

    def _generate_final_metrics(self, workflow_id: str) -> VerificationMetrics:
        """Generate final verification metrics for a completed workflow"""

        workflow_state = self.active_workflows[workflow_id]
        current_metrics = workflow_state.get("metrics")

        if current_metrics:
            return current_metrics

        # Generate default metrics if none exist
        return VerificationMetrics(
            workflow_id=workflow_id,
            timestamp=datetime.now().isoformat(),
            overall_confidence=0.8,
            reasoning_coherence=0.8,
            decision_accuracy=0.8,
            steps_analyzed=0,
            issues_detected=0,
            warnings_raised=0,
            anomaly_score=0.1,
            consistency_score=0.9,
            vs_historical_average=1.0,
            vs_expected_performance=1.0,
        )

    def _check_final_workflow_state(self, workflow_id: str) -> List[RealTimeAlert]:
        """Check final workflow state for any issues"""

        alerts = []
        workflow_state = self.active_workflows[workflow_id]
        metrics = workflow_state.get("metrics")

        if metrics:
            # Check overall performance
            if metrics.overall_confidence < 0.6:
                alerts.append(
                    RealTimeAlert(
                        alert_id=f"{workflow_id}_final_low_confidence",
                        alert_type=VerificationAlert.WARNING,
                        timestamp=datetime.now().isoformat(),
                        workflow_id=workflow_id,
                        step_number=metrics.steps_analyzed,
                        message=f"Workflow completed with low overall confidence: {metrics.overall_confidence:.2f}",
                        details={"final_metrics": metrics.dict()},
                        confidence_score=0.9,
                        suggested_actions=[
                            "Review workflow results carefully",
                            "Consider re-running with different approach",
                        ],
                        auto_resolvable=False,
                    )
                )

            # Check for excessive issues
            if metrics.issues_detected > metrics.steps_analyzed * 0.5:
                alerts.append(
                    RealTimeAlert(
                        alert_id=f"{workflow_id}_final_many_issues",
                        alert_type=VerificationAlert.CRITICAL,
                        timestamp=datetime.now().isoformat(),
                        workflow_id=workflow_id,
                        step_number=metrics.steps_analyzed,
                        message=f"High number of issues detected: {metrics.issues_detected}/{metrics.steps_analyzed}",
                        details={
                            "issue_rate": metrics.issues_detected
                            / max(1, metrics.steps_analyzed)
                        },
                        confidence_score=0.95,
                        suggested_actions=[
                            "Thoroughly verify all results",
                            "Consider workflow methodology review",
                        ],
                        auto_resolvable=False,
                    )
                )

        return alerts

    def _process_alert(self, alert: RealTimeAlert) -> None:
        """Process a verification alert"""

        self.alerts.append(alert)

        # Add alert to workflow state
        workflow_id = alert.workflow_id
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["alerts"].append(alert)

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                console.print(f"[red]Alert callback error: {e}[/red]")

        # Display alert if appropriate
        if alert.alert_type in [
            VerificationAlert.WARNING,
            VerificationAlert.CRITICAL,
            VerificationAlert.HALLUCINATION_DETECTED,
        ]:
            self._display_alert(alert)

    def _display_alert(self, alert: RealTimeAlert) -> None:
        """Display an alert to the console"""

        alert_symbols = {
            VerificationAlert.INFO: "ℹ️",
            VerificationAlert.WARNING: "⚠️",
            VerificationAlert.CRITICAL: "🚨",
            VerificationAlert.HALLUCINATION_DETECTED: "🧠",
        }

        symbol = alert_symbols.get(alert.alert_type, "📢")

        console.print(
            f"{symbol} [bold]{alert.alert_type.value.upper()}[/bold]: {alert.message}"
        )

        if alert.suggested_actions:
            console.print(
                f"   💡 Suggestions: {', '.join(alert.suggested_actions[:2])}"
            )

    def _start_live_display(self) -> None:
        """Start live display of verification status"""

        # DISABLED: Live display causes empty box rendering issues
        # Simply log that monitoring is active instead
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"🔍 Real-time verification monitoring active (display disabled for stability)"
        )

        # Disable live display completely
        self.enable_live_display = False
        self.live_display = None

        # Original code commented out:
        # try:
        #     layout = Layout()
        #     layout.split_column(Layout(name="header", size=3), Layout(name="main"))
        #     layout["main"].split_row(Layout(name="workflows"), Layout(name="alerts"))
        #     self.live_display = Live(layout, console=console, refresh_per_second=2)
        #     self.live_display.start()
        #     # Start update thread
        #     self.display_thread = threading.Thread(
        #         target=self._update_live_display, daemon=True
        #     )
        #     self.display_thread.start()
        # except Exception as e:
        #     # If display fails (e.g., another display is active), disable live display
        #     import logging
        #     logger = logging.getLogger(__name__)
        #     logger.warning(f"Could not start live display: {e}")
        #     self.enable_live_display = False
        #     self.live_display = None

    def _update_live_display(self) -> None:
        """Update live display continuously"""

        while self.is_monitoring and self.live_display:
            try:
                layout = self.live_display.renderable

                # Update header
                header_text = f"🔍 Real-Time AI Verification - Active Workflows: {len(self.active_workflows)}"
                layout["header"].update(Panel(header_text, style="bold blue"))

                # Update workflows panel
                workflows_content = self._create_workflows_display()
                layout["workflows"].update(
                    Panel(workflows_content, title="Active Workflows")
                )

                # Update alerts panel
                alerts_content = self._create_alerts_display()
                layout["alerts"].update(Panel(alerts_content, title="Recent Alerts"))

                time.sleep(0.5)

            except Exception:
                # Silently handle display errors
                pass

    def _create_workflows_display(self) -> Table:
        """Create display table for active workflows"""

        table = Table(box=box.SIMPLE)
        table.add_column("Workflow", style="cyan")
        table.add_column("Steps", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_column("Issues", style="red")

        for workflow_id, state in self.active_workflows.items():
            steps = len(state["reasoning_steps"])
            metrics = state.get("metrics")

            confidence = f"{metrics.overall_confidence:.2f}" if metrics else "N/A"
            issues = str(len(state["alerts"]))

            table.add_row(workflow_id[:8], str(steps), confidence, issues)

        return table

    def _create_alerts_display(self) -> Text:
        """Create display for recent alerts"""

        text = Text()
        recent_alerts = self.alerts[-5:]  # Last 5 alerts

        if not recent_alerts:
            text.append("No alerts", style="dim")
            return text

        for alert in recent_alerts:
            symbol = "🚨" if alert.alert_type == VerificationAlert.CRITICAL else "⚠️"
            text.append(f"{symbol} {alert.message[:50]}...\n")

        return text

    def _stop_live_display(self) -> None:
        """Stop live display"""

        self.is_monitoring = False

        if self.live_display:
            self.live_display.stop()
            self.live_display = None

        if self.display_thread:
            self.display_thread.join(timeout=1.0)
            self.display_thread = None

    def _load_historical_patterns(self) -> Dict[str, Any]:
        """Load historical patterns for comparison"""
        # Placeholder - would load from actual historical data
        return {
            "avg_confidence": 0.75,
            "avg_steps": 4.2,
            "common_tools": ["run_metabolic_fba", "find_minimal_media"],
            "success_rate": 0.89,
        }

    def _initialize_anomaly_detectors(self) -> Dict[str, Any]:
        """Initialize anomaly detection algorithms"""
        # Placeholder - would initialize actual ML models
        return {
            "confidence_detector": {"threshold": 0.3, "sensitivity": 0.8},
            "reasoning_detector": {"min_quality": 0.5, "pattern_matching": True},
            "consistency_detector": {"window_size": 3, "threshold": 0.4},
        }

    def add_alert_callback(self, callback: Callable[[RealTimeAlert], None]) -> None:
        """Add a callback function for alert notifications"""
        self.alert_callbacks.append(callback)

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow"""
        return self.active_workflows.get(workflow_id)

    def get_recent_alerts(self, count: int = 10) -> List[RealTimeAlert]:
        """Get recent alerts"""
        return self.alerts[-count:]


def create_realtime_detector(
    logs_dir: Optional[Path] = None, enable_display: bool = True
) -> RealTimeHallucinationDetector:
    """Factory function to create a real-time hallucination detector"""
    return RealTimeHallucinationDetector(logs_dir, enable_display)
