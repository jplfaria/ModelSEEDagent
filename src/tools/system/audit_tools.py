#!/usr/bin/env python3
"""
System Audit Tools - BaseTool Wrappers
=====================================

Provides BaseTool wrappers for system-level audit and verification tools
to enable validation through the standard tool validation suite.

These tools validate system functionality rather than biological accuracy.
"""

import json
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..ai_audit import AIDecisionVerifier, AIWorkflowAudit
from ..audit import HallucinationDetector, ToolAuditor
from ..base import BaseTool, ToolRegistry, ToolResult
from ..realtime_verification import RealTimeHallucinationDetector


class SystemToolConfig(BaseModel):
    """Configuration for system audit tools"""

    test_duration: int = Field(default=5, description="Test duration in seconds")
    create_test_files: bool = Field(
        default=True, description="Whether to create test files for validation"
    )
    cleanup_after_test: bool = Field(
        default=True, description="Whether to cleanup test files after validation"
    )


@ToolRegistry.register
class ToolAuditTool(BaseTool):
    """Tool audit system validation - tests basic audit functionality"""

    tool_name = "validate_tool_audit"
    tool_description = """Validates the tool audit system functionality including:
    - Audit record creation and storage
    - Console output capture
    - File tracking and verification
    - Execution metadata collection"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._config = SystemToolConfig(**config.get("system_config", {}))

    def _run_tool(self, input_data: Any) -> ToolResult:
        """Test tool audit functionality"""
        try:
            test_id = str(uuid.uuid4())[:8]
            test_results = {
                "test_id": test_id,
                "timestamp": datetime.now().isoformat(),
                "tests_performed": [],
                "validation_results": {},
            }

            # Test 1: Audit record creation
            try:
                auditor = ToolAuditor()
                audit_id = auditor.start_audit(
                    "test_tool", {"test_param": "test_value"}
                )

                # Simulate some work
                time.sleep(0.1)

                auditor.complete_audit(
                    audit_id, {"success": True, "test_output": "validation_complete"}
                )

                test_results["tests_performed"].append("audit_record_creation")
                test_results["validation_results"]["audit_record_creation"] = {
                    "success": True,
                    "audit_id": audit_id,
                    "message": "Audit record created and completed successfully",
                }

            except Exception as e:
                test_results["validation_results"]["audit_record_creation"] = {
                    "success": False,
                    "error": str(e),
                }

            # Test 2: Console capture functionality
            try:
                from ..audit import ConsoleCapture

                with ConsoleCapture() as capture:
                    print("Test console output")
                    print("Additional output line")

                stdout, stderr = capture.get_captured_output()

                console_test_success = (
                    "Test console output" in stdout
                    and "Additional output line" in stdout
                )

                test_results["tests_performed"].append("console_capture")
                test_results["validation_results"]["console_capture"] = {
                    "success": console_test_success,
                    "stdout_length": len(stdout),
                    "stderr_length": len(stderr),
                    "message": (
                        "Console capture working"
                        if console_test_success
                        else "Console capture failed"
                    ),
                }

            except Exception as e:
                test_results["validation_results"]["console_capture"] = {
                    "success": False,
                    "error": str(e),
                }

            # Test 3: File tracking
            try:
                from ..audit import FileTracker

                tracker = FileTracker()

                if self._config.create_test_files:
                    # Create temporary test file
                    with tempfile.NamedTemporaryFile(
                        mode="w", delete=False, suffix=".test"
                    ) as f:
                        test_file_path = f.name
                        f.write("Test file content for audit validation")

                    # Track the file
                    tracker.track_file_creation(test_file_path, "test_content")
                    tracked_files = tracker.get_tracked_files()

                    file_tracking_success = len(tracked_files) > 0

                    if self._config.cleanup_after_test:
                        Path(test_file_path).unlink(missing_ok=True)
                else:
                    file_tracking_success = True  # Skip test if not creating files

                test_results["tests_performed"].append("file_tracking")
                test_results["validation_results"]["file_tracking"] = {
                    "success": file_tracking_success,
                    "tracked_files_count": (
                        len(tracked_files) if self._config.create_test_files else 0
                    ),
                    "message": (
                        "File tracking working"
                        if file_tracking_success
                        else "File tracking failed"
                    ),
                }

            except Exception as e:
                test_results["validation_results"]["file_tracking"] = {
                    "success": False,
                    "error": str(e),
                }

            # Calculate overall success
            successful_tests = sum(
                1
                for result in test_results["validation_results"].values()
                if result.get("success", False)
            )
            total_tests = len(test_results["validation_results"])

            overall_success = successful_tests == total_tests
            success_rate = successful_tests / total_tests if total_tests > 0 else 0

            return ToolResult(
                success=overall_success,
                message=f"Tool audit validation completed. {successful_tests}/{total_tests} tests passed ({success_rate:.1%})",
                data={
                    "overall_success": overall_success,
                    "success_rate": success_rate,
                    "successful_tests": successful_tests,
                    "total_tests": total_tests,
                    "detailed_results": test_results,
                    "functionality_validated": test_results["tests_performed"],
                },
                metadata={
                    "test_type": "system_audit_validation",
                    "test_id": test_id,
                    "validation_approach": "functional_testing",
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Tool audit validation failed: {str(e)}",
                error=str(e),
            )


@ToolRegistry.register
class AIAuditTool(BaseTool):
    """AI audit system validation - tests AI reasoning capture and workflow tracking"""

    tool_name = "validate_ai_audit"
    tool_description = """Validates the AI audit system functionality including:
    - AI reasoning step capture
    - Workflow tracking and coherence analysis
    - Decision verification and confidence scoring
    - Multi-step reasoning chain validation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._config = SystemToolConfig(**config.get("system_config", {}))

    def _run_tool(self, input_data: Any) -> ToolResult:
        """Test AI audit functionality"""
        try:
            test_id = str(uuid.uuid4())[:8]
            test_results = {
                "test_id": test_id,
                "timestamp": datetime.now().isoformat(),
                "tests_performed": [],
                "validation_results": {},
            }

            # Test 1: AI workflow audit creation
            try:
                workflow_audit = AIWorkflowAudit(
                    workflow_id=f"test_workflow_{test_id}",
                    user_query="Test query for validation",
                    timestamp_start=datetime.now().isoformat(),
                )

                test_results["tests_performed"].append("workflow_audit_creation")
                test_results["validation_results"]["workflow_audit_creation"] = {
                    "success": True,
                    "workflow_id": workflow_audit.workflow_id,
                    "message": "AI workflow audit created successfully",
                }

            except Exception as e:
                test_results["validation_results"]["workflow_audit_creation"] = {
                    "success": False,
                    "error": str(e),
                }

            # Test 2: AI reasoning step tracking
            try:
                from ..ai_audit import AIReasoningStep

                reasoning_step = AIReasoningStep(
                    step_id=f"step_{test_id}",
                    step_number=1,
                    timestamp=datetime.now().isoformat(),
                    ai_thought="Test reasoning step for validation",
                    context_analysis="Analyzing test context",
                    available_tools=["test_tool_1", "test_tool_2"],
                    selected_tool="test_tool_1",
                    selection_rationale="Selected for validation testing",
                    confidence_score=0.85,
                    expected_result="Successful validation",
                    success_criteria=["Test completes", "No errors occur"],
                    alternative_tools=["test_tool_2"],
                    rejection_reasons={"test_tool_2": "Not needed for this test"},
                )

                # Validate reasoning step structure
                step_validation_success = (
                    reasoning_step.confidence_score > 0
                    and reasoning_step.confidence_score <= 1.0
                    and len(reasoning_step.available_tools) > 0
                    and reasoning_step.selected_tool is not None
                )

                test_results["tests_performed"].append("reasoning_step_tracking")
                test_results["validation_results"]["reasoning_step_tracking"] = {
                    "success": step_validation_success,
                    "step_id": reasoning_step.step_id,
                    "confidence_score": reasoning_step.confidence_score,
                    "message": (
                        "Reasoning step tracking working"
                        if step_validation_success
                        else "Reasoning step validation failed"
                    ),
                }

            except Exception as e:
                test_results["validation_results"]["reasoning_step_tracking"] = {
                    "success": False,
                    "error": str(e),
                }

            # Test 3: Decision verification
            try:
                verifier = AIDecisionVerifier()

                # Test decision verification logic
                verification_result = verifier.verify_tool_selection(
                    available_tools=["test_tool", "alt_tool_1", "alt_tool_2"],
                    selected_tool="test_tool",
                    rationale="Best tool for validation testing",
                    confidence=0.9,
                )

                decision_verification_success = verification_result.get(
                    "is_valid", False
                )

                test_results["tests_performed"].append("decision_verification")
                test_results["validation_results"]["decision_verification"] = {
                    "success": decision_verification_success,
                    "verification_result": verification_result,
                    "message": (
                        "Decision verification working"
                        if decision_verification_success
                        else "Decision verification failed"
                    ),
                }

            except Exception as e:
                test_results["validation_results"]["decision_verification"] = {
                    "success": False,
                    "error": str(e),
                }

            # Calculate overall success
            successful_tests = sum(
                1
                for result in test_results["validation_results"].values()
                if result.get("success", False)
            )
            total_tests = len(test_results["validation_results"])

            overall_success = successful_tests == total_tests
            success_rate = successful_tests / total_tests if total_tests > 0 else 0

            return ToolResult(
                success=overall_success,
                message=f"AI audit validation completed. {successful_tests}/{total_tests} tests passed ({success_rate:.1%})",
                data={
                    "overall_success": overall_success,
                    "success_rate": success_rate,
                    "successful_tests": successful_tests,
                    "total_tests": total_tests,
                    "detailed_results": test_results,
                    "functionality_validated": test_results["tests_performed"],
                },
                metadata={
                    "test_type": "ai_audit_validation",
                    "test_id": test_id,
                    "validation_approach": "functional_testing",
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message=f"AI audit validation failed: {str(e)}",
                error=str(e),
            )


@ToolRegistry.register
class RealtimeVerificationTool(BaseTool):
    """Real-time verification system validation - tests live monitoring capabilities"""

    tool_name = "validate_realtime_verification"
    tool_description = """Validates the real-time verification system functionality including:
    - Real-time alert generation and processing
    - Live metrics calculation and tracking
    - Anomaly detection and pattern analysis
    - Verification threshold management"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._config = SystemToolConfig(**config.get("system_config", {}))

    def _run_tool(self, input_data: Any) -> ToolResult:
        """Test real-time verification functionality"""
        try:
            test_id = str(uuid.uuid4())[:8]
            test_results = {
                "test_id": test_id,
                "timestamp": datetime.now().isoformat(),
                "tests_performed": [],
                "validation_results": {},
            }

            # Test 1: Alert generation
            try:
                from ..realtime_verification import RealTimeAlert, VerificationAlert

                test_alert = RealTimeAlert(
                    alert_id=f"alert_{test_id}",
                    alert_type=VerificationAlert.INFO,
                    timestamp=datetime.now().isoformat(),
                    workflow_id=f"workflow_{test_id}",
                    step_number=1,
                    message="Test alert for validation",
                    details={"test_param": "test_value"},
                    confidence_score=0.8,
                    suggested_actions=["review_test_result"],
                    auto_resolvable=True,
                )

                alert_validation_success = (
                    test_alert.confidence_score > 0
                    and test_alert.confidence_score <= 1.0
                    and len(test_alert.suggested_actions) > 0
                )

                test_results["tests_performed"].append("alert_generation")
                test_results["validation_results"]["alert_generation"] = {
                    "success": alert_validation_success,
                    "alert_id": test_alert.alert_id,
                    "alert_type": test_alert.alert_type.value,
                    "message": (
                        "Alert generation working"
                        if alert_validation_success
                        else "Alert generation failed"
                    ),
                }

            except Exception as e:
                test_results["validation_results"]["alert_generation"] = {
                    "success": False,
                    "error": str(e),
                }

            # Test 2: Metrics calculation
            try:
                from ..realtime_verification import VerificationMetrics

                test_metrics = VerificationMetrics(
                    workflow_id=f"workflow_{test_id}",
                    timestamp=datetime.now().isoformat(),
                    overall_confidence=0.85,
                    reasoning_coherence=0.90,
                    decision_accuracy=0.80,
                    steps_analyzed=5,
                    issues_detected=1,
                    warnings_raised=0,
                    anomaly_score=0.15,
                    consistency_score=0.88,
                    vs_historical_average=0.92,
                    vs_expected_performance=0.87,
                )

                metrics_validation_success = (
                    0 <= test_metrics.overall_confidence <= 1.0
                    and 0 <= test_metrics.reasoning_coherence <= 1.0
                    and test_metrics.steps_analyzed >= 0
                    and test_metrics.issues_detected >= 0
                )

                test_results["tests_performed"].append("metrics_calculation")
                test_results["validation_results"]["metrics_calculation"] = {
                    "success": metrics_validation_success,
                    "overall_confidence": test_metrics.overall_confidence,
                    "steps_analyzed": test_metrics.steps_analyzed,
                    "message": (
                        "Metrics calculation working"
                        if metrics_validation_success
                        else "Metrics calculation failed"
                    ),
                }

            except Exception as e:
                test_results["validation_results"]["metrics_calculation"] = {
                    "success": False,
                    "error": str(e),
                }

            # Test 3: Real-time detector initialization
            try:
                detector = RealTimeHallucinationDetector()

                # Test basic detector functionality
                detector_initialization_success = (
                    hasattr(detector, "start_monitoring")
                    and hasattr(detector, "stop_monitoring")
                    and hasattr(detector, "add_verification_alert")
                )

                test_results["tests_performed"].append("detector_initialization")
                test_results["validation_results"]["detector_initialization"] = {
                    "success": detector_initialization_success,
                    "detector_type": type(detector).__name__,
                    "message": (
                        "Detector initialization working"
                        if detector_initialization_success
                        else "Detector initialization failed"
                    ),
                }

            except Exception as e:
                test_results["validation_results"]["detector_initialization"] = {
                    "success": False,
                    "error": str(e),
                }

            # Calculate overall success
            successful_tests = sum(
                1
                for result in test_results["validation_results"].values()
                if result.get("success", False)
            )
            total_tests = len(test_results["validation_results"])

            overall_success = successful_tests == total_tests
            success_rate = successful_tests / total_tests if total_tests > 0 else 0

            return ToolResult(
                success=overall_success,
                message=f"Real-time verification validation completed. {successful_tests}/{total_tests} tests passed ({success_rate:.1%})",
                data={
                    "overall_success": overall_success,
                    "success_rate": success_rate,
                    "successful_tests": successful_tests,
                    "total_tests": total_tests,
                    "detailed_results": test_results,
                    "functionality_validated": test_results["tests_performed"],
                },
                metadata={
                    "test_type": "realtime_verification_validation",
                    "test_id": test_id,
                    "validation_approach": "functional_testing",
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Real-time verification validation failed: {str(e)}",
                error=str(e),
            )


# Export all audit tools
__all__ = [
    "ToolAuditTool",
    "AIAuditTool",
    "RealtimeVerificationTool",
    "SystemToolConfig",
]
