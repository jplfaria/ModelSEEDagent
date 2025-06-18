#!/usr/bin/env python3
"""
Tool Execution Audit System

Comprehensive auditing system for tracking and verifying all tool executions
to detect hallucinations and provide transparency into AI tool behavior.

Phase 4 Implementation:
- Automatic capture of tool inputs, outputs, and console logs
- Standardized audit format for analysis and review
- Session-aware storage for organized audit trails
"""

import contextlib
import io
import json
import os
import re
import statistics
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class AuditRecord(BaseModel):
    """Standardized audit record for tool execution"""

    audit_id: str = Field(description="Unique identifier for this audit record")
    session_id: Optional[str] = Field(
        description="Session ID if executed within a session"
    )
    tool_name: str = Field(description="Name of the executed tool")
    timestamp: str = Field(description="ISO timestamp of execution start")

    input: Dict[str, Any] = Field(description="Input data and parameters")
    output: Dict[str, Any] = Field(description="Complete output capture")
    execution: Dict[str, Any] = Field(description="Execution metadata")

    # Additional context
    environment: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables and context"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat(), Path: lambda v: str(v)}


class ConsoleCapture:
    """Context manager to capture stdout and stderr during tool execution"""

    def __init__(self):
        self.stdout_capture = io.StringIO()
        self.stderr_capture = io.StringIO()
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Create tee-like behavior to capture AND display
        sys.stdout = TeeOutput(self.original_stdout, self.stdout_capture)
        sys.stderr = TeeOutput(self.original_stderr, self.stderr_capture)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def get_captured_output(self) -> Tuple[str, str]:
        """Get captured stdout and stderr"""
        return self.stdout_capture.getvalue(), self.stderr_capture.getvalue()


class TeeOutput:
    """Output stream that writes to both original and capture streams"""

    def __init__(self, original, capture):
        self.original = original
        self.capture = capture

    def write(self, text):
        self.original.write(text)
        self.capture.write(text)

    def flush(self):
        self.original.flush()
        self.capture.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


class FileTracker:
    """Track files created during tool execution"""

    def __init__(self, watch_dirs: List[str] = None):
        self.watch_dirs = watch_dirs or ["."]
        self.initial_files = set()
        self.final_files = set()

    def __enter__(self):
        self.initial_files = self._scan_files()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_files = self._scan_files()

    def _scan_files(self) -> set:
        """Scan for files in watched directories"""
        files = set()
        for watch_dir in self.watch_dirs:
            try:
                watch_path = Path(watch_dir)
                if watch_path.exists():
                    for file_path in watch_path.rglob("*"):
                        if file_path.is_file():
                            files.add(str(file_path.absolute()))
            except Exception:
                # Ignore scanning errors
                pass
        return files

    def get_new_files(self) -> List[str]:
        """Get list of files created during execution"""
        return list(self.final_files - self.initial_files)


class ToolAuditor:
    """Main auditing system for tool execution tracking"""

    def __init__(self, base_audit_dir: str = "logs"):
        self.base_audit_dir = Path(base_audit_dir)
        self.current_session_id = None

    def set_session_id(self, session_id: str):
        """Set the current session ID for audit organization"""
        self.current_session_id = session_id

    def get_audit_dir(self, session_id: Optional[str] = None) -> Path:
        """Get the audit directory for a session"""
        session_id = session_id or self.current_session_id or "default"
        audit_dir = self.base_audit_dir / session_id / "tool_audits"
        audit_dir.mkdir(parents=True, exist_ok=True)
        return audit_dir

    def create_audit_record(
        self,
        tool_name: str,
        input_data: Any,
        result: Any,
        console_output: Tuple[str, str],
        execution_time: float,
        created_files: List[str],
        error: Optional[str] = None,
    ) -> AuditRecord:
        """Create a standardized audit record"""

        audit_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Capture environment context
        environment = {
            "python_version": sys.version,
            "working_directory": str(Path.cwd()),
            "user": os.getenv("USER", "unknown"),
        }

        # Process console output
        stdout, stderr = console_output

        # Build structured output
        output = {
            "structured": self._serialize_result(result),
            "console": {"stdout": stdout, "stderr": stderr},
            "files": created_files,
        }

        # Build execution metadata
        execution = {
            "duration_seconds": execution_time,
            "success": (
                getattr(result, "success", True)
                if hasattr(result, "success")
                else (error is None)
            ),
            "error": error,
            "timestamp_end": datetime.now().isoformat(),
        }

        return AuditRecord(
            audit_id=audit_id,
            session_id=self.current_session_id,
            tool_name=tool_name,
            timestamp=timestamp,
            input=self._serialize_input(input_data),
            output=output,
            execution=execution,
            environment=environment,
        )

    def save_audit_record(self, audit_record: AuditRecord) -> Path:
        """Save audit record to disk"""
        audit_dir = self.get_audit_dir(audit_record.session_id)

        # Create filename with timestamp and audit ID
        timestamp_str = datetime.fromisoformat(audit_record.timestamp).strftime(
            "%Y%m%d_%H%M%S"
        )
        filename = (
            f"{timestamp_str}_{audit_record.tool_name}_{audit_record.audit_id[:8]}.json"
        )

        audit_file = audit_dir / filename

        with open(audit_file, "w") as f:
            json.dump(audit_record.dict(), f, indent=2, default=str)

        return audit_file

    def audit_tool_execution(
        self,
        tool_name: str,
        input_data: Any,
        execution_func,
        watch_dirs: List[str] = None,
    ) -> Tuple[Any, Path]:
        """
        Audit a tool execution with comprehensive capture

        Returns:
            Tuple of (tool_result, audit_file_path)
        """
        start_time = time.time()
        result = None
        error = None

        # Set up monitoring
        with ConsoleCapture() as console, FileTracker(watch_dirs) as file_tracker:
            try:
                # Execute the tool
                result = execution_func(input_data)
            except Exception as e:
                error = str(e)
                raise
            finally:
                # Always create audit record, even on failure
                execution_time = time.time() - start_time
                console_output = console.get_captured_output()
                created_files = file_tracker.get_new_files()

                # Create and save audit record
                audit_record = self.create_audit_record(
                    tool_name=tool_name,
                    input_data=input_data,
                    result=result,
                    console_output=console_output,
                    execution_time=execution_time,
                    created_files=created_files,
                    error=error,
                )

                audit_file = self.save_audit_record(audit_record)

        return result, audit_file

    def _serialize_input(self, input_data: Any) -> Dict[str, Any]:
        """Serialize input data for audit storage"""
        try:
            if hasattr(input_data, "dict"):
                return input_data.dict()
            elif isinstance(input_data, dict):
                return input_data
            else:
                return {"raw_input": str(input_data), "type": type(input_data).__name__}
        except Exception:
            return {"raw_input": str(input_data), "serialization_error": True}

    def _serialize_result(self, result: Any) -> Dict[str, Any]:
        """Serialize tool result for audit storage with Smart Summarization metrics"""
        try:
            if hasattr(result, "dict"):
                result_dict = result.dict()

                # Add Smart Summarization metrics if available
                if (
                    hasattr(result, "has_smart_summarization")
                    and result.has_smart_summarization()
                ):
                    result_dict["smart_summarization_metrics"] = {
                        "enabled": True,
                        "has_key_findings": result.key_findings is not None,
                        "has_summary_dict": result.summary_dict is not None,
                        "has_full_data_path": result.full_data_path is not None,
                        "key_findings_count": (
                            len(result.key_findings) if result.key_findings else 0
                        ),
                        "schema_version": result.schema_version,
                    }

                    # Add size metrics if available in metadata
                    if result.metadata:
                        if "summarization_reduction_pct" in result.metadata:
                            result_dict["smart_summarization_metrics"][
                                "reduction_percentage"
                            ] = result.metadata["summarization_reduction_pct"]
                        if "original_data_size" in result.metadata:
                            result_dict["smart_summarization_metrics"][
                                "original_size_bytes"
                            ] = result.metadata["original_data_size"]
                        if "summarized_size_bytes" in result.metadata:
                            result_dict["smart_summarization_metrics"][
                                "summarized_size_bytes"
                            ] = result.metadata["summarized_size_bytes"]
                else:
                    result_dict["smart_summarization_metrics"] = {"enabled": False}

                return result_dict
            elif isinstance(result, dict):
                return result
            else:
                return {"raw_result": str(result), "type": type(result).__name__}
        except Exception as e:
            return {
                "raw_result": str(result),
                "serialization_error": True,
                "error": str(e),
            }


# Global auditor instance
_global_auditor = ToolAuditor()


def get_auditor() -> ToolAuditor:
    """Get the global tool auditor instance"""
    return _global_auditor


def set_audit_session(session_id: str):
    """Set the current audit session ID"""
    _global_auditor.set_session_id(session_id)


# Phase 4.3: Advanced Hallucination Detection Helpers


class HallucinationDetector:
    """
    Advanced hallucination detection system for tool execution audits

    Performs sophisticated analysis to detect discrepancies between AI claims
    and actual tool execution results.
    """

    def __init__(self, logs_dir: Optional[Union[str, Path]] = None):
        """Initialize the hallucination detector"""
        self.logs_dir = Path(logs_dir or "logs")
        self.verification_patterns = self._load_verification_patterns()

    def _load_verification_patterns(self) -> Dict[str, Any]:
        """Load common hallucination patterns and verification rules"""
        return {
            # Common words that indicate success/failure
            "success_indicators": [
                "successful",
                "completed",
                "finished",
                "done",
                "passed",
                "optimal",
                "converged",
                "solved",
                "generated",
                "created",
            ],
            "failure_indicators": [
                "failed",
                "error",
                "exception",
                "crashed",
                "aborted",
                "timeout",
                "invalid",
                "corrupted",
                "missing",
                "broken",
            ],
            # Numerical claim patterns
            "numerical_patterns": {
                "growth_rate": r"growth rate:?\s*([0-9]+\.?[0-9]*)",
                "objective_value": r"objective value:?\s*([0-9]+\.?[0-9]*)",
                "execution_time": r"(?:took|completed in|duration):?\s*([0-9]+\.?[0-9]*)\s*(?:s|sec|second)",
                "file_count": r"(?:created|generated|saved)\s*([0-9]+)\s*files?",
                "reaction_count": r"([0-9]+)\s*reactions?",
                "metabolite_count": r"([0-9]+)\s*metabolites?",
            },
            # File reference patterns
            "file_patterns": {
                "file_saved": r"saved to:?\s*([^\s]+\.(?:csv|json|xml|txt|html))",
                "file_created": r"created:?\s*([^\s]+\.(?:csv|json|xml|txt|html))",
                "file_written": r"written to:?\s*([^\s]+\.(?:csv|json|xml|txt|html))",
            },
        }

    def verify_tool_claims(self, audit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare AI tool claims in message vs actual data output

        Args:
            audit_data: Complete audit record for a tool execution

        Returns:
            Verification result with detailed analysis
        """
        result = {
            "verification_type": "tool_claims",
            "audit_id": audit_data.get("audit_id", "unknown"),
            "tool_name": audit_data.get("tool_name", "unknown"),
            "timestamp": audit_data.get("timestamp", ""),
            "issues": [],
            "verifications": [],
            "confidence_score": 1.0,
            "details": {},
        }

        # Extract AI message and actual data
        output_data = audit_data.get("output", {})
        structured_output = output_data.get("structured", {})
        ai_message = structured_output.get("message", "")
        actual_data = structured_output.get("data", {})

        if not ai_message or not actual_data:
            result["issues"].append("Missing AI message or actual data for comparison")
            result["confidence_score"] = 0.0
            return result

        # 1. Success/Failure Consistency Check
        self._verify_success_failure_claims(ai_message, structured_output, result)

        # 2. Numerical Claims Verification
        self._verify_numerical_claims(ai_message, actual_data, result)

        # 3. File Claims Verification
        self._verify_file_claims(ai_message, actual_data, output_data, result)

        # 4. Data Structure Claims Verification
        self._verify_data_structure_claims(ai_message, actual_data, result)

        # Calculate overall confidence score
        total_checks = len(result["verifications"]) + len(result["issues"])
        if total_checks > 0:
            result["confidence_score"] = len(result["verifications"]) / total_checks

        return result

    def _verify_success_failure_claims(
        self, message: str, structured_output: Dict, result: Dict
    ):
        """Verify success/failure claims consistency"""
        message_lower = message.lower()
        actual_success = structured_output.get("success", True)

        # Check for success indicators in message
        success_found = any(
            indicator in message_lower
            for indicator in self.verification_patterns["success_indicators"]
        )
        failure_found = any(
            indicator in message_lower
            for indicator in self.verification_patterns["failure_indicators"]
        )

        if success_found and not actual_success:
            result["issues"].append("AI claims success but actual execution failed")
        elif failure_found and actual_success:
            result["issues"].append("AI claims failure but actual execution succeeded")
        elif success_found and actual_success:
            result["verifications"].append(
                "Success claims consistent with actual results"
            )
        elif not success_found and not failure_found:
            result["verifications"].append(
                "Neutral tone matches uncertain execution status"
            )

    def _verify_numerical_claims(self, message: str, actual_data: Dict, result: Dict):
        """Verify numerical claims in AI message against actual data"""
        numerical_claims = {}

        # Extract numerical claims from message
        for claim_type, pattern in self.verification_patterns[
            "numerical_patterns"
        ].items():
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                try:
                    numerical_claims[claim_type] = float(matches[0])
                except ValueError:
                    continue

        if not numerical_claims:
            result["verifications"].append("No specific numerical claims to verify")
            return

        # Verify each numerical claim
        for claim_type, claimed_value in numerical_claims.items():
            actual_value = self._extract_actual_numerical_value(claim_type, actual_data)

            if actual_value is not None:
                # Allow for reasonable floating point tolerance
                tolerance = (
                    abs(claimed_value) * 0.001 + 1e-6
                )  # 0.1% relative + small absolute

                if abs(claimed_value - actual_value) <= tolerance:
                    result["verifications"].append(
                        f"{claim_type}: claimed {claimed_value} matches actual {actual_value}"
                    )
                else:
                    result["issues"].append(
                        f"{claim_type}: claimed {claimed_value} but actual {actual_value}"
                    )
            else:
                result["issues"].append(
                    f"{claim_type}: claimed {claimed_value} but no actual value found"
                )

        result["details"]["numerical_claims"] = numerical_claims

    def _extract_actual_numerical_value(
        self, claim_type: str, actual_data: Dict
    ) -> Optional[float]:
        """Extract actual numerical value from data based on claim type"""
        if claim_type == "growth_rate" or claim_type == "objective_value":
            return actual_data.get("objective_value")
        elif claim_type == "reaction_count":
            return actual_data.get("num_reactions") or len(
                actual_data.get("significant_fluxes", {})
            )
        elif claim_type == "metabolite_count":
            return actual_data.get("num_metabolites")
        elif claim_type == "file_count":
            return len(actual_data.get("files", [])) if "files" in actual_data else None

        return None

    def _verify_file_claims(
        self, message: str, actual_data: Dict, output_data: Dict, result: Dict
    ):
        """Verify file creation claims in AI message"""
        file_claims = []

        # Extract file claims from message
        for pattern_name, pattern in self.verification_patterns[
            "file_patterns"
        ].items():
            matches = re.findall(pattern, message, re.IGNORECASE)
            file_claims.extend(matches)

        # Get actual files created
        actual_files = output_data.get("files", [])
        result_files = actual_data.get("result_file")
        if result_files:
            if isinstance(result_files, str):
                actual_files.append(result_files)
            elif isinstance(result_files, list):
                actual_files.extend(result_files)

        if not file_claims and not actual_files:
            result["verifications"].append("No file creation claims to verify")
            return

        # Verify file claims
        if file_claims and not actual_files:
            result["issues"].append(
                f"AI claims files created ({file_claims}) but no actual files found"
            )
        elif not file_claims and actual_files:
            result["issues"].append(
                f"Files created ({actual_files}) but not mentioned by AI"
            )
        elif file_claims and actual_files:
            # Check if claimed files match actual files
            claimed_names = [Path(f).name for f in file_claims]
            actual_names = [Path(f).name for f in actual_files]

            missing_claims = set(claimed_names) - set(actual_names)
            unexpected_files = set(actual_names) - set(claimed_names)

            if missing_claims:
                result["issues"].append(
                    f"AI claims files not actually created: {list(missing_claims)}"
                )
            if unexpected_files:
                result["issues"].append(
                    f"Files created but not mentioned by AI: {list(unexpected_files)}"
                )

            matching_files = set(claimed_names) & set(actual_names)
            if matching_files:
                result["verifications"].append(
                    f"File claims verified: {list(matching_files)}"
                )

        result["details"]["file_claims"] = file_claims
        result["details"]["actual_files"] = actual_files

    def _verify_data_structure_claims(
        self, message: str, actual_data: Dict, result: Dict
    ):
        """Verify claims about data structure and content"""
        # Check for claims about data keys/fields
        claimed_fields = self._extract_field_claims(message)
        actual_fields = (
            set(actual_data.keys()) if isinstance(actual_data, dict) else set()
        )

        if claimed_fields:
            missing_fields = claimed_fields - actual_fields
            if missing_fields:
                result["issues"].append(
                    f"AI claims fields not found in data: {list(missing_fields)}"
                )

            present_fields = claimed_fields & actual_fields
            if present_fields:
                result["verifications"].append(
                    f"Data structure claims verified: {list(present_fields)}"
                )

        # Check for claims about data completeness
        if "empty" in message.lower() or "no data" in message.lower():
            if actual_data:
                result["issues"].append("AI claims no data but actual data is present")
            else:
                result["verifications"].append("Empty data claim verified")

        result["details"]["claimed_fields"] = (
            list(claimed_fields) if claimed_fields else []
        )
        result["details"]["actual_fields"] = list(actual_fields)

    def _extract_field_claims(self, message: str) -> set:
        """Extract field/key claims from AI message"""
        # Common field patterns in metabolic modeling
        field_patterns = [
            r"(?:contains?|includes?|has|with)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:field|key|column|data)",
            r"(?:the|a)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:is|was|are|were)",
        ]

        claimed_fields = set()
        for pattern in field_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                # Filter for reasonable field names
                if len(match) > 2 and match.lower() not in [
                    "the",
                    "and",
                    "for",
                    "with",
                    "data",
                    "file",
                ]:
                    claimed_fields.add(match.lower())

        return claimed_fields

    def validate_file_outputs(self, audit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate file outputs exist and match descriptions

        Args:
            audit_data: Complete audit record for a tool execution

        Returns:
            File validation result with detailed analysis
        """
        result = {
            "verification_type": "file_validation",
            "audit_id": audit_data.get("audit_id", "unknown"),
            "tool_name": audit_data.get("tool_name", "unknown"),
            "timestamp": audit_data.get("timestamp", ""),
            "issues": [],
            "verifications": [],
            "confidence_score": 1.0,
            "details": {},
        }

        # Get claimed files from audit data
        output_data = audit_data.get("output", {})
        claimed_files = output_data.get("files", [])

        if not claimed_files:
            result["verifications"].append("No file outputs claimed")
            return result

        file_validations = []
        for file_path in claimed_files:
            file_result = self._validate_single_file(file_path, audit_data)
            file_validations.append(file_result)

            if file_result["valid"]:
                result["verifications"].append(f"File validated: {file_path}")
            else:
                result["issues"].extend(file_result["issues"])

        result["details"]["file_validations"] = file_validations

        # Calculate confidence score
        if file_validations:
            valid_count = sum(1 for fv in file_validations if fv["valid"])
            result["confidence_score"] = valid_count / len(file_validations)

        return result

    def _validate_single_file(self, file_path: str, audit_data: Dict) -> Dict[str, Any]:
        """Validate a single file output"""
        file_obj = Path(file_path)
        result = {"file_path": file_path, "valid": True, "issues": [], "details": {}}

        # Check if file exists
        if not file_obj.exists():
            result["valid"] = False
            result["issues"].append(f"File does not exist: {file_path}")
            return result

        # Basic file properties
        try:
            stat = file_obj.stat()
            result["details"] = {
                "size_bytes": stat.st_size,
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "readable": file_obj.is_file() and stat.st_size > 0,
            }

            # Check if file is empty (potential issue)
            if stat.st_size == 0:
                result["issues"].append(f"File is empty: {file_path}")

            # Validate file format based on extension
            self._validate_file_format(file_obj, result)

        except Exception as e:
            result["valid"] = False
            result["issues"].append(f"Error accessing file {file_path}: {str(e)}")

        return result

    def _validate_file_format(self, file_obj: Path, result: Dict):
        """Validate file format based on extension"""
        extension = file_obj.suffix.lower()

        try:
            if extension == ".json":
                with open(file_obj, "r") as f:
                    json.load(f)
                result["details"]["format_valid"] = True
            elif extension == ".csv":
                # Basic CSV validation - check if it has at least one comma or tab
                with open(file_obj, "r") as f:
                    first_line = f.readline()
                    if "," in first_line or "\t" in first_line:
                        result["details"]["format_valid"] = True
                    else:
                        result["issues"].append(
                            "CSV file appears to have no delimiters"
                        )
            elif extension in [".xml", ".sbml"]:
                # Basic XML validation - check for XML tags
                with open(file_obj, "r") as f:
                    content = f.read(1000)  # Read first 1KB
                    if "<" in content and ">" in content:
                        result["details"]["format_valid"] = True
                    else:
                        result["issues"].append(
                            "XML/SBML file appears to have no XML tags"
                        )
            else:
                result["details"]["format_valid"] = "unknown_format"

        except Exception as e:
            result["issues"].append(f"Format validation failed: {str(e)}")
            result["details"]["format_valid"] = False

    def cross_reference_console_output(
        self, audit_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Cross-reference console output with structured results

        Args:
            audit_data: Complete audit record for a tool execution

        Returns:
            Cross-reference analysis result
        """
        result = {
            "verification_type": "console_crossref",
            "audit_id": audit_data.get("audit_id", "unknown"),
            "tool_name": audit_data.get("tool_name", "unknown"),
            "timestamp": audit_data.get("timestamp", ""),
            "issues": [],
            "verifications": [],
            "confidence_score": 1.0,
            "details": {},
        }

        # Extract console and structured outputs
        output_data = audit_data.get("output", {})
        console_output = output_data.get("console", {})
        structured_output = output_data.get("structured", {})

        stdout = (
            console_output.get("stdout", "")
            if isinstance(console_output, dict)
            else str(console_output)
        )
        stderr = (
            console_output.get("stderr", "") if isinstance(console_output, dict) else ""
        )

        if not stdout and not stderr:
            result["verifications"].append("No console output to cross-reference")
            return result

        # 1. Error consistency check
        self._check_error_consistency(stdout, stderr, structured_output, result)

        # 2. Progress vs completion consistency
        self._check_progress_consistency(stdout, stderr, structured_output, result)

        # 3. Numerical value consistency
        self._check_console_numerical_consistency(
            stdout, stderr, structured_output, result
        )

        # 4. File operation consistency
        self._check_file_operation_consistency(stdout, stderr, output_data, result)

        # Calculate confidence score
        total_checks = len(result["verifications"]) + len(result["issues"])
        if total_checks > 0:
            result["confidence_score"] = len(result["verifications"]) / total_checks

        return result

    def _check_error_consistency(
        self, stdout: str, stderr: str, structured_output: Dict, result: Dict
    ):
        """Check consistency between console errors and structured success status"""
        console_text = (stdout + " " + stderr).lower()
        has_console_errors = any(
            error_word in console_text
            for error_word in ["error", "exception", "failed", "traceback", "critical"]
        )

        structured_success = structured_output.get("success", True)
        structured_error = structured_output.get("error")

        if has_console_errors and structured_success and not structured_error:
            result["issues"].append(
                "Console shows errors but structured output indicates success"
            )
        elif not has_console_errors and not structured_success:
            result["issues"].append(
                "Structured output shows failure but no console errors"
            )
        elif has_console_errors and not structured_success:
            result["verifications"].append(
                "Console errors consistent with structured failure"
            )
        else:
            result["verifications"].append("Error reporting consistency verified")

    def _check_progress_consistency(
        self, stdout: str, stderr: str, structured_output: Dict, result: Dict
    ):
        """Check consistency between progress messages and completion status"""
        console_text = stdout + " " + stderr

        # Look for progress indicators
        progress_patterns = [
            r"(\d+)%",
            r"(\d+)/(\d+)",
            r"step (\d+)",
            r"iteration (\d+)",
            r"processing",
            r"computing",
            r"analyzing",
            r"running",
        ]

        has_progress = any(
            re.search(pattern, console_text, re.IGNORECASE)
            for pattern in progress_patterns
        )
        has_completion = any(
            word in console_text.lower()
            for word in ["completed", "finished", "done", "success", "optimal"]
        )

        structured_success = structured_output.get("success", True)

        if has_progress and not has_completion and structured_success:
            result["issues"].append(
                "Console shows progress but no completion message despite success"
            )
        elif has_completion and structured_success:
            result["verifications"].append(
                "Console completion messages match structured success"
            )
        elif not has_progress and not has_completion:
            result["verifications"].append(
                "Minimal console output consistent with structured results"
            )

    def _check_console_numerical_consistency(
        self, stdout: str, stderr: str, structured_output: Dict, result: Dict
    ):
        """Check numerical values mentioned in console vs structured output"""
        console_text = stdout + " " + stderr
        structured_data = structured_output.get("data", {})

        # Extract numbers from console output
        console_numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)", console_text)
        console_numbers = [
            float(n) for n in console_numbers if n.replace(".", "").isdigit()
        ]

        # Extract numbers from structured data
        structured_numbers = []
        self._extract_numbers_recursive(structured_data, structured_numbers)

        if console_numbers and structured_numbers:
            # Check if any console numbers match structured numbers (with tolerance)
            matches = 0
            for console_num in console_numbers:
                for struct_num in structured_numbers:
                    if abs(console_num - struct_num) < max(
                        abs(console_num) * 0.01, 1e-6
                    ):
                        matches += 1
                        break

            if matches > 0:
                result["verifications"].append(
                    f"Console numbers consistent with structured data ({matches} matches)"
                )
            else:
                result["issues"].append(
                    "No numerical consistency between console and structured output"
                )

        result["details"]["console_numbers"] = console_numbers[:10]  # Limit for brevity
        result["details"]["structured_numbers"] = structured_numbers[:10]

    def _extract_numbers_recursive(self, obj: Any, numbers: List[float]):
        """Recursively extract numerical values from structured data"""
        if isinstance(obj, (int, float)):
            numbers.append(float(obj))
        elif isinstance(obj, dict):
            for value in obj.values():
                self._extract_numbers_recursive(value, numbers)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_numbers_recursive(item, numbers)

    def _check_file_operation_consistency(
        self, stdout: str, stderr: str, output_data: Dict, result: Dict
    ):
        """Check file operation mentions in console vs actual file outputs"""
        console_text = stdout + " " + stderr
        actual_files = output_data.get("files", [])

        # Look for file operation mentions in console
        file_mentions = re.findall(
            r"(?:writ|sav|creat)\w*\s+(?:to\s+)?([^\s]+\.(?:csv|json|xml|txt|html))",
            console_text,
            re.IGNORECASE,
        )

        if file_mentions and actual_files:
            mentioned_names = {Path(f).name for f in file_mentions}
            actual_names = {Path(f).name for f in actual_files}

            if mentioned_names & actual_names:
                result["verifications"].append(
                    "Console file operations match actual file outputs"
                )
            else:
                result["issues"].append("Console mentions files not in actual outputs")
        elif file_mentions and not actual_files:
            result["issues"].append(
                "Console mentions file operations but no files created"
            )
        elif not file_mentions and actual_files:
            result["verifications"].append(
                "Files created without console noise (acceptable)"
            )
        else:
            result["verifications"].append("No file operations to cross-reference")


# Statistical Analysis and Pattern Detection Functions


def analyze_multiple_tool_runs(
    logs_dir: Optional[Union[str, Path]] = None,
    tool_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform statistical analysis across multiple tool runs for pattern detection

    Args:
        logs_dir: Directory containing audit logs
        tool_name: Filter by specific tool name
        session_id: Filter by specific session

    Returns:
        Statistical analysis results with pattern detection
    """
    detector = HallucinationDetector(logs_dir)
    logs_path = detector.logs_dir

    # Collect audit files
    audit_files = []
    search_pattern = "*/tool_audits/*.json"

    if session_id:
        search_pattern = f"{session_id}/tool_audits/*.json"

    for audit_file in logs_path.glob(search_pattern):
        try:
            with open(audit_file, "r") as f:
                audit_data = json.load(f)

            if tool_name and audit_data.get("tool_name") != tool_name:
                continue

            audit_files.append(audit_data)
        except Exception:
            continue

    if not audit_files:
        return {
            "error": "No audit files found matching criteria",
            "tool_name": tool_name,
            "session_id": session_id,
            "total_files": 0,
        }

    # Statistical analysis
    analysis = {
        "tool_name": tool_name,
        "session_id": session_id,
        "total_runs": len(audit_files),
        "date_range": _calculate_date_range(audit_files),
        "success_rate": _calculate_success_rate(audit_files),
        "execution_times": _analyze_execution_times(audit_files),
        "error_patterns": _analyze_error_patterns(audit_files),
        "file_output_patterns": _analyze_file_patterns(audit_files),
        "hallucination_indicators": _detect_hallucination_patterns(
            audit_files, detector
        ),
        "recommendations": [],
    }

    # Generate recommendations
    analysis["recommendations"] = _generate_recommendations(analysis)

    return analysis


def _calculate_date_range(audit_files: List[Dict]) -> Dict[str, str]:
    """Calculate date range of audit files"""
    timestamps = [
        audit.get("timestamp", "") for audit in audit_files if audit.get("timestamp")
    ]
    if not timestamps:
        return {"start": "unknown", "end": "unknown"}

    timestamps.sort()
    return {"start": timestamps[0][:19], "end": timestamps[-1][:19]}


def _calculate_success_rate(audit_files: List[Dict]) -> Dict[str, Any]:
    """Calculate success rate statistics"""
    successes = sum(
        1 for audit in audit_files if audit.get("execution", {}).get("success", False)
    )
    total = len(audit_files)

    return {
        "successful": successes,
        "failed": total - successes,
        "total": total,
        "percentage": (successes / total * 100) if total > 0 else 0,
    }


def _analyze_execution_times(audit_files: List[Dict]) -> Dict[str, Any]:
    """Analyze execution time patterns"""
    times = [
        audit.get("execution", {}).get("duration_seconds", 0) for audit in audit_files
    ]
    times = [t for t in times if t > 0]

    if not times:
        return {"error": "No execution times available"}

    return {
        "count": len(times),
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
        "outliers": _detect_time_outliers(times),
    }


def _detect_time_outliers(times: List[float]) -> List[float]:
    """Detect execution time outliers using IQR method"""
    if len(times) < 4:
        return []

    sorted_times = sorted(times)
    q1_idx = len(sorted_times) // 4
    q3_idx = 3 * len(sorted_times) // 4

    q1 = sorted_times[q1_idx]
    q3 = sorted_times[q3_idx]
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return [t for t in times if t < lower_bound or t > upper_bound]


def _analyze_error_patterns(audit_files: List[Dict]) -> Dict[str, Any]:
    """Analyze error patterns across runs"""
    error_types = {}
    console_errors = []

    for audit in audit_files:
        execution = audit.get("execution", {})
        if not execution.get("success", True):
            error = execution.get("error")
            if error:
                error_types[str(error)] = error_types.get(str(error), 0) + 1

        # Check console output for errors
        console = audit.get("output", {}).get("console", {})
        stdout = (
            console.get("stdout", "") if isinstance(console, dict) else str(console)
        )
        stderr = console.get("stderr", "") if isinstance(console, dict) else ""

        console_text = (stdout + " " + stderr).lower()
        if any(
            error_word in console_text
            for error_word in ["error", "exception", "failed"]
        ):
            console_errors.append(audit.get("audit_id", "unknown"))

    return {
        "structural_errors": error_types,
        "console_error_count": len(console_errors),
        "console_error_audits": console_errors[:10],  # Limit for brevity
        "most_common_error": (
            max(error_types.items(), key=lambda x: x[1]) if error_types else None
        ),
    }


def _analyze_file_patterns(audit_files: List[Dict]) -> Dict[str, Any]:
    """Analyze file output patterns"""
    file_extensions = {}
    file_counts = []
    missing_files = 0

    for audit in audit_files:
        files = audit.get("output", {}).get("files", [])
        file_counts.append(len(files))

        for file_path in files:
            if not Path(file_path).exists():
                missing_files += 1

            ext = Path(file_path).suffix.lower()
            if ext:
                file_extensions[ext] = file_extensions.get(ext, 0) + 1

    return {
        "file_count_stats": {
            "mean": statistics.mean(file_counts) if file_counts else 0,
            "median": statistics.median(file_counts) if file_counts else 0,
            "max": max(file_counts) if file_counts else 0,
            "total_files": sum(file_counts),
        },
        "file_extensions": dict(
            sorted(file_extensions.items(), key=lambda x: x[1], reverse=True)
        ),
        "missing_files": missing_files,
        "file_reliability": (
            ((sum(file_counts) - missing_files) / sum(file_counts) * 100)
            if sum(file_counts) > 0
            else 100
        ),
    }


def _detect_hallucination_patterns(
    audit_files: List[Dict], detector: HallucinationDetector
) -> Dict[str, Any]:
    """Detect hallucination patterns across multiple runs"""
    hallucination_scores = []
    consistent_failures = 0
    message_data_mismatches = 0

    for audit in audit_files:
        # Run hallucination detection on each audit
        verification = detector.verify_tool_claims(audit)
        hallucination_scores.append(verification["confidence_score"])

        if len(verification["issues"]) > 0:
            if any("claims" in issue for issue in verification["issues"]):
                message_data_mismatches += 1

        # Check for consistent failure patterns
        execution = audit.get("execution", {})
        output = audit.get("output", {}).get("structured", {})
        if not execution.get("success") and output.get("success"):
            consistent_failures += 1

    avg_confidence = (
        statistics.mean(hallucination_scores) if hallucination_scores else 1.0
    )

    return {
        "average_confidence": avg_confidence,
        "low_confidence_runs": sum(1 for score in hallucination_scores if score < 0.7),
        "message_data_mismatches": message_data_mismatches,
        "consistent_failure_patterns": consistent_failures,
        "reliability_grade": _calculate_reliability_grade(
            avg_confidence, consistent_failures, len(audit_files)
        ),
    }


def _calculate_reliability_grade(
    avg_confidence: float, failures: int, total: int
) -> str:
    """Calculate overall reliability grade"""
    failure_rate = failures / total if total > 0 else 0

    if avg_confidence >= 0.9 and failure_rate <= 0.05:
        return "A+ (Excellent)"
    elif avg_confidence >= 0.8 and failure_rate <= 0.1:
        return "A (Very Good)"
    elif avg_confidence >= 0.7 and failure_rate <= 0.2:
        return "B (Good)"
    elif avg_confidence >= 0.6 and failure_rate <= 0.3:
        return "C (Fair)"
    else:
        return "D (Poor - Review Recommended)"


def _generate_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on analysis"""
    recommendations = []

    # Success rate recommendations
    success_rate = analysis["success_rate"]["percentage"]
    if success_rate < 80:
        recommendations.append(
            f"Low success rate ({success_rate:.1f}%) - investigate tool reliability"
        )

    # Execution time recommendations
    exec_times = analysis.get("execution_times", {})
    if "outliers" in exec_times and len(exec_times["outliers"]) > 0:
        recommendations.append(
            f"Found {len(exec_times['outliers'])} execution time outliers - check for performance issues"
        )

    # Error pattern recommendations
    error_patterns = analysis["error_patterns"]
    if error_patterns["console_error_count"] > 0:
        recommendations.append(
            f"{error_patterns['console_error_count']} runs had console errors - review error handling"
        )

    # File reliability recommendations
    file_patterns = analysis["file_output_patterns"]
    if file_patterns["missing_files"] > 0:
        recommendations.append(
            f"{file_patterns['missing_files']} claimed files are missing - verify file creation logic"
        )

    # Hallucination recommendations
    hallucination = analysis["hallucination_indicators"]
    if hallucination["average_confidence"] < 0.8:
        recommendations.append(
            f"Low hallucination confidence ({hallucination['average_confidence']:.2f}) - review AI output accuracy"
        )

    if hallucination["message_data_mismatches"] > 0:
        recommendations.append(
            f"{hallucination['message_data_mismatches']} message-data mismatches - check AI claim accuracy"
        )

    if not recommendations:
        recommendations.append("Tool execution appears reliable - no issues detected")

    return recommendations


def detect_common_hallucination_patterns(
    logs_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Detect common hallucination patterns across all tools and sessions

    Args:
        logs_dir: Directory containing audit logs

    Returns:
        Common hallucination patterns and their frequencies
    """
    detector = HallucinationDetector(logs_dir)
    logs_path = detector.logs_dir

    # Collect all audit files
    audit_files = []
    for audit_file in logs_path.glob("*/tool_audits/*.json"):
        try:
            with open(audit_file, "r") as f:
                audit_data = json.load(f)
            audit_files.append(audit_data)
        except Exception:
            continue

    if not audit_files:
        return {"error": "No audit files found", "total_files": 0}

    # Pattern detection
    patterns = {
        "total_audits": len(audit_files),
        "date_range": _calculate_date_range(audit_files),
        "common_patterns": {
            "success_failure_mismatches": 0,
            "numerical_claim_errors": 0,
            "file_claim_errors": 0,
            "console_structure_mismatches": 0,
            "empty_data_claims": 0,
            "missing_ai_messages": 0,
        },
        "pattern_examples": {},
        "tool_reliability_ranking": {},
        "overall_confidence": 1.0,
    }

    confidence_scores = []
    tool_issues = {}

    for audit in audit_files:
        tool_name = audit.get("tool_name", "unknown")
        if tool_name not in tool_issues:
            tool_issues[tool_name] = {"total": 0, "issues": 0}

        tool_issues[tool_name]["total"] += 1

        # Run comprehensive verification
        claim_verification = detector.verify_tool_claims(audit)
        file_verification = detector.validate_file_outputs(audit)
        console_verification = detector.cross_reference_console_output(audit)

        # Aggregate confidence scores
        avg_confidence = statistics.mean(
            [
                claim_verification["confidence_score"],
                file_verification["confidence_score"],
                console_verification["confidence_score"],
            ]
        )
        confidence_scores.append(avg_confidence)

        # Count pattern types
        all_issues = (
            claim_verification["issues"]
            + file_verification["issues"]
            + console_verification["issues"]
        )

        if all_issues:
            tool_issues[tool_name]["issues"] += 1

            # Categorize issues
            for issue in all_issues:
                issue_lower = issue.lower()
                if "success" in issue_lower and "fail" in issue_lower:
                    patterns["common_patterns"]["success_failure_mismatches"] += 1
                elif "claim" in issue_lower and any(
                    word in issue_lower for word in ["number", "value", "count"]
                ):
                    patterns["common_patterns"]["numerical_claim_errors"] += 1
                elif "file" in issue_lower:
                    patterns["common_patterns"]["file_claim_errors"] += 1
                elif "console" in issue_lower:
                    patterns["common_patterns"]["console_structure_mismatches"] += 1
                elif "empty" in issue_lower or "no data" in issue_lower:
                    patterns["common_patterns"]["empty_data_claims"] += 1
                elif "missing" in issue_lower and "message" in issue_lower:
                    patterns["common_patterns"]["missing_ai_messages"] += 1

    # Calculate overall confidence
    patterns["overall_confidence"] = (
        statistics.mean(confidence_scores) if confidence_scores else 1.0
    )

    # Rank tools by reliability
    tool_rankings = []
    for tool_name, stats in tool_issues.items():
        if stats["total"] > 0:
            reliability = 1.0 - (stats["issues"] / stats["total"])
            tool_rankings.append(
                (tool_name, reliability, stats["total"], stats["issues"])
            )

    tool_rankings.sort(key=lambda x: x[1], reverse=True)  # Sort by reliability
    patterns["tool_reliability_ranking"] = {
        tool: {"reliability": rel, "total_runs": total, "issues": issues}
        for tool, rel, total, issues in tool_rankings
    }

    # Generate pattern examples
    patterns["pattern_examples"] = {
        "most_reliable_tool": tool_rankings[0][0] if tool_rankings else None,
        "least_reliable_tool": tool_rankings[-1][0] if tool_rankings else None,
        "total_issues_found": sum(patterns["common_patterns"].values()),
        "issue_rate": (
            sum(patterns["common_patterns"].values()) / len(audit_files)
            if audit_files
            else 0
        ),
    }

    return patterns
