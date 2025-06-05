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
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        """Serialize tool result for audit storage"""
        try:
            if hasattr(result, "dict"):
                return result.dict()
            elif isinstance(result, dict):
                return result
            else:
                return {"raw_result": str(result), "type": type(result).__name__}
        except Exception:
            return {"raw_result": str(result), "serialization_error": True}


# Global auditor instance
_global_auditor = ToolAuditor()


def get_auditor() -> ToolAuditor:
    """Get the global tool auditor instance"""
    return _global_auditor


def set_audit_session(session_id: str):
    """Set the current audit session ID"""
    _global_auditor.set_session_id(session_id)
