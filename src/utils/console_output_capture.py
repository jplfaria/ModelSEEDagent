"""
Console Output Capture System for ModelSEEDagent

This module provides functionality to capture and persist valuable CLI debug information
that includes AI reasoning flows, formatted results, and console output that is normally
only displayed to the user but not saved in persistent logs.

Part of the CLI Debug Capture implementation roadmap Phase 1.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConsoleOutputCapture:
    """
    Captures console debug output and AI reasoning flows for persistent storage.

    This class provides the foundation for capturing valuable CLI debug information
    that is currently only displayed in the console but not saved to logs.

    Features:
    - AI reasoning step capture
    - Formatted output preservation
    - Console debug message logging
    - Configurable capture levels
    - Performance-optimized file operations
    """

    def __init__(
        self,
        run_dir: Path,
        enabled: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize console output capture system.

        Args:
            run_dir: Directory for storing capture files
            enabled: Whether capture is enabled (default: False for safety)
            config: Optional configuration dictionary
        """
        self.run_dir = Path(run_dir)
        self.enabled = enabled
        self.config = config or {}

        # File paths for different types of captured data
        self.console_log_file = self.run_dir / "console_debug_output.jsonl"
        self.reasoning_flow_file = self.run_dir / "ai_reasoning_flow.json"
        self.formatted_results_file = self.run_dir / "formatted_results.json"

        # Configuration options with defaults
        self.max_size_mb = self.config.get("console_output_max_size_mb", 50)
        self.capture_level = self.config.get("debug_capture_level", "basic")

        # Internal state
        self.reasoning_steps = []
        self.total_size_bytes = 0

        # Create directory if it doesn't exist
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Console output capture initialized: {self.run_dir}")

    def capture_reasoning_step(
        self, step_type: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Capture AI reasoning steps as they occur.

        Args:
            step_type: Type of reasoning step ("tool_selection", "decision_analysis", "conclusion")
            content: The actual reasoning content or AI output
            metadata: Additional context information
        """
        if not self.enabled:
            return

        # Check size limits
        if self._check_size_limit(content):
            logger.warning("Console capture size limit reached, skipping capture")
            return

        reasoning_entry = {
            "timestamp": datetime.now().isoformat(),
            "step_type": step_type,
            "content": content,
            "metadata": metadata or {},
            "capture_level": self.capture_level,
        }

        # Store in memory for later retrieval
        self.reasoning_steps.append(reasoning_entry)

        try:
            # Append to JSONL file for streaming/incremental processing
            with open(self.console_log_file, "a", encoding="utf-8") as f:
                json_line = json.dumps(reasoning_entry, default=str)
                f.write(json_line + "\n")
                self.total_size_bytes += len(json_line.encode("utf-8"))

            logger.debug(f"Captured reasoning step: {step_type}")

        except Exception as e:
            logger.error(f"Failed to capture reasoning step: {e}")

    def capture_formatted_output(
        self, output_type: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Capture final formatted results.

        Args:
            output_type: Type of output ("final_analysis", "tool_result", "summary")
            content: The formatted output content
            metadata: Additional context about the output
        """
        if not self.enabled:
            return

        # Check size limits
        if self._check_size_limit(content):
            logger.warning(
                "Console capture size limit reached, skipping formatted output"
            )
            return

        formatted_entry = {
            "timestamp": datetime.now().isoformat(),
            "output_type": output_type,
            "content": content,
            "metadata": metadata or {},
            "content_length": len(content),
        }

        try:
            # Load existing data or create new
            existing_data = []
            if self.formatted_results_file.exists():
                with open(self.formatted_results_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)

            # Append new entry
            existing_data.append(formatted_entry)

            # Save back to file
            with open(self.formatted_results_file, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, default=str)

            logger.debug(f"Captured formatted output: {output_type}")

        except Exception as e:
            logger.error(f"Failed to capture formatted output: {e}")

    def capture_console_message(
        self, level: str, message: str, context: Optional[Dict[str, Any]] = None
    ):
        """
        Capture console debug messages with context.

        Args:
            level: Log level ("DEBUG", "INFO", "WARNING", "ERROR")
            message: The console message
            context: Additional context information
        """
        if not self.enabled:
            return

        # Only capture if level is appropriate for current capture level
        if not self._should_capture_level(level):
            return

        console_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "context": context or {},
            "capture_level": self.capture_level,
        }

        self.capture_reasoning_step(
            "console_message", json.dumps(console_entry), context
        )

    def get_complete_reasoning_flow(self) -> List[Dict[str, Any]]:
        """
        Retrieve complete reasoning flow for analysis.

        Returns:
            List of all captured reasoning steps
        """
        if not self.enabled:
            return []

        return self.reasoning_steps.copy()

    def get_formatted_results(self) -> List[Dict[str, Any]]:
        """
        Retrieve all captured formatted results.

        Returns:
            List of formatted output entries
        """
        if not self.enabled or not self.formatted_results_file.exists():
            return []

        try:
            with open(self.formatted_results_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read formatted results: {e}")
            return []

    def get_capture_summary(self) -> Dict[str, Any]:
        """
        Get summary of captured information.

        Returns:
            Summary statistics about captured data
        """
        if not self.enabled:
            return {"enabled": False}

        reasoning_count = len(self.reasoning_steps)
        formatted_count = len(self.get_formatted_results())

        return {
            "enabled": True,
            "capture_level": self.capture_level,
            "reasoning_steps_captured": reasoning_count,
            "formatted_results_captured": formatted_count,
            "total_size_mb": self.total_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_mb,
            "console_log_file": str(self.console_log_file),
            "reasoning_flow_file": str(self.reasoning_flow_file),
            "formatted_results_file": str(self.formatted_results_file),
        }

    def save_reasoning_flow(self):
        """
        Save complete reasoning flow to dedicated file.
        """
        if not self.enabled or not self.reasoning_steps:
            return

        reasoning_data = {
            "capture_session": {
                "timestamp": datetime.now().isoformat(),
                "capture_level": self.capture_level,
                "total_steps": len(self.reasoning_steps),
            },
            "reasoning_steps": self.reasoning_steps,
        }

        try:
            with open(self.reasoning_flow_file, "w", encoding="utf-8") as f:
                json.dump(reasoning_data, f, indent=2, default=str)
            logger.info(f"Saved reasoning flow: {self.reasoning_flow_file}")
        except Exception as e:
            logger.error(f"Failed to save reasoning flow: {e}")

    def cleanup_old_captures(self, max_sessions: int = 10):
        """
        Clean up old capture files to manage disk usage.

        Args:
            max_sessions: Maximum number of capture sessions to retain
        """
        if not self.enabled:
            return

        try:
            # Get all capture directories
            parent_dir = self.run_dir.parent
            if not parent_dir.exists():
                return

            # Find all run directories
            run_dirs = [
                d
                for d in parent_dir.iterdir()
                if d.is_dir() and d.name.startswith("realtime_run_")
            ]

            # Sort by creation time
            run_dirs.sort(key=lambda x: x.stat().st_ctime, reverse=True)

            # Remove oldest sessions beyond the limit
            for old_dir in run_dirs[max_sessions:]:
                try:
                    # Remove capture files only, preserve other logs
                    for capture_file in [
                        "console_debug_output.jsonl",
                        "ai_reasoning_flow.json",
                        "formatted_results.json",
                    ]:
                        file_path = old_dir / capture_file
                        if file_path.exists():
                            file_path.unlink()

                    logger.debug(f"Cleaned up old capture files: {old_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {old_dir}: {e}")

        except Exception as e:
            logger.error(f"Failed to cleanup old captures: {e}")

    def _check_size_limit(self, content: str) -> bool:
        """Check if adding content would exceed size limit."""
        content_size = len(content.encode("utf-8"))
        max_bytes = self.max_size_mb * 1024 * 1024
        return (self.total_size_bytes + content_size) > max_bytes

    def _should_capture_level(self, level: str) -> bool:
        """Determine if message level should be captured based on capture level."""
        level_hierarchy = {
            "basic": ["ERROR", "WARNING"],
            "detailed": ["ERROR", "WARNING", "INFO"],
            "comprehensive": ["ERROR", "WARNING", "INFO", "DEBUG"],
        }

        allowed_levels = level_hierarchy.get(self.capture_level, ["ERROR"])
        return level in allowed_levels

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save data before closing."""
        if self.enabled:
            self.save_reasoning_flow()
