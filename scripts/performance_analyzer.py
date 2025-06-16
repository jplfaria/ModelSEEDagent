#!/usr/bin/env python3
"""
Performance Analysis Tool for ModelSEEDagent

Analyzes log files to extract performance metrics and establish baselines.
Used for comparing before/after optimization results.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""

    # Session-level metrics
    session_id: str
    total_session_time: float
    total_tools_executed: int

    # LLM-related metrics
    total_llm_calls: int
    total_llm_time: float
    avg_llm_response_time: float

    # Tool execution metrics
    tool_execution_times: Dict[str, float]
    avg_tool_execution_time: float

    # System overhead metrics
    argollm_initializations: int
    model_load_events: int
    glpk_warnings: int

    # Memory/data metrics
    largest_prompt_size: int
    total_data_size_mb: float

    # Success metrics
    success_rate: float
    errors_encountered: int


class PerformanceAnalyzer:
    """Analyzes ModelSEEDagent performance logs"""

    def __init__(self, logs_dir: str):
        self.logs_dir = Path(logs_dir)

    def analyze_session(self, session_dir: str) -> PerformanceMetrics:
        """Analyze a single session directory for performance metrics"""
        session_path = Path(session_dir)
        session_id = session_path.name

        print(f"🔍 Analyzing session: {session_id}")

        # Extract metrics from different log files
        console_metrics = self._analyze_console_logs(session_path)
        audit_metrics = self._analyze_audit_trail(session_path)
        # reasoning_metrics = self._analyze_reasoning_flow(session_path)  # Currently unused

        # Combine all metrics
        metrics = PerformanceMetrics(
            session_id=session_id,
            total_session_time=console_metrics.get("total_time", 0),
            total_tools_executed=audit_metrics.get("tools_executed", 0),
            total_llm_calls=console_metrics.get("llm_calls", 0),
            total_llm_time=console_metrics.get("llm_time", 0),
            avg_llm_response_time=console_metrics.get("avg_llm_time", 0),
            tool_execution_times=audit_metrics.get("tool_times", {}),
            avg_tool_execution_time=audit_metrics.get("avg_tool_time", 0),
            argollm_initializations=console_metrics.get("argollm_inits", 0),
            model_load_events=console_metrics.get("model_loads", 0),
            glpk_warnings=console_metrics.get("glpk_warnings", 0),
            largest_prompt_size=console_metrics.get("max_prompt_size", 0),
            total_data_size_mb=console_metrics.get("total_data_mb", 0),
            success_rate=audit_metrics.get("success_rate", 0),
            errors_encountered=audit_metrics.get("errors", 0),
        )

        return metrics

    def _analyze_console_logs(self, session_path: Path) -> Dict[str, Any]:
        """Extract metrics from console debug logs"""
        console_file = session_path / "console_debug_output.jsonl"

        if not console_file.exists():
            print(f"⚠️  Console log not found: {console_file}")
            return {}

        metrics = {
            "llm_calls": 0,
            "llm_times": [],
            "argollm_inits": 0,
            "model_loads": 0,
            "glpk_warnings": 0,
            "prompt_sizes": [],
            "total_data_mb": 0,
        }

        start_time = None
        end_time = None

        try:
            with open(console_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        entry = json.loads(line)
                        timestamp = entry.get("timestamp")
                        content = entry.get("content", "")
                        metadata = entry.get("metadata", {})

                        # Track session timing
                        if timestamp:
                            ts = datetime.fromisoformat(
                                timestamp.replace("Z", "+00:00")
                            )
                            if start_time is None:
                                start_time = ts
                            end_time = ts

                        # Count LLM calls and timing
                        if "execution_time" in metadata:
                            metrics["llm_calls"] += 1
                            metrics["llm_times"].append(metadata["execution_time"])

                        # Track prompt sizes
                        if "prompt_length_chars" in metadata:
                            metrics["prompt_sizes"].append(
                                metadata["prompt_length_chars"]
                            )

                        # Look for specific patterns in content
                        if "ArgoLLM initialized" in content:
                            metrics["argollm_inits"] += 1

                        if "Successfully loaded model" in content:
                            metrics["model_loads"] += 1

                        if (
                            "glpk doesn't support setting the optimality tolerance"
                            in content
                        ):
                            metrics["glpk_warnings"] += 1

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"❌ Error analyzing console logs: {e}")
            return {}

        # Calculate derived metrics
        if start_time and end_time:
            metrics["total_time"] = (end_time - start_time).total_seconds()

        if metrics["llm_times"]:
            metrics["llm_time"] = sum(metrics["llm_times"])
            metrics["avg_llm_time"] = metrics["llm_time"] / len(metrics["llm_times"])

        if metrics["prompt_sizes"]:
            metrics["max_prompt_size"] = max(metrics["prompt_sizes"])

        return metrics

    def _analyze_audit_trail(self, session_path: Path) -> Dict[str, Any]:
        """Extract metrics from audit trail"""
        audit_file = session_path / "complete_audit_trail.json"

        if not audit_file.exists():
            print(f"⚠️  Audit trail not found: {audit_file}")
            return {}

        try:
            with open(audit_file, "r") as f:
                audit_data = json.load(f)

            execution_sequence = audit_data.get("execution_sequence", [])

            tool_times = {}
            total_tools = 0
            successful_tools = 0
            errors = 0

            for step in execution_sequence:
                if step.get("step") == "tool_execution":
                    total_tools += 1
                    tool_name = step.get("data", {}).get("tool_name", "unknown")

                    # Extract timing if available
                    if "execution_time" in step.get("data", {}):
                        execution_time = step["data"]["execution_time"]
                        tool_times[tool_name] = execution_time

                    # Check success
                    if step.get("data", {}).get("success", False):
                        successful_tools += 1
                    else:
                        errors += 1

            success_rate = (successful_tools / total_tools) if total_tools > 0 else 0
            avg_tool_time = (
                sum(tool_times.values()) / len(tool_times) if tool_times else 0
            )

            return {
                "tools_executed": total_tools,
                "tool_times": tool_times,
                "avg_tool_time": avg_tool_time,
                "success_rate": success_rate,
                "errors": errors,
            }

        except Exception as e:
            print(f"❌ Error analyzing audit trail: {e}")
            return {}

    def _analyze_reasoning_flow(self, session_path: Path) -> Dict[str, Any]:
        """Extract metrics from AI reasoning flow"""
        reasoning_file = session_path / "ai_reasoning_flow.json"

        if not reasoning_file.exists():
            return {}

        try:
            with open(reasoning_file, "r") as f:
                reasoning_data = json.load(f)

            # Extract reasoning-specific metrics
            return {
                "reasoning_steps": len(reasoning_data.get("steps", [])),
                "decision_points": len(
                    [
                        s
                        for s in reasoning_data.get("steps", [])
                        if "decision" in s.get("type", "")
                    ]
                ),
            }

        except Exception as e:
            print(f"❌ Error analyzing reasoning flow: {e}")
            return {}

    def create_baseline_report(self, session_dirs: List[str]) -> Dict[str, Any]:
        """Create a comprehensive baseline performance report"""
        all_metrics = []

        for session_dir in session_dirs:
            if os.path.exists(session_dir):
                metrics = self.analyze_session(session_dir)
                all_metrics.append(metrics)

        if not all_metrics:
            raise ValueError("No valid sessions found for analysis")

        # Aggregate metrics
        report = {
            "baseline_timestamp": datetime.now().isoformat(),
            "sessions_analyzed": len(all_metrics),
            "session_details": [],
            "aggregate_metrics": {},
        }

        # Individual session details
        for metrics in all_metrics:
            report["session_details"].append(
                {
                    "session_id": metrics.session_id,
                    "total_time": metrics.total_session_time,
                    "tools_executed": metrics.total_tools_executed,
                    "llm_calls": metrics.total_llm_calls,
                    "argollm_inits": metrics.argollm_initializations,
                    "model_loads": metrics.model_load_events,
                    "glpk_warnings": metrics.glpk_warnings,
                    "success_rate": metrics.success_rate,
                }
            )

        # Aggregate statistics
        total_times = [m.total_session_time for m in all_metrics]
        llm_calls = [m.total_llm_calls for m in all_metrics]
        argollm_inits = [m.argollm_initializations for m in all_metrics]
        model_loads = [m.model_load_events for m in all_metrics]

        report["aggregate_metrics"] = {
            "avg_session_time": sum(total_times) / len(total_times),
            "avg_llm_calls": sum(llm_calls) / len(llm_calls),
            "avg_argollm_inits": sum(argollm_inits) / len(argollm_inits),
            "avg_model_loads": sum(model_loads) / len(model_loads),
            "total_sessions": len(all_metrics),
        }

        return report

    def save_baseline(self, report: Dict[str, Any], output_file: str):
        """Save baseline report to file"""
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"📊 Baseline report saved to: {output_file}")

    def print_summary(self, report: Dict[str, Any]):
        """Print a human-readable summary of the baseline"""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE BASELINE SUMMARY")
        print("=" * 60)

        agg = report["aggregate_metrics"]
        print(f"Sessions Analyzed: {report['sessions_analyzed']}")
        print(f"Average Session Time: {agg['avg_session_time']:.1f}s")
        print(f"Average LLM Calls: {agg['avg_llm_calls']:.1f}")
        print(f"Average ArgoLLM Inits: {agg['avg_argollm_inits']:.1f}")
        print(f"Average Model Loads: {agg['avg_model_loads']:.1f}")

        print("\n🎯 OPTIMIZATION TARGETS:")
        print(
            f"  • Reduce ArgoLLM initializations from {agg['avg_argollm_inits']:.0f} to 1-2"
        )
        print(f"  • Reduce model loads from {agg['avg_model_loads']:.0f} to 1")
        print(
            f"  • Target session time: <90s (current: {agg['avg_session_time']:.1f}s)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ModelSEEDagent performance logs"
    )
    parser.add_argument(
        "--session-dir", type=str, help="Single session directory to analyze"
    )
    parser.add_argument(
        "--baseline-sessions",
        nargs="+",
        help="Multiple session directories for baseline creation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="performance_baseline.json",
        help="Output file for baseline report",
    )

    args = parser.parse_args()

    analyzer = PerformanceAnalyzer("logs")

    if args.session_dir:
        # Analyze single session
        metrics = analyzer.analyze_session(args.session_dir)
        print(f"Session {metrics.session_id}:")
        print(f"  Total time: {metrics.total_session_time:.1f}s")
        print(f"  Tools executed: {metrics.total_tools_executed}")
        print(f"  LLM calls: {metrics.total_llm_calls}")
        print(f"  ArgoLLM inits: {metrics.argollm_initializations}")

    elif args.baseline_sessions:
        # Create baseline from multiple sessions
        report = analyzer.create_baseline_report(args.baseline_sessions)
        analyzer.print_summary(report)
        analyzer.save_baseline(report, args.output)

    else:
        print("Please specify either --session-dir or --baseline-sessions")


if __name__ == "__main__":
    main()
