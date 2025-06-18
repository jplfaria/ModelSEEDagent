#!/usr/bin/env python3
"""
Performance Test Script for ModelSEEDagent

Runs reproducible test scenarios to compare performance before/after optimizations.
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path


class PerformanceTest:
    """Runs reproducible performance tests"""

    def __init__(self):
        self.repo_root = Path(__file__).parent.parent
        self.baseline_file = self.repo_root / "scripts" / "baseline_metrics.json"

    def load_baseline(self):
        """Load baseline metrics for comparison"""
        if self.baseline_file.exists():
            with open(self.baseline_file, "r") as f:
                return json.load(f)
        return None

    def run_test_scenario(self, scenario_name: str = "e_coli_analysis") -> dict:
        """Run the standard E. coli analysis test scenario"""

        print(f"🧪 Running performance test: {scenario_name}")
        start_time = time.time()

        # Create a test input script
        test_script = f"""
# E. coli Comprehensive Analysis Test
# This script runs the exact same queries as the baseline

# Query 1: Initial comprehensive analysis
I need a comprehensive metabolic analysis of E. col for our data examples ecoli cor emodel

# Query 2: Detailed growth rate analysis
explore the predicted growth rate in more detail and give me a summary of what you learn

# Exit
exit
"""

        # Run the test using subprocess to capture output
        try:
            # Use modelseed-agent in batch mode with input
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(test_script)
                input_file = f.name

            print(f"⏱️  Test started at: {datetime.now().strftime('%H:%M:%S')}")

            # Run the command with full path
            agent_path = self.repo_root / "modelseed-agent"
            result = subprocess.run(
                [str(agent_path), "interactive"],
                input=test_script,
                text=True,
                capture_output=True,
                timeout=600,  # 10 minute timeout
            )

            end_time = time.time()
            total_time = end_time - start_time

            print(f"✅ Test completed in: {total_time:.1f}s")

            # Clean up
            os.unlink(input_file)

            # Extract metrics from output
            metrics = self._extract_metrics_from_output(
                result.stdout, result.stderr, total_time
            )

            return {
                "scenario_name": scenario_name,
                "timestamp": datetime.now().isoformat(),
                "total_time": total_time,
                "success": result.returncode == 0,
                "metrics": metrics,
                "stdout_sample": result.stdout[-1000:] if result.stdout else "",
                "stderr_sample": result.stderr[-1000:] if result.stderr else "",
            }

        except subprocess.TimeoutExpired:
            print("❌ Test timed out after 10 minutes")
            return {"error": "timeout", "total_time": 600}
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return {"error": str(e), "total_time": 0}

    def _extract_metrics_from_output(
        self, stdout: str, stderr: str, total_time: float
    ) -> dict:
        """Extract performance metrics from command output"""

        metrics = {
            "total_time": total_time,
            "argollm_initializations": 0,
            "model_loads": 0,
            "glpk_warnings": 0,
            "tools_executed": 0,
            "session_created": False,
            "errors": [],
        }

        combined_output = stdout + stderr

        # Count ArgoLLM initializations
        metrics["argollm_initializations"] = combined_output.count(
            "ArgoLLM initialized"
        )

        # Count model loads
        metrics["model_loads"] = combined_output.count("Successfully loaded model")

        # Count GLPK warnings
        metrics["glpk_warnings"] = combined_output.count(
            "glpk doesn't support setting the optimality tolerance"
        )

        # Count tools executed (look for tool completion messages)
        tool_patterns = [
            "completed successfully",
            "Tool .* executed",
            "run_metabolic_fba",
            "run_flux_variability_analysis",
            "find_minimal_media",
            "analyze_essentiality",
        ]

        for pattern in tool_patterns:
            import re

            matches = re.findall(pattern, combined_output, re.IGNORECASE)
            metrics["tools_executed"] += len(matches)

        # Check for session creation
        metrics["session_created"] = "Created new session" in combined_output

        # Look for errors
        error_patterns = ["ERROR", "FAILED", "Exception", "Error:"]
        for pattern in error_patterns:
            if pattern in combined_output:
                metrics["errors"].append(pattern)

        return metrics

    def compare_with_baseline(self, test_result: dict) -> dict:
        """Compare test results with baseline metrics"""

        baseline = self.load_baseline()
        if not baseline:
            print("⚠️  No baseline found for comparison")
            return {"comparison": "no_baseline"}

        baseline_time = baseline["session_metrics"]["total_session_time"]
        test_time = test_result["total_time"]
        speedup = baseline_time / test_time if test_time > 0 else 0

        comparison = {
            "baseline_time": baseline_time,
            "test_time": test_time,
            "speedup": speedup,
            "improvements": {},
        }

        # Compare specific metrics
        baseline_issues = baseline["performance_issues"]
        test_metrics = test_result["metrics"]

        comparison["improvements"] = {
            "argollm_inits": {
                "baseline": baseline_issues["argollm_initializations"],
                "current": test_metrics["argollm_initializations"],
                "improvement": baseline_issues["argollm_initializations"]
                - test_metrics["argollm_initializations"],
            },
            "model_loads": {
                "baseline": baseline_issues["model_loads_from_disk"],
                "current": test_metrics["model_loads"],
                "improvement": baseline_issues["model_loads_from_disk"]
                - test_metrics["model_loads"],
            },
            "glpk_warnings": {
                "baseline": baseline_issues["glpk_tolerance_warnings"],
                "current": test_metrics["glpk_warnings"],
                "improvement": baseline_issues["glpk_tolerance_warnings"]
                - test_metrics["glpk_warnings"],
            },
        }

        return comparison

    def print_results(self, test_result: dict, comparison: dict = None):
        """Print human-readable test results"""

        print("\n" + "=" * 60)
        print("🧪 PERFORMANCE TEST RESULTS")
        print("=" * 60)

        print(f"Scenario: {test_result.get('scenario_name', 'unknown')}")
        print(f"Total Time: {test_result['total_time']:.1f}s")
        print(f"Success: {'✅' if test_result.get('success', False) else '❌'}")

        if "metrics" in test_result:
            metrics = test_result["metrics"]
            print(f"\n📊 Metrics:")
            print(f"  • ArgoLLM Initializations: {metrics['argollm_initializations']}")
            print(f"  • Model Loads: {metrics['model_loads']}")
            print(f"  • GLPK Warnings: {metrics['glpk_warnings']}")
            print(f"  • Tools Executed: {metrics['tools_executed']}")

        if comparison and "speedup" in comparison:
            print(f"\n🚀 Performance Comparison:")
            print(f"  • Baseline Time: {comparison['baseline_time']:.1f}s")
            print(f"  • Current Time: {comparison['test_time']:.1f}s")
            print(f"  • Speedup: {comparison['speedup']:.1f}x")

            if "improvements" in comparison:
                imp = comparison["improvements"]
                print(f"\n✨ Improvements:")
                for metric, data in imp.items():
                    improvement = data["improvement"]
                    symbol = (
                        "✅" if improvement > 0 else "❌" if improvement < 0 else "➖"
                    )
                    print(
                        f"  • {metric}: {data['baseline']} → {data['current']} ({symbol} {improvement:+d})"
                    )

    def save_results(self, test_result: dict, output_file: str):
        """Save test results to file"""
        with open(output_file, "w") as f:
            json.dump(test_result, f, indent=2)
        print(f"💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run ModelSEEDagent performance tests")
    parser.add_argument(
        "--scenario", default="e_coli_analysis", help="Test scenario to run"
    )
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument(
        "--compare-baseline", action="store_true", help="Compare results with baseline"
    )

    args = parser.parse_args()

    tester = PerformanceTest()

    # Run the test
    result = tester.run_test_scenario(args.scenario)

    # Compare with baseline if requested
    comparison = None
    if args.compare_baseline:
        comparison = tester.compare_with_baseline(result)

    # Print results
    tester.print_results(result, comparison)

    # Save results if output file specified
    if args.output:
        tester.save_results(result, args.output)


if __name__ == "__main__":
    main()
