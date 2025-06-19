#!/usr/bin/env python3
"""
Validation Comparison Utility

Tool for comparing validation results across different runs to track
performance trends, regressions, and improvements during development.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_validation_summary(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load validation summary from JSON file"""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def find_validation_files(results_dir: Path) -> List[Path]:
    """Find all validation summary files"""
    pattern = "validation_summary_*.json"
    files = list(results_dir.glob(pattern))

    # Add latest file if it exists
    latest_file = results_dir / "latest_validation_summary.json"
    if latest_file.exists():
        files.append(latest_file)

    return sorted(files)


def extract_key_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics for comparison"""
    return {
        "total_tests": summary.get("total_tests", 0),
        "passed_tests": summary.get("passed_tests", 0),
        "success_rate": summary.get("system_performance", {}).get(
            "overall_success_rate", 0.0
        ),
        "quality_score": summary.get("average_quality_score", 0.0),
        "execution_time": summary.get("average_execution_time", 0.0),
        "artifacts_generated": summary.get("system_performance", {}).get(
            "total_artifacts_generated", 0
        ),
        "hypotheses_generated": summary.get("system_performance", {}).get(
            "total_hypotheses_generated", 0
        ),
    }


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 100.0 if new_value > 0 else 0.0
    return ((new_value - old_value) / old_value) * 100


def format_change(change: float, is_improvement_positive: bool = True) -> str:
    """Format change with appropriate indicators"""
    if abs(change) < 0.1:
        return "±0.0%"

    indicator = ""
    if change > 0:
        indicator = "📈" if is_improvement_positive else "📉"
    else:
        indicator = "📉" if is_improvement_positive else "📈"

    return f"{indicator} {change:+.1f}%"


def compare_two_validations(
    old_summary: Dict[str, Any], new_summary: Dict[str, Any]
) -> None:
    """Compare two validation summaries"""
    old_metrics = extract_key_metrics(old_summary)
    new_metrics = extract_key_metrics(new_summary)

    old_date = old_summary.get("validation_date", "Unknown")
    new_date = new_summary.get("validation_date", "Unknown")

    print("=" * 80)
    print("VALIDATION COMPARISON REPORT")
    print("=" * 80)
    print(f"Previous: {old_date}")
    print(f"Current:  {new_date}")
    print()

    print("📊 PERFORMANCE COMPARISON")
    print("-" * 50)

    metrics_info = [
        ("success_rate", "Success Rate", True, "%"),
        ("quality_score", "Quality Score", True, ""),
        ("execution_time", "Execution Time", False, "s"),
        ("artifacts_generated", "Artifacts Generated", True, ""),
        ("hypotheses_generated", "Hypotheses Generated", True, ""),
    ]

    for metric_key, metric_name, improvement_positive, unit in metrics_info:
        old_val = old_metrics.get(metric_key, 0)
        new_val = new_metrics.get(metric_key, 0)
        change = calculate_percentage_change(old_val, new_val)
        change_str = format_change(change, improvement_positive)

        if unit == "%":
            print(
                f"{metric_name:20}: {old_val*100:5.1f}% → {new_val*100:5.1f}% {change_str}"
            )
        elif unit == "s":
            print(f"{metric_name:20}: {old_val:5.1f}s → {new_val:5.1f}s {change_str}")
        else:
            print(f"{metric_name:20}: {old_val:5.0f} → {new_val:5.0f} {change_str}")

    print()
    print("🎯 TEST CATEGORIES COMPARISON")
    print("-" * 50)

    categories = [
        "integration_results",
        "performance_results",
        "quality_results",
        "regression_results",
    ]
    for category in categories:
        old_cat = old_summary.get(category, {})
        new_cat = new_summary.get(category, {})

        old_rate = old_cat.get("passed", 0) / max(old_cat.get("total", 1), 1) * 100
        new_rate = new_cat.get("passed", 0) / max(new_cat.get("total", 1), 1) * 100

        category_name = category.replace("_results", "").title()
        change = calculate_percentage_change(old_rate, new_rate)
        change_str = format_change(change, True)

        print(f"{category_name:15}: {old_rate:5.1f}% → {new_rate:5.1f}% {change_str}")

    print()
    print("⚠️  ALERTS")
    print("-" * 50)

    alerts = []

    # Success rate regression
    if new_metrics["success_rate"] < old_metrics["success_rate"]:
        alerts.append("🚨 SUCCESS RATE DECREASED - Investigate test failures")

    # Quality regression (>5% drop)
    quality_change = calculate_percentage_change(
        old_metrics["quality_score"], new_metrics["quality_score"]
    )
    if quality_change < -5:
        alerts.append(f"🚨 QUALITY REGRESSION: {quality_change:.1f}% - Review changes")

    # Performance regression (>20% slower)
    time_change = calculate_percentage_change(
        old_metrics["execution_time"], new_metrics["execution_time"]
    )
    if time_change > 20:
        alerts.append(f"🚨 PERFORMANCE REGRESSION: {time_change:.1f}% slower")

    # Low artifact/hypothesis generation
    if new_metrics["artifacts_generated"] < old_metrics["artifacts_generated"]:
        alerts.append("⚠️  ARTIFACT GENERATION DECREASED")

    if new_metrics["hypotheses_generated"] < old_metrics["hypotheses_generated"]:
        alerts.append("⚠️  HYPOTHESIS GENERATION DECREASED")

    if alerts:
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("  ✅ No significant issues detected")

    print()
    print("💡 RECOMMENDATIONS")
    print("-" * 50)

    recommendations = []

    if quality_change > 5:
        recommendations.append(
            "📈 Quality improved significantly - Consider documenting changes"
        )
    elif quality_change < -2:
        recommendations.append("📉 Quality decreased - Review recent changes")

    if time_change < -10:
        recommendations.append("⚡ Performance improved - Great optimization work!")
    elif time_change > 10:
        recommendations.append("🐌 Performance decreased - Consider optimization")

    if new_metrics["success_rate"] == 1.0:
        recommendations.append("✅ Perfect test success rate maintained")

    if recommendations:
        for rec in recommendations:
            print(f"  {rec}")
    else:
        print("  📊 Performance is stable - Continue current approach")


def show_trend_analysis(results_dir: Path, max_files: int = 10) -> None:
    """Show trend analysis across multiple validation runs"""
    files = find_validation_files(results_dir)

    if len(files) < 2:
        print("❌ Need at least 2 validation files for trend analysis")
        return

    # Load recent files
    recent_files = files[-max_files:]
    summaries = []

    for file_path in recent_files:
        summary = load_validation_summary(file_path)
        if summary:
            summaries.append((file_path.name, summary))

    if len(summaries) < 2:
        print("❌ Could not load enough validation files for trend analysis")
        return

    print("=" * 80)
    print("VALIDATION TREND ANALYSIS")
    print("=" * 80)
    print(f"Analyzing {len(summaries)} recent validation runs")
    print()

    print("📈 QUALITY SCORE TREND")
    print("-" * 50)

    for i, (filename, summary) in enumerate(summaries):
        quality = summary.get("average_quality_score", 0) * 100
        success = (
            summary.get("system_performance", {}).get("overall_success_rate", 0) * 100
        )
        date = summary.get("validation_date", "Unknown")[:19]  # Trim microseconds

        trend_indicator = ""
        if i > 0:
            prev_quality = summaries[i - 1][1].get("average_quality_score", 0) * 100
            if quality > prev_quality + 1:
                trend_indicator = " 📈"
            elif quality < prev_quality - 1:
                trend_indicator = " 📉"
            else:
                trend_indicator = " ➡️"

        print(
            f"{date}: Quality {quality:5.1f}% | Success {success:5.1f}%{trend_indicator}"
        )

    # Calculate overall trend
    first_quality = summaries[0][1].get("average_quality_score", 0)
    last_quality = summaries[-1][1].get("average_quality_score", 0)
    overall_change = calculate_percentage_change(first_quality, last_quality)

    print()
    print(
        f"📊 Overall Trend: {format_change(overall_change, True)} over {len(summaries)} runs"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare validation results across runs"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/reasoning_validation",
        help="Directory containing validation results",
    )
    parser.add_argument(
        "--mode",
        choices=["latest", "compare", "trend"],
        default="latest",
        help="Comparison mode",
    )
    parser.add_argument(
        "--file1", type=str, help="First file for comparison (compare mode)"
    )
    parser.add_argument(
        "--file2", type=str, help="Second file for comparison (compare mode)"
    )
    parser.add_argument(
        "--max-files", type=int, default=10, help="Maximum files for trend analysis"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)

    if args.mode == "latest":
        # Compare latest with previous
        files = find_validation_files(results_dir)
        if len(files) < 2:
            print("❌ Need at least 2 validation files for comparison")
            sys.exit(1)

        # Get latest and second-to-latest
        latest_file = results_dir / "latest_validation_summary.json"
        if latest_file.exists():
            second_latest = [f for f in files if f != latest_file][-1]
        else:
            latest_file = files[-1]
            second_latest = files[-2]

        old_summary = load_validation_summary(second_latest)
        new_summary = load_validation_summary(latest_file)

        if old_summary and new_summary:
            compare_two_validations(old_summary, new_summary)
        else:
            print("❌ Could not load validation files")

    elif args.mode == "compare":
        if not args.file1 or not args.file2:
            print("❌ --file1 and --file2 required for compare mode")
            sys.exit(1)

        file1_path = results_dir / args.file1
        file2_path = results_dir / args.file2

        old_summary = load_validation_summary(file1_path)
        new_summary = load_validation_summary(file2_path)

        if old_summary and new_summary:
            compare_two_validations(old_summary, new_summary)
        else:
            print("❌ Could not load validation files")

    elif args.mode == "trend":
        show_trend_analysis(results_dir, args.max_files)


if __name__ == "__main__":
    main()
