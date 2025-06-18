"""
Reasoning Validation Suite for ModelSEEDagent Phase 3

Automated systematic testing and validation framework for comprehensive
reasoning quality assessment across all dimensions and bias detection.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import Phase 3 components
try:
    from ..reasoning.composite_metrics import (
        CompositeMetricsCalculator,
        QualityBenchmarkManager,
    )
    from ..reasoning.quality_validator import (
        QualityAssessment,
        ReasoningQualityValidator,
        ReasoningTrace,
    )
    from .reasoning_diversity_checker import (
        BiasDetectionResult,
        DiversityMetrics,
        ReasoningDiversityChecker,
    )
except ImportError:
    from reasoning_diversity_checker import (
        BiasDetectionResult,
        DiversityMetrics,
        ReasoningDiversityChecker,
    )

    from reasoning.composite_metrics import (
        CompositeMetricsCalculator,
        QualityBenchmarkManager,
    )
    from reasoning.quality_validator import (
        QualityAssessment,
        ReasoningQualityValidator,
        ReasoningTrace,
    )

logger = logging.getLogger(__name__)


class ValidationTestCase:
    """Individual validation test case"""

    def __init__(
        self,
        name: str,
        reasoning_trace: ReasoningTrace,
        expected_outcomes: Optional[Dict[str, Any]] = None,
        test_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.reasoning_trace = reasoning_trace
        self.expected_outcomes = expected_outcomes
        self.test_metadata = test_metadata or {}
        self.result: Optional[Dict[str, Any]] = None
        self.timestamp: Optional[datetime] = None


class ValidationSuite:
    """
    Comprehensive automated validation suite for reasoning quality assessment

    Provides systematic testing across multiple dimensions:
    - Individual quality assessment validation
    - Composite metrics calculation verification
    - Bias detection and diversity analysis
    - Performance benchmarking
    - Regression testing
    - Quality trend analysis
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("validation_results")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize validation components
        self.quality_validator = ReasoningQualityValidator()
        self.metrics_calculator = CompositeMetricsCalculator()
        self.diversity_checker = ReasoningDiversityChecker()
        self.benchmark_manager = QualityBenchmarkManager()

        # Test suite state
        self.test_cases: List[ValidationTestCase] = []
        self.validation_results: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}

        logger.info(
            f"ValidationSuite initialized with output directory: {self.output_dir}"
        )

    def add_test_case(
        self,
        name: str,
        reasoning_trace: ReasoningTrace,
        expected_outcomes: Optional[Dict[str, Any]] = None,
        test_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a test case to the validation suite"""
        test_case = ValidationTestCase(
            name, reasoning_trace, expected_outcomes, test_metadata
        )
        self.test_cases.append(test_case)
        logger.info(f"Added test case: {name}")

    def run_comprehensive_validation(
        self,
        include_performance_tests: bool = True,
        include_bias_detection: bool = True,
        include_benchmark_comparison: bool = True,
    ) -> Dict[str, Any]:
        """
        Run comprehensive validation across all test cases

        Args:
            include_performance_tests: Include performance benchmarking
            include_bias_detection: Include bias and diversity analysis
            include_benchmark_comparison: Include benchmark comparisons

        Returns:
            Comprehensive validation results
        """
        try:
            start_time = time.time()

            logger.info(
                f"Starting comprehensive validation of {len(self.test_cases)} test cases"
            )

            # Run individual test case validation
            individual_results = self._run_individual_validations()

            # Run composite metrics validation
            composite_results = self._run_composite_metrics_validation()

            # Run bias detection and diversity analysis
            bias_diversity_results = {}
            if include_bias_detection:
                bias_diversity_results = self._run_bias_diversity_validation()

            # Run performance benchmarking
            performance_results = {}
            if include_performance_tests:
                performance_results = self._run_performance_validation()

            # Run benchmark comparisons
            benchmark_results = {}
            if include_benchmark_comparison:
                benchmark_results = self._run_benchmark_validation()

            # Aggregate results
            total_time = time.time() - start_time

            comprehensive_results = {
                "validation_summary": {
                    "total_test_cases": len(self.test_cases),
                    "validation_timestamp": datetime.now().isoformat(),
                    "total_validation_time": total_time,
                    "validation_successful": True,
                },
                "individual_validations": individual_results,
                "composite_metrics": composite_results,
                "bias_diversity_analysis": bias_diversity_results,
                "performance_benchmarks": performance_results,
                "benchmark_comparisons": benchmark_results,
                "quality_trends": self._analyze_quality_trends(),
                "recommendations": self._generate_validation_recommendations(),
            }

            # Save results
            self._save_validation_results(comprehensive_results)

            logger.info(
                f"Comprehensive validation completed in {total_time:.2f} seconds"
            )

            return comprehensive_results

        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            raise

    def run_regression_testing(
        self, baseline_results: Dict[str, Any], tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """
        Run regression testing against baseline results

        Args:
            baseline_results: Previous validation results for comparison
            tolerance: Acceptable degradation threshold (0.05 = 5%)

        Returns:
            Regression test results
        """
        try:
            current_results = self.run_comprehensive_validation()

            regression_analysis = {
                "regression_timestamp": datetime.now().isoformat(),
                "baseline_timestamp": baseline_results.get(
                    "validation_summary", {}
                ).get("validation_timestamp"),
                "tolerance_threshold": tolerance,
                "dimension_comparisons": {},
                "overall_regression_status": "pass",
                "failed_dimensions": [],
                "performance_regression": {},
            }

            # Compare individual dimension performance
            current_individual = current_results.get("individual_validations", {})
            baseline_individual = baseline_results.get("individual_validations", {})

            for test_name in current_individual:
                if test_name in baseline_individual:
                    comparison = self._compare_test_results(
                        current_individual[test_name],
                        baseline_individual[test_name],
                        tolerance,
                    )
                    regression_analysis["dimension_comparisons"][test_name] = comparison

                    if not comparison["within_tolerance"]:
                        regression_analysis["failed_dimensions"].append(test_name)
                        regression_analysis["overall_regression_status"] = "fail"

            # Compare performance metrics
            current_perf = current_results.get("performance_benchmarks", {})
            baseline_perf = baseline_results.get("performance_benchmarks", {})

            regression_analysis["performance_regression"] = (
                self._compare_performance_metrics(
                    current_perf, baseline_perf, tolerance
                )
            )

            logger.info(
                f"Regression testing completed: {regression_analysis['overall_regression_status']}"
            )

            return regression_analysis

        except Exception as e:
            logger.error(f"Regression testing failed: {e}")
            raise

    def generate_quality_report(
        self, validation_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate comprehensive quality assessment report

        Args:
            validation_results: Optional validation results, runs new validation if None

        Returns:
            Formatted quality report
        """
        if validation_results is None:
            validation_results = self.run_comprehensive_validation()

        report_sections = []

        # Executive Summary
        report_sections.append(self._generate_executive_summary(validation_results))

        # Quality Dimension Analysis
        report_sections.append(self._generate_dimension_analysis(validation_results))

        # Bias and Diversity Assessment
        report_sections.append(self._generate_bias_diversity_report(validation_results))

        # Performance Analysis
        report_sections.append(self._generate_performance_report(validation_results))

        # Recommendations
        report_sections.append(
            self._generate_recommendations_report(validation_results)
        )

        # Technical Details
        report_sections.append(self._generate_technical_details(validation_results))

        full_report = "\n\n".join(report_sections)

        # Save report
        report_file = (
            self.output_dir
            / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(report_file, "w") as f:
            f.write(full_report)

        logger.info(f"Quality report generated: {report_file}")

        return full_report

    def _run_individual_validations(self) -> Dict[str, Any]:
        """Run individual quality validation for each test case"""
        individual_results = {}

        for test_case in self.test_cases:
            try:
                start_time = time.time()

                # Run quality validation
                quality_assessment = self.quality_validator.validate_reasoning_quality(
                    test_case.reasoning_trace, test_case.expected_outcomes
                )

                validation_time = time.time() - start_time

                # Store results
                individual_results[test_case.name] = {
                    "quality_assessment": {
                        "overall_score": quality_assessment.overall_score,
                        "grade": quality_assessment.grade,
                        "dimension_scores": {
                            dim_name: dim.score
                            for dim_name, dim in quality_assessment.dimensions.items()
                        },
                        "composite_metrics": quality_assessment.composite_metrics,
                        "bias_flags": quality_assessment.bias_flags,
                        "recommendations": quality_assessment.recommendations,
                    },
                    "validation_metadata": {
                        "validation_time": validation_time,
                        "test_metadata": test_case.test_metadata,
                        "trace_id": test_case.reasoning_trace.trace_id,
                    },
                }

                test_case.result = individual_results[test_case.name]
                test_case.timestamp = datetime.now()

            except Exception as e:
                logger.error(f"Individual validation failed for {test_case.name}: {e}")
                individual_results[test_case.name] = {
                    "error": str(e),
                    "validation_time": 0.0,
                }

        return individual_results

    def _run_composite_metrics_validation(self) -> Dict[str, Any]:
        """Run composite metrics validation across all test cases"""
        composite_results = {
            "metric_distributions": {},
            "correlation_analysis": {},
            "weight_optimization": {},
            "consistency_analysis": {},
        }

        try:
            # Collect all quality scores
            all_scores = []
            for test_case in self.test_cases:
                if test_case.result and "quality_assessment" in test_case.result:
                    dimension_scores = test_case.result["quality_assessment"][
                        "dimension_scores"
                    ]
                    all_scores.append(dimension_scores)

            if not all_scores:
                return composite_results

            # Calculate metric distributions
            for dimension in all_scores[0].keys():
                scores = [
                    scores[dimension] for scores in all_scores if dimension in scores
                ]
                composite_results["metric_distributions"][dimension] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "std_dev": self._calculate_std_dev(scores),
                }

            # Calculate composite scores for all test cases
            composite_scores = []
            for scores in all_scores:
                composite_score = self.metrics_calculator.calculate_composite_score(
                    scores
                )
                composite_scores.append(composite_score)

            # Analyze composite score consistency
            overall_scores = [cs.overall_score for cs in composite_scores]
            composite_results["consistency_analysis"] = {
                "mean_composite_score": sum(overall_scores) / len(overall_scores),
                "composite_score_std_dev": self._calculate_std_dev(overall_scores),
                "grade_distribution": self._calculate_grade_distribution(
                    composite_scores
                ),
            }

        except Exception as e:
            logger.error(f"Composite metrics validation failed: {e}")
            composite_results["error"] = str(e)

        return composite_results

    def _run_bias_diversity_validation(self) -> Dict[str, Any]:
        """Run bias detection and diversity analysis"""
        bias_diversity_results = {
            "diversity_metrics": {},
            "bias_detection_summary": {},
            "diversity_trends": {},
            "bias_risk_assessment": {},
        }

        try:
            # Prepare reasoning traces for diversity analysis
            trace_data = []
            for test_case in self.test_cases:
                trace_dict = {
                    "trace_id": test_case.reasoning_trace.trace_id,
                    "final_conclusion": test_case.reasoning_trace.final_conclusion,
                    "steps": test_case.reasoning_trace.steps,
                    "tools_used": test_case.reasoning_trace.tools_used,
                    "confidence_claims": test_case.reasoning_trace.confidence_claims,
                }
                trace_data.append(trace_dict)

            # Run diversity assessment
            diversity_metrics = self.diversity_checker.assess_reasoning_diversity(
                trace_data
            )
            bias_diversity_results["diversity_metrics"] = {
                "vocabulary_diversity": diversity_metrics.vocabulary_diversity,
                "structural_diversity": diversity_metrics.structural_diversity,
                "tool_usage_diversity": diversity_metrics.tool_usage_diversity,
                "approach_diversity": diversity_metrics.approach_diversity,
                "hypothesis_diversity": diversity_metrics.hypothesis_diversity,
                "overall_diversity_score": diversity_metrics.overall_diversity_score,
                "bias_risk_level": diversity_metrics.bias_risk_level,
                "diversity_grade": diversity_metrics.diversity_grade,
            }

            # Run bias detection
            bias_results = self.diversity_checker.detect_bias_patterns(trace_data)
            bias_diversity_results["bias_detection_summary"] = {
                "total_bias_patterns_detected": len(bias_results),
                "bias_types": [bias.bias_type for bias in bias_results],
                "severity_distribution": self._analyze_bias_severity_distribution(
                    bias_results
                ),
                "high_severity_biases": [
                    {
                        "type": bias.bias_type,
                        "severity": bias.severity,
                        "recommendation": bias.recommendation,
                    }
                    for bias in bias_results
                    if bias.severity in ["high", "critical"]
                ],
            }

            # Generate diversity recommendations
            diversity_recommendations = (
                self.diversity_checker.generate_diversity_recommendations(
                    diversity_metrics, bias_results
                )
            )
            bias_diversity_results["diversity_recommendations"] = (
                diversity_recommendations
            )

        except Exception as e:
            logger.error(f"Bias/diversity validation failed: {e}")
            bias_diversity_results["error"] = str(e)

        return bias_diversity_results

    def _run_performance_validation(self) -> Dict[str, Any]:
        """Run performance benchmarking validation"""
        performance_results = {
            "validation_performance": {},
            "memory_usage": {},
            "throughput_analysis": {},
            "scalability_assessment": {},
        }

        try:
            # Measure validation performance
            start_time = time.time()
            memory_start = self._get_memory_usage()

            # Run a subset of validations for performance measurement
            test_subset = self.test_cases[: min(5, len(self.test_cases))]
            validation_times = []

            for test_case in test_subset:
                case_start = time.time()
                self.quality_validator.validate_reasoning_quality(
                    test_case.reasoning_trace, test_case.expected_outcomes
                )
                case_time = time.time() - case_start
                validation_times.append(case_time)

            total_time = time.time() - start_time
            memory_end = self._get_memory_usage()

            performance_results["validation_performance"] = {
                "total_validation_time": total_time,
                "average_case_time": sum(validation_times) / len(validation_times),
                "min_case_time": min(validation_times),
                "max_case_time": max(validation_times),
                "cases_per_second": len(test_subset) / total_time,
            }

            performance_results["memory_usage"] = {
                "memory_delta": memory_end - memory_start,
                "memory_per_case": (memory_end - memory_start) / len(test_subset),
            }

            # Throughput analysis
            performance_results["throughput_analysis"] = {
                "estimated_hourly_capacity": (len(test_subset) / total_time) * 3600,
                "scalability_factor": len(self.test_cases) / len(test_subset),
            }

        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            performance_results["error"] = str(e)

        return performance_results

    def _run_benchmark_validation(self) -> Dict[str, Any]:
        """Run benchmark comparison validation"""
        benchmark_results = {
            "benchmark_comparisons": {},
            "percentile_rankings": {},
            "benchmark_trends": {},
        }

        try:
            # Compare each test case against benchmarks
            for test_case in self.test_cases:
                if test_case.result and "quality_assessment" in test_case.result:
                    dimension_scores = test_case.result["quality_assessment"][
                        "dimension_scores"
                    ]

                    comparison = self.benchmark_manager.compare_to_benchmarks(
                        dimension_scores
                    )
                    benchmark_results["benchmark_comparisons"][
                        test_case.name
                    ] = comparison

            # Calculate overall benchmark performance
            all_comparisons = benchmark_results["benchmark_comparisons"].values()
            if all_comparisons:
                avg_percentile = sum(
                    comp["percentile_ranking"] for comp in all_comparisons
                ) / len(all_comparisons)

                benchmark_results["overall_benchmark_performance"] = {
                    "average_percentile_ranking": avg_percentile,
                    "above_median_count": sum(
                        1 for comp in all_comparisons if comp["percentile_ranking"] > 50
                    ),
                    "total_comparisons": len(all_comparisons),
                }

        except Exception as e:
            logger.error(f"Benchmark validation failed: {e}")
            benchmark_results["error"] = str(e)

        return benchmark_results

    def _analyze_quality_trends(self) -> Dict[str, Any]:
        """Analyze quality trends across test cases"""
        trends = {
            "dimension_trends": {},
            "performance_correlations": {},
            "quality_patterns": {},
        }

        try:
            # Collect dimension scores across all test cases
            dimension_data = defaultdict(list)

            for test_case in self.test_cases:
                if test_case.result and "quality_assessment" in test_case.result:
                    dimension_scores = test_case.result["quality_assessment"][
                        "dimension_scores"
                    ]
                    for dim, score in dimension_scores.items():
                        dimension_data[dim].append(score)

            # Calculate trends for each dimension
            for dimension, scores in dimension_data.items():
                if len(scores) > 1:
                    trends["dimension_trends"][dimension] = {
                        "mean": sum(scores) / len(scores),
                        "trend_direction": "stable",  # Simplified - could implement trend analysis
                        "volatility": self._calculate_std_dev(scores),
                        "consistency_score": 1.0
                        - (
                            self._calculate_std_dev(scores)
                            / max(0.1, sum(scores) / len(scores))
                        ),
                    }

        except Exception as e:
            logger.error(f"Quality trend analysis failed: {e}")
            trends["error"] = str(e)

        return trends

    def _generate_validation_recommendations(self) -> List[str]:
        """Generate validation-based recommendations"""
        recommendations = []

        try:
            # Analyze results and generate recommendations
            total_test_cases = len(self.test_cases)
            successful_validations = len(
                [tc for tc in self.test_cases if tc.result and "error" not in tc.result]
            )

            if successful_validations < total_test_cases:
                recommendations.append(
                    f"Address validation failures: {total_test_cases - successful_validations} of {total_test_cases} validations failed"
                )

            # Collect common issues
            common_low_dimensions = defaultdict(int)
            for test_case in self.test_cases:
                if test_case.result and "quality_assessment" in test_case.result:
                    dimension_scores = test_case.result["quality_assessment"][
                        "dimension_scores"
                    ]
                    for dim, score in dimension_scores.items():
                        if score < 0.7:  # Below acceptable threshold
                            common_low_dimensions[dim] += 1

            # Generate dimension-specific recommendations
            for dimension, count in common_low_dimensions.items():
                if count > len(self.test_cases) * 0.5:  # More than 50% of cases
                    recommendations.append(
                        f"Systematic improvement needed in {dimension.replace('_', ' ')}: "
                        f"low scores in {count}/{len(self.test_cases)} test cases"
                    )

            if not recommendations:
                recommendations.append(
                    "Validation results are satisfactory - continue current quality practices"
                )

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            recommendations.append(
                "Unable to generate recommendations due to analysis error"
            )

        return recommendations

    # Helper methods for validation analysis

    def _compare_test_results(
        self, current: Dict[str, Any], baseline: Dict[str, Any], tolerance: float
    ) -> Dict[str, Any]:
        """Compare current test results against baseline"""
        comparison = {
            "within_tolerance": True,
            "dimension_changes": {},
            "overall_score_change": 0.0,
        }

        try:
            current_scores = current.get("quality_assessment", {}).get(
                "dimension_scores", {}
            )
            baseline_scores = baseline.get("quality_assessment", {}).get(
                "dimension_scores", {}
            )

            for dimension in current_scores:
                if dimension in baseline_scores:
                    current_score = current_scores[dimension]
                    baseline_score = baseline_scores[dimension]
                    change = current_score - baseline_score

                    comparison["dimension_changes"][dimension] = {
                        "current": current_score,
                        "baseline": baseline_score,
                        "change": change,
                        "within_tolerance": abs(change) <= tolerance,
                    }

                    if abs(change) > tolerance:
                        comparison["within_tolerance"] = False

            # Overall score comparison
            current_overall = current.get("quality_assessment", {}).get(
                "overall_score", 0.0
            )
            baseline_overall = baseline.get("quality_assessment", {}).get(
                "overall_score", 0.0
            )
            comparison["overall_score_change"] = current_overall - baseline_overall

        except Exception as e:
            logger.error(f"Test result comparison failed: {e}")
            comparison["error"] = str(e)

        return comparison

    def _compare_performance_metrics(
        self, current: Dict[str, Any], baseline: Dict[str, Any], tolerance: float
    ) -> Dict[str, Any]:
        """Compare performance metrics against baseline"""
        comparison = {"performance_regression": False, "metric_changes": {}}

        try:
            performance_metrics = [
                "total_validation_time",
                "average_case_time",
                "cases_per_second",
            ]

            current_perf = current.get("validation_performance", {})
            baseline_perf = baseline.get("validation_performance", {})

            for metric in performance_metrics:
                if metric in current_perf and metric in baseline_perf:
                    current_val = current_perf[metric]
                    baseline_val = baseline_perf[metric]

                    # For time metrics, higher is worse; for throughput, lower is worse
                    if "time" in metric:
                        change_ratio = (current_val - baseline_val) / baseline_val
                        regression = change_ratio > tolerance
                    else:  # throughput metrics
                        change_ratio = (baseline_val - current_val) / baseline_val
                        regression = change_ratio > tolerance

                    comparison["metric_changes"][metric] = {
                        "current": current_val,
                        "baseline": baseline_val,
                        "change_ratio": change_ratio,
                        "regression": regression,
                    }

                    if regression:
                        comparison["performance_regression"] = True

        except Exception as e:
            logger.error(f"Performance comparison failed: {e}")
            comparison["error"] = str(e)

        return comparison

    def _analyze_bias_severity_distribution(
        self, bias_results: List[BiasDetectionResult]
    ) -> Dict[str, int]:
        """Analyze distribution of bias severity levels"""
        severity_dist = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for bias in bias_results:
            if bias.severity in severity_dist:
                severity_dist[bias.severity] += 1

        return severity_dist

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def _calculate_grade_distribution(self, composite_scores) -> Dict[str, int]:
        """Calculate distribution of quality grades"""
        grade_dist = defaultdict(int)
        for score in composite_scores:
            grade_dist[score.grade] += 1
        return dict(grade_dist)

    def _get_memory_usage(self) -> float:
        """Get current memory usage (simplified)"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0  # Fallback if psutil not available

    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"validation_results_{timestamp}.json"

        try:
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Validation results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")

    # Report Generation Methods

    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary section"""
        summary = results.get("validation_summary", {})

        return f"""# Phase 3 Reasoning Quality Validation Report

## Executive Summary

**Validation Date:** {summary.get('validation_timestamp', 'Unknown')}
**Total Test Cases:** {summary.get('total_test_cases', 0)}
**Validation Time:** {summary.get('total_validation_time', 0.0):.2f} seconds
**Status:** {'✅ PASSED' if summary.get('validation_successful', False) else '❌ FAILED'}

This report presents a comprehensive assessment of reasoning quality across {summary.get('total_test_cases', 0)} test cases, evaluating biological accuracy, reasoning transparency, synthesis effectiveness, confidence calibration, and methodological rigor."""

    def _generate_dimension_analysis(self, results: Dict[str, Any]) -> str:
        """Generate quality dimension analysis section"""
        composite = results.get("composite_metrics", {})
        distributions = composite.get("metric_distributions", {})

        analysis = "## Quality Dimension Analysis\n\n"

        for dimension, stats in distributions.items():
            analysis += f"### {dimension.replace('_', ' ').title()}\n"
            analysis += f"- **Mean Score:** {stats.get('mean', 0.0):.3f}\n"
            analysis += f"- **Range:** {stats.get('min', 0.0):.3f} - {stats.get('max', 0.0):.3f}\n"
            analysis += f"- **Consistency:** {1.0 - min(1.0, stats.get('std_dev', 0.0)):.3f}\n\n"

        return analysis

    def _generate_bias_diversity_report(self, results: Dict[str, Any]) -> str:
        """Generate bias and diversity report section"""
        bias_diversity = results.get("bias_diversity_analysis", {})
        diversity = bias_diversity.get("diversity_metrics", {})
        bias_summary = bias_diversity.get("bias_detection_summary", {})

        report = "## Bias Detection and Diversity Analysis\n\n"

        report += "### Diversity Assessment\n"
        report += f"- **Overall Diversity Score:** {diversity.get('overall_diversity_score', 0.0):.3f}\n"
        report += f"- **Diversity Grade:** {diversity.get('diversity_grade', 'N/A')}\n"
        report += (
            f"- **Bias Risk Level:** {diversity.get('bias_risk_level', 'Unknown')}\n\n"
        )

        report += "### Bias Detection Summary\n"
        report += f"- **Total Bias Patterns:** {bias_summary.get('total_bias_patterns_detected', 0)}\n"
        report += f"- **High Severity Biases:** {len(bias_summary.get('high_severity_biases', []))}\n\n"

        return report

    def _generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate performance analysis section"""
        performance = results.get("performance_benchmarks", {})
        validation_perf = performance.get("validation_performance", {})

        report = "## Performance Analysis\n\n"
        report += f"- **Average Validation Time:** {validation_perf.get('average_case_time', 0.0):.3f} seconds\n"
        report += f"- **Throughput:** {validation_perf.get('cases_per_second', 0.0):.1f} cases/second\n"
        report += f"- **Scalability Factor:** {performance.get('throughput_analysis', {}).get('scalability_factor', 1.0):.1f}x\n\n"

        return report

    def _generate_recommendations_report(self, results: Dict[str, Any]) -> str:
        """Generate recommendations section"""
        recommendations = results.get("recommendations", [])

        report = "## Recommendations\n\n"
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"

        return report

    def _generate_technical_details(self, results: Dict[str, Any]) -> str:
        """Generate technical details section"""
        return """## Technical Implementation Details

### Validation Framework Components
- **Quality Validator:** 5-dimensional assessment system
- **Composite Metrics Calculator:** Advanced scoring with multiple methodologies
- **Diversity Checker:** 8 bias detection methods
- **Benchmark Manager:** Historical performance tracking

### Assessment Dimensions
1. **Biological Accuracy** (30% weight): Domain knowledge correctness
2. **Reasoning Transparency** (25% weight): Explanation quality and clarity
3. **Synthesis Effectiveness** (20% weight): Cross-tool integration capability
4. **Confidence Calibration** (15% weight): Uncertainty estimate accuracy
5. **Methodological Rigor** (10% weight): Systematic approach adherence

### Bias Detection Coverage
- Tool selection bias
- Confirmation bias
- Anchoring bias
- Availability heuristic bias
- Template over-reliance
- Vocabulary limitations
- Approach rigidity
- Hypothesis narrowing

This validation framework provides comprehensive assessment capabilities for maintaining high-quality biochemical reasoning across all ModelSEEDagent operations."""


# Helper imports
from collections import defaultdict


def create_sample_reasoning_trace(
    trace_id: str,
    query: str,
    tools_used: List[str],
    conclusion: str,
    steps: Optional[List[Dict[str, Any]]] = None,
) -> ReasoningTrace:
    """Create a sample reasoning trace for testing"""
    return ReasoningTrace(
        trace_id=trace_id,
        query=query,
        steps=steps
        or [{"step": i, "action": f"Used {tool}"} for i, tool in enumerate(tools_used)],
        tools_used=tools_used,
        final_conclusion=conclusion,
        confidence_claims=[{"claim": "High confidence", "level": 0.8}],
        evidence_citations=["Tool output data", "Biochemical knowledge"],
        duration=2.5,
        timestamp=datetime.now(),
    )


def run_validation_demo():
    """Run a demonstration of the validation suite"""
    suite = ValidationSuite()

    # Add sample test cases
    suite.add_test_case(
        "growth_analysis_test",
        create_sample_reasoning_trace(
            "test_001",
            "Analyze E. coli growth on glucose minimal media",
            ["run_metabolic_fba", "analyze_essentiality"],
            "E. coli demonstrates robust growth on glucose minimal media with a predicted growth rate of 0.85 h⁻¹. Essential gene analysis reveals 312 genes critical for growth, representing 14.2% of the genome. The high growth rate indicates efficient glucose utilization through glycolysis and the TCA cycle.",
        ),
        {"expected_growth_rate": 0.85, "expected_essential_genes": 312},
    )

    suite.add_test_case(
        "media_optimization_test",
        create_sample_reasoning_trace(
            "test_002",
            "Optimize media composition for maximum growth",
            ["analyze_media_composition", "select_optimal_media"],
            "Media optimization analysis suggests reducing glucose concentration to 5 mM and supplementing with trace elements including iron and magnesium. This balanced composition should improve growth efficiency while reducing cost by approximately 20%.",
        ),
        {"expected_cost_reduction": 0.2},
    )

    # Run comprehensive validation
    results = suite.run_comprehensive_validation()

    # Generate quality report
    report = suite.generate_quality_report(results)

    print("Validation Suite Demonstration Complete")
    print(f"Results saved to: {suite.output_dir}")

    return results, report


if __name__ == "__main__":
    # Run demonstration
    run_validation_demo()
