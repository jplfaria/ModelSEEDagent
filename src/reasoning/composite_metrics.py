"""
Composite Metrics Calculator for ModelSEEDagent Phase 3

Advanced composite scoring system for multi-dimensional quality optimization,
providing balanced assessment across biological accuracy, transparency, synthesis,
confidence calibration, and methodological rigor.
"""

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MetricConfiguration:
    """Configuration for composite metric calculation"""

    weights: Dict[str, float]
    normalization_method: str = (
        "weighted_average"  # weighted_average, geometric_mean, harmonic_mean
    )
    penalty_threshold: float = 0.5  # Threshold below which penalties apply
    penalty_factor: float = 0.1  # Penalty multiplier for low scores
    consistency_weight: float = 0.1  # Weight for consistency bonus
    excellence_threshold: float = 0.9  # Threshold for excellence bonus
    excellence_bonus: float = 0.05  # Bonus for exceptional performance


@dataclass
class CompositeScore:
    """Comprehensive composite scoring result"""

    overall_score: float
    weighted_score: float
    geometric_mean: float
    harmonic_mean: float
    consistency_bonus: float
    excellence_bonus: float
    penalty_applied: float
    grade: str
    ranking_percentile: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


class CompositeMetricsCalculator:
    """
    Advanced composite metrics calculation system

    Provides multiple scoring methodologies to balance different quality aspects:
    - Weighted averages for standard scoring
    - Geometric means for multiplicative effects
    - Harmonic means for conservative estimates
    - Consistency bonuses for balanced performance
    - Excellence bonuses for exceptional quality
    - Penalty systems for critical deficiencies
    """

    def __init__(self):
        self.default_weights = {
            "biological_accuracy": 0.30,
            "reasoning_transparency": 0.25,
            "synthesis_effectiveness": 0.20,
            "confidence_calibration": 0.15,
            "methodological_rigor": 0.10,
        }

        self.scoring_methods = {
            "weighted_average": self._calculate_weighted_average,
            "geometric_mean": self._calculate_geometric_mean,
            "harmonic_mean": self._calculate_harmonic_mean,
            "robust_composite": self._calculate_robust_composite,
        }

        self.grade_boundaries = {
            "A+": 0.95,
            "A": 0.90,
            "B+": 0.85,
            "B": 0.80,
            "C+": 0.75,
            "C": 0.70,
            "D": 0.60,
            "F": 0.0,
        }

        logger.info(
            "CompositeMetricsCalculator initialized with advanced scoring methods"
        )

    def calculate_composite_score(
        self,
        quality_scores: Dict[str, float],
        metric_config: Optional[MetricConfiguration] = None,
    ) -> CompositeScore:
        """
        Calculate comprehensive composite score with multiple methodologies

        Args:
            quality_scores: Dictionary of dimension scores (0.0-1.0)
            metric_config: Optional configuration for calculation

        Returns:
            Comprehensive composite scoring result
        """
        try:
            if not quality_scores:
                raise ValueError("Quality scores dictionary cannot be empty")

            config = metric_config or self._get_default_config()

            # Validate input scores
            self._validate_scores(quality_scores)

            # Calculate base scores using different methods
            weighted_score = self._calculate_weighted_average(
                quality_scores, config.weights
            )
            geometric_mean = self._calculate_geometric_mean(
                quality_scores, config.weights
            )
            harmonic_mean = self._calculate_harmonic_mean(
                quality_scores, config.weights
            )

            # Calculate consistency metrics
            consistency_bonus = self._calculate_consistency_bonus(
                quality_scores, config.consistency_weight
            )

            # Calculate excellence bonus
            excellence_bonus = self._calculate_excellence_bonus(
                quality_scores, config.excellence_threshold, config.excellence_bonus
            )

            # Calculate penalties for critical deficiencies
            penalty_applied = self._calculate_penalties(
                quality_scores, config.penalty_threshold, config.penalty_factor
            )

            # Determine primary score based on configuration
            if config.normalization_method == "weighted_average":
                primary_score = weighted_score
            elif config.normalization_method == "geometric_mean":
                primary_score = geometric_mean
            elif config.normalization_method == "harmonic_mean":
                primary_score = harmonic_mean
            else:
                primary_score = weighted_score

            # Apply bonuses and penalties
            overall_score = (
                primary_score + consistency_bonus + excellence_bonus - penalty_applied
            )

            # Ensure score stays within bounds
            overall_score = max(0.0, min(1.0, overall_score))

            # Assign grade
            grade = self._assign_grade(overall_score)

            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                quality_scores, overall_score
            )

            composite_score = CompositeScore(
                overall_score=overall_score,
                weighted_score=weighted_score,
                geometric_mean=geometric_mean,
                harmonic_mean=harmonic_mean,
                consistency_bonus=consistency_bonus,
                excellence_bonus=excellence_bonus,
                penalty_applied=penalty_applied,
                grade=grade,
                confidence_interval=confidence_interval,
            )

            logger.info(
                f"Composite score calculated: {overall_score:.3f} ({grade}) "
                f"with {config.normalization_method} method"
            )

            return composite_score

        except Exception as e:
            logger.error(f"Composite score calculation failed: {e}")
            raise

    def calculate_ranking_percentile(
        self, current_score: float, historical_scores: List[float]
    ) -> float:
        """Calculate percentile ranking against historical scores"""
        if not historical_scores:
            return 50.0  # Default to median if no history

        sorted_scores = sorted(historical_scores)
        position = sum(1 for score in sorted_scores if score <= current_score)
        percentile = (position / len(sorted_scores)) * 100

        return percentile

    def optimize_weights(
        self,
        training_data: List[Dict[str, Any]],
        target_outcomes: List[float],
        optimization_method: str = "gradient_descent",
    ) -> Dict[str, float]:
        """
        Optimize dimension weights based on training data

        Args:
            training_data: List of quality score dictionaries
            target_outcomes: List of target composite scores
            optimization_method: Optimization algorithm to use

        Returns:
            Optimized weights dictionary
        """
        try:
            if optimization_method == "grid_search":
                return self._optimize_weights_grid_search(
                    training_data, target_outcomes
                )
            elif optimization_method == "gradient_descent":
                return self._optimize_weights_gradient_descent(
                    training_data, target_outcomes
                )
            else:
                logger.warning(f"Unknown optimization method: {optimization_method}")
                return self.default_weights

        except Exception as e:
            logger.error(f"Weight optimization failed: {e}")
            return self.default_weights

    def _get_default_config(self) -> MetricConfiguration:
        """Get default metric configuration"""
        return MetricConfiguration(
            weights=self.default_weights.copy(),
            normalization_method="weighted_average",
            penalty_threshold=0.5,
            penalty_factor=0.1,
            consistency_weight=0.05,
            excellence_threshold=0.9,
            excellence_bonus=0.03,
        )

    def _validate_scores(self, scores: Dict[str, float]):
        """Validate input scores are within valid range"""
        for dimension, score in scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Score for {dimension} ({score}) must be between 0.0 and 1.0"
                )

    def _calculate_weighted_average(
        self, scores: Dict[str, float], weights: Dict[str, float]
    ) -> float:
        """Calculate weighted average score"""
        total_weighted_score = 0.0
        total_weight = 0.0

        for dimension, score in scores.items():
            weight = weights.get(dimension, 0.0)
            total_weighted_score += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_weighted_score / total_weight

    def _calculate_geometric_mean(
        self, scores: Dict[str, float], weights: Dict[str, float]
    ) -> float:
        """Calculate weighted geometric mean"""
        product = 1.0
        total_weight = 0.0

        for dimension, score in scores.items():
            weight = weights.get(dimension, 0.0)
            if score > 0:  # Avoid log(0)
                product *= math.pow(score, weight)
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return math.pow(product, 1.0 / total_weight) if product > 0 else 0.0

    def _calculate_harmonic_mean(
        self, scores: Dict[str, float], weights: Dict[str, float]
    ) -> float:
        """Calculate weighted harmonic mean"""
        weighted_reciprocal_sum = 0.0
        total_weight = 0.0

        for dimension, score in scores.items():
            weight = weights.get(dimension, 0.0)
            if score > 0:  # Avoid division by zero
                weighted_reciprocal_sum += weight / score
                total_weight += weight

        if total_weight == 0 or weighted_reciprocal_sum == 0:
            return 0.0

        return total_weight / weighted_reciprocal_sum

    def _calculate_robust_composite(
        self, scores: Dict[str, float], weights: Dict[str, float]
    ) -> float:
        """Calculate robust composite using multiple methods"""
        weighted_avg = self._calculate_weighted_average(scores, weights)
        geometric_mean = self._calculate_geometric_mean(scores, weights)
        harmonic_mean = self._calculate_harmonic_mean(scores, weights)

        # Combine methods with emphasis on conservative estimates
        robust_score = weighted_avg * 0.5 + geometric_mean * 0.3 + harmonic_mean * 0.2

        return robust_score

    def _calculate_consistency_bonus(
        self, scores: Dict[str, float], consistency_weight: float
    ) -> float:
        """Calculate bonus for consistent performance across dimensions"""
        if not scores:
            return 0.0

        score_values = list(scores.values())
        mean_score = sum(score_values) / len(score_values)

        # Calculate standard deviation
        variance = sum((score - mean_score) ** 2 for score in score_values) / len(
            score_values
        )
        std_deviation = math.sqrt(variance)

        # Consistency bonus inversely related to standard deviation
        # Higher consistency (lower std dev) gets higher bonus
        consistency_score = max(0.0, 1.0 - (std_deviation / 0.5))  # Normalize to 0-1
        consistency_bonus = consistency_score * consistency_weight

        return consistency_bonus

    def _calculate_excellence_bonus(
        self, scores: Dict[str, float], threshold: float, bonus: float
    ) -> float:
        """Calculate bonus for exceptional performance"""
        excellent_dimensions = sum(1 for score in scores.values() if score >= threshold)
        total_dimensions = len(scores)

        if total_dimensions == 0:
            return 0.0

        excellence_ratio = excellent_dimensions / total_dimensions

        # Progressive bonus based on proportion of excellent dimensions
        if excellence_ratio >= 0.8:  # 80% or more dimensions excellent
            return bonus
        elif excellence_ratio >= 0.6:  # 60-79% excellent
            return bonus * 0.7
        elif excellence_ratio >= 0.4:  # 40-59% excellent
            return bonus * 0.4
        else:
            return 0.0

    def _calculate_penalties(
        self, scores: Dict[str, float], threshold: float, penalty_factor: float
    ) -> float:
        """Calculate penalties for critical deficiencies"""
        total_penalty = 0.0

        for dimension, score in scores.items():
            if score < threshold:
                # Progressive penalty - worse scores get higher penalties
                deficiency = threshold - score
                dimension_penalty = deficiency * penalty_factor

                # Critical dimensions get higher penalties
                if dimension in ["biological_accuracy", "methodological_rigor"]:
                    dimension_penalty *= 1.5

                total_penalty += dimension_penalty

        return total_penalty

    def _assign_grade(self, score: float) -> str:
        """Assign letter grade based on score"""
        for grade, threshold in self.grade_boundaries.items():
            if score >= threshold:
                return grade
        return "F"

    def _calculate_confidence_interval(
        self, scores: Dict[str, float], overall_score: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the composite score"""
        if not scores:
            return (overall_score, overall_score)

        # Simple confidence interval based on score variability
        score_values = list(scores.values())
        std_deviation = math.sqrt(
            sum((score - overall_score) ** 2 for score in score_values)
            / len(score_values)
        )

        # 95% confidence interval (approximately ±2 standard deviations)
        margin_of_error = 1.96 * std_deviation / math.sqrt(len(score_values))

        lower_bound = max(0.0, overall_score - margin_of_error)
        upper_bound = min(1.0, overall_score + margin_of_error)

        return (lower_bound, upper_bound)

    def _optimize_weights_grid_search(
        self, training_data: List[Dict[str, Any]], targets: List[float]
    ) -> Dict[str, float]:
        """Optimize weights using grid search"""
        # Simplified grid search implementation
        best_weights = self.default_weights.copy()
        best_error = float("inf")

        # Define search space (simplified)
        search_points = [0.1, 0.2, 0.3, 0.4, 0.5]

        for bio_weight in search_points:
            for trans_weight in search_points:
                for synth_weight in search_points:
                    remaining = 1.0 - bio_weight - trans_weight - synth_weight
                    if remaining < 0:
                        continue

                    weights = {
                        "biological_accuracy": bio_weight,
                        "reasoning_transparency": trans_weight,
                        "synthesis_effectiveness": synth_weight,
                        "confidence_calibration": remaining * 0.6,
                        "methodological_rigor": remaining * 0.4,
                    }

                    error = self._calculate_optimization_error(
                        training_data, targets, weights
                    )

                    if error < best_error:
                        best_error = error
                        best_weights = weights.copy()

        return best_weights

    def _optimize_weights_gradient_descent(
        self, training_data: List[Dict[str, Any]], targets: List[float]
    ) -> Dict[str, float]:
        """Optimize weights using gradient descent"""
        # Simplified gradient descent implementation
        weights = self.default_weights.copy()
        learning_rate = 0.01
        iterations = 100

        for _ in range(iterations):
            gradients = self._calculate_gradients(training_data, targets, weights)

            # Update weights
            for dimension in weights:
                weights[dimension] -= learning_rate * gradients[dimension]
                weights[dimension] = max(
                    0.01, min(0.8, weights[dimension])
                )  # Constrain

            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            for dimension in weights:
                weights[dimension] /= total_weight

        return weights

    def _calculate_optimization_error(
        self,
        training_data: List[Dict[str, Any]],
        targets: List[float],
        weights: Dict[str, float],
    ) -> float:
        """Calculate optimization error for given weights"""
        total_error = 0.0

        for i, data in enumerate(training_data):
            if i >= len(targets):
                break

            predicted = self._calculate_weighted_average(data, weights)
            actual = targets[i]
            total_error += (predicted - actual) ** 2

        return total_error / len(training_data)

    def _calculate_gradients(
        self,
        training_data: List[Dict[str, Any]],
        targets: List[float],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate gradients for weight optimization"""
        gradients = {dim: 0.0 for dim in weights}

        for i, data in enumerate(training_data):
            if i >= len(targets):
                break

            predicted = self._calculate_weighted_average(data, weights)
            actual = targets[i]
            error = predicted - actual

            # Calculate gradient for each dimension
            for dimension in weights:
                if dimension in data:
                    gradients[dimension] += 2 * error * data[dimension]

        # Average gradients
        for dimension in gradients:
            gradients[dimension] /= len(training_data)

        return gradients


class QualityBenchmarkManager:
    """
    Manager for quality benchmarking and performance tracking
    """

    def __init__(self, benchmark_file: Optional[str] = None):
        self.benchmark_file = benchmark_file
        self.benchmarks = self._load_benchmarks()
        self.calculator = CompositeMetricsCalculator()

        logger.info("QualityBenchmarkManager initialized")

    def add_benchmark(
        self, name: str, scores: Dict[str, float], metadata: Dict[str, Any]
    ):
        """Add a new quality benchmark"""
        composite_score = self.calculator.calculate_composite_score(scores)

        self.benchmarks[name] = {
            "scores": scores,
            "composite_score": composite_score.overall_score,
            "grade": composite_score.grade,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
        }

        self._save_benchmarks()
        logger.info(
            f"Added benchmark '{name}' with score {composite_score.overall_score:.3f}"
        )

    def compare_to_benchmarks(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Compare current scores to established benchmarks"""
        composite_score = self.calculator.calculate_composite_score(scores)

        comparisons = {}
        for name, benchmark in self.benchmarks.items():
            comparison = {
                "benchmark_score": benchmark["composite_score"],
                "current_score": composite_score.overall_score,
                "difference": composite_score.overall_score
                - benchmark["composite_score"],
                "better": composite_score.overall_score > benchmark["composite_score"],
            }
            comparisons[name] = comparison

        return {
            "current_performance": {
                "score": composite_score.overall_score,
                "grade": composite_score.grade,
            },
            "benchmark_comparisons": comparisons,
            "percentile_ranking": self._calculate_percentile_ranking(
                composite_score.overall_score
            ),
        }

    def _load_benchmarks(self) -> Dict[str, Any]:
        """Load benchmarks from file"""
        if not self.benchmark_file:
            return {}

        try:
            with open(self.benchmark_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_benchmarks(self):
        """Save benchmarks to file"""
        if not self.benchmark_file:
            return

        try:
            with open(self.benchmark_file, "w") as f:
                json.dump(self.benchmarks, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save benchmarks: {e}")

    def _calculate_percentile_ranking(self, current_score: float) -> float:
        """Calculate percentile ranking against all benchmarks"""
        all_scores = [b["composite_score"] for b in self.benchmarks.values()]
        if not all_scores:
            return 50.0

        return self.calculator.calculate_ranking_percentile(current_score, all_scores)
