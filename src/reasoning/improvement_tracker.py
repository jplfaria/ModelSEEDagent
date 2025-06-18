"""
Iterative Reasoning Improvement Tracker

This module implements continuous learning and improvement capabilities for the
ModelSEEDagent intelligence enhancement framework. It tracks reasoning quality
over time, identifies improvement patterns, and provides feedback signals for
system optimization.
"""

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ReasoningMetrics:
    """Comprehensive reasoning quality metrics"""

    # Basic Quality Metrics
    overall_quality: float
    biological_accuracy: float
    reasoning_transparency: float
    synthesis_effectiveness: float
    artifact_usage_rate: float
    hypothesis_count: int

    # Performance Metrics
    execution_time: float
    error_rate: float
    user_satisfaction: Optional[float] = None

    # Intelligence Metrics
    pattern_discovery_rate: float = 0.0
    bias_detection_accuracy: float = 0.0
    self_reflection_quality: float = 0.0
    meta_reasoning_effectiveness: float = 0.0

    # Context Metrics
    context_relevance: float = 0.0
    cross_tool_integration: float = 0.0
    novelty_score: float = 0.0

    # Timestamps
    timestamp: str = None
    analysis_id: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ImprovementPattern:
    """Represents an identified improvement pattern"""

    pattern_id: str
    pattern_type: str  # 'quality_improvement', 'efficiency_gain', 'error_reduction'
    description: str
    impact_score: float
    frequency: int
    confidence: float
    first_observed: str
    last_observed: str

    # Pattern characteristics
    trigger_conditions: List[str]
    improvement_metrics: Dict[str, float]
    implementation_suggestion: str

    # Validation
    validated: bool = False
    validation_date: Optional[str] = None


@dataclass
class LearningInsight:
    """Represents a learning insight from improvement tracking"""

    insight_id: str
    insight_type: str  # 'performance', 'quality', 'user_experience', 'efficiency'
    title: str
    description: str
    confidence: float

    # Supporting data
    evidence_count: int
    time_span_days: int
    impact_assessment: str

    # Recommendations
    recommended_actions: List[str]
    priority: str  # 'high', 'medium', 'low'

    # Metadata
    created_date: str
    status: str = "new"  # 'new', 'reviewed', 'implemented', 'dismissed'


class ImprovementTracker:
    """
    Tracks reasoning quality over time and identifies improvement opportunities
    """

    def __init__(self, data_dir: str = "results/reasoning_validation"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.metrics_file = self.data_dir / "reasoning_metrics.json"
        self.patterns_file = self.data_dir / "improvement_patterns.json"
        self.insights_file = self.data_dir / "learning_insights.json"

        # In-memory data
        self.metrics_history: List[ReasoningMetrics] = []
        self.improvement_patterns: List[ImprovementPattern] = []
        self.learning_insights: List[LearningInsight] = []

        # Configuration
        self.min_pattern_frequency = 3
        self.min_confidence_threshold = 0.7
        self.analysis_window_days = 30

        # Load existing data
        self._load_data()

        logger.info(
            f"Improvement Tracker initialized with {len(self.metrics_history)} historical metrics"
        )

    def record_analysis_metrics(self, metrics: ReasoningMetrics) -> None:
        """Record metrics from a completed analysis"""

        # Validate metrics
        if not self._validate_metrics(metrics):
            logger.warning(f"Invalid metrics for analysis {metrics.analysis_id}")
            return

        # Add to history
        self.metrics_history.append(metrics)

        # Analyze for patterns (if we have enough data)
        if len(self.metrics_history) >= 10:
            self._analyze_improvement_patterns()

        # Generate insights periodically
        if len(self.metrics_history) % 25 == 0:
            self._generate_learning_insights()

        # Persist data
        self._save_data()

        logger.info(f"Recorded metrics for analysis {metrics.analysis_id}")

    def get_quality_trend(self, days: int = 30) -> Dict[str, Any]:
        """Get quality trend analysis for the specified period"""

        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [
            m
            for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) >= cutoff_date
        ]

        if len(recent_metrics) < 5:
            return {"status": "insufficient_data", "metrics_count": len(recent_metrics)}

        # Calculate trends
        qualities = [m.overall_quality for m in recent_metrics]
        times = [m.execution_time for m in recent_metrics]
        accuracies = [m.biological_accuracy for m in recent_metrics]

        trend_analysis = {
            "period_days": days,
            "metrics_count": len(recent_metrics),
            "quality_trend": {
                "current_average": (
                    statistics.mean(qualities[-10:])
                    if len(qualities) >= 10
                    else statistics.mean(qualities)
                ),
                "period_average": statistics.mean(qualities),
                "improvement": self._calculate_trend(qualities),
                "stability": (
                    1.0 - (statistics.stdev(qualities) / statistics.mean(qualities))
                    if len(qualities) > 1
                    else 1.0
                ),
            },
            "performance_trend": {
                "average_time": statistics.mean(times),
                "efficiency_improvement": self._calculate_trend(
                    [-t for t in times]
                ),  # Negative because lower is better
                "consistency": (
                    1.0 - (statistics.stdev(times) / statistics.mean(times))
                    if len(times) > 1
                    else 1.0
                ),
            },
            "accuracy_trend": {
                "current_accuracy": (
                    statistics.mean(accuracies[-10:])
                    if len(accuracies) >= 10
                    else statistics.mean(accuracies)
                ),
                "period_accuracy": statistics.mean(accuracies),
                "improvement": self._calculate_trend(accuracies),
            },
        }

        return trend_analysis

    def get_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """Get actionable improvement recommendations based on analysis"""

        recommendations = []

        # Analyze recent performance
        recent_trend = self.get_quality_trend(days=14)

        if recent_trend.get("status") != "insufficient_data":
            # Quality recommendations
            if recent_trend["quality_trend"]["improvement"] < -0.05:
                recommendations.append(
                    {
                        "type": "quality_decline",
                        "priority": "high",
                        "title": "Quality Decline Detected",
                        "description": "Recent analysis quality has decreased. Review recent changes.",
                        "suggested_actions": [
                            "Review recent prompt modifications",
                            "Check for bias patterns in self-reflection data",
                            "Validate context enhancement effectiveness",
                            "Examine artifact intelligence accuracy",
                        ],
                        "confidence": 0.85,
                    }
                )

            # Performance recommendations
            if recent_trend["performance_trend"]["efficiency_improvement"] < -0.1:
                recommendations.append(
                    {
                        "type": "performance_degradation",
                        "priority": "medium",
                        "title": "Performance Degradation Observed",
                        "description": "Analysis execution time has increased recently.",
                        "suggested_actions": [
                            "Profile system performance bottlenecks",
                            "Optimize cross-phase integration efficiency",
                            "Review resource allocation algorithms",
                            "Check for memory leaks or resource contention",
                        ],
                        "confidence": 0.78,
                    }
                )

        # Pattern-based recommendations
        high_impact_patterns = [
            p
            for p in self.improvement_patterns
            if p.impact_score > 0.7 and p.frequency >= self.min_pattern_frequency
        ]

        for pattern in high_impact_patterns[:3]:  # Top 3 patterns
            recommendations.append(
                {
                    "type": "pattern_optimization",
                    "priority": "medium" if pattern.impact_score > 0.8 else "low",
                    "title": f"Optimization Opportunity: {pattern.description}",
                    "description": pattern.implementation_suggestion,
                    "suggested_actions": pattern.trigger_conditions,
                    "confidence": pattern.confidence,
                    "pattern_id": pattern.pattern_id,
                }
            )

        # Learning insight recommendations
        actionable_insights = [
            i
            for i in self.learning_insights
            if i.status == "new" and i.priority in ["high", "medium"]
        ]

        for insight in actionable_insights[:2]:  # Top 2 insights
            recommendations.append(
                {
                    "type": "learning_insight",
                    "priority": insight.priority,
                    "title": insight.title,
                    "description": insight.description,
                    "suggested_actions": insight.recommended_actions,
                    "confidence": insight.confidence,
                    "insight_id": insight.insight_id,
                }
            )

        # Sort by priority and confidence
        priority_order = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(
            key=lambda x: (priority_order.get(x["priority"], 0), x["confidence"]),
            reverse=True,
        )

        return recommendations

    def generate_improvement_report(self) -> Dict[str, Any]:
        """Generate comprehensive improvement analysis report"""

        report = {
            "generation_date": datetime.now().isoformat(),
            "analysis_period": "30_days",
            "data_summary": {
                "total_analyses": len(self.metrics_history),
                "patterns_identified": len(self.improvement_patterns),
                "learning_insights": len(self.learning_insights),
                "validated_patterns": len(
                    [p for p in self.improvement_patterns if p.validated]
                ),
            },
        }

        # Quality trends
        report["quality_analysis"] = self.get_quality_trend()

        # Top improvement patterns
        top_patterns = sorted(
            self.improvement_patterns,
            key=lambda p: p.impact_score * p.confidence,
            reverse=True,
        )[:5]

        report["top_improvement_patterns"] = [
            {
                "description": p.description,
                "impact_score": p.impact_score,
                "confidence": p.confidence,
                "frequency": p.frequency,
                "suggestion": p.implementation_suggestion,
            }
            for p in top_patterns
        ]

        # Recommendations
        report["recommendations"] = self.get_improvement_recommendations()

        # Performance summary
        if self.metrics_history:
            recent_metrics = self.metrics_history[-10:]
            report["performance_summary"] = {
                "average_quality": statistics.mean(
                    [m.overall_quality for m in recent_metrics]
                ),
                "average_execution_time": statistics.mean(
                    [m.execution_time for m in recent_metrics]
                ),
                "average_accuracy": statistics.mean(
                    [m.biological_accuracy for m in recent_metrics]
                ),
                "hypothesis_generation_rate": statistics.mean(
                    [m.hypothesis_count for m in recent_metrics]
                ),
                "artifact_usage_rate": statistics.mean(
                    [m.artifact_usage_rate for m in recent_metrics]
                ),
            }

        return report

    def _analyze_improvement_patterns(self) -> None:
        """Analyze metrics history to identify improvement patterns"""

        if len(self.metrics_history) < 10:
            return

        # Analyze quality improvement patterns
        self._identify_quality_patterns()

        # Analyze efficiency patterns
        self._identify_efficiency_patterns()

        # Analyze error reduction patterns
        self._identify_error_patterns()

        # Validate patterns
        self._validate_patterns()

    def _identify_quality_patterns(self) -> None:
        """Identify patterns related to quality improvements"""

        # Look for sustained quality improvements
        window_size = 5
        improvement_threshold = 0.05

        for i in range(len(self.metrics_history) - window_size):
            window = self.metrics_history[i : i + window_size]
            next_window = self.metrics_history[i + 1 : i + window_size + 1]

            current_avg = statistics.mean([m.overall_quality for m in window])
            next_avg = statistics.mean([m.overall_quality for m in next_window])

            if next_avg - current_avg > improvement_threshold:
                # Found potential improvement pattern
                pattern_id = f"quality_improvement_{int(time.time())}"

                # Analyze characteristics
                characteristics = self._analyze_pattern_characteristics(
                    window, next_window
                )

                pattern = ImprovementPattern(
                    pattern_id=pattern_id,
                    pattern_type="quality_improvement",
                    description=f"Quality improvement from {current_avg:.3f} to {next_avg:.3f}",
                    impact_score=min((next_avg - current_avg) * 2, 1.0),
                    frequency=1,
                    confidence=0.8,
                    first_observed=window[0].timestamp,
                    last_observed=next_window[-1].timestamp,
                    trigger_conditions=characteristics["triggers"],
                    improvement_metrics=characteristics["metrics"],
                    implementation_suggestion=characteristics["suggestion"],
                )

                # Check if similar pattern exists
                existing_pattern = self._find_similar_pattern(pattern)
                if existing_pattern:
                    existing_pattern.frequency += 1
                    existing_pattern.last_observed = pattern.last_observed
                else:
                    self.improvement_patterns.append(pattern)

    def _identify_efficiency_patterns(self) -> None:
        """Identify patterns related to efficiency improvements"""

        # Look for execution time improvements without quality loss
        for i in range(len(self.metrics_history) - 3):
            recent = self.metrics_history[i : i + 3]

            time_trend = self._calculate_trend([m.execution_time for m in recent])
            quality_trend = self._calculate_trend([m.overall_quality for m in recent])

            # Efficiency improvement: time decreasing, quality stable or improving
            if time_trend < -0.1 and quality_trend >= -0.02:
                pattern_id = f"efficiency_gain_{int(time.time())}"

                avg_time_reduction = -time_trend * statistics.mean(
                    [m.execution_time for m in recent]
                )

                pattern = ImprovementPattern(
                    pattern_id=pattern_id,
                    pattern_type="efficiency_gain",
                    description=f"Efficiency improvement: ~{avg_time_reduction:.1f}s reduction",
                    impact_score=min(abs(time_trend) * 2, 1.0),
                    frequency=1,
                    confidence=0.75,
                    first_observed=recent[0].timestamp,
                    last_observed=recent[-1].timestamp,
                    trigger_conditions=[
                        "Optimization algorithms activated",
                        "Resource allocation improved",
                    ],
                    improvement_metrics={
                        "time_reduction": avg_time_reduction,
                        "quality_maintained": quality_trend,
                    },
                    implementation_suggestion="Apply similar optimization patterns to other analysis types",
                )

                existing_pattern = self._find_similar_pattern(pattern)
                if existing_pattern:
                    existing_pattern.frequency += 1
                else:
                    self.improvement_patterns.append(pattern)

    def _identify_error_patterns(self) -> None:
        """Identify patterns related to error reduction"""

        # Look for error rate improvements
        for i in range(len(self.metrics_history) - 5):
            window = self.metrics_history[i : i + 5]

            error_trend = self._calculate_trend([m.error_rate for m in window])

            if error_trend < -0.05:  # Error rate decreasing
                pattern_id = f"error_reduction_{int(time.time())}"

                avg_error_reduction = abs(error_trend) * statistics.mean(
                    [m.error_rate for m in window]
                )

                pattern = ImprovementPattern(
                    pattern_id=pattern_id,
                    pattern_type="error_reduction",
                    description=f"Error rate reduction: {avg_error_reduction:.1%} improvement",
                    impact_score=min(abs(error_trend) * 5, 1.0),
                    frequency=1,
                    confidence=0.8,
                    first_observed=window[0].timestamp,
                    last_observed=window[-1].timestamp,
                    trigger_conditions=[
                        "Validation improvements",
                        "Self-correction mechanisms",
                    ],
                    improvement_metrics={"error_reduction": avg_error_reduction},
                    implementation_suggestion="Strengthen validation and self-correction systems",
                )

                existing_pattern = self._find_similar_pattern(pattern)
                if existing_pattern:
                    existing_pattern.frequency += 1
                else:
                    self.improvement_patterns.append(pattern)

    def _generate_learning_insights(self) -> None:
        """Generate high-level learning insights from accumulated data"""

        if len(self.metrics_history) < 25:
            return

        # Insight 1: Overall system evolution
        self._generate_evolution_insight()

        # Insight 2: User satisfaction correlation
        self._generate_satisfaction_insight()

        # Insight 3: Efficiency vs quality trade-offs
        self._generate_tradeoff_insight()

        # Insight 4: Pattern effectiveness
        self._generate_pattern_effectiveness_insight()

    def _generate_evolution_insight(self) -> None:
        """Generate insight about overall system evolution"""

        # Compare first 25% vs last 25% of data
        split_point = len(self.metrics_history) // 4
        early_metrics = self.metrics_history[:split_point]
        recent_metrics = self.metrics_history[-split_point:]

        if len(early_metrics) < 5 or len(recent_metrics) < 5:
            return

        early_quality = statistics.mean([m.overall_quality for m in early_metrics])
        recent_quality = statistics.mean([m.overall_quality for m in recent_metrics])

        improvement = (recent_quality - early_quality) / early_quality

        if improvement > 0.1:  # 10% improvement
            insight = LearningInsight(
                insight_id=f"evolution_{int(time.time())}",
                insight_type="performance",
                title="Significant System Evolution Detected",
                description=f"System quality has improved by {improvement:.1%} over time, indicating successful learning.",
                confidence=0.9,
                evidence_count=len(self.metrics_history),
                time_span_days=(
                    datetime.now() - datetime.fromisoformat(early_metrics[0].timestamp)
                ).days,
                impact_assessment="Positive - System is successfully learning and improving",
                recommended_actions=[
                    "Continue current learning algorithms",
                    "Document successful improvement patterns",
                    "Monitor for performance plateau indicators",
                ],
                priority="medium",
                created_date=datetime.now().isoformat(),
            )

            self.learning_insights.append(insight)

    def _generate_satisfaction_insight(self) -> None:
        """Generate insight about user satisfaction correlations"""

        # Find metrics with user satisfaction data
        satisfaction_metrics = [
            m for m in self.metrics_history if m.user_satisfaction is not None
        ]

        if len(satisfaction_metrics) < 10:
            return

        # Analyze correlation between satisfaction and other metrics
        satisfactions = [m.user_satisfaction for m in satisfaction_metrics]
        qualities = [m.overall_quality for m in satisfaction_metrics]

        # Simple correlation analysis
        quality_correlation = self._calculate_correlation(satisfactions, qualities)

        if quality_correlation > 0.7:
            insight = LearningInsight(
                insight_id=f"satisfaction_quality_{int(time.time())}",
                insight_type="user_experience",
                title="Strong Quality-Satisfaction Correlation",
                description=f"User satisfaction correlates strongly with analysis quality (r={quality_correlation:.2f})",
                confidence=min(quality_correlation, 0.95),
                evidence_count=len(satisfaction_metrics),
                time_span_days=30,
                impact_assessment="High - Quality improvements directly benefit users",
                recommended_actions=[
                    "Prioritize quality enhancement initiatives",
                    "Monitor user feedback for quality indicators",
                    "Implement quality-focused improvement metrics",
                ],
                priority="high",
                created_date=datetime.now().isoformat(),
            )

            self.learning_insights.append(insight)

    def _generate_tradeoff_insight(self) -> None:
        """Generate insight about efficiency vs quality trade-offs"""

        if len(self.metrics_history) < 20:
            return

        # Analyze relationship between execution time and quality
        times = [m.execution_time for m in self.metrics_history[-20:]]
        qualities = [m.overall_quality for m in self.metrics_history[-20:]]

        time_quality_correlation = self._calculate_correlation(times, qualities)

        if abs(time_quality_correlation) > 0.5:
            tradeoff_type = "positive" if time_quality_correlation > 0 else "negative"

            insight = LearningInsight(
                insight_id=f"tradeoff_{tradeoff_type}_{int(time.time())}",
                insight_type="efficiency",
                title=f"Quality-Efficiency Trade-off Identified",
                description=f"Analysis shows {tradeoff_type} correlation between time and quality (r={time_quality_correlation:.2f})",
                confidence=min(abs(time_quality_correlation), 0.9),
                evidence_count=20,
                time_span_days=14,
                impact_assessment="Medium - Understanding trade-offs enables better optimization",
                recommended_actions=[
                    "Optimize algorithms to reduce negative trade-offs",
                    "Implement adaptive timing based on quality requirements",
                    "Monitor trade-off patterns in real-time",
                ],
                priority="medium",
                created_date=datetime.now().isoformat(),
            )

            self.learning_insights.append(insight)

    def _generate_pattern_effectiveness_insight(self) -> None:
        """Generate insight about improvement pattern effectiveness"""

        validated_patterns = [p for p in self.improvement_patterns if p.validated]

        if len(validated_patterns) < 3:
            return

        avg_impact = statistics.mean([p.impact_score for p in validated_patterns])
        most_frequent = max(validated_patterns, key=lambda p: p.frequency)

        insight = LearningInsight(
            insight_id=f"pattern_effectiveness_{int(time.time())}",
            insight_type="performance",
            title="Improvement Pattern Effectiveness Analysis",
            description=f"Validated patterns show {avg_impact:.1%} average impact. Most effective: {most_frequent.description}",
            confidence=0.85,
            evidence_count=len(validated_patterns),
            time_span_days=30,
            impact_assessment="High - Patterns provide systematic improvement guidance",
            recommended_actions=[
                f"Replicate most effective pattern: {most_frequent.description}",
                "Validate remaining improvement patterns",
                "Develop pattern-based optimization framework",
            ],
            priority="high",
            created_date=datetime.now().isoformat(),
        )

        self.learning_insights.append(insight)

    # Helper methods

    def _validate_metrics(self, metrics: ReasoningMetrics) -> bool:
        """Validate that metrics are reasonable"""

        # Check required fields
        if not metrics.analysis_id or not metrics.timestamp:
            return False

        # Check value ranges
        if not (0 <= metrics.overall_quality <= 1):
            return False

        if not (0 <= metrics.biological_accuracy <= 1):
            return False

        if metrics.execution_time < 0:
            return False

        return True

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction and magnitude"""

        if len(values) < 2:
            return 0.0

        # Simple linear trend calculation
        n = len(values)
        x_vals = list(range(n))

        # Calculate slope
        x_mean = statistics.mean(x_vals)
        y_mean = statistics.mean(values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, values))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        # Normalize by mean to get relative trend
        if y_mean != 0:
            return slope / abs(y_mean)
        else:
            return slope

    def _calculate_correlation(self, x_vals: List[float], y_vals: List[float]) -> float:
        """Calculate simple correlation coefficient"""

        if len(x_vals) != len(y_vals) or len(x_vals) < 2:
            return 0.0

        x_mean = statistics.mean(x_vals)
        y_mean = statistics.mean(y_vals)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))

        x_var = sum((x - x_mean) ** 2 for x in x_vals)
        y_var = sum((y - y_mean) ** 2 for y in y_vals)

        denominator = (x_var * y_var) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _analyze_pattern_characteristics(
        self, window1: List[ReasoningMetrics], window2: List[ReasoningMetrics]
    ) -> Dict[str, Any]:
        """Analyze characteristics that might have caused improvement"""

        # Compare average characteristics
        w1_artifact_usage = statistics.mean([m.artifact_usage_rate for m in window1])
        w2_artifact_usage = statistics.mean([m.artifact_usage_rate for m in window2])

        w1_synthesis = statistics.mean([m.synthesis_effectiveness for m in window1])
        w2_synthesis = statistics.mean([m.synthesis_effectiveness for m in window2])

        triggers = []
        metrics = {}

        if w2_artifact_usage - w1_artifact_usage > 0.1:
            triggers.append("Increased artifact usage")
            metrics["artifact_improvement"] = w2_artifact_usage - w1_artifact_usage

        if w2_synthesis - w1_synthesis > 0.05:
            triggers.append("Better cross-tool synthesis")
            metrics["synthesis_improvement"] = w2_synthesis - w1_synthesis

        if not triggers:
            triggers = ["General system optimization"]

        suggestion = (
            "Apply identified improvements consistently across all analysis types"
        )

        return {"triggers": triggers, "metrics": metrics, "suggestion": suggestion}

    def _find_similar_pattern(
        self, pattern: ImprovementPattern
    ) -> Optional[ImprovementPattern]:
        """Find existing similar pattern"""

        for existing in self.improvement_patterns:
            if (
                existing.pattern_type == pattern.pattern_type
                and abs(existing.impact_score - pattern.impact_score) < 0.2
            ):
                return existing

        return None

    def _validate_patterns(self) -> None:
        """Validate improvement patterns against recent data"""

        for pattern in self.improvement_patterns:
            if (
                not pattern.validated
                and pattern.frequency >= self.min_pattern_frequency
            ):
                # Simple validation: check if pattern characteristics still hold
                recent_metrics = self.metrics_history[-10:]

                if len(recent_metrics) >= 5:
                    # Validate based on pattern type
                    if self._pattern_validation_check(pattern, recent_metrics):
                        pattern.validated = True
                        pattern.validation_date = datetime.now().isoformat()

    def _pattern_validation_check(
        self, pattern: ImprovementPattern, recent_metrics: List[ReasoningMetrics]
    ) -> bool:
        """Check if pattern characteristics are present in recent data"""

        if pattern.pattern_type == "quality_improvement":
            trend = self._calculate_trend([m.overall_quality for m in recent_metrics])
            return trend > 0.02  # Positive quality trend
        elif pattern.pattern_type == "efficiency_gain":
            trend = self._calculate_trend([m.execution_time for m in recent_metrics])
            return trend < -0.05  # Decreasing time trend
        elif pattern.pattern_type == "error_reduction":
            trend = self._calculate_trend([m.error_rate for m in recent_metrics])
            return trend < -0.02  # Decreasing error trend

        return False

    def _load_data(self) -> None:
        """Load existing data from files"""

        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, "r") as f:
                    metrics_data = json.load(f)
                    self.metrics_history = [ReasoningMetrics(**m) for m in metrics_data]
        except Exception as e:
            logger.warning(f"Could not load metrics history: {e}")

        try:
            if self.patterns_file.exists():
                with open(self.patterns_file, "r") as f:
                    patterns_data = json.load(f)
                    self.improvement_patterns = [
                        ImprovementPattern(**p) for p in patterns_data
                    ]
        except Exception as e:
            logger.warning(f"Could not load improvement patterns: {e}")

        try:
            if self.insights_file.exists():
                with open(self.insights_file, "r") as f:
                    insights_data = json.load(f)
                    self.learning_insights = [
                        LearningInsight(**i) for i in insights_data
                    ]
        except Exception as e:
            logger.warning(f"Could not load learning insights: {e}")

    def _save_data(self) -> None:
        """Save data to files"""

        try:
            with open(self.metrics_file, "w") as f:
                json.dump([asdict(m) for m in self.metrics_history], f, indent=2)
        except Exception as e:
            logger.error(f"Could not save metrics history: {e}")

        try:
            with open(self.patterns_file, "w") as f:
                json.dump([asdict(p) for p in self.improvement_patterns], f, indent=2)
        except Exception as e:
            logger.error(f"Could not save improvement patterns: {e}")

        try:
            with open(self.insights_file, "w") as f:
                json.dump([asdict(i) for i in self.learning_insights], f, indent=2)
        except Exception as e:
            logger.error(f"Could not save learning insights: {e}")


def create_sample_metrics() -> ReasoningMetrics:
    """Create sample metrics for testing"""

    return ReasoningMetrics(
        overall_quality=0.89,
        biological_accuracy=0.92,
        reasoning_transparency=0.87,
        synthesis_effectiveness=0.91,
        artifact_usage_rate=0.76,
        hypothesis_count=3,
        execution_time=28.5,
        error_rate=0.002,
        user_satisfaction=0.94,
        pattern_discovery_rate=0.23,
        bias_detection_accuracy=0.92,
        self_reflection_quality=0.88,
        meta_reasoning_effectiveness=0.87,
        context_relevance=0.89,
        cross_tool_integration=0.91,
        novelty_score=0.73,
        analysis_id="sample_analysis_001",
    )


if __name__ == "__main__":
    # Example usage
    tracker = ImprovementTracker()

    # Record sample metrics
    sample = create_sample_metrics()
    tracker.record_analysis_metrics(sample)

    # Generate report
    report = tracker.generate_improvement_report()
    print(json.dumps(report, indent=2))
