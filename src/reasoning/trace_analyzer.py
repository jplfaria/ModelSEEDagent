"""
Reasoning Trace Analyzer for ModelSEEDagent

Analyzes reasoning traces to assess quality, identify patterns, and provide
insights for improving AI decision-making processes.
"""

import json
import logging
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .trace_logger import (
    DecisionPoint,
    DecisionType,
    HypothesisTrace,
    ReasoningTrace,
    load_reasoning_trace,
)

logger = logging.getLogger(__name__)


class ReasoningQualityMetrics:
    """Quality metrics for reasoning assessment"""

    def __init__(self):
        self.biological_accuracy: Optional[float] = None
        self.reasoning_transparency: Optional[float] = None
        self.synthesis_effectiveness: Optional[float] = None
        self.hypothesis_quality: Optional[float] = None
        self.decision_consistency: Optional[float] = None
        self.overall_score: Optional[float] = None


class ReasoningTraceAnalyzer:
    """Analyzes reasoning traces for quality assessment and pattern identification"""

    def __init__(self, trace_dir: Optional[Path] = None):
        self.trace_dir = (
            trace_dir or Path.home() / ".modelseed-agent" / "reasoning_traces"
        )
        self.trace_cache: Dict[str, ReasoningTrace] = {}

    def analyze_session_quality(self, session_id: str) -> ReasoningQualityMetrics:
        """Analyze the quality of reasoning in a specific session"""
        trace = self._get_trace(session_id)
        if not trace:
            logger.error(f"Could not load trace for session {session_id}")
            return ReasoningQualityMetrics()

        metrics = ReasoningQualityMetrics()

        # Analyze different aspects of reasoning quality
        metrics.reasoning_transparency = self._assess_transparency(trace)
        metrics.decision_consistency = self._assess_consistency(trace)
        metrics.synthesis_effectiveness = self._assess_synthesis(trace)
        metrics.hypothesis_quality = self._assess_hypothesis_quality(trace)
        metrics.biological_accuracy = self._assess_biological_accuracy(trace)

        # Calculate overall score
        scores = [
            metrics.reasoning_transparency,
            metrics.decision_consistency,
            metrics.synthesis_effectiveness,
            metrics.hypothesis_quality,
            metrics.biological_accuracy,
        ]

        valid_scores = [s for s in scores if s is not None]
        if valid_scores:
            metrics.overall_score = sum(valid_scores) / len(valid_scores)

        return metrics

    def analyze_decision_patterns(
        self, session_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze patterns in decision-making across sessions"""
        if session_ids is None:
            session_ids = self._get_all_session_ids()

        traces = [self._get_trace(sid) for sid in session_ids]
        traces = [t for t in traces if t is not None]

        if not traces:
            return {"error": "No valid traces found"}

        patterns = {
            "decision_type_frequency": self._analyze_decision_frequency(traces),
            "confidence_patterns": self._analyze_confidence_patterns(traces),
            "tool_selection_patterns": self._analyze_tool_selection_patterns(traces),
            "hypothesis_patterns": self._analyze_hypothesis_patterns(traces),
            "workflow_adaptation_patterns": self._analyze_workflow_patterns(traces),
            "temporal_patterns": self._analyze_temporal_patterns(traces),
        }

        return patterns

    def identify_reasoning_issues(self, session_id: str) -> List[Dict[str, Any]]:
        """Identify potential issues in reasoning quality"""
        trace = self._get_trace(session_id)
        if not trace:
            return [{"type": "error", "message": f"Could not load trace {session_id}"}]

        issues = []

        # Check for low confidence decisions
        low_confidence_decisions = [
            d for d in trace.decision_points if d.confidence < 0.6
        ]
        if low_confidence_decisions:
            issues.append(
                {
                    "type": "low_confidence",
                    "count": len(low_confidence_decisions),
                    "decisions": [d.decision_id for d in low_confidence_decisions],
                    "severity": "medium",
                }
            )

        # Check for short reasoning explanations
        short_reasoning = [
            d for d in trace.decision_points if len(d.reasoning.strip()) < 30
        ]
        if short_reasoning:
            issues.append(
                {
                    "type": "insufficient_reasoning",
                    "count": len(short_reasoning),
                    "decisions": [d.decision_id for d in short_reasoning],
                    "severity": "high",
                }
            )

        # Check for decisions without alternatives
        no_alternatives = [
            d for d in trace.decision_points if not d.alternatives_considered
        ]
        if no_alternatives:
            issues.append(
                {
                    "type": "no_alternatives_considered",
                    "count": len(no_alternatives),
                    "decisions": [d.decision_id for d in no_alternatives],
                    "severity": "medium",
                }
            )

        # Check for hypotheses without testable predictions
        weak_hypotheses = [
            h for h in trace.hypotheses if not h.predictions or len(h.predictions) == 0
        ]
        if weak_hypotheses:
            issues.append(
                {
                    "type": "weak_hypotheses",
                    "count": len(weak_hypotheses),
                    "hypotheses": [h.hypothesis_id for h in weak_hypotheses],
                    "severity": "high",
                }
            )

        # Check for long decision chains without validation
        unvalidated_chains = self._find_unvalidated_decision_chains(trace)
        if unvalidated_chains:
            issues.append(
                {
                    "type": "unvalidated_decision_chains",
                    "count": len(unvalidated_chains),
                    "chains": unvalidated_chains,
                    "severity": "medium",
                }
            )

        return issues

    def compare_reasoning_quality(
        self, session_id_a: str, session_id_b: str
    ) -> Dict[str, Any]:
        """Compare reasoning quality between two sessions"""
        metrics_a = self.analyze_session_quality(session_id_a)
        metrics_b = self.analyze_session_quality(session_id_b)

        comparison = {
            "session_a": session_id_a,
            "session_b": session_id_b,
            "metrics_comparison": {},
            "winner": None,
        }

        metric_names = [
            "reasoning_transparency",
            "decision_consistency",
            "synthesis_effectiveness",
            "hypothesis_quality",
            "biological_accuracy",
            "overall_score",
        ]

        scores_a = []
        scores_b = []

        for metric in metric_names:
            value_a = getattr(metrics_a, metric)
            value_b = getattr(metrics_b, metric)

            comparison["metrics_comparison"][metric] = {
                "session_a": value_a,
                "session_b": value_b,
                "difference": (value_b - value_a) if (value_a and value_b) else None,
            }

            if value_a and value_b:
                scores_a.append(value_a)
                scores_b.append(value_b)

        # Determine overall winner
        if scores_a and scores_b:
            avg_a = sum(scores_a) / len(scores_a)
            avg_b = sum(scores_b) / len(scores_b)

            if avg_a > avg_b:
                comparison["winner"] = "session_a"
            elif avg_b > avg_a:
                comparison["winner"] = "session_b"
            else:
                comparison["winner"] = "tie"

        return comparison

    def generate_reasoning_report(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive reasoning analysis report"""
        trace = self._get_trace(session_id)
        if not trace:
            return {"error": f"Could not load trace {session_id}"}

        quality_metrics = self.analyze_session_quality(session_id)
        issues = self.identify_reasoning_issues(session_id)

        # Calculate session statistics
        decision_stats = self._calculate_decision_statistics(trace)
        hypothesis_stats = self._calculate_hypothesis_statistics(trace)

        # Analyze tool usage effectiveness
        tool_effectiveness = self._analyze_tool_effectiveness(trace)

        report = {
            "session_id": session_id,
            "query": trace.query,
            "analysis_period": {
                "start": trace.start_time.isoformat(),
                "end": trace.end_time.isoformat() if trace.end_time else None,
                "duration_seconds": (
                    (trace.end_time - trace.start_time).total_seconds()
                    if trace.end_time
                    else None
                ),
            },
            "quality_metrics": {
                "reasoning_transparency": quality_metrics.reasoning_transparency,
                "decision_consistency": quality_metrics.decision_consistency,
                "synthesis_effectiveness": quality_metrics.synthesis_effectiveness,
                "hypothesis_quality": quality_metrics.hypothesis_quality,
                "biological_accuracy": quality_metrics.biological_accuracy,
                "overall_score": quality_metrics.overall_score,
            },
            "decision_statistics": decision_stats,
            "hypothesis_statistics": hypothesis_stats,
            "tool_effectiveness": tool_effectiveness,
            "identified_issues": issues,
            "recommendations": self._generate_recommendations(quality_metrics, issues),
            "summary": self._generate_summary(trace, quality_metrics, issues),
        }

        return report

    def _get_trace(self, session_id: str) -> Optional[ReasoningTrace]:
        """Get trace from cache or load from disk"""
        if session_id not in self.trace_cache:
            trace = load_reasoning_trace(session_id, self.trace_dir)
            if trace:
                self.trace_cache[session_id] = trace
            return trace
        return self.trace_cache[session_id]

    def _get_all_session_ids(self) -> List[str]:
        """Get all available session IDs"""
        session_ids = []

        if not self.trace_dir.exists():
            return session_ids

        for file_path in self.trace_dir.glob("trace_*.json"):
            # Extract session ID from filename
            filename = file_path.stem
            if filename.startswith("trace_"):
                session_id = filename[6:]  # Remove "trace_" prefix
                if session_id.endswith("_final"):
                    session_id = session_id[:-6]  # Remove "_final" suffix
                session_ids.append(session_id)

        return list(set(session_ids))  # Remove duplicates

    def _assess_transparency(self, trace: ReasoningTrace) -> float:
        """Assess reasoning transparency quality"""
        if not trace.decision_points:
            return 0.0

        scores = []

        for decision in trace.decision_points:
            score = 0.0

            # Check reasoning length and quality
            reasoning_length = len(decision.reasoning.strip())
            if reasoning_length > 50:
                score += 0.3
            elif reasoning_length > 20:
                score += 0.15

            # Check if alternatives were considered
            if decision.alternatives_considered:
                score += 0.2

            # Check if evidence was provided
            if decision.evidence:
                score += 0.2

            # Check if assumptions were documented
            if decision.assumptions:
                score += 0.1

            # Check confidence alignment with reasoning quality
            if decision.confidence > 0.8 and reasoning_length > 40:
                score += 0.1
            elif decision.confidence < 0.6 and reasoning_length > 30:
                score += 0.1

            scores.append(min(1.0, score))

        return sum(scores) / len(scores)

    def _assess_consistency(self, trace: ReasoningTrace) -> float:
        """Assess decision consistency across the session"""
        if len(trace.decision_points) < 2:
            return 1.0  # Perfect consistency with one decision

        # Check confidence consistency for similar decision types
        type_confidences = defaultdict(list)
        for decision in trace.decision_points:
            type_confidences[decision.decision_type].append(decision.confidence)

        consistency_scores = []

        for decision_type, confidences in type_confidences.items():
            if len(confidences) > 1:
                # Calculate coefficient of variation (lower is more consistent)
                if statistics.mean(confidences) > 0:
                    cv = statistics.stdev(confidences) / statistics.mean(confidences)
                    consistency_score = max(0.0, 1.0 - cv)
                    consistency_scores.append(consistency_score)

        if not consistency_scores:
            return 0.8  # Default reasonable score

        return sum(consistency_scores) / len(consistency_scores)

    def _assess_synthesis(self, trace: ReasoningTrace) -> float:
        """Assess synthesis and integration effectiveness"""
        score = 0.0

        # Check if synthesis reasoning was provided
        if trace.synthesis_reasoning and len(trace.synthesis_reasoning.strip()) > 50:
            score += 0.4

        # Check for synthesis-type decisions
        synthesis_decisions = [
            d
            for d in trace.decision_points
            if d.decision_type == DecisionType.SYNTHESIS
        ]

        if synthesis_decisions:
            score += 0.3

            # Check quality of synthesis decisions
            for decision in synthesis_decisions:
                if decision.evidence and len(decision.evidence) > 1:
                    score += 0.1
                if decision.confidence > 0.7:
                    score += 0.1

        # Check for cross-references between decisions
        decision_references = 0
        for decision in trace.decision_points:
            if decision.parent_decision_id or any(
                other.decision_id in decision.reasoning
                for other in trace.decision_points
                if other != decision
            ):
                decision_references += 1

        if decision_references > 0:
            reference_score = min(0.2, decision_references / len(trace.decision_points))
            score += reference_score

        return min(1.0, score)

    def _assess_hypothesis_quality(self, trace: ReasoningTrace) -> float:
        """Assess quality of hypothesis formation and testing"""
        if not trace.hypotheses:
            return 0.5  # Neutral score if no hypotheses

        scores = []

        for hypothesis in trace.hypotheses:
            score = 0.0

            # Check hypothesis statement quality
            if len(hypothesis.statement.strip()) > 30:
                score += 0.2

            # Check rationale quality
            if len(hypothesis.rationale.strip()) > 50:
                score += 0.2

            # Check for testable predictions
            if hypothesis.predictions and len(hypothesis.predictions) > 0:
                score += 0.3
                if len(hypothesis.predictions) > 1:
                    score += 0.1

            # Check for evidence consideration
            if hypothesis.evidence_for:
                score += 0.1
            if hypothesis.evidence_against:
                score += 0.1

            # Check confidence calibration
            if 0.3 <= hypothesis.confidence <= 0.9:  # Reasonable uncertainty
                score += 0.1

            scores.append(min(1.0, score))

        return sum(scores) / len(scores)

    def _assess_biological_accuracy(self, trace: ReasoningTrace) -> float:
        """Assess biological accuracy of reasoning (simplified heuristic)"""
        # This is a simplified heuristic - in practice, this would require
        # domain expert validation or comparison with known correct answers

        score = 0.5  # Start with neutral score

        # Look for biological terminology usage
        biological_terms = [
            "metabolic",
            "pathway",
            "enzyme",
            "reaction",
            "gene",
            "protein",
            "growth",
            "biomass",
            "flux",
            "essential",
            "auxotroph",
            "media",
            "substrate",
            "product",
            "cofactor",
            "regulation",
            "synthesis",
        ]

        term_usage = 0
        total_text = " ".join([d.reasoning for d in trace.decision_points])
        total_text += " ".join([h.rationale for h in trace.hypotheses])
        total_text = total_text.lower()

        for term in biological_terms:
            if term in total_text:
                term_usage += 1

        # Bonus for appropriate biological terminology
        terminology_score = min(0.3, term_usage / 20.0)
        score += terminology_score

        # Check for quantitative reasoning
        quantitative_indicators = [
            "rate",
            "concentration",
            "ratio",
            "percentage",
            "fold",
        ]
        quant_usage = sum(
            1 for indicator in quantitative_indicators if indicator in total_text
        )
        quant_score = min(0.2, quant_usage / 10.0)
        score += quant_score

        return min(1.0, score)

    def _analyze_decision_frequency(
        self, traces: List[ReasoningTrace]
    ) -> Dict[str, Any]:
        """Analyze frequency of different decision types"""
        type_counts = Counter()

        for trace in traces:
            for decision in trace.decision_points:
                type_counts[decision.decision_type.value] += 1

        total_decisions = sum(type_counts.values())

        return {
            "counts": dict(type_counts),
            "percentages": {
                k: (v / total_decisions) * 100 for k, v in type_counts.items()
            },
            "total_decisions": total_decisions,
        }

    def _analyze_confidence_patterns(
        self, traces: List[ReasoningTrace]
    ) -> Dict[str, Any]:
        """Analyze confidence patterns across decisions"""
        all_confidences = []
        confidence_by_type = defaultdict(list)

        for trace in traces:
            for decision in trace.decision_points:
                all_confidences.append(decision.confidence)
                confidence_by_type[decision.decision_type.value].append(
                    decision.confidence
                )

        patterns = {
            "overall_stats": {
                "mean": statistics.mean(all_confidences),
                "median": statistics.median(all_confidences),
                "std_dev": (
                    statistics.stdev(all_confidences) if len(all_confidences) > 1 else 0
                ),
                "min": min(all_confidences),
                "max": max(all_confidences),
            },
            "by_decision_type": {},
        }

        for decision_type, confidences in confidence_by_type.items():
            if confidences:
                patterns["by_decision_type"][decision_type] = {
                    "mean": statistics.mean(confidences),
                    "count": len(confidences),
                    "std_dev": (
                        statistics.stdev(confidences) if len(confidences) > 1 else 0
                    ),
                }

        return patterns

    def _analyze_tool_selection_patterns(
        self, traces: List[ReasoningTrace]
    ) -> Dict[str, Any]:
        """Analyze patterns in tool selection decisions"""
        tool_selections = []

        for trace in traces:
            for decision in trace.decision_points:
                if decision.decision_type == DecisionType.TOOL_SELECTION:
                    tool_selections.append(
                        {
                            "tool": decision.chosen_option,
                            "confidence": decision.confidence,
                            "alternatives_count": len(decision.alternatives_considered),
                            "reasoning_length": len(decision.reasoning),
                        }
                    )

        if not tool_selections:
            return {"message": "No tool selection decisions found"}

        tool_counts = Counter(sel["tool"] for sel in tool_selections)
        avg_confidence_by_tool = defaultdict(list)

        for sel in tool_selections:
            avg_confidence_by_tool[sel["tool"]].append(sel["confidence"])

        return {
            "tool_frequency": dict(tool_counts),
            "avg_confidence_by_tool": {
                tool: statistics.mean(confidences)
                for tool, confidences in avg_confidence_by_tool.items()
            },
            "total_tool_selections": len(tool_selections),
            "avg_alternatives_considered": statistics.mean(
                [sel["alternatives_count"] for sel in tool_selections]
            ),
            "avg_reasoning_length": statistics.mean(
                [sel["reasoning_length"] for sel in tool_selections]
            ),
        }

    def _analyze_hypothesis_patterns(
        self, traces: List[ReasoningTrace]
    ) -> Dict[str, Any]:
        """Analyze patterns in hypothesis formation"""
        all_hypotheses = []

        for trace in traces:
            all_hypotheses.extend(trace.hypotheses)

        if not all_hypotheses:
            return {"message": "No hypotheses found"}

        status_counts = Counter(h.status for h in all_hypotheses)
        avg_confidence = statistics.mean([h.confidence for h in all_hypotheses])

        prediction_counts = [len(h.predictions) for h in all_hypotheses]
        avg_predictions = statistics.mean(prediction_counts) if prediction_counts else 0

        return {
            "total_hypotheses": len(all_hypotheses),
            "status_distribution": dict(status_counts),
            "avg_confidence": avg_confidence,
            "avg_predictions_per_hypothesis": avg_predictions,
            "hypotheses_per_session": (
                len(all_hypotheses) / len(traces) if traces else 0
            ),
        }

    def _analyze_workflow_patterns(
        self, traces: List[ReasoningTrace]
    ) -> Dict[str, Any]:
        """Analyze workflow adaptation patterns"""
        adaptations = []

        for trace in traces:
            for decision in trace.decision_points:
                if decision.decision_type == DecisionType.WORKFLOW_ADAPTATION:
                    adaptations.append(decision)

        if not adaptations:
            return {"message": "No workflow adaptations found"}

        return {
            "total_adaptations": len(adaptations),
            "adaptations_per_session": len(adaptations) / len(traces) if traces else 0,
            "avg_confidence": statistics.mean([a.confidence for a in adaptations]),
            "adaptation_triggers": [
                a.context.get("trigger_event", "unknown") for a in adaptations
            ],
        }

    def _analyze_temporal_patterns(
        self, traces: List[ReasoningTrace]
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in reasoning"""
        durations = []
        decisions_per_session = []

        for trace in traces:
            if trace.end_time:
                duration = (trace.end_time - trace.start_time).total_seconds()
                durations.append(duration)

            decisions_per_session.append(len(trace.decision_points))

        patterns = {
            "session_durations": {
                "avg_seconds": statistics.mean(durations) if durations else 0,
                "median_seconds": statistics.median(durations) if durations else 0,
                "total_sessions": len(durations),
            },
            "decisions_per_session": {
                "avg": (
                    statistics.mean(decisions_per_session)
                    if decisions_per_session
                    else 0
                ),
                "median": (
                    statistics.median(decisions_per_session)
                    if decisions_per_session
                    else 0
                ),
                "min": min(decisions_per_session) if decisions_per_session else 0,
                "max": max(decisions_per_session) if decisions_per_session else 0,
            },
        }

        return patterns

    def _find_unvalidated_decision_chains(
        self, trace: ReasoningTrace
    ) -> List[List[str]]:
        """Find decision chains that lack outcome validation"""
        unvalidated_chains = []

        # Find root decisions (no parent)
        root_decisions = [d for d in trace.decision_points if not d.parent_decision_id]

        for root in root_decisions:
            chain = self._build_decision_chain(trace, root.decision_id)

            # Check if any decision in chain has been validated
            validated = any(d.outcome_validated is not None for d in chain)

            if not validated and len(chain) > 2:  # Only flag longer chains
                unvalidated_chains.append([d.decision_id for d in chain])

        return unvalidated_chains

    def _build_decision_chain(
        self, trace: ReasoningTrace, root_id: str
    ) -> List[DecisionPoint]:
        """Build a chain of related decisions"""
        chain = []
        decision_map = {d.decision_id: d for d in trace.decision_points}

        # Find all decisions in this chain
        to_visit = [root_id]
        visited = set()

        while to_visit:
            current_id = to_visit.pop(0)
            if current_id in visited or current_id not in decision_map:
                continue

            visited.add(current_id)
            decision = decision_map[current_id]
            chain.append(decision)

            # Find child decisions
            children = [
                d.decision_id
                for d in trace.decision_points
                if d.parent_decision_id == current_id
            ]
            to_visit.extend(children)

        return sorted(chain, key=lambda d: d.step_number)

    def _calculate_decision_statistics(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """Calculate comprehensive decision statistics"""
        if not trace.decision_points:
            return {"total_decisions": 0}

        confidences = [d.confidence for d in trace.decision_points]
        reasoning_lengths = [len(d.reasoning) for d in trace.decision_points]

        return {
            "total_decisions": len(trace.decision_points),
            "avg_confidence": statistics.mean(confidences),
            "confidence_std": (
                statistics.stdev(confidences) if len(confidences) > 1 else 0
            ),
            "avg_reasoning_length": statistics.mean(reasoning_lengths),
            "decisions_with_evidence": len(
                [d for d in trace.decision_points if d.evidence]
            ),
            "decisions_with_alternatives": len(
                [d for d in trace.decision_points if d.alternatives_considered]
            ),
            "validated_decisions": len(
                [d for d in trace.decision_points if d.outcome_validated is not None]
            ),
        }

    def _calculate_hypothesis_statistics(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """Calculate hypothesis-specific statistics"""
        if not trace.hypotheses:
            return {"total_hypotheses": 0}

        confidences = [h.confidence for h in trace.hypotheses]
        prediction_counts = [len(h.predictions) for h in trace.hypotheses]

        return {
            "total_hypotheses": len(trace.hypotheses),
            "avg_confidence": statistics.mean(confidences),
            "avg_predictions": statistics.mean(prediction_counts),
            "status_distribution": Counter([h.status for h in trace.hypotheses]),
            "hypotheses_with_evidence": len(
                [h for h in trace.hypotheses if h.evidence_for or h.evidence_against]
            ),
        }

    def _analyze_tool_effectiveness(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """Analyze effectiveness of tool usage in the session"""
        tool_decisions = [
            d
            for d in trace.decision_points
            if d.decision_type == DecisionType.TOOL_SELECTION
        ]

        if not tool_decisions:
            return {"message": "No tool selection decisions found"}

        tool_confidence = {}
        tool_usage = Counter()

        for decision in tool_decisions:
            tool = decision.chosen_option

            # Handle case where chosen_option might be a list
            if isinstance(tool, list):
                tool = tool[0] if tool else "unknown_tool"
            elif not isinstance(tool, str):
                tool = str(tool)

            tool_usage[tool] += 1

            if tool not in tool_confidence:
                tool_confidence[tool] = []
            tool_confidence[tool].append(decision.confidence)

        # Calculate average confidence per tool
        avg_confidence = {
            tool: statistics.mean(confidences)
            for tool, confidences in tool_confidence.items()
        }

        return {
            "tools_used": dict(tool_usage),
            "unique_tools": len(tool_usage),
            "avg_confidence_by_tool": avg_confidence,
            "most_used_tool": tool_usage.most_common(1)[0] if tool_usage else None,
            "tool_diversity": (
                len(tool_usage) / sum(tool_usage.values()) if tool_usage else 0
            ),
        }

    def _generate_recommendations(
        self, metrics: ReasoningQualityMetrics, issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        # Transparency recommendations
        if metrics.reasoning_transparency and metrics.reasoning_transparency < 0.7:
            recommendations.append(
                "Improve reasoning transparency by providing more detailed explanations for decisions"
            )

        # Consistency recommendations
        if metrics.decision_consistency and metrics.decision_consistency < 0.6:
            recommendations.append(
                "Work on decision consistency by maintaining similar confidence levels for similar types of decisions"
            )

        # Synthesis recommendations
        if metrics.synthesis_effectiveness and metrics.synthesis_effectiveness < 0.5:
            recommendations.append(
                "Enhance synthesis by explicitly connecting findings from different tools and analysis steps"
            )

        # Hypothesis recommendations
        if metrics.hypothesis_quality and metrics.hypothesis_quality < 0.6:
            recommendations.append(
                "Improve hypothesis quality by including more testable predictions and stronger rationales"
            )

        # Issue-specific recommendations
        for issue in issues:
            if issue["type"] == "low_confidence" and issue["count"] > 2:
                recommendations.append(
                    "Consider gathering more evidence before making decisions to increase confidence"
                )
            elif issue["type"] == "insufficient_reasoning" and issue["count"] > 1:
                recommendations.append(
                    "Provide more detailed reasoning explanations for better transparency"
                )
            elif issue["type"] == "weak_hypotheses":
                recommendations.append(
                    "Strengthen hypotheses by including specific, testable predictions"
                )

        return recommendations

    def _generate_summary(
        self,
        trace: ReasoningTrace,
        metrics: ReasoningQualityMetrics,
        issues: List[Dict[str, Any]],
    ) -> str:
        """Generate a text summary of the analysis"""
        summary_parts = []

        # Basic session info
        duration = ""
        if trace.end_time:
            duration_sec = (trace.end_time - trace.start_time).total_seconds()
            duration = f" over {duration_sec:.1f} seconds"

        summary_parts.append(
            f"Session {trace.session_id} involved {len(trace.decision_points)} decisions and {len(trace.hypotheses)} hypotheses{duration}."
        )

        # Quality assessment
        if metrics.overall_score:
            if metrics.overall_score >= 0.8:
                quality_desc = "excellent"
            elif metrics.overall_score >= 0.6:
                quality_desc = "good"
            elif metrics.overall_score >= 0.4:
                quality_desc = "moderate"
            else:
                quality_desc = "needs improvement"

            summary_parts.append(
                f"Overall reasoning quality is {quality_desc} (score: {metrics.overall_score:.2f})."
            )

        # Key strengths and weaknesses
        strengths = []
        weaknesses = []

        if metrics.reasoning_transparency and metrics.reasoning_transparency > 0.7:
            strengths.append("good transparency")
        elif metrics.reasoning_transparency and metrics.reasoning_transparency < 0.5:
            weaknesses.append("poor transparency")

        if metrics.synthesis_effectiveness and metrics.synthesis_effectiveness > 0.7:
            strengths.append("effective synthesis")
        elif metrics.synthesis_effectiveness and metrics.synthesis_effectiveness < 0.5:
            weaknesses.append("weak synthesis")

        if strengths:
            summary_parts.append(f"Key strengths include {', '.join(strengths)}.")

        if weaknesses:
            summary_parts.append(
                f"Areas for improvement include {', '.join(weaknesses)}."
            )

        # Issues summary
        high_severity_issues = [i for i in issues if i.get("severity") == "high"]
        if high_severity_issues:
            summary_parts.append(
                f"There are {len(high_severity_issues)} high-severity issues that should be addressed."
            )

        return " ".join(summary_parts)
