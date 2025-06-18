#!/usr/bin/env python3
"""
Cross-Model Learning and Pattern Memory - Phase 8.4

Implements AI learning capabilities that accumulate insights across multiple
model analyses, identify patterns, and improve tool selection and reasoning
based on experience.

Key Features:
- Pattern recognition across multiple analyses
- Experience-based tool selection improvement
- Metabolic insight accumulation and retrieval
- Analysis strategy optimization based on history
- Cross-model comparison and learning
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field


class AnalysisPattern(BaseModel):
    """Represents a learned pattern from analysis history"""

    model_config = ConfigDict(protected_namespaces=())

    pattern_id: str = Field(description="Unique pattern identifier")
    pattern_type: str = Field(
        description="Type of pattern (tool_sequence, insight_correlation, etc.)"
    )

    # Pattern definition
    description: str = Field(description="Human-readable pattern description")
    conditions: Dict[str, Any] = Field(description="Conditions where pattern applies")
    outcomes: Dict[str, Any] = Field(description="Expected outcomes of pattern")

    # Evidence and validation
    occurrence_count: int = Field(
        default=1, description="Number of times pattern observed"
    )
    success_rate: float = Field(description="Success rate when pattern is applied")
    confidence: float = Field(description="Confidence in pattern validity")

    # Context
    first_observed: str = Field(description="When pattern was first identified")
    last_observed: str = Field(description="Most recent pattern observation")
    model_types: Set[str] = Field(
        default_factory=set, description="Model types where pattern applies"
    )

    # Usage
    times_applied: int = Field(default=0, description="Times pattern was actively used")
    application_success_rate: float = Field(
        default=0.0, description="Success rate when actively applied"
    )


class MetabolicInsight(BaseModel):
    """Represents accumulated knowledge about metabolic systems"""

    model_config = ConfigDict(protected_namespaces=())

    insight_id: str = Field(description="Unique insight identifier")
    insight_category: str = Field(description="Category of insight")

    # Insight content
    summary: str = Field(description="Concise insight summary")
    detailed_description: str = Field(description="Detailed insight explanation")
    evidence_sources: List[str] = Field(
        description="Analysis sessions that support this insight"
    )

    # Applicability
    organisms: Set[str] = Field(
        default_factory=set, description="Organisms this insight applies to"
    )
    model_characteristics: Dict[str, Any] = Field(
        description="Model features where insight is relevant"
    )

    # Validation
    confidence_score: float = Field(description="Confidence in insight accuracy")
    validation_count: int = Field(
        default=1, description="Number of times insight was validated"
    )

    # Metadata
    discovered_date: str = Field(description="When insight was first discovered")
    last_validated: str = Field(description="Most recent validation")


class AnalysisExperience(BaseModel):
    """Records experience from a completed analysis"""

    model_config = ConfigDict(protected_namespaces=())

    experience_id: str = Field(description="Unique experience identifier")
    session_id: str = Field(description="Source analysis session")
    timestamp: str = Field(description="When analysis was completed")

    # Analysis characteristics
    user_query: str = Field(description="Original user query")
    model_characteristics: Dict[str, Any] = Field(description="Model properties")
    tools_used: List[str] = Field(description="Tools used in analysis")
    tool_sequence: List[str] = Field(description="Order of tool execution")

    # Outcomes
    success: bool = Field(description="Whether analysis was successful")
    insights_discovered: List[str] = Field(description="Key insights from analysis")
    execution_time: float = Field(description="Total analysis time")
    user_satisfaction: Optional[float] = Field(
        default=None, description="User satisfaction score"
    )

    # Learning opportunities
    effective_strategies: List[str] = Field(description="Strategies that worked well")
    ineffective_strategies: List[str] = Field(description="Strategies that didn't work")
    missed_opportunities: List[str] = Field(
        description="Potential improvements identified"
    )

    # Smart Summarization Effectiveness - Phase C Enhancement
    summarization_metrics: Optional[Dict[str, Any]] = Field(
        default=None, description="Smart Summarization effectiveness metrics"
    )


class PatternExtractor:
    """Extracts patterns from analysis history"""

    def __init__(self, llm):
        """Initialize pattern extractor"""
        self.llm = llm
        self.pattern_templates = self._load_pattern_templates()

    def extract_patterns_from_experiences(
        self, experiences: List[AnalysisExperience]
    ) -> List[AnalysisPattern]:
        """Extract patterns from multiple analysis experiences"""

        patterns = []

        # Extract tool sequence patterns
        tool_patterns = self._extract_tool_sequence_patterns(experiences)
        patterns.extend(tool_patterns)

        # Extract query-outcome patterns
        query_patterns = self._extract_query_outcome_patterns(experiences)
        patterns.extend(query_patterns)

        # Extract model characteristic patterns
        model_patterns = self._extract_model_characteristic_patterns(experiences)
        patterns.extend(model_patterns)

        return patterns

    def _extract_tool_sequence_patterns(
        self, experiences: List[AnalysisExperience]
    ) -> List[AnalysisPattern]:
        """Extract common tool sequence patterns"""

        # Group experiences by success and analyze successful tool sequences
        successful_experiences = [exp for exp in experiences if exp.success]

        if len(successful_experiences) < 3:
            return []  # Need minimum data for pattern detection

        # Use AI to identify tool sequence patterns
        # Prepare analysis data
        analysis_data = [
            {
                "query_type": exp.user_query[:100],
                "tools_used": exp.tool_sequence,
                "success": exp.success,
                "insights": exp.insights_discovered[:2],  # First 2 insights
            }
            for exp in successful_experiences[:10]
        ]

        sequence_prompt = f"""
        Analyze these successful analysis tool sequences to identify common patterns:

        Successful Analyses:
        {json.dumps(analysis_data, indent=2)}

        Identify 2-3 common tool sequence patterns that appear in successful analyses.
        For each pattern:
        1. Describe the sequence and when it's effective
        2. Estimate its success rate
        3. Identify conditions where it applies

        Format as JSON:
        [
            {{
                "description": "Pattern description",
                "tool_sequence": ["tool1", "tool2", "tool3"],
                "conditions": {{"query_type": "growth analysis", "model_size": "medium"}},
                "estimated_success_rate": 0.85,
                "rationale": "Why this pattern works"
            }}
        ]
        """

        response = self.llm._generate_response(sequence_prompt)

        try:
            pattern_data = json.loads(response.text)
            patterns = []

            for pattern_info in pattern_data:
                pattern = AnalysisPattern(
                    pattern_id=str(uuid.uuid4())[:8],
                    pattern_type="tool_sequence",
                    description=pattern_info["description"],
                    conditions=pattern_info["conditions"],
                    outcomes={"tool_sequence": pattern_info["tool_sequence"]},
                    occurrence_count=self._count_pattern_occurrences(
                        pattern_info, successful_experiences
                    ),
                    success_rate=pattern_info["estimated_success_rate"],
                    confidence=min(pattern_info["estimated_success_rate"], 0.9),
                    first_observed=datetime.now().isoformat(),
                    last_observed=datetime.now().isoformat(),
                    model_types=set(),
                )
                patterns.append(pattern)

            return patterns

        except json.JSONDecodeError:
            return []

    def _extract_query_outcome_patterns(
        self, experiences: List[AnalysisExperience]
    ) -> List[AnalysisPattern]:
        """Extract patterns linking query types to successful outcomes"""

        # Group experiences by query characteristics
        query_groups = {}
        for exp in experiences:
            query_type = self._classify_query_type(exp.user_query)
            if query_type not in query_groups:
                query_groups[query_type] = []
            query_groups[query_type].append(exp)

        patterns = []
        for query_type, group_experiences in query_groups.items():
            if len(group_experiences) >= 2:  # Need at least 2 examples
                pattern = self._create_query_outcome_pattern(
                    query_type, group_experiences
                )
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _extract_model_characteristic_patterns(
        self, experiences: List[AnalysisExperience]
    ) -> List[AnalysisPattern]:
        """Extract patterns based on model characteristics"""

        # Analyze correlation between model characteristics and successful strategies
        patterns = []

        # Group by model size
        size_groups = {"small": [], "medium": [], "large": []}
        for exp in experiences:
            model_size = exp.model_characteristics.get("size_category", "medium")
            if model_size in size_groups:
                size_groups[model_size].append(exp)

        # Create patterns for each size category
        for size, group_experiences in size_groups.items():
            if len(group_experiences) >= 2:
                successful = [exp for exp in group_experiences if exp.success]
                if successful:
                    pattern = AnalysisPattern(
                        pattern_id=str(uuid.uuid4())[:8],
                        pattern_type="model_size_strategy",
                        description=f"Effective strategies for {size} models",
                        conditions={"model_size": size},
                        outcomes={
                            "effective_tools": self._get_most_effective_tools(
                                successful
                            )
                        },
                        occurrence_count=len(successful),
                        success_rate=len(successful) / len(group_experiences),
                        confidence=0.7,
                        first_observed=datetime.now().isoformat(),
                        last_observed=datetime.now().isoformat(),
                        model_types=set(),
                    )
                    patterns.append(pattern)

        return patterns

    def _classify_query_type(self, query: str) -> str:
        """Classify query into type categories"""

        query_lower = query.lower()

        if any(word in query_lower for word in ["comprehensive", "complete", "full"]):
            return "comprehensive_analysis"
        elif any(word in query_lower for word in ["growth", "biomass", "rate"]):
            return "growth_analysis"
        elif any(word in query_lower for word in ["essential", "gene", "knockout"]):
            return "essentiality_analysis"
        elif any(word in query_lower for word in ["media", "nutrient", "auxotroph"]):
            return "media_analysis"
        elif any(word in query_lower for word in ["flux", "pathway", "variability"]):
            return "flux_analysis"
        else:
            return "general_analysis"

    def _count_pattern_occurrences(
        self, pattern_info: Dict, experiences: List[AnalysisExperience]
    ) -> int:
        """Count how many experiences match a pattern"""

        count = 0
        pattern_sequence = pattern_info.get("tool_sequence", [])

        for exp in experiences:
            if self._sequence_matches_pattern(exp.tool_sequence, pattern_sequence):
                count += 1

        return count

    def _sequence_matches_pattern(
        self, actual_sequence: List[str], pattern_sequence: List[str]
    ) -> bool:
        """Check if actual sequence matches pattern (allowing for gaps)"""

        if not pattern_sequence:
            return False

        pattern_index = 0
        for tool in actual_sequence:
            if (
                pattern_index < len(pattern_sequence)
                and tool == pattern_sequence[pattern_index]
            ):
                pattern_index += 1

        # Pattern matches if we found all pattern tools in order
        return pattern_index == len(pattern_sequence)

    def _create_query_outcome_pattern(
        self, query_type: str, experiences: List[AnalysisExperience]
    ) -> Optional[AnalysisPattern]:
        """Create pattern for query type and outcomes"""

        successful = [exp for exp in experiences if exp.success]
        if not successful:
            return None

        # Find common characteristics of successful analyses for this query type
        common_tools = self._find_common_tools(successful)
        avg_time = sum(exp.execution_time for exp in successful) / len(successful)

        pattern = AnalysisPattern(
            pattern_id=str(uuid.uuid4())[:8],
            pattern_type="query_outcome",
            description=f"Successful approach for {query_type} queries",
            conditions={"query_type": query_type},
            outcomes={
                "recommended_tools": common_tools,
                "expected_time": avg_time,
                "success_indicators": self._extract_success_indicators(successful),
            },
            occurrence_count=len(successful),
            success_rate=len(successful) / len(experiences),
            confidence=0.8,
            first_observed=datetime.now().isoformat(),
            last_observed=datetime.now().isoformat(),
            model_types=set(),
        )

        return pattern

    def _find_common_tools(self, experiences: List[AnalysisExperience]) -> List[str]:
        """Find tools commonly used in successful analyses"""

        tool_counts = {}
        for exp in experiences:
            for tool in exp.tools_used:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        # Return tools used in at least half of successful analyses
        threshold = len(experiences) / 2
        return [tool for tool, count in tool_counts.items() if count >= threshold]

    def _get_most_effective_tools(
        self, experiences: List[AnalysisExperience]
    ) -> List[str]:
        """Get tools that appear most frequently in effective strategies"""

        effective_tools = {}
        for exp in experiences:
            for strategy in exp.effective_strategies:
                if "tool" in strategy.lower():
                    for tool in exp.tools_used:
                        if tool.lower() in strategy.lower():
                            effective_tools[tool] = effective_tools.get(tool, 0) + 1

        # Sort by frequency and return top tools
        sorted_tools = sorted(effective_tools.items(), key=lambda x: x[1], reverse=True)
        return [tool for tool, count in sorted_tools[:5]]

    def _extract_success_indicators(
        self, experiences: List[AnalysisExperience]
    ) -> List[str]:
        """Extract common indicators of successful analyses"""

        indicators = []

        # Analyze insights to find common success patterns
        all_insights = []
        for exp in experiences:
            all_insights.extend(exp.insights_discovered)

        # Simple frequency analysis for now (could be enhanced with NLP)
        insight_words = {}
        for insight in all_insights:
            words = insight.lower().split()
            for word in words:
                if len(word) > 4:  # Focus on meaningful words
                    insight_words[word] = insight_words.get(word, 0) + 1

        # Top insight themes
        top_themes = sorted(insight_words.items(), key=lambda x: x[1], reverse=True)[:3]
        indicators.extend([f"Discovers {theme}" for theme, count in top_themes])

        return indicators

    def _load_pattern_templates(self) -> Dict[str, Any]:
        """Load predefined pattern templates"""

        templates = {
            "comprehensive_workflow": {
                "typical_sequence": [
                    "run_metabolic_fba",
                    "analyze_metabolic_model",
                    "analyze_essentiality",
                ],
                "conditions": {"query_type": "comprehensive"},
            },
            "growth_investigation": {
                "typical_sequence": [
                    "run_metabolic_fba",
                    "find_minimal_media",
                    "identify_auxotrophies",
                ],
                "conditions": {"growth_focus": True},
            },
        }

        return templates


class LearningMemory:
    """Manages learning memory and pattern application"""

    def __init__(self, llm, storage_path: Optional[Path] = None):
        """Initialize learning memory"""
        self.llm = llm
        self.storage_path = storage_path or Path("logs/learning_memory")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Memory components
        self.patterns = {}
        self.insights = {}
        self.experiences = []

        # Load existing memory
        self._load_memory()

    def record_analysis_experience(self, experience: AnalysisExperience):
        """Record a completed analysis experience"""

        self.experiences.append(experience)

        # Extract new patterns if we have enough data
        if len(self.experiences) % 5 == 0:  # Every 5 experiences
            self._update_patterns()

        # Save to storage
        self._save_experience(experience)

    def record_analysis(
        self,
        query: str,
        model_characteristics: Dict[str, Any],
        tools_used: List[str],
        outcome: str,
        confidence: float,
    ):
        """Record analysis for learning purposes - compatibility method"""
        try:
            from datetime import datetime

            # Create an AnalysisExperience from the parameters
            experience = AnalysisExperience(
                experience_id=str(datetime.now().timestamp()),
                session_id="pattern_memory_session",  # Default session ID
                timestamp=datetime.now().isoformat(),
                user_query=query,
                model_characteristics=model_characteristics,
                tools_used=tools_used,
                tool_sequence=tools_used,  # Same as tools_used for simplicity
                success=confidence > 0.5,
                insights_discovered=[],  # Changed from insights_gained
                execution_time=0.0,  # Not provided
                effective_strategies=[],  # Required field
                ineffective_strategies=[],  # Required field
                missed_opportunities=[],  # Required field
            )

            self.record_analysis_experience(experience)
        except Exception as e:
            # Silently ignore errors to avoid breaking the main workflow
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Could not record analysis experience: {e}")

    def record_summarization_effectiveness(
        self,
        tool_name: str,
        original_size_bytes: int,
        summarized_size_bytes: int,
        reduction_percentage: float,
        user_satisfaction_score: Optional[float] = None,
        information_completeness_score: Optional[float] = None,
        context_window_savings: Optional[int] = None,
    ) -> None:
        """Record Smart Summarization effectiveness metrics for learning

        Args:
            tool_name: Name of the tool that was summarized
            original_size_bytes: Size of original tool output
            summarized_size_bytes: Size of summarized output
            reduction_percentage: Percentage reduction achieved
            user_satisfaction_score: User satisfaction with summarized results (0.0-1.0)
            information_completeness_score: How much information was retained (0.0-1.0)
            context_window_savings: Token savings in LLM context window
        """
        try:
            # Find recent experiences to attach summarization metrics to
            if self.experiences:
                latest_experience = self.experiences[-1]

                # Initialize summarization metrics if not present
                if latest_experience.summarization_metrics is None:
                    latest_experience.summarization_metrics = {}

                # Add summarization effectiveness data
                tool_metrics = {
                    "original_size_bytes": original_size_bytes,
                    "summarized_size_bytes": summarized_size_bytes,
                    "reduction_percentage": reduction_percentage,
                    "user_satisfaction_score": user_satisfaction_score,
                    "information_completeness_score": information_completeness_score,
                    "context_window_savings": context_window_savings,
                    "timestamp": str(datetime.now().isoformat()),
                }

                latest_experience.summarization_metrics[tool_name] = tool_metrics

                # Update effective/ineffective strategies based on performance
                if reduction_percentage > 90 and (user_satisfaction_score or 0.8) > 0.7:
                    strategy = f"Smart Summarization for {tool_name} (≥90% reduction, high satisfaction)"
                    if strategy not in latest_experience.effective_strategies:
                        latest_experience.effective_strategies.append(strategy)

                elif (
                    reduction_percentage < 50 or (user_satisfaction_score or 0.3) < 0.5
                ):
                    strategy = f"Smart Summarization for {tool_name} (low effectiveness or satisfaction)"
                    if strategy not in latest_experience.ineffective_strategies:
                        latest_experience.ineffective_strategies.append(strategy)

                # Save updated experience
                self._save_experience(latest_experience)

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Could not record summarization effectiveness: {e}")

    def get_summarization_insights(self) -> Dict[str, Any]:
        """Analyze Smart Summarization effectiveness across all experiences

        Returns:
            Dict containing insights about which tools benefit most from summarization,
            average reduction rates, user satisfaction patterns, etc.
        """
        try:
            from datetime import datetime

            tool_effectiveness = {}
            total_metrics = {
                "total_tools_summarized": 0,
                "average_reduction_percentage": 0.0,
                "average_user_satisfaction": 0.0,
                "total_context_savings": 0,
                "most_effective_tools": [],
                "least_effective_tools": [],
            }

            # Analyze all experiences with summarization metrics
            for experience in self.experiences:
                if experience.summarization_metrics:
                    for tool_name, metrics in experience.summarization_metrics.items():
                        if tool_name not in tool_effectiveness:
                            tool_effectiveness[tool_name] = {
                                "count": 0,
                                "total_reduction": 0.0,
                                "total_satisfaction": 0.0,
                                "total_context_savings": 0,
                                "satisfaction_scores": [],
                            }

                        stats = tool_effectiveness[tool_name]
                        stats["count"] += 1
                        stats["total_reduction"] += metrics.get(
                            "reduction_percentage", 0
                        )

                        if metrics.get("user_satisfaction_score"):
                            stats["total_satisfaction"] += metrics[
                                "user_satisfaction_score"
                            ]
                            stats["satisfaction_scores"].append(
                                metrics["user_satisfaction_score"]
                            )

                        if metrics.get("context_window_savings"):
                            stats["total_context_savings"] += metrics[
                                "context_window_savings"
                            ]

            # Calculate averages and insights
            if tool_effectiveness:
                for tool_name, stats in tool_effectiveness.items():
                    stats["average_reduction"] = (
                        stats["total_reduction"] / stats["count"]
                    )
                    stats["average_satisfaction"] = (
                        stats["total_satisfaction"] / len(stats["satisfaction_scores"])
                        if stats["satisfaction_scores"]
                        else 0.0
                    )

                # Identify most/least effective tools
                sorted_by_effectiveness = sorted(
                    tool_effectiveness.items(),
                    key=lambda x: x[1]["average_reduction"]
                    * (x[1]["average_satisfaction"] or 0.8),
                    reverse=True,
                )

                total_metrics["most_effective_tools"] = sorted_by_effectiveness[:3]
                total_metrics["least_effective_tools"] = sorted_by_effectiveness[-3:]

                # Calculate overall averages
                all_reductions = [
                    stats["average_reduction"] for stats in tool_effectiveness.values()
                ]
                all_satisfactions = [
                    stats["average_satisfaction"]
                    for stats in tool_effectiveness.values()
                    if stats["average_satisfaction"] > 0
                ]

                total_metrics["total_tools_summarized"] = len(tool_effectiveness)
                total_metrics["average_reduction_percentage"] = (
                    sum(all_reductions) / len(all_reductions) if all_reductions else 0.0
                )
                total_metrics["average_user_satisfaction"] = (
                    sum(all_satisfactions) / len(all_satisfactions)
                    if all_satisfactions
                    else 0.0
                )
                total_metrics["total_context_savings"] = sum(
                    stats["total_context_savings"]
                    for stats in tool_effectiveness.values()
                )

            return {
                "tool_effectiveness": tool_effectiveness,
                "overall_metrics": total_metrics,
                "insights_generated": datetime.now().isoformat(),
            }

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Could not generate summarization insights: {e}")
            return {"error": str(e)}

    def get_recommended_approach(
        self, query: str, model_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get recommended approach based on learned patterns"""

        recommendations = {
            "recommended_tools": [],
            "suggested_sequence": [],
            "confidence": 0.0,
            "rationale": "",
            "applicable_patterns": [],
        }

        # Find applicable patterns
        applicable_patterns = self._find_applicable_patterns(
            query, model_characteristics
        )

        if applicable_patterns:
            # Combine recommendations from multiple patterns
            all_tools = []
            all_sequences = []
            total_confidence = 0

            for pattern in applicable_patterns:
                if "tool_sequence" in pattern.outcomes:
                    all_sequences.append(pattern.outcomes["tool_sequence"])
                if "recommended_tools" in pattern.outcomes:
                    all_tools.extend(pattern.outcomes["recommended_tools"])
                total_confidence += pattern.confidence
                recommendations["applicable_patterns"].append(pattern.pattern_id)

            # Deduplicate and prioritize
            recommendations["recommended_tools"] = list(set(all_tools))
            if all_sequences:
                recommendations["suggested_sequence"] = self._merge_sequences(
                    all_sequences
                )
            recommendations["confidence"] = min(
                total_confidence / len(applicable_patterns), 1.0
            )

            # Generate rationale
            recommendations["rationale"] = self._generate_recommendation_rationale(
                applicable_patterns
            )

        return recommendations

    def get_relevant_insights(
        self, query: str, model_characteristics: Dict[str, Any]
    ) -> List[MetabolicInsight]:
        """Get insights relevant to current analysis"""

        relevant_insights = []

        for insight in self.insights.values():
            if self._insight_is_relevant(insight, query, model_characteristics):
                relevant_insights.append(insight)

        # Sort by relevance/confidence
        relevant_insights.sort(key=lambda x: x.confidence_score, reverse=True)

        return relevant_insights[:5]  # Return top 5 most relevant

    def update_pattern_success(self, pattern_id: str, was_successful: bool):
        """Update pattern success rate based on application outcome"""

        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.times_applied += 1

            if was_successful:
                successful_applications = (
                    pattern.application_success_rate * (pattern.times_applied - 1) + 1
                )
            else:
                successful_applications = pattern.application_success_rate * (
                    pattern.times_applied - 1
                )

            pattern.application_success_rate = (
                successful_applications / pattern.times_applied
            )

            # Update overall confidence based on application results
            pattern.confidence = (
                pattern.success_rate + pattern.application_success_rate
            ) / 2

            self._save_patterns()

    def _find_applicable_patterns(
        self, query: str, model_characteristics: Dict[str, Any]
    ) -> List[AnalysisPattern]:
        """Find patterns applicable to current situation"""

        applicable = []
        query_type = self._classify_query_type(query)

        for pattern in self.patterns.values():
            if self._pattern_matches_context(
                pattern, query_type, model_characteristics
            ):
                applicable.append(pattern)

        # Sort by confidence and success rate
        applicable.sort(key=lambda x: x.confidence * x.success_rate, reverse=True)

        return applicable

    def _pattern_matches_context(
        self,
        pattern: AnalysisPattern,
        query_type: str,
        model_characteristics: Dict[str, Any],
    ) -> bool:
        """Check if pattern matches current context"""

        # Check query type match
        if "query_type" in pattern.conditions:
            if pattern.conditions["query_type"] != query_type:
                return False

        # Check model characteristics match
        for key, value in pattern.conditions.items():
            if key in model_characteristics:
                if model_characteristics[key] != value:
                    return False

        return True

    def _insight_is_relevant(
        self,
        insight: MetabolicInsight,
        query: str,
        model_characteristics: Dict[str, Any],
    ) -> bool:
        """Check if insight is relevant to current analysis"""

        # Simple keyword matching for now
        query_lower = query.lower()
        insight_lower = insight.summary.lower()

        # Check for keyword overlap
        query_words = set(query_lower.split())
        insight_words = set(insight_lower.split())
        overlap = len(query_words.intersection(insight_words))

        return overlap > 0

    def _merge_sequences(self, sequences: List[List[str]]) -> List[str]:
        """Merge multiple tool sequences into a recommended sequence"""

        # Simple approach: find most common starting tools and build sequence
        if not sequences:
            return []

        # Count frequency of each tool at each position
        max_length = max(len(seq) for seq in sequences)
        position_counts = [{} for _ in range(max_length)]

        for seq in sequences:
            for i, tool in enumerate(seq):
                if tool not in position_counts[i]:
                    position_counts[i][tool] = 0
                position_counts[i][tool] += 1

        # Build merged sequence
        merged_sequence = []
        for position_count in position_counts:
            if position_count:
                most_common_tool = max(position_count.items(), key=lambda x: x[1])[0]
                if most_common_tool not in merged_sequence:  # Avoid duplicates
                    merged_sequence.append(most_common_tool)

        return merged_sequence

    def _generate_recommendation_rationale(
        self, patterns: List[AnalysisPattern]
    ) -> str:
        """Generate explanation for recommendations"""

        if not patterns:
            return "No specific patterns found - using general approach"

        rationale_parts = []
        for pattern in patterns:
            confidence_desc = (
                "high"
                if pattern.confidence > 0.8
                else "moderate" if pattern.confidence > 0.6 else "low"
            )
            rationale_parts.append(
                f"{pattern.description} (observed {pattern.occurrence_count} times, {confidence_desc} confidence)"
            )

        return "Based on learned patterns: " + "; ".join(rationale_parts)

    def _update_patterns(self):
        """Update patterns based on recent experiences"""

        extractor = PatternExtractor(self.llm)
        new_patterns = extractor.extract_patterns_from_experiences(
            self.experiences[-10:]
        )  # Last 10 experiences

        for pattern in new_patterns:
            if pattern.pattern_id not in self.patterns:
                self.patterns[pattern.pattern_id] = pattern
            else:
                # Update existing pattern
                existing = self.patterns[pattern.pattern_id]
                existing.occurrence_count += pattern.occurrence_count
                existing.last_observed = pattern.last_observed

        self._save_patterns()

    def _classify_query_type(self, query: str) -> str:
        """Classify query type (reuse from PatternExtractor)"""

        query_lower = query.lower()

        if any(word in query_lower for word in ["comprehensive", "complete", "full"]):
            return "comprehensive_analysis"
        elif any(word in query_lower for word in ["growth", "biomass", "rate"]):
            return "growth_analysis"
        elif any(word in query_lower for word in ["essential", "gene", "knockout"]):
            return "essentiality_analysis"
        elif any(word in query_lower for word in ["media", "nutrient", "auxotroph"]):
            return "media_analysis"
        elif any(word in query_lower for word in ["flux", "pathway", "variability"]):
            return "flux_analysis"
        else:
            return "general_analysis"

    def _load_memory(self):
        """Load memory from storage"""

        # Load patterns
        patterns_file = self.storage_path / "patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file) as f:
                    patterns_data = json.load(f)
                    for pattern_data in patterns_data:
                        pattern = AnalysisPattern(**pattern_data)
                        self.patterns[pattern.pattern_id] = pattern
            except Exception as e:
                print(f"Warning: Could not load patterns: {e}")

        # Load insights
        insights_file = self.storage_path / "insights.json"
        if insights_file.exists():
            try:
                with open(insights_file) as f:
                    insights_data = json.load(f)
                    for insight_data in insights_data:
                        insight = MetabolicInsight(**insight_data)
                        self.insights[insight.insight_id] = insight
            except Exception as e:
                print(f"Warning: Could not load insights: {e}")

    def _save_patterns(self):
        """Save patterns to storage"""

        patterns_file = self.storage_path / "patterns.json"
        try:
            patterns_data = [pattern.dict() for pattern in self.patterns.values()]
            with open(patterns_file, "w") as f:
                json.dump(patterns_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save patterns: {e}")

    def _save_experience(self, experience: AnalysisExperience):
        """Save individual experience"""

        experiences_file = self.storage_path / "experiences.json"
        try:
            experiences_data = []
            if experiences_file.exists():
                with open(experiences_file) as f:
                    experiences_data = json.load(f)

            experiences_data.append(experience.dict())

            with open(experiences_file, "w") as f:
                json.dump(experiences_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save experience: {e}")


def create_learning_system(llm, storage_path: Optional[Path] = None):
    """Factory function to create learning and pattern memory system"""

    return LearningMemory(llm, storage_path)
