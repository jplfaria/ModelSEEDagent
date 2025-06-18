"""FluxVariabilityAnalysis Summarizer

Priority implementation for Smart Summarization Framework.
Reduces FVA output from 200+ KB to 2KB while preserving critical insights.
"""

import json
from typing import Any, Dict, List, Optional, Union

from ..base import ToolResult
from ..smart_summarization import BaseSummarizer, summarizer_registry


class FluxVariabilitySummarizer(BaseSummarizer):
    """Smart summarizer for FluxVariabilityAnalysis tool

    Transforms detailed FVA results into three-tier summarization:
    - key_findings: Critical insights about reaction variability (≤2KB)
    - summary_dict: Statistical summary and top reactions (≤5KB)
    - full_data_path: Complete FVA DataFrame on disk
    """

    def get_tool_name(self) -> str:
        return "run_flux_variability_analysis"

    def summarize(
        self,
        raw_output: Any,
        artifact_path: str,
        model_stats: Optional[Dict[str, Union[str, int]]] = None,
    ) -> ToolResult:
        """Summarize FVA output preserving critical negative evidence

        Args:
            raw_output: Original FVA tool output with data/summary structure
            artifact_path: Path where full FVA data is stored
            model_stats: Model metadata (reactions, genes, etc.)

        Returns:
            ToolResult with three-tier FVA summarization
        """
        # Extract FVA data from tool output
        if isinstance(raw_output, dict) and "fva_results" in raw_output:
            fva_data = raw_output["fva_results"]
            summary_data = raw_output.get("summary", {})
        else:
            # Fallback for direct DataFrame input
            fva_data = raw_output
            summary_data = {}

        # Parse the reaction categorization from our bloat-fixed FVA tool
        if "blocked_reactions" in summary_data:
            # New streamlined format (after bloat fix)
            blocked_reactions = summary_data["blocked_reactions"]
            variable_reactions = summary_data["variable_reactions"]
            fixed_reactions = summary_data["fixed_reactions"]
            essential_reactions = summary_data["essential_reactions"]
        else:
            # Fallback: analyze raw FVA data
            (
                blocked_reactions,
                variable_reactions,
                fixed_reactions,
                essential_reactions,
            ) = self._analyze_fva_data(fva_data)

        # Extract total reaction count
        total_reactions = (
            len(fva_data.get("minimum", {}))
            if isinstance(fva_data, dict)
            else len(fva_data)
        )
        model_id = model_stats.get("model_id", "unknown") if model_stats else "unknown"

        # Generate key findings (≤2KB) - Focus on critical insights for LLM
        key_findings = self._generate_key_findings(
            total_reactions,
            blocked_reactions,
            variable_reactions,
            fixed_reactions,
            essential_reactions,
            model_id,
        )

        # Generate summary dict (≤5KB) - Structured data for analysis
        summary_dict = self._generate_summary_dict(
            total_reactions,
            blocked_reactions,
            variable_reactions,
            fixed_reactions,
            essential_reactions,
            fva_data,
            model_stats,
        )

        # Enhance key findings with smart bucketing insights if available
        if "smart_flux_buckets" in summary_dict:
            buckets = summary_dict["smart_flux_buckets"]
            if "insights" in buckets:
                insights = buckets["insights"]
                optimization_potential = insights.get("optimization_potential", 0)
                if optimization_potential > 0:
                    key_findings.append(
                        f"High optimization potential: {optimization_potential} highly variable reactions"
                    )

        # Validate size limits
        self.validate_size_limits(key_findings, summary_dict)

        return ToolResult(
            success=True,
            message=f"FVA analysis summarized: {total_reactions} reactions analyzed",
            full_data_path=artifact_path,
            summary_dict=summary_dict,
            key_findings=key_findings,
            tool_name=self.get_tool_name(),
            model_stats=model_stats,
            schema_version="1.0",
        )

    def _analyze_fva_data(self, fva_data: Any) -> tuple:
        """Analyze raw FVA data to categorize reactions"""
        # This is a fallback for cases where summary isn't available
        blocked_reactions = []
        variable_reactions = []
        fixed_reactions = []
        essential_reactions = []

        if (
            isinstance(fva_data, dict)
            and "minimum" in fva_data
            and "maximum" in fva_data
        ):
            # Standard FVA DataFrame format converted to dict
            min_fluxes = fva_data["minimum"]
            max_fluxes = fva_data["maximum"]

            for rxn_id in min_fluxes.keys():
                min_flux = min_fluxes[rxn_id]
                max_flux = max_fluxes[rxn_id]
                flux_range = abs(max_flux - min_flux)

                # Categorize based on flux analysis
                if abs(min_flux) < 1e-6 and abs(max_flux) < 1e-6:
                    blocked_reactions.append(rxn_id)
                elif flux_range < 1e-6:
                    fixed_reactions.append(rxn_id)
                else:
                    variable_reactions.append(rxn_id)

                # Essential reactions carry flux in one direction
                if min_flux > 1e-6 or max_flux < -1e-6:
                    essential_reactions.append(rxn_id)

        return (
            blocked_reactions,
            variable_reactions,
            fixed_reactions,
            essential_reactions,
        )

    def _generate_key_findings(
        self,
        total_reactions: int,
        blocked: List[str],
        variable: List[str],
        fixed: List[str],
        essential: List[str],
        model_id: str,
    ) -> List[str]:
        """Generate critical insights for LLM consumption (≤2KB)"""
        blocked_count = len(blocked)
        variable_count = len(variable)
        fixed_count = len(fixed)
        essential_count = len(essential)

        # Calculate percentages
        blocked_pct = (
            (blocked_count / total_reactions * 100) if total_reactions > 0 else 0
        )
        variable_pct = (
            (variable_count / total_reactions * 100) if total_reactions > 0 else 0
        )
        essential_pct = (
            (essential_count / total_reactions * 100) if total_reactions > 0 else 0
        )

        key_findings = [
            f"FVA analysis of {model_id}: {total_reactions} reactions analyzed",
            f"Blocked reactions: {blocked_count} ({blocked_pct:.1f}%) - cannot carry flux",
            f"Variable reactions: {variable_count} ({variable_pct:.1f}%) - flux can vary",
            f"Fixed reactions: {fixed_count} ({(100-blocked_pct-variable_pct):.1f}%) - carry fixed flux",
            f"Essential reactions: {essential_count} ({essential_pct:.1f}%) - required for growth",
        ]

        # Add critical insights about network flexibility
        if variable_count == 0:
            key_findings.append(
                "WARNING: No variable reactions found - model may be over-constrained"
            )
        elif variable_pct < 10:
            key_findings.append(
                "WARNING: Very low flux variability - limited metabolic flexibility"
            )
        elif variable_pct > 50:
            key_findings.append(
                "High flux variability - significant metabolic flexibility"
            )

        # Critical blocked reactions warning (preserve negative evidence)
        if blocked_count > total_reactions * 0.5:
            key_findings.append("WARNING: >50% reactions blocked - check media constraints")

        # Examples of top variable reactions (if available)
        if len(variable) > 0:
            top_variable = variable[:3]  # Show top 3
            key_findings.append(f"Top variable reactions: {', '.join(top_variable)}")

        return key_findings

    def _generate_summary_dict(
        self,
        total_reactions: int,
        blocked: List[str],
        variable: List[str],
        fixed: List[str],
        essential: List[str],
        fva_data: Any,
        model_stats: Optional[Dict[str, Union[str, int]]],
    ) -> Dict[str, Any]:
        """Generate structured summary for analysis (≤5KB)"""
        summary_dict = {
            "statistics": {
                "total_reactions": total_reactions,
                "blocked_count": len(blocked),
                "variable_count": len(variable),
                "fixed_count": len(fixed),
                "essential_count": len(essential),
                "network_flexibility_score": (
                    len(variable) / total_reactions if total_reactions > 0 else 0
                ),
            },
            "reaction_categories": {
                "blocked_reactions": blocked[
                    :10
                ],  # Top 10 blocked (critical for debugging)
                "variable_reactions": variable[
                    :10
                ],  # Top 10 variable (optimization targets)
                "essential_reactions": essential[
                    :10
                ],  # Top 10 essential (critical pathways)
            },
            "model_context": model_stats or {},
            "analysis_metadata": {
                "fva_method": "flux_variability_analysis",
                "categorization_threshold": 1e-6,
                "framework_version": "1.0",
            },
        }

        # Add flux statistics if raw FVA data is available
        if isinstance(fva_data, dict) and "minimum" in fva_data:
            min_fluxes = list(fva_data["minimum"].values())
            max_fluxes = list(fva_data["maximum"].values())

            # Smart bucketing for flux ranges
            flux_buckets = self._smart_flux_bucketing(fva_data, variable)

            summary_dict["flux_statistics"] = {
                "min_flux_observed": min(min_fluxes) if min_fluxes else 0,
                "max_flux_observed": max(max_fluxes) if max_fluxes else 0,
                "avg_flux_range": (
                    sum(
                        abs(max_fluxes[i] - min_fluxes[i])
                        for i in range(len(min_fluxes))
                    )
                    / len(min_fluxes)
                    if min_fluxes
                    else 0
                ),
            }

            # Add smart bucketing results
            summary_dict["smart_flux_buckets"] = flux_buckets

        return summary_dict

    def _smart_flux_bucketing(
        self, fva_data: Dict[str, Any], variable_reactions: List[str]
    ) -> Dict[str, Any]:
        """Intelligent bucketing of flux ranges for better insights

        Categorizes variable reactions by flux range magnitude and distribution
        to identify different types of metabolic flexibility.
        """
        if not isinstance(fva_data, dict) or "minimum" not in fva_data:
            return {}

        min_fluxes = fva_data["minimum"]
        max_fluxes = fva_data["maximum"]

        # Calculate flux ranges for variable reactions
        flux_ranges = []
        for rxn_id in variable_reactions:
            if rxn_id in min_fluxes and rxn_id in max_fluxes:
                flux_range = abs(max_fluxes[rxn_id] - min_fluxes[rxn_id])
                flux_ranges.append((rxn_id, flux_range))

        if not flux_ranges:
            return {}

        # Sort by flux range
        flux_ranges.sort(key=lambda x: x[1], reverse=True)

        # Smart bucketing based on flux range distribution
        total_ranges = len(flux_ranges)
        if total_ranges == 0:
            return {}

        # Calculate dynamic thresholds based on data distribution
        all_ranges = [r[1] for r in flux_ranges]
        max_range = max(all_ranges)
        mean_range = sum(all_ranges) / len(all_ranges)

        # Define smart buckets with adaptive thresholds
        high_threshold = max(mean_range * 3, max_range * 0.3)  # Top tier variability
        medium_threshold = max(mean_range * 0.5, max_range * 0.1)  # Medium variability
        low_threshold = mean_range * 0.1  # Low but significant variability

        # Categorize reactions into smart buckets
        high_variability = []
        medium_variability = []
        low_variability = []
        minimal_variability = []

        for rxn_id, flux_range in flux_ranges:
            if flux_range >= high_threshold:
                high_variability.append(
                    {"reaction_id": rxn_id, "flux_range": round(flux_range, 4)}
                )
            elif flux_range >= medium_threshold:
                medium_variability.append(
                    {"reaction_id": rxn_id, "flux_range": round(flux_range, 4)}
                )
            elif flux_range >= low_threshold:
                low_variability.append(
                    {"reaction_id": rxn_id, "flux_range": round(flux_range, 4)}
                )
            else:
                minimal_variability.append(
                    {"reaction_id": rxn_id, "flux_range": round(flux_range, 4)}
                )

        return {
            "bucketing_thresholds": {
                "high_variability": round(high_threshold, 4),
                "medium_variability": round(medium_threshold, 4),
                "low_variability": round(low_threshold, 4),
                "max_range_observed": round(max_range, 4),
                "mean_range": round(mean_range, 4),
            },
            "variability_categories": {
                "high_variability": {
                    "count": len(high_variability),
                    "reactions": high_variability[:5],  # Top 5 most variable
                    "description": "Major flux alternatives - optimization targets",
                },
                "medium_variability": {
                    "count": len(medium_variability),
                    "reactions": medium_variability[:3],  # Top 3 medium variable
                    "description": "Moderate flux flexibility - adaptation pathways",
                },
                "low_variability": {
                    "count": len(low_variability),
                    "reactions": low_variability[:2],  # Top 2 low variable
                    "description": "Minor flux adjustments - fine-tuning",
                },
                "minimal_variability": {
                    "count": len(minimal_variability),
                    "description": "Barely variable - near-fixed flux",
                },
            },
            "insights": {
                "total_variable_reactions": total_ranges,
                "optimization_potential": len(high_variability)
                + len(medium_variability),
                "flexibility_distribution": {
                    "high_flex_pct": round(
                        len(high_variability) / total_ranges * 100, 1
                    ),
                    "medium_flex_pct": round(
                        len(medium_variability) / total_ranges * 100, 1
                    ),
                    "low_flex_pct": round(len(low_variability) / total_ranges * 100, 1),
                },
            },
        }


# Register the summarizer
flux_variability_summarizer = FluxVariabilitySummarizer()
summarizer_registry.register(flux_variability_summarizer)
