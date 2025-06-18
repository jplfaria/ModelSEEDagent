"""FBA Summarizer for Smart Summarization Framework

Provides three-tier summarization for Flux Balance Analysis results:
- key_findings: Essential FBA insights for LLM consumption
- summary_dict: Core metrics and statistics
- full_data_path: Complete flux data and detailed results
"""

from typing import Any, Dict, List, Optional, Union

from ..base import ToolResult
from ..smart_summarization import BaseSummarizer, artifact_storage, summarizer_registry


class FBASummarizer(BaseSummarizer):
    """Summarizer for FBA tool results"""

    def get_tool_name(self) -> str:
        return "run_metabolic_fba"

    def summarize(
        self,
        raw_output: Any,
        artifact_path: str,
        model_stats: Optional[Dict[str, Union[str, int]]] = None,
    ) -> ToolResult:
        """Summarize FBA results into three-tier hierarchy"""

        model_id = model_stats.get("model_id", "unknown") if model_stats else "unknown"

        # Extract core FBA data
        objective_value = raw_output.get("objective_value", 0.0)
        status = raw_output.get("status", "unknown")
        significant_fluxes = raw_output.get("significant_fluxes", {})
        subsystem_fluxes = raw_output.get("subsystem_fluxes", {})

        # Generate key findings (≤2KB) - Focus on critical insights for LLM
        key_findings = self._generate_key_findings(
            objective_value, status, significant_fluxes, subsystem_fluxes, model_id
        )

        # Generate summary dict (≤5KB) - Structured data for follow-up analysis
        summary_dict = self._generate_summary_dict(
            objective_value, status, significant_fluxes, subsystem_fluxes, model_stats
        )

        # Validate size constraints
        self.validate_size_limits(key_findings, summary_dict)

        return ToolResult(
            success=True,
            message=f"FBA analysis completed for {model_id} (growth rate: {objective_value:.4f} h⁻¹)",
            data=raw_output,  # Legacy compatibility
            key_findings=key_findings,
            summary_dict=summary_dict,
            full_data_path=artifact_path,
            tool_name=self.get_tool_name(),
            model_stats=model_stats,
            schema_version="1.0",
        )

    def _generate_key_findings(
        self,
        objective_value: float,
        status: str,
        significant_fluxes: Dict[str, Any],
        subsystem_fluxes: Dict[str, Any],
        model_id: str,
    ) -> List[str]:
        """Generate key findings for LLM consumption"""
        findings = []

        # Growth rate finding
        if status == "optimal":
            findings.append(
                f"Optimal growth rate achieved: {objective_value:.4f} h⁻¹ for {model_id}"
            )
        elif status == "infeasible":
            findings.append(
                f"No feasible growth solution found for {model_id} under current conditions"
            )
        else:
            findings.append(
                f"FBA status: {status} with objective value {objective_value:.4f}"
            )

        # Flux activity insights
        active_fluxes = len([f for f in significant_fluxes.values() if abs(f) > 1e-6])
        findings.append(
            f"Network utilization: {active_fluxes}/{len(significant_fluxes)} reactions carry significant flux"
        )

        # High flux reactions
        if significant_fluxes:
            high_flux_reactions = [
                reaction
                for reaction, flux in significant_fluxes.items()
                if abs(flux) > objective_value * 0.1  # >10% of growth rate
            ]
            if high_flux_reactions:
                findings.append(
                    f"High-flux reactions ({len(high_flux_reactions)}): {', '.join(high_flux_reactions[:5])}"
                )

        # Subsystem activity
        if subsystem_fluxes:
            active_subsystems = []
            for sys, flux_data in subsystem_fluxes.items():
                # Handle both single values and lists
                if isinstance(flux_data, (list, tuple)):
                    total_flux = sum(
                        abs(f) for f in flux_data if isinstance(f, (int, float))
                    )
                else:
                    total_flux = (
                        abs(flux_data) if isinstance(flux_data, (int, float)) else 0
                    )

                if total_flux > 1e-6:
                    active_subsystems.append(sys)

            if active_subsystems:
                findings.append(
                    f"Active metabolic subsystems: {', '.join(active_subsystems)}"
                )

        return findings

    def _generate_summary_dict(
        self,
        objective_value: float,
        status: str,
        significant_fluxes: Dict[str, Any],
        subsystem_fluxes: Dict[str, Any],
        model_stats: Optional[Dict[str, Union[str, int]]],
    ) -> Dict[str, Any]:
        """Generate structured summary for follow-up analysis"""

        # Calculate flux statistics
        flux_values = [abs(f) for f in significant_fluxes.values() if abs(f) > 1e-6]

        summary = {
            "growth_metrics": {
                "objective_value": round(objective_value, 6),
                "status": status,
                "feasible": status == "optimal",
            },
            "flux_statistics": {
                "total_reactions": len(significant_fluxes),
                "active_reactions": len(flux_values),
                "inactive_reactions": len(significant_fluxes) - len(flux_values),
                "activity_percentage": (
                    round((len(flux_values) / len(significant_fluxes) * 100), 1)
                    if significant_fluxes
                    else 0
                ),
            },
        }

        # Add flux distribution if available
        if flux_values:
            summary["flux_statistics"].update(
                {
                    "max_flux": round(max(flux_values), 6),
                    "mean_active_flux": round(sum(flux_values) / len(flux_values), 6),
                    "high_flux_reactions": len(
                        [f for f in flux_values if f > objective_value * 0.1]
                    ),
                }
            )

        # Add subsystem information
        if subsystem_fluxes:
            subsystem_activity = {}
            for subsys, flux_data in subsystem_fluxes.items():
                # Handle both single values and lists
                if isinstance(flux_data, (list, tuple)):
                    total_flux = sum(
                        f for f in flux_data if isinstance(f, (int, float))
                    )
                else:
                    total_flux = flux_data if isinstance(flux_data, (int, float)) else 0
                subsystem_activity[subsys] = round(total_flux, 6)
            summary["subsystem_activity"] = subsystem_activity

        # Add model metadata if available
        if model_stats:
            summary["model_info"] = {
                "model_id": model_stats.get("model_id", "unknown"),
                "reactions": model_stats.get("reactions", len(significant_fluxes)),
                "metabolites": model_stats.get("metabolites", "unknown"),
            }

        return summary


# Create and register the FBA summarizer
fba_summarizer = FBASummarizer()
summarizer_registry.register(fba_summarizer)
