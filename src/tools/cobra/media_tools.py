"""
AI-Friendly Media Tools Suite
=============================

Specialized tools for AI agents to intelligently work with media and models.
Each tool has a focused purpose for better AI decision-making.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cobra
from pydantic import BaseModel, Field

from ..base import BaseTool, ToolRegistry, ToolResult
from .media_library import MediaLibrary, get_media_library
from .modelseedpy_integration import ModelSEEDpyEnhancement, get_modelseedpy_enhancement
from .utils import ModelUtils


class MediaConfig(BaseModel):
    """Configuration for media tools"""

    default_data_dir: Optional[str] = None
    enable_ai_suggestions: bool = True
    max_media_suggestions: int = 5


@ToolRegistry.register
class MediaSelectorTool(BaseTool):
    """AI tool to intelligently select appropriate media for a given model"""

    tool_name = "select_optimal_media"
    tool_description = """Select the most appropriate media type for a metabolic model.
    Analyzes model characteristics and suggests compatible media with growth predictions."""

    def model_post_init(self, __context):
        """Initialize components after Pydantic model creation"""
        super().model_post_init(__context)
        self._media_library = get_media_library()
        self._enhancement = get_modelseedpy_enhancement()
        self._model_utils = ModelUtils()

    def _run_tool(self, input_data: Dict[str, Any]) -> ToolResult:
        """Select optimal media for a model"""
        try:
            model_path = input_data.get("model_path")
            model_object = input_data.get("model_object")
            target_growth = input_data.get("target_growth", None)
            exclude_media = input_data.get("exclude_media", [])

            # Load model
            if model_object:
                model = model_object
            elif model_path:
                model = self._model_utils.load_model(model_path)
            else:
                raise ValueError("Either model_path or model_object required")

            # Analyze model characteristics
            analysis = self._enhancement.analyze_modelseedpy_model(model)

            # Test all available media
            media_results = {}
            recommendations = []

            for media_name in self._media_library.list_available_media():
                if media_name in exclude_media:
                    continue

                try:
                    # Test growth
                    model_with_media = self._enhancement.apply_media_with_cobrakbase(
                        model, media_name
                    )
                    solution = model_with_media.optimize()
                    growth_rate = (
                        solution.objective_value
                        if solution.status == "optimal"
                        else 0.0
                    )

                    # Check compatibility
                    compatibility = self._media_library.analyze_media_compatibility(
                        model, media_name
                    )

                    media_results[media_name] = {
                        "growth_rate": growth_rate,
                        "status": solution.status,
                        "feasible": growth_rate > 1e-6,
                        "compatibility_score": compatibility["compatibility_score"],
                        "media_description": self._media_library.get_media(
                            media_name
                        ).description,
                    }

                    # Add to recommendations if feasible
                    if growth_rate > 1e-6:
                        recommendations.append(
                            {
                                "media_name": media_name,
                                "growth_rate": growth_rate,
                                "compatibility": compatibility["compatibility_score"],
                                "recommendation_score": growth_rate
                                * compatibility["compatibility_score"],
                            }
                        )

                except Exception as e:
                    media_results[media_name] = {
                        "error": str(e),
                        "feasible": False,
                    }

            # Sort recommendations by score
            recommendations.sort(key=lambda x: x["recommendation_score"], reverse=True)

            # Generate AI suggestions
            best_media = recommendations[0]["media_name"] if recommendations else None
            ai_suggestion = self._generate_media_suggestion(
                model, analysis, recommendations, target_growth
            )

            return ToolResult(
                success=True,
                message=f"Analyzed {len(media_results)} media types. Best: {best_media}",
                data={
                    "model_analysis": analysis,
                    "media_results": media_results,
                    "recommendations": recommendations[:5],  # Top 5
                    "best_media": best_media,
                    "ai_suggestion": ai_suggestion,
                },
                metadata={
                    "model_id": model.id,
                    "total_media_tested": len(media_results),
                    "feasible_media_count": len(recommendations),
                    "best_growth_rate": (
                        recommendations[0]["growth_rate"] if recommendations else 0
                    ),
                },
            )

        except Exception as e:
            return ToolResult(
                success=False, message=f"Media selection failed: {str(e)}", error=str(e)
            )

    def _generate_media_suggestion(
        self, model, analysis, recommendations, target_growth
    ):
        """Generate AI-driven media suggestions"""
        if not recommendations:
            return {
                "suggestion": "No compatible media found",
                "reason": "Model may have missing exchange reactions or metabolic gaps",
                "action": "Consider using minimal media or checking model completeness",
            }

        best = recommendations[0]

        if target_growth and best["growth_rate"] < target_growth:
            return {
                "suggestion": f"Use {best['media_name']} but consider media optimization",
                "reason": f"Best growth ({best['growth_rate']:.3f}) below target ({target_growth:.3f})",
                "action": "Try adding nutrients or using MediaOptimizationTool",
            }

        if analysis["is_modelseedpy"]:
            return {
                "suggestion": f"Recommended: {best['media_name']}",
                "reason": f"Optimal for ModelSEED models: {best['growth_rate']:.3f} h⁻¹",
                "action": f"Apply {best['media_name']} for reliable growth simulation",
            }
        else:
            return {
                "suggestion": f"Recommended: {best['media_name']}",
                "reason": f"Best compatibility with BIGG models: {best['growth_rate']:.3f} h⁻¹",
                "action": f"Use {best['media_name']} with possible compound mapping",
            }


@ToolRegistry.register
class MediaManipulatorTool(BaseTool):
    """AI tool to modify media compositions using natural language commands"""

    tool_name = "manipulate_media_composition"
    tool_description = """Modify media compositions using AI-driven natural language commands.
    Examples: 'make anaerobic', 'add vitamins', 'increase glucose uptake', 'remove amino acids'."""

    def model_post_init(self, __context):
        """Initialize components after Pydantic model creation"""
        super().model_post_init(__context)
        self._media_library = get_media_library()
        self._enhancement = get_modelseedpy_enhancement()

    def _run_tool(self, input_data: Dict[str, Any]) -> ToolResult:
        """Manipulate media using AI commands"""
        try:
            base_media = input_data.get("base_media", "GMM")
            ai_command = input_data.get("ai_command")
            model_path = input_data.get("model_path")
            model_object = input_data.get("model_object")
            test_growth = input_data.get("test_growth", True)
            save_to_file = input_data.get("save_to_file")

            if not ai_command:
                raise ValueError("ai_command is required")

            # Create modified media
            original_media = self._media_library.get_media(base_media)
            modified_media = self._media_library.manipulate_media(
                base_media, ai_command
            )

            # Calculate changes
            changes = self._analyze_media_changes(original_media, modified_media)

            result_data = {
                "original_media": {
                    "name": original_media.name,
                    "compounds": len(original_media.compounds),
                    "description": original_media.description,
                },
                "modified_media": {
                    "name": modified_media.name,
                    "compounds": len(modified_media.compounds),
                    "description": modified_media.description,
                },
                "changes": changes,
                "ai_command": ai_command,
            }

            # Test growth if model provided
            if test_growth and (model_path or model_object):
                growth_results = self._test_media_growth(
                    model_path, model_object, base_media, modified_media
                )
                result_data["growth_comparison"] = growth_results

            # Save to file if requested
            if save_to_file:
                self._media_library.save_media_to_file(
                    modified_media, save_to_file, "tsv"
                )
                result_data["saved_file"] = save_to_file

            return ToolResult(
                success=True,
                message=f"Media manipulation completed: {ai_command}",
                data=result_data,
                metadata={
                    "base_media": base_media,
                    "command_applied": ai_command,
                    "compounds_added": changes["added_count"],
                    "compounds_removed": changes["removed_count"],
                    "compounds_modified": changes["modified_count"],
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Media manipulation failed: {str(e)}",
                error=str(e),
            )

    def _analyze_media_changes(self, original, modified):
        """Analyze what changed between media compositions"""
        orig_compounds = set(original.compounds.keys())
        mod_compounds = set(modified.compounds.keys())

        added = mod_compounds - orig_compounds
        removed = orig_compounds - mod_compounds
        common = orig_compounds & mod_compounds

        modified_compounds = []
        for compound in common:
            if original.compounds[compound] != modified.compounds[compound]:
                modified_compounds.append(
                    {
                        "compound": compound,
                        "original_rate": original.compounds[compound],
                        "modified_rate": modified.compounds[compound],
                    }
                )

        return {
            "added_compounds": list(added),
            "removed_compounds": list(removed),
            "modified_compounds": modified_compounds,
            "added_count": len(added),
            "removed_count": len(removed),
            "modified_count": len(modified_compounds),
        }

    def _test_media_growth(
        self, model_path, model_object, base_media_name, modified_media
    ):
        """Test growth with original vs modified media"""
        try:
            # Load model
            if model_object:
                model = model_object
            elif model_path:
                model_utils = ModelUtils()
                model = model_utils.load_model(model_path)
            else:
                return {"error": "No model provided"}

            # Test original media
            orig_model = self._enhancement.apply_media_with_cobrakbase(
                model, base_media_name
            )
            orig_solution = orig_model.optimize()
            orig_growth = (
                orig_solution.objective_value
                if orig_solution.status == "optimal"
                else 0
            )

            # Test modified media
            mod_model = model.copy()
            mod_model = self._media_library.apply_media_to_model(
                mod_model, modified_media.name
            )
            mod_solution = mod_model.optimize()
            mod_growth = (
                mod_solution.objective_value if mod_solution.status == "optimal" else 0
            )

            growth_change = mod_growth - orig_growth

            return {
                "original_growth": orig_growth,
                "modified_growth": mod_growth,
                "growth_change": growth_change,
                "improvement": growth_change > 1e-6,
                "original_status": orig_solution.status,
                "modified_status": mod_solution.status,
            }

        except Exception as e:
            return {"error": str(e)}


@ToolRegistry.register
class MediaCompatibilityTool(BaseTool):
    """AI tool to analyze media-model compatibility and suggest improvements"""

    tool_name = "analyze_media_compatibility"
    tool_description = """Analyze compatibility between media and models.
    Identifies mapping issues and suggests media modifications for better compatibility."""

    def model_post_init(self, __context):
        """Initialize components after Pydantic model creation"""
        super().model_post_init(__context)
        self._media_library = get_media_library()
        self._enhancement = get_modelseedpy_enhancement()

    def _run_tool(self, input_data: Dict[str, Any]) -> ToolResult:
        """Analyze media-model compatibility"""
        try:
            model_path = input_data.get("model_path")
            model_object = input_data.get("model_object")
            media_names = input_data.get("media_names", [])
            suggest_fixes = input_data.get("suggest_fixes", True)

            # Load model
            if model_object:
                model = model_object
            elif model_path:
                model_utils = ModelUtils()
                model = model_utils.load_model(model_path)
            else:
                raise ValueError("Either model_path or model_object required")

            # Use all media if none specified
            if not media_names:
                media_names = self._media_library.list_available_media()

            # Analyze each media
            compatibility_results = {}
            overall_issues = {"missing_transporters": set(), "format_mismatches": []}

            for media_name in media_names:
                compatibility = self._media_library.analyze_media_compatibility(
                    model, media_name
                )

                # Test actual growth
                try:
                    test_model = self._enhancement.apply_media_with_cobrakbase(
                        model, media_name
                    )
                    solution = test_model.optimize()
                    growth_rate = (
                        solution.objective_value if solution.status == "optimal" else 0
                    )
                    growth_feasible = growth_rate > 1e-6
                except Exception:
                    growth_rate = 0
                    growth_feasible = False

                compatibility_results[media_name] = {
                    **compatibility,
                    "actual_growth": growth_rate,
                    "growth_feasible": growth_feasible,
                }

                # Track issues
                overall_issues["missing_transporters"].update(
                    compatibility["unmapped_compounds"]
                )
                if compatibility["model_format"] != compatibility["media_format"]:
                    overall_issues["format_mismatches"].append(media_name)

            # Generate suggestions
            suggestions = []
            if suggest_fixes:
                suggestions = self._generate_compatibility_suggestions(
                    model, compatibility_results, overall_issues
                )

            return ToolResult(
                success=True,
                message=f"Compatibility analysis completed for {len(media_names)} media types",
                data={
                    "model_analysis": {
                        "model_id": model.id,
                        "total_reactions": len(model.reactions),
                        "exchange_reactions": len(model.exchanges),
                        "model_format": self._detect_model_format(model),
                    },
                    "compatibility_results": compatibility_results,
                    "overall_issues": {
                        "missing_transporters": list(
                            overall_issues["missing_transporters"]
                        ),
                        "format_mismatches": overall_issues["format_mismatches"],
                    },
                    "suggestions": suggestions,
                },
                metadata={
                    "total_media_analyzed": len(media_names),
                    "compatible_media": sum(
                        1
                        for r in compatibility_results.values()
                        if r["growth_feasible"]
                    ),
                    "avg_compatibility": sum(
                        r["compatibility_score"] for r in compatibility_results.values()
                    )
                    / len(compatibility_results),
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Compatibility analysis failed: {str(e)}",
                error=str(e),
            )

    def _detect_model_format(self, model):
        """Detect model ID format"""
        return (
            "bigg" if any("_" in rxn.id for rxn in model.exchanges[:5]) else "modelseed"
        )

    def _generate_compatibility_suggestions(self, model, results, issues):
        """Generate AI-driven compatibility improvement suggestions"""
        suggestions = []

        # Find best compatibility
        best_media = max(
            results.keys(), key=lambda x: results[x]["compatibility_score"]
        )

        # General suggestions
        if len(issues["missing_transporters"]) > 10:
            suggestions.append(
                {
                    "type": "model_limitation",
                    "priority": "high",
                    "suggestion": "Model lacks many nutrient transporters",
                    "action": f"Focus on minimal media like {best_media} or consider model gap-filling",
                    "compounds": list(issues["missing_transporters"])[:10],
                }
            )

        if issues["format_mismatches"]:
            suggestions.append(
                {
                    "type": "format_mismatch",
                    "priority": "medium",
                    "suggestion": "Media format doesn't match model format",
                    "action": "Use compound mapping or convert media format",
                    "affected_media": issues["format_mismatches"],
                }
            )

        # Media-specific suggestions
        for media_name, result in results.items():
            if result["growth_feasible"] and result["compatibility_score"] < 0.8:
                suggestions.append(
                    {
                        "type": "partial_compatibility",
                        "priority": "low",
                        "suggestion": f"{media_name} works but has mapping issues",
                        "action": f"Consider removing unmapped compounds: {result['unmapped_compounds'][:5]}",
                        "media": media_name,
                    }
                )

        return suggestions


@ToolRegistry.register
class MediaComparatorTool(BaseTool):
    """AI tool to compare model performance across different media conditions"""

    tool_name = "compare_media_performance"
    tool_description = """Compare how a model performs across different media conditions.
    Provides detailed growth analysis and identifies optimal conditions."""

    def model_post_init(self, __context):
        """Initialize components after Pydantic model creation"""
        super().model_post_init(__context)
        self._media_library = get_media_library()
        self._enhancement = get_modelseedpy_enhancement()

    def _run_tool(self, input_data: Dict[str, Any]) -> ToolResult:
        """Compare model performance across media"""
        try:
            model_path = input_data.get("model_path")
            model_object = input_data.get("model_object")
            media_names = input_data.get("media_names", [])
            include_ai_modifications = input_data.get("include_ai_modifications", False)
            ai_modifications = input_data.get(
                "ai_modifications", ["make anaerobic", "add vitamins"]
            )

            # Load model
            if model_object:
                model = model_object
            elif model_path:
                model_utils = ModelUtils()
                model = model_utils.load_model(model_path)
            else:
                raise ValueError("Either model_path or model_object required")

            # Use all media if none specified
            if not media_names:
                media_names = self._media_library.list_available_media()

            # Test base media
            comparison_results = {}
            for media_name in media_names:
                result = self._test_single_media(model, media_name)
                comparison_results[media_name] = result

                # Test AI modifications if requested
                if include_ai_modifications and result["feasible"]:
                    result["ai_modifications"] = {}
                    for modification in ai_modifications:
                        try:
                            mod_result = self._test_modified_media(
                                model, media_name, modification
                            )
                            result["ai_modifications"][modification] = mod_result
                        except Exception as e:
                            result["ai_modifications"][modification] = {"error": str(e)}

            # Generate summary and rankings
            summary = self._generate_comparison_summary(comparison_results)

            return ToolResult(
                success=True,
                message=f"Media comparison completed for {len(media_names)} conditions",
                data={
                    "model_info": {
                        "model_id": model.id,
                        "reactions": len(model.reactions),
                        "metabolites": len(model.metabolites),
                    },
                    "comparison_results": comparison_results,
                    "summary": summary,
                },
                metadata={
                    "total_conditions_tested": len(media_names),
                    "feasible_conditions": summary["feasible_count"],
                    "best_growth_rate": summary["best_growth_rate"],
                    "best_media": summary["best_media"],
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Media comparison failed: {str(e)}",
                error=str(e),
            )

    def _test_single_media(self, model, media_name):
        """Test growth with a single media"""
        try:
            test_model = self._enhancement.apply_media_with_cobrakbase(
                model, media_name
            )
            solution = test_model.optimize()

            growth_rate = (
                solution.objective_value if solution.status == "optimal" else 0
            )

            # Get media info
            media_comp = self._media_library.get_media(media_name)

            return {
                "growth_rate": growth_rate,
                "status": solution.status,
                "feasible": growth_rate > 1e-6,
                "media_description": media_comp.description,
                "compound_count": len(media_comp.compounds),
                "conditions": media_comp.conditions,
            }
        except Exception as e:
            return {
                "error": str(e),
                "feasible": False,
                "growth_rate": 0,
            }

    def _test_modified_media(self, model, base_media, modification):
        """Test growth with AI-modified media"""
        try:
            # Create modified media
            modified_media = self._media_library.manipulate_media(
                base_media, modification
            )

            # Apply to model
            model_copy = model.copy()
            modified_model = self._media_library.apply_media_to_model(
                model_copy, modified_media.name
            )

            solution = modified_model.optimize()
            growth_rate = (
                solution.objective_value if solution.status == "optimal" else 0
            )

            return {
                "growth_rate": growth_rate,
                "status": solution.status,
                "feasible": growth_rate > 1e-6,
                "modification": modification,
            }
        except Exception as e:
            return {"error": str(e), "feasible": False}

    def _generate_comparison_summary(self, results):
        """Generate summary statistics and rankings"""
        feasible_results = {
            name: result
            for name, result in results.items()
            if result.get("feasible", False)
        }

        if not feasible_results:
            return {
                "feasible_count": 0,
                "best_media": None,
                "best_growth_rate": 0,
                "rankings": [],
                "insights": ["No feasible media conditions found"],
            }

        # Rank by growth rate
        rankings = sorted(
            feasible_results.items(), key=lambda x: x[1]["growth_rate"], reverse=True
        )

        best_media, best_result = rankings[0]

        # Generate insights
        insights = []
        if len(feasible_results) == 1:
            insights.append(f"Only {best_media} supports growth")
        else:
            insights.append(
                f"Best performance: {best_media} ({best_result['growth_rate']:.3f} h⁻¹)"
            )

            # Compare rich vs minimal media
            rich_media = [
                name
                for name, result in feasible_results.items()
                if "rich" in result.get("conditions", {})
            ]
            if rich_media:
                insights.append(f"Rich media available: {', '.join(rich_media)}")

        return {
            "feasible_count": len(feasible_results),
            "best_media": best_media,
            "best_growth_rate": best_result["growth_rate"],
            "rankings": [(name, result["growth_rate"]) for name, result in rankings],
            "insights": insights,
        }


# Export all tools
__all__ = [
    "MediaSelectorTool",
    "MediaManipulatorTool",
    "MediaCompatibilityTool",
    "MediaComparatorTool",
]
