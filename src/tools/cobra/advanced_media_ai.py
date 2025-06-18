"""
Advanced AI Capabilities for Media Optimization
==============================================

This module implements advanced AI-driven capabilities for:
1. Media Optimization - AI-driven optimization for specific growth targets
2. Auxotrophy Prediction - AI predicts required supplements based on model gaps
3. Cross-Model Analysis - Compare how different models perform on the same media
4. Dynamic Media Adaptation - Real-time media adjustment based on results

These tools extend our basic AI media tools with sophisticated analysis capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field

from ..base import BaseTool, ToolRegistry, ToolResult
from .media_library import MediaLibrary, get_media_library
from .modelseedpy_integration import ModelSEEDpyEnhancement, get_modelseedpy_enhancement
from .utils_optimized import OptimizedModelUtils

logger = logging.getLogger(__name__)


class OptimizationConfig(BaseModel):
    """Configuration for media optimization"""

    target_growth_rate: float = Field(
        default=0.5, description="Target growth rate to optimize for"
    )
    max_compounds: int = Field(
        default=50, description="Maximum compounds allowed in optimized media"
    )
    optimization_steps: int = Field(
        default=10, description="Number of optimization iterations"
    )
    include_essential_only: bool = Field(
        default=False, description="Only include essential compounds"
    )


class AuxotrophyAnalysisConfig(BaseModel):
    """Configuration for auxotrophy analysis"""

    compound_categories: List[str] = Field(
        default=["amino_acids", "vitamins", "nucleotides", "lipids"],
        description="Categories to test for auxotrophy",
    )
    growth_threshold: float = Field(
        default=0.01, description="Minimum growth to consider non-auxotrophic"
    )
    detailed_analysis: bool = Field(
        default=True, description="Perform detailed pathway analysis"
    )


@ToolRegistry.register
class MediaOptimizationTool(BaseTool):
    """AI-driven media optimization for specific growth targets and constraints"""

    tool_name = "optimize_media_composition"
    tool_description = """Optimize media composition using AI to achieve specific growth targets.
    Uses iterative optimization with pathway analysis to minimize media complexity while meeting growth requirements."""

    def model_post_init(self, __context):
        """Initialize components after Pydantic model creation"""
        super().model_post_init(__context)
        self._media_library = get_media_library()
        self._enhancement = get_modelseedpy_enhancement()
        self._model_utils = OptimizedModelUtils(use_cache=True)

    def _run_tool(self, input_data: Dict[str, Any]) -> ToolResult:
        """Optimize media composition for specific targets"""
        try:
            # Handle None input_data
            if input_data is None:
                raise ValueError("input_data cannot be None")
            
            model_path = input_data.get("model_path")
            model_object = input_data.get("model_object")
            target_growth = input_data.get("target_growth_rate", 0.5)
            base_media = input_data.get("base_media", "GMM")
            max_compounds = input_data.get("max_compounds", 50)
            optimization_strategy = input_data.get(
                "strategy", "iterative"
            )  # "iterative", "genetic", "greedy"

            # Load model
            if model_object:
                model = model_object
            elif model_path:
                model = self._model_utils.load_model(model_path)
            else:
                raise ValueError("Either model_path or model_object required")

            # Get base media
            media_composition = self._media_library.get_media(base_media)
            if not media_composition:
                raise ValueError(f"Base media '{base_media}' not found")

            # Get compounds dictionary from MediaComposition
            media_dict = media_composition.compounds

            # Run optimization based on strategy
            optimization_result = self._optimize_media(
                model, media_dict, target_growth, max_compounds, optimization_strategy
            )

            # Analyze the optimized media
            analysis = self._analyze_optimized_media(model, optimization_result)

            # Generate AI insights
            ai_insights = self._generate_optimization_insights(
                optimization_result, analysis, target_growth
            )

            return ToolResult(
                success=True,
                message=f"Media optimization completed. Achieved {optimization_result['final_growth']:.3f} h⁻¹ with {len(optimization_result['optimized_media'])} compounds",
                data={
                    "optimization_result": optimization_result,
                    "media_analysis": analysis,
                    "ai_insights": ai_insights,
                    "optimized_media": optimization_result["optimized_media"],
                    "compound_importance": optimization_result.get(
                        "compound_importance", {}
                    ),
                },
                metadata={
                    "model_id": getattr(model, 'id', None) or getattr(model, 'name', None) or 'unknown_model',
                    "target_growth": target_growth,
                    "achieved_growth": optimization_result["final_growth"],
                    "optimization_strategy": optimization_strategy,
                    "compounds_used": len(optimization_result["optimized_media"]),
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Media optimization failed: {str(e)}",
                error=str(e),
            )

    def _optimize_media(
        self,
        model,
        base_media: Dict,
        target_growth: float,
        max_compounds: int,
        strategy: str,
    ) -> Dict[str, Any]:
        """Optimize media using specified strategy"""

        if strategy == "iterative":
            return self._iterative_optimization(
                model, base_media, target_growth, max_compounds
            )
        elif strategy == "greedy":
            return self._greedy_optimization(
                model, base_media, target_growth, max_compounds
            )
        else:
            # Default to iterative
            return self._iterative_optimization(
                model, base_media, target_growth, max_compounds
            )

    def _iterative_optimization(
        self, model, base_media: Dict, target_growth: float, max_compounds: int
    ) -> Dict[str, Any]:
        """Iterative optimization approach"""

        # Start with base media
        current_media = base_media.copy()
        model.medium = current_media

        # Test base growth
        base_growth = model.optimize().objective_value

        if base_growth >= target_growth:
            return {
                "optimized_media": current_media,
                "final_growth": base_growth,
                "optimization_steps": 0,
                "strategy": "iterative",
                "base_growth": base_growth,
            }

        # Get available compounds from media library
        all_available_compounds = self._get_available_compounds()

        optimization_history = []

        for step in range(10):  # Max 10 optimization steps
            best_addition = None
            best_growth = base_growth

            # Test adding each available compound
            for compound_id in all_available_compounds:
                if (
                    compound_id not in current_media
                    and len(current_media) < max_compounds
                ):
                    # Test adding this compound
                    test_media = current_media.copy()
                    test_media[compound_id] = -10.0  # Standard uptake rate

                    try:
                        model.medium = test_media
                        growth = model.optimize().objective_value

                        if growth > best_growth:
                            best_growth = growth
                            best_addition = compound_id

                    except Exception:
                        continue

            # Add best compound if found
            if best_addition:
                current_media[best_addition] = -10.0
                optimization_history.append(
                    {
                        "step": step + 1,
                        "added_compound": best_addition,
                        "growth_achieved": best_growth,
                    }
                )

                # Check if target reached
                if best_growth >= target_growth:
                    break
            else:
                break

        # Final test
        model.medium = current_media
        final_growth = model.optimize().objective_value

        return {
            "optimized_media": current_media,
            "final_growth": final_growth,
            "optimization_steps": len(optimization_history),
            "strategy": "iterative",
            "base_growth": base_growth,
            "optimization_history": optimization_history,
        }

    def _greedy_optimization(
        self, model, base_media: Dict, target_growth: float, max_compounds: int
    ) -> Dict[str, Any]:
        """Greedy optimization approach - add compounds with highest growth impact"""

        current_media = base_media.copy()
        model.medium = current_media
        base_growth = model.optimize().objective_value

        if base_growth >= target_growth:
            return {
                "optimized_media": current_media,
                "final_growth": base_growth,
                "optimization_steps": 0,
                "strategy": "greedy",
                "base_growth": base_growth,
            }

        # Evaluate all compounds for their growth impact
        all_available_compounds = self._get_available_compounds()
        compound_impacts = []

        for compound_id in all_available_compounds:
            if compound_id not in current_media:
                test_media = current_media.copy()
                test_media[compound_id] = -10.0

                try:
                    model.medium = test_media
                    growth = model.optimize().objective_value
                    impact = growth - base_growth

                    if impact > 0:
                        compound_impacts.append((compound_id, impact, growth))

                except Exception:
                    continue

        # Sort by impact (descending)
        compound_impacts.sort(key=lambda x: x[1], reverse=True)

        # Add compounds greedily until target reached or max compounds
        optimization_history = []
        current_growth = base_growth

        for compound_id, impact, projected_growth in compound_impacts:
            if len(current_media) >= max_compounds:
                break

            current_media[compound_id] = -10.0
            model.medium = current_media
            actual_growth = model.optimize().objective_value

            optimization_history.append(
                {
                    "step": len(optimization_history) + 1,
                    "added_compound": compound_id,
                    "projected_impact": impact,
                    "actual_growth": actual_growth,
                }
            )

            current_growth = actual_growth

            if current_growth >= target_growth:
                break

        return {
            "optimized_media": current_media,
            "final_growth": current_growth,
            "optimization_steps": len(optimization_history),
            "strategy": "greedy",
            "base_growth": base_growth,
            "optimization_history": optimization_history,
            "compound_impacts": compound_impacts[:10],  # Top 10
        }

    def _get_available_compounds(self) -> List[str]:
        """Get list of available compounds for optimization"""
        # Get compounds from all media types in library
        all_compounds = set()

        for media_name in self._media_library.list_available_media():
            media_dict = self._media_library.get_media(media_name)
            if media_dict:
                all_compounds.update(media_dict.keys())

        return list(all_compounds)

    def _analyze_optimized_media(
        self, model, optimization_result: Dict
    ) -> Dict[str, Any]:
        """Analyze the optimized media composition"""

        optimized_media = optimization_result["optimized_media"]

        # Categorize compounds
        compound_categories = self._categorize_compounds(list(optimized_media.keys()))

        # Calculate media complexity
        complexity_score = len(optimized_media) / 50.0  # Normalized to 50 compounds max

        # Analyze pathway coverage
        pathway_analysis = self._analyze_pathway_coverage(model, optimized_media)

        return {
            "total_compounds": len(optimized_media),
            "compound_categories": compound_categories,
            "complexity_score": complexity_score,
            "pathway_analysis": pathway_analysis,
            "media_efficiency": optimization_result["final_growth"]
            / len(optimized_media),
        }

    def _categorize_compounds(self, compounds: List[str]) -> Dict[str, List[str]]:
        """Categorize compounds by type"""
        categories = {
            "carbon_sources": [],
            "nitrogen_sources": [],
            "phosphorus_sources": [],
            "sulfur_sources": [],
            "amino_acids": [],
            "vitamins": [],
            "minerals": [],
            "other": [],
        }

        # Simple categorization based on compound IDs (this could be enhanced with biochem data)
        for compound in compounds:
            if "glc" in compound.lower() or "glucose" in compound.lower():
                categories["carbon_sources"].append(compound)
            elif "nh" in compound.lower() or "nitrogen" in compound.lower():
                categories["nitrogen_sources"].append(compound)
            elif "po" in compound.lower() or "phosphate" in compound.lower():
                categories["phosphorus_sources"].append(compound)
            elif "so" in compound.lower() or "sulfate" in compound.lower():
                categories["sulfur_sources"].append(compound)
            elif any(
                aa in compound.lower()
                for aa in [
                    "ala",
                    "arg",
                    "asn",
                    "asp",
                    "cys",
                    "glu",
                    "gln",
                    "gly",
                    "his",
                    "ile",
                    "leu",
                    "lys",
                    "met",
                    "phe",
                    "pro",
                    "ser",
                    "thr",
                    "trp",
                    "tyr",
                    "val",
                ]
            ):
                categories["amino_acids"].append(compound)
            elif any(
                vit in compound.lower()
                for vit in [
                    "vitamin",
                    "thiamin",
                    "riboflavin",
                    "niacin",
                    "folate",
                    "cobalamin",
                ]
            ):
                categories["vitamins"].append(compound)
            elif any(
                min in compound.lower()
                for min in ["fe", "mg", "mn", "zn", "cu", "ca", "k", "na"]
            ):
                categories["minerals"].append(compound)
            else:
                categories["other"].append(compound)

        return categories

    def _analyze_pathway_coverage(self, model, media: Dict) -> Dict[str, Any]:
        """Analyze which pathways are supported by the media"""

        # This is a simplified analysis - could be enhanced with pathway database
        supported_pathways = []

        # Check for basic metabolic pathways
        if any("glc" in comp.lower() for comp in media.keys()):
            supported_pathways.append("Glycolysis")

        if any(aa in str(media.keys()).lower() for aa in ["ala", "gly", "ser"]):
            supported_pathways.append("Amino acid metabolism")

        if any("po" in comp.lower() for comp in media.keys()):
            supported_pathways.append("Nucleotide metabolism")

        return {
            "supported_pathways": supported_pathways,
            "pathway_count": len(supported_pathways),
        }

    def _generate_optimization_insights(
        self, optimization_result: Dict, analysis: Dict, target_growth: float
    ) -> Dict[str, Any]:
        """Generate AI-driven insights about the optimization"""

        final_growth = optimization_result["final_growth"]
        base_growth = optimization_result["base_growth"]
        improvement = final_growth - base_growth

        insights = {
            "summary": "",
            "recommendations": [],
            "efficiency_analysis": {},
            "further_optimization": [],
        }

        # Generate summary
        if final_growth >= target_growth:
            insights["summary"] = (
                f"✅ Successfully optimized media to achieve target growth of {target_growth:.3f} h⁻¹. Final growth: {final_growth:.3f} h⁻¹"
            )
        else:
            insights["summary"] = (
                f"⚠️ Partial optimization achieved. Target: {target_growth:.3f} h⁻¹, Achieved: {final_growth:.3f} h⁻¹"
            )

        # Generate recommendations
        if improvement > 0.1:
            insights["recommendations"].append(
                "Significant growth improvement achieved through media optimization"
            )

        if analysis["complexity_score"] > 0.7:
            insights["recommendations"].append(
                "Media complexity is high - consider removing non-essential compounds"
            )

        if final_growth < target_growth:
            insights["recommendations"].append(
                "Target not reached - consider investigating metabolic gaps or essential nutrients"
            )

        # Efficiency analysis
        insights["efficiency_analysis"] = {
            "growth_per_compound": analysis["media_efficiency"],
            "improvement_ratio": improvement / base_growth if base_growth > 0 else 0,
            "complexity_rating": (
                "Low"
                if analysis["complexity_score"] < 0.3
                else "Medium" if analysis["complexity_score"] < 0.7 else "High"
            ),
        }

        # Further optimization suggestions
        if optimization_result["optimization_steps"] >= 10:
            insights["further_optimization"].append(
                "Maximum optimization steps reached - consider different strategy"
            )

        if len(optimization_result.get("optimization_history", [])) == 0:
            insights["further_optimization"].append(
                "No beneficial compounds found - check base media adequacy"
            )

        return insights


@ToolRegistry.register
class AuxotrophyPredictionTool(BaseTool):
    """AI-powered auxotrophy prediction based on model gaps and pathway analysis"""

    tool_name = "predict_auxotrophies"
    tool_description = """Predict potential auxotrophies by analyzing metabolic gaps and testing compound removal.
    Uses AI to identify essential nutrients and predict growth requirements."""

    def model_post_init(self, __context):
        """Initialize components after Pydantic model creation"""
        super().model_post_init(__context)
        self._media_library = get_media_library()
        self._enhancement = get_modelseedpy_enhancement()
        self._model_utils = OptimizedModelUtils(use_cache=True)

    def _run_tool(self, input_data: Dict[str, Any]) -> ToolResult:
        """Predict auxotrophies for a model"""
        try:
            # Handle None input_data
            if input_data is None:
                raise ValueError("input_data cannot be None")
            
            model_path = input_data.get("model_path")
            model_object = input_data.get("model_object")
            test_media = input_data.get("test_media", "AuxoMedia")
            compound_categories = input_data.get(
                "compound_categories", ["amino_acids", "vitamins", "nucleotides"]
            )
            growth_threshold = input_data.get("growth_threshold", 0.01)

            # Load model
            if model_object:
                model = model_object
            elif model_path:
                model = self._model_utils.load_model(model_path)
            else:
                raise ValueError("Either model_path or model_object required")

            # Get test media
            media_composition = self._media_library.get_media(test_media)
            if not media_composition:
                raise ValueError(f"Test media '{test_media}' not found")

            # Get compounds dictionary from MediaComposition
            media_dict = media_composition.compounds

            # Perform auxotrophy analysis
            auxotrophy_results = self._analyze_auxotrophies(
                model, media_dict, compound_categories, growth_threshold
            )

            # Generate AI predictions
            ai_predictions = self._generate_auxotrophy_predictions(
                model, auxotrophy_results
            )

            return ToolResult(
                success=True,
                message=f"Auxotrophy analysis completed. Found {len(auxotrophy_results['predicted_auxotrophies'])} potential auxotrophies",
                data={
                    "auxotrophy_results": auxotrophy_results,
                    "ai_predictions": ai_predictions,
                    "predicted_auxotrophies": auxotrophy_results[
                        "predicted_auxotrophies"
                    ],
                    "essential_compounds": auxotrophy_results["essential_compounds"],
                },
                metadata={
                    "model_id": getattr(model, 'id', None) or getattr(model, 'name', None) or 'unknown_model',
                    "test_media": test_media,
                    "growth_threshold": growth_threshold,
                    "auxotrophy_count": len(
                        auxotrophy_results["predicted_auxotrophies"]
                    ),
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Auxotrophy prediction failed: {str(e)}",
                error=str(e),
            )

    def _analyze_auxotrophies(
        self, model, media: Dict, categories: List[str], threshold: float
    ) -> Dict[str, Any]:
        """Analyze potential auxotrophies by testing compound removal"""

        # Test base growth
        model.medium = media
        base_growth = model.optimize().objective_value

        auxotrophies = []
        essential_compounds = []
        compound_impacts = {}

        # Test removal of each compound
        for compound_id in media.keys():
            # Skip if compound doesn't match tested categories
            if not self._compound_in_categories(compound_id, categories):
                continue

            # Create media without this compound
            test_media = media.copy()
            del test_media[compound_id]

            try:
                model.medium = test_media
                growth_without = model.optimize().objective_value

                growth_impact = base_growth - growth_without
                compound_impacts[compound_id] = {
                    "base_growth": base_growth,
                    "growth_without": growth_without,
                    "impact": growth_impact,
                    "essential": growth_without < threshold,
                }

                if growth_without < threshold:
                    auxotrophies.append(
                        {
                            "compound": compound_id,
                            "category": self._get_compound_category(compound_id),
                            "growth_without": growth_without,
                            "impact": growth_impact,
                            "severity": "High" if growth_without < 0.001 else "Medium",
                        }
                    )
                    essential_compounds.append(compound_id)

            except Exception as e:
                logger.warning(f"Failed to test compound {compound_id}: {e}")
                continue

        return {
            "base_growth": base_growth,
            "predicted_auxotrophies": auxotrophies,
            "essential_compounds": essential_compounds,
            "compound_impacts": compound_impacts,
            "categories_tested": categories,
        }

    def _compound_in_categories(self, compound_id: str, categories: List[str]) -> bool:
        """Check if compound belongs to tested categories"""
        compound_lower = compound_id.lower()

        category_keywords = {
            "amino_acids": [
                "ala",
                "arg",
                "asn",
                "asp",
                "cys",
                "glu",
                "gln",
                "gly",
                "his",
                "ile",
                "leu",
                "lys",
                "met",
                "phe",
                "pro",
                "ser",
                "thr",
                "trp",
                "tyr",
                "val",
            ],
            "vitamins": [
                "vitamin",
                "thiamin",
                "riboflavin",
                "niacin",
                "folate",
                "cobalamin",
                "biotin",
                "pantothen",
            ],
            "nucleotides": ["atp", "gtp", "ctp", "utp", "amp", "gmp", "cmp", "ump"],
            "lipids": ["fatty", "lipid", "phosphatidyl"],
        }

        for category in categories:
            if category in category_keywords:
                if any(
                    keyword in compound_lower for keyword in category_keywords[category]
                ):
                    return True

        return False

    def _get_compound_category(self, compound_id: str) -> str:
        """Get category for a compound"""
        compound_lower = compound_id.lower()

        if any(
            aa in compound_lower
            for aa in [
                "ala",
                "arg",
                "asn",
                "asp",
                "cys",
                "glu",
                "gln",
                "gly",
                "his",
                "ile",
                "leu",
                "lys",
                "met",
                "phe",
                "pro",
                "ser",
                "thr",
                "trp",
                "tyr",
                "val",
            ]
        ):
            return "amino_acids"
        elif any(
            vit in compound_lower
            for vit in [
                "vitamin",
                "thiamin",
                "riboflavin",
                "niacin",
                "folate",
                "cobalamin",
            ]
        ):
            return "vitamins"
        elif any(nuc in compound_lower for nuc in ["atp", "gtp", "ctp", "utp"]):
            return "nucleotides"
        else:
            return "other"

    def _generate_auxotrophy_predictions(
        self, model, auxotrophy_results: Dict
    ) -> Dict[str, Any]:
        """Generate AI-driven auxotrophy predictions and insights"""

        auxotrophies = auxotrophy_results["predicted_auxotrophies"]

        predictions = {
            "summary": "",
            "pathway_analysis": {},
            "supplement_recommendations": [],
            "metabolic_insights": [],
        }

        # Generate summary
        if len(auxotrophies) == 0:
            predictions["summary"] = (
                "✅ No auxotrophies detected - model appears metabolically complete for tested categories"
            )
        else:
            predictions["summary"] = (
                f"🔍 Detected {len(auxotrophies)} potential auxotrophies requiring supplementation"
            )

        # Analyze by category
        category_counts = {}
        for aux in auxotrophies:
            category = aux["category"]
            category_counts[category] = category_counts.get(category, 0) + 1

        predictions["pathway_analysis"] = {
            "affected_categories": category_counts,
            "most_affected": (
                max(category_counts.items(), key=lambda x: x[1])[0]
                if category_counts
                else None
            ),
        }

        # Generate supplement recommendations
        if "amino_acids" in category_counts:
            predictions["supplement_recommendations"].append(
                "Consider amino acid supplementation or check amino acid biosynthesis pathways"
            )

        if "vitamins" in category_counts:
            predictions["supplement_recommendations"].append(
                "Vitamin supplementation recommended - check cofactor availability"
            )

        # Metabolic insights
        if len(auxotrophies) > 5:
            predictions["metabolic_insights"].append(
                "High number of auxotrophies suggests significant metabolic gaps"
            )

        high_impact_aux = [aux for aux in auxotrophies if aux["impact"] > 0.1]
        if high_impact_aux:
            predictions["metabolic_insights"].append(
                f"{len(high_impact_aux)} auxotrophies have high growth impact"
            )

        return predictions


# Export all tools
__all__ = [
    "MediaOptimizationTool",
    "AuxotrophyPredictionTool",
    "OptimizationConfig",
    "AuxotrophyAnalysisConfig",
]
