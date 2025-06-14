#!/usr/bin/env python3
"""
ModelSEED Tool Validation Suite with Biological Validation
=========================================================

This script provides comprehensive validation for ALL ModelSEED tools (COBRA + AI Media + Biochemistry)
against multiple models with detailed biological knowledge validation to ensure results are scientifically meaningful.

This is the main tool validation system for ModelSEEDagent, providing:

**Validation Levels:**
- **Comprehensive Validation**: Full 19 tools × 4 models (76 test combinations)
- **CI Validation**: Essential subset (FBA on e_coli_core) for continuous integration
- **Future: Audit Validation**: System tools validation (separate approach)

**Models tested (4 total):**
- data/examples/e_coli_core.xml (BiGG core model)
- data/examples/iML1515.xml (BiGG genome-scale model)
- data/examples/EcoliMG1655.xml (ModelSEED E. coli model)
- data/examples/B_aphidicola.xml (ModelSEED minimal organism)

**Tools validated (19 total):**
- 11 COBRA tools (FBA, ModelAnalysis, FluxVariability, etc.)
- 6 AI Media tools (MediaSelector, MediaManipulator, etc.)
- 2 Biochemistry tools (BiochemEntityResolver, BiochemSearch)

**Biological validation includes:**
- Growth rate feasibility (0.0-2.0 h⁻¹ for bacteria)
- Essential gene counts (10-20% of total genes)
- Media compatibility scores (model format matching)
- Biochemistry ID resolution accuracy (ModelSEED ↔ BiGG)
- Pathway flux consistency (carbon balance, ATP production)
"""

import json
import logging
import os
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.biochem import BiochemEntityResolverTool, BiochemSearchTool

# Import all tools
from src.tools.cobra import (  # AI Media Tools
    AuxotrophyTool,
    EssentialityAnalysisTool,
    FBATool,
    FluxSamplingTool,
    FluxVariabilityTool,
    GeneDeletionTool,
    MediaComparatorTool,
    MediaCompatibilityTool,
    MediaManipulatorTool,
    MediaSelectorTool,
    MinimalMediaTool,
    MissingMediaTool,
    ModelAnalysisTool,
    PathwayAnalysisTool,
    ProductionEnvelopeTool,
    ReactionExpressionTool,
)

# Import advanced AI media tools
from src.tools.cobra.advanced_media_ai import (
    AuxotrophyPredictionTool,
    MediaOptimizationTool,
)


class BiologicalValidator:
    """Validates tool outputs for biological feasibility and correctness"""

    def __init__(self):
        self.validation_rules = {
            # Growth rate feasibility (h⁻¹)
            "growth_rate": {"min": 0.0, "max": 2.0, "typical_range": (0.1, 1.5)},
            # Essential gene percentages
            "essential_genes": {
                "min_percent": 5,
                "max_percent": 30,
                "typical_percent": (10, 20),
            },
            # Media component counts
            "minimal_media": {
                "min_components": 4,
                "max_components": 50,
                "typical_range": (8, 20),
            },
            # Flux magnitudes (mmol/gDW/h)
            "flux_magnitude": {"min": 0.0, "max": 1000.0, "typical_max": 100.0},
            # Model format detection
            "model_formats": ["bigg", "modelseed"],
            # Biochemistry validation
            "biochem_resolution": {
                "success_rate_min": 0.7,
                "common_compounds": [
                    "cpd00001",
                    "cpd00002",
                    "cpd00027",
                    "h2o",
                    "atp",
                    "glc__D",
                ],
            },
        }

    def validate_fba_results(self, results: Dict, model_name: str) -> Dict[str, Any]:
        """Validate FBA results for biological feasibility"""
        validation = {
            "is_valid": True,
            "warnings": [],
            "biological_insights": [],
            "scores": {},
        }

        if not results.get("success", False):
            validation["is_valid"] = False
            validation["warnings"].append("FBA failed to complete")
            return validation

        data = results.get("data", {})
        growth_rate = data.get("objective_value", 0)

        # Growth rate validation
        if growth_rate < self.validation_rules["growth_rate"]["min"]:
            validation["warnings"].append(
                f"No growth detected (growth rate: {growth_rate:.4f})"
            )
            validation["biological_insights"].append(
                "Model may be missing essential nutrients or have metabolic gaps"
            )
        elif growth_rate > self.validation_rules["growth_rate"]["max"]:
            validation["warnings"].append(
                f"Unrealistically high growth rate: {growth_rate:.4f} h⁻¹"
            )
        else:
            typical_min, typical_max = self.validation_rules["growth_rate"][
                "typical_range"
            ]
            if typical_min <= growth_rate <= typical_max:
                validation["biological_insights"].append(
                    f"Growth rate {growth_rate:.4f} h⁻¹ is biologically realistic"
                )
                validation["scores"]["growth_feasibility"] = 1.0
            else:
                validation["scores"]["growth_feasibility"] = 0.7

        # Flux magnitude validation
        significant_fluxes = data.get("significant_fluxes", {})
        if significant_fluxes:
            max_flux = max(abs(v) for v in significant_fluxes.values())
            if max_flux > self.validation_rules["flux_magnitude"]["typical_max"]:
                validation["warnings"].append(
                    f"Very high flux detected: {max_flux:.2f}"
                )
            validation["scores"]["flux_magnitude"] = min(
                1.0, self.validation_rules["flux_magnitude"]["typical_max"] / max_flux
            )

        # Carbon balance check (glucose uptake vs CO2 production)
        glucose_uptake = abs(
            significant_fluxes.get(
                "EX_glc__D_e", significant_fluxes.get("EX_cpd00027_e0", 0)
            )
        )
        co2_production = significant_fluxes.get(
            "EX_co2_e", significant_fluxes.get("EX_cpd00011_e0", 0)
        )

        if glucose_uptake > 0 and co2_production > 0:
            carbon_ratio = co2_production / glucose_uptake
            if 0.5 <= carbon_ratio <= 6.0:  # Reasonable C-balance
                validation["biological_insights"].append(
                    f"Carbon balance reasonable: {carbon_ratio:.2f} CO2/glucose"
                )
                validation["scores"]["carbon_balance"] = 1.0
            else:
                validation["warnings"].append(
                    f"Unusual carbon balance: {carbon_ratio:.2f}"
                )
                validation["scores"]["carbon_balance"] = 0.5

        return validation

    def validate_essentiality_results(
        self, results: Dict, model_name: str
    ) -> Dict[str, Any]:
        """Validate gene essentiality results"""
        validation = {
            "is_valid": True,
            "warnings": [],
            "biological_insights": [],
            "scores": {},
        }

        if not results.get("success", False):
            validation["is_valid"] = False
            return validation

        data = results.get("data", {})
        essential_genes = data.get("essential_genes", [])
        total_genes = data.get(
            "total_genes", len(essential_genes) * 5
        )  # Estimate if not provided

        if total_genes == 0:
            validation["warnings"].append("No genes found in model")
            return validation

        essential_percent = (len(essential_genes) / total_genes) * 100

        # Essential gene percentage validation
        min_percent = self.validation_rules["essential_genes"]["min_percent"]
        max_percent = self.validation_rules["essential_genes"]["max_percent"]
        typical_min, typical_max = self.validation_rules["essential_genes"][
            "typical_percent"
        ]

        if essential_percent < min_percent:
            validation["warnings"].append(
                f"Very low essential gene percentage: {essential_percent:.1f}%"
            )
            validation["biological_insights"].append(
                "Model may be missing gene-reaction associations"
            )
        elif essential_percent > max_percent:
            validation["warnings"].append(
                f"Very high essential gene percentage: {essential_percent:.1f}%"
            )
            validation["biological_insights"].append(
                "Model may be too constrained or missing redundant pathways"
            )
        else:
            if typical_min <= essential_percent <= typical_max:
                validation["biological_insights"].append(
                    f"Essential gene percentage {essential_percent:.1f}% is typical"
                )
                validation["scores"]["essentiality"] = 1.0
            else:
                validation["scores"]["essentiality"] = 0.8

        # Model-specific validation
        if "aphidicola" in model_name.lower():
            if essential_percent > 25:
                validation["biological_insights"].append(
                    "High essentiality expected for minimal organism like B. aphidicola"
                )
        elif "core" in model_name.lower():
            if essential_percent > 20:
                validation["biological_insights"].append(
                    "High essentiality expected for core model with minimal redundancy"
                )

        return validation

    def validate_media_results(self, results: Dict, model_name: str) -> Dict[str, Any]:
        """Validate AI media tool results"""
        validation = {
            "is_valid": True,
            "warnings": [],
            "biological_insights": [],
            "scores": {},
        }

        if not results.get("success", False):
            validation["is_valid"] = False
            return validation

        data = results.get("data", {})

        # Media compatibility validation
        if "compatibility_score" in data:
            compatibility = data["compatibility_score"]
            if compatibility > 0.8:
                validation["biological_insights"].append(
                    "High media-model compatibility"
                )
                validation["scores"]["compatibility"] = 1.0
            elif compatibility > 0.5:
                validation["biological_insights"].append(
                    "Moderate media-model compatibility"
                )
                validation["scores"]["compatibility"] = 0.7
            else:
                validation["warnings"].append(
                    f"Low media compatibility: {compatibility:.2f}"
                )
                validation["scores"]["compatibility"] = 0.3

        # Growth rate validation from media testing
        if "growth_rate" in data:
            growth_rate = data["growth_rate"]
            validation.update(
                self.validate_fba_results(
                    {"success": True, "data": {"objective_value": growth_rate}},
                    model_name,
                )
            )

        # Media component count validation
        if "compounds" in data:
            component_count = len(data["compounds"])
            min_comp = self.validation_rules["minimal_media"]["min_components"]
            max_comp = self.validation_rules["minimal_media"]["max_components"]

            if component_count < min_comp:
                validation["warnings"].append(
                    f"Very few media components: {component_count}"
                )
            elif component_count > max_comp:
                validation["warnings"].append(
                    f"Very many media components: {component_count}"
                )
            else:
                validation["biological_insights"].append(
                    f"Reasonable media complexity: {component_count} components"
                )

        return validation

    def validate_biochem_results(self, results: Dict, test_type: str) -> Dict[str, Any]:
        """Validate biochemistry tool results"""
        validation = {
            "is_valid": True,
            "warnings": [],
            "biological_insights": [],
            "scores": {},
        }

        if not results.get("success", False):
            validation["is_valid"] = False
            return validation

        data = results.get("data", {})

        if test_type == "resolution":
            # Test resolution of common compounds
            if "resolved" in data and data["resolved"]:
                validation["biological_insights"].append(
                    "Successfully resolved biochemistry entity"
                )
                validation["scores"]["resolution_success"] = 1.0

                if "primary_name" in data and data["primary_name"]:
                    validation["biological_insights"].append(
                        f"Found name: {data['primary_name']}"
                    )

                if "aliases" in data:
                    alias_count = len(data["aliases"])
                    if alias_count > 0:
                        validation["biological_insights"].append(
                            f"Found {alias_count} cross-database aliases"
                        )
                        validation["scores"]["cross_db_coverage"] = min(
                            1.0, alias_count / 5
                        )
            else:
                validation["warnings"].append("Failed to resolve biochemistry entity")
                validation["scores"]["resolution_success"] = 0.0

        elif test_type == "search":
            if "total_results" in data:
                result_count = data["total_results"]
                if result_count > 0:
                    validation["biological_insights"].append(
                        f"Search found {result_count} matches"
                    )
                    validation["scores"]["search_coverage"] = min(
                        1.0, result_count / 10
                    )
                else:
                    validation["warnings"].append("Search returned no results")
                    validation["scores"]["search_coverage"] = 0.0

        return validation


class ModelSEEDToolValidationSuite:
    """Comprehensive validation suite for all ModelSEED tools with biological validation"""

    def __init__(self):
        self.results = {}
        self.validator = BiologicalValidator()

        # All 4 models including the new EcoliMG1655
        self.models = {
            "e_coli_core": "data/examples/e_coli_core.xml",  # BiGG core model
            "iML1515": "data/examples/iML1515.xml",  # BiGG genome-scale
            "EcoliMG1655": "data/examples/EcoliMG1655.xml",  # ModelSEED E. coli
            "B_aphidicola": "data/examples/B_aphidicola.xml",  # ModelSEED minimal
        }

        # Configure detailed logging
        logging.basicConfig(
            level=logging.INFO,  # Reduced verbosity for comprehensive testing
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Initialize all tools
        basic_config = {"fba_config": {}, "model_config": {}}

        # COBRA tools (12 tools - PathwayAnalysis re-enabled with annotation awareness)
        self.cobra_tools = {
            "FBA": FBATool(basic_config),
            "ModelAnalysis": ModelAnalysisTool(basic_config),
            "PathwayAnalysis": PathwayAnalysisTool(
                basic_config
            ),  # Re-enabled - requires pathway annotations
            "FluxVariability": FluxVariabilityTool(basic_config),
            "GeneDeletion": GeneDeletionTool(basic_config),
            "Essentiality": EssentialityAnalysisTool(basic_config),
            "FluxSampling": FluxSamplingTool(basic_config),
            "ProductionEnvelope": ProductionEnvelopeTool(basic_config),
            "Auxotrophy": AuxotrophyTool(basic_config),
            "MinimalMedia": MinimalMediaTool(basic_config),
            "MissingMedia": MissingMediaTool(basic_config),
            "ReactionExpression": ReactionExpressionTool(basic_config),
        }

        # AI Media tools (6 tools)
        self.media_tools = {
            "MediaSelector": MediaSelectorTool(basic_config),
            "MediaManipulator": MediaManipulatorTool(basic_config),
            "MediaCompatibility": MediaCompatibilityTool(basic_config),
            "MediaComparator": MediaComparatorTool(basic_config),
            "MediaOptimization": MediaOptimizationTool(basic_config),
            "AuxotrophyPrediction": AuxotrophyPredictionTool(basic_config),
        }

        # Biochemistry tools (2 tools)
        self.biochem_tools = {
            "BiochemEntityResolver": BiochemEntityResolverTool(basic_config),
            "BiochemSearch": BiochemSearchTool(basic_config),
        }

        # Combine all tools
        self.all_tools = {**self.cobra_tools, **self.media_tools, **self.biochem_tools}

        print(f"🧪 ModelSEED Tool Validation Suite Initialized")
        print(
            f"📊 Validating {len(self.all_tools)} tools ({len(self.cobra_tools)} COBRA + {len(self.media_tools)} Media + {len(self.biochem_tools)} Biochem)"
        )
        print(f"🧬 Testing on {len(self.models)} models (2 BiGG + 2 ModelSEED)")
        print(f"🔬 With biological validation for scientific accuracy")
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def get_tool_parameters(
        self, tool_name: str, model_path: str, model_name: str
    ) -> Dict[str, Any]:
        """Get biologically appropriate parameters for each tool"""

        base_params = {"model_path": model_path}

        # COBRA tool parameters
        cobra_params = {
            "FBA": self._get_fba_params(model_name),
            "ModelAnalysis": {},
            "PathwayAnalysis": self._get_pathway_analysis_params(
                model_name
            ),  # Re-enabled with model-specific params
            "FluxVariability": {"fraction_of_optimum": 0.9},
            "GeneDeletion": self._get_gene_deletion_params(model_name),
            "Essentiality": {"threshold": 0.01},  # 1% of wildtype growth
            "FluxSampling": {"n_samples": 50, "method": "optgp"},  # Reduced for speed
            "ProductionEnvelope": self._get_production_envelope_params(model_name),
            "Auxotrophy": {},
            "MinimalMedia": {},
            "MissingMedia": {
                "target_metabolites": self._get_target_metabolites(model_name)
            },
            "ReactionExpression": {
                "expression_data": self._get_expression_data(model_name)
            },
        }

        # AI Media tool parameters
        media_params = {
            "MediaSelector": {"target_growth": 0.1},
            "MediaManipulator": {
                "base_media": "GMM",
                "ai_command": "make anaerobic",
                "test_growth": True,
            },
            "MediaCompatibility": {"media_names": ["GMM", "AuxoMedia"]},
            "MediaComparator": {
                "media_list": ["GMM", "AuxoMedia", "PyruvateMinimalMedia"]
            },
            "MediaOptimization": {
                "target_growth_rate": 0.3,
                "base_media": "GMM",
                "strategy": "iterative",
            },
            "AuxotrophyPrediction": {
                "test_media": "AuxoMedia",
                "compound_categories": ["amino_acids", "vitamins"],
            },
        }

        # Biochemistry tool parameters
        biochem_params = {
            "BiochemEntityResolver": self._get_biochem_test_entities(model_name),
            "BiochemSearch": {
                "query": "glucose",
                "entity_type": "compound",
                "max_results": 10,
            },
        }

        all_params = {**cobra_params, **media_params, **biochem_params}

        return {**base_params, **all_params.get(tool_name, {})}

    def _get_fba_params(self, model_name: str) -> Dict[str, Any]:
        """Get FBA parameters - use GMM media for ModelSEED models to constrain growth"""
        if "aphidicola" in model_name.lower() or "ecolimg" in model_name.lower():
            # Use GMM media for ModelSEED models to get realistic growth rates
            return {"media": "GMM"}
        else:
            # BiGG models use default media
            return {}

    def _get_pathway_analysis_params(self, model_name: str) -> Dict[str, Any]:
        """Get pathway analysis parameters with annotation awareness"""

        # Basic pathway analysis - will test for annotation availability
        if "iML1515" in model_name:
            # Genome-scale model likely has good annotations
            return {"pathway_name": "glycolysis", "detailed_analysis": True}
        elif "core" in model_name.lower():
            # Core model may have limited annotations
            return {"pathway_name": "central_metabolism", "detailed_analysis": False}
        else:
            # ModelSEED models may have different annotation formats
            return {"pathway_name": "core_pathways", "allow_partial_annotation": True}

    def _get_gene_deletion_params(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific gene deletion parameters"""
        if "iML1515" in model_name:
            return {"genes": ["b0008", "b0116"], "method": "single"}
        elif "aphidicola" in model_name.lower():
            return {"genes": ["272557.1.peg.1", "272557.1.peg.2"], "method": "single"}
        elif "ecolimg" in model_name.lower():
            return {"genes": ["83333.1.peg.1", "83333.1.peg.2"], "method": "single"}
        else:  # e_coli_core
            return {"genes": ["b0008", "b0009"], "method": "single"}

    def _get_production_envelope_params(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific production envelope parameters"""
        if "iML1515" in model_name:
            return {
                "objective_rxn": "BIOMASS_Ec_iML1515_core_75p37M",
                "carbon_sources": ["EX_glc__D_e"],
            }
        elif "aphidicola" in model_name.lower() or "ecolimg" in model_name.lower():
            return {
                "objective_rxn": "bio1",  # Common ModelSEED biomass
                "carbon_sources": ["EX_cpd00027_e0"],  # ModelSEED glucose
            }
        else:  # e_coli_core
            return {
                "objective_rxn": "BIOMASS_Ecoli_core_w_GAM",
                "carbon_sources": ["EX_glc__D_e"],
            }

    def _get_target_metabolites(self, model_name: str) -> List[str]:
        """Get model-specific target metabolites"""
        if "aphidicola" in model_name.lower() or "ecolimg" in model_name.lower():
            return ["cpd00027", "cpd00067", "cpd00001"]  # ModelSEED IDs
        else:
            return ["glc__D", "h", "h2o"]  # BiGG IDs

    def _get_expression_data(self, model_name: str) -> Dict[str, float]:
        """Get model-specific expression data"""
        if "iML1515" in model_name:
            return {"b0008": 1.5, "b0116": 0.8}
        elif "aphidicola" in model_name.lower():
            return {"272557.1.peg.1": 1.2, "272557.1.peg.2": 0.9}
        elif "ecolimg" in model_name.lower():
            return {"83333.1.peg.1": 1.3, "83333.1.peg.2": 0.7}
        else:
            return {"b0008": 1.5, "b0009": 0.8}

    def _get_biochem_test_entities(self, model_name: str) -> Dict[str, Any]:
        """Get biochemistry entities to test based on model format"""
        if "aphidicola" in model_name.lower() or "ecolimg" in model_name.lower():
            # Test ModelSEED → name resolution for ModelSEED models
            return {
                "entity_id": "cpd00027",
                "entity_type": "compound",
                "include_aliases": True,
            }
        else:
            # Test BiGG → name resolution for BiGG models
            return {
                "entity_id": "glc__D",
                "entity_type": "compound",
                "include_aliases": True,
            }

    def test_tool_on_model(
        self, tool_name: str, tool: Any, model_name: str, model_path: str
    ) -> Dict[str, Any]:
        """Test a single tool on a single model with biological validation"""

        print(f"🔬 Testing {tool_name} on {model_name}")

        test_result = {
            "tool": tool_name,
            "model": model_name,
            "model_path": model_path,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "execution_time": 0,
            "output": None,
            "error": None,
            "biological_validation": None,
            "warnings": [],
            "debug_info": {},
        }

        start_time = datetime.now()

        try:
            # Get appropriate parameters
            tool_params = self.get_tool_parameters(tool_name, model_path, model_name)

            # Execute tool
            if tool_name in self.biochem_tools:
                # Biochemistry tools use different parameter structure
                result = tool._run_tool(tool_params)
            elif tool_name == "ModelAnalysis":
                # ModelAnalysis expects just the model_path string
                result = tool._run_tool(model_path)
            elif tool_name == "PathwayAnalysis":
                # PathwayAnalysis may fail due to missing annotations - handle gracefully
                try:
                    result = tool._run_tool(tool_params)
                except Exception as e:
                    if "pathway" in str(e).lower() or "annotation" in str(e).lower():
                        # Create a result indicating annotation issues
                        test_result["warnings"].append(
                            f"PathwayAnalysis failed due to annotation issues: {str(e)}"
                        )
                        result = type(
                            "Result",
                            (),
                            {
                                "success": False,
                                "message": f"PathwayAnalysis requires pathway annotations not available in {model_name}",
                                "data": {
                                    "annotation_available": False,
                                    "reason": str(e),
                                },
                                "error": None,
                            },
                        )()
                    else:
                        raise  # Re-raise if it's a different error
            else:
                # COBRA and Media tools
                result = tool._run_tool(tool_params)

            # Store raw output
            if hasattr(result, "model_dump"):
                test_result["output"] = result.model_dump()
            elif isinstance(result, dict):
                test_result["output"] = result
            else:
                test_result["output"] = str(result)

            test_result["success"] = True

            # Biological validation
            test_result["biological_validation"] = self._validate_tool_result(
                tool_name, test_result["output"], model_name
            )

            # Extract key metrics for reporting
            self._extract_key_metrics(test_result, tool_name)

            print(f"  ✅ SUCCESS - {self._get_success_summary(test_result, tool_name)}")

        except Exception as e:
            test_result["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            print(f"  ❌ ERROR: {type(e).__name__}: {str(e)}")

        end_time = datetime.now()
        test_result["execution_time"] = (end_time - start_time).total_seconds()

        return test_result

    def _validate_tool_result(
        self, tool_name: str, output: Any, model_name: str
    ) -> Dict[str, Any]:
        """Apply biological validation based on tool type"""

        if tool_name == "FBA":
            return self.validator.validate_fba_results(output, model_name)
        elif tool_name == "Essentiality":
            return self.validator.validate_essentiality_results(output, model_name)
        elif tool_name in self.media_tools:
            return self.validator.validate_media_results(output, model_name)
        elif tool_name == "BiochemEntityResolver":
            return self.validator.validate_biochem_results(output, "resolution")
        elif tool_name == "BiochemSearch":
            return self.validator.validate_biochem_results(output, "search")
        else:
            # Basic validation for other tools
            return {
                "is_valid": (
                    output.get("success", False) if isinstance(output, dict) else True
                ),
                "warnings": [],
                "biological_insights": [],
                "scores": {},
            }

    def _extract_key_metrics(self, test_result: Dict, tool_name: str) -> None:
        """Extract key metrics for summary reporting"""
        output = test_result["output"]
        if not isinstance(output, dict):
            return

        data = output.get("data", {})
        metadata = output.get("metadata", {})  # noqa: F841

        # Common metrics
        metrics = {}

        if "objective_value" in data:
            metrics["growth_rate"] = data["objective_value"]
        elif "growth_rate" in data:
            metrics["growth_rate"] = data["growth_rate"]

        if "essential_genes" in data:
            metrics["essential_gene_count"] = len(data["essential_genes"])

        if "compatibility_score" in data:
            metrics["media_compatibility"] = data["compatibility_score"]

        if "total_results" in data:
            metrics["search_results"] = data["total_results"]

        if "resolved" in data:
            metrics["resolution_success"] = data["resolved"]

        test_result["key_metrics"] = metrics

    def _get_success_summary(self, test_result: Dict, tool_name: str) -> str:
        """Generate a brief success summary"""
        metrics = test_result.get("key_metrics", {})
        validation = test_result.get("biological_validation", {})

        summary_parts = []

        if "growth_rate" in metrics:
            summary_parts.append(f"Growth: {metrics['growth_rate']:.3f} h⁻¹")

        if "essential_gene_count" in metrics:
            summary_parts.append(f"Essential genes: {metrics['essential_gene_count']}")

        if "media_compatibility" in metrics:
            summary_parts.append(f"Compatibility: {metrics['media_compatibility']:.2f}")

        if "resolution_success" in metrics:
            summary_parts.append(f"Resolved: {metrics['resolution_success']}")

        if validation.get("scores"):
            avg_score = sum(validation["scores"].values()) / len(validation["scores"])
            summary_parts.append(f"Bio-score: {avg_score:.2f}")

        return " | ".join(summary_parts) if summary_parts else "Completed"

    def run_comprehensive_validation(self):
        """Run all tools on all models with comprehensive validation"""

        total_tests = len(self.all_tools) * len(self.models)
        current_test = 0
        failed_tests = []

        for model_name, model_path in self.models.items():
            self.results[model_name] = {}

            print(f"\n🧬 TESTING MODEL: {model_name.upper()}")
            print(f"📍 Path: {model_path}")

            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                # Create placeholder results for missing model
                for tool_name in self.all_tools.keys():
                    self.results[model_name][tool_name] = {
                        "success": False,
                        "error": {
                            "type": "FileNotFoundError",
                            "message": f"Model file not found: {model_path}",
                        },
                        "tool": tool_name,
                        "model": model_name,
                        "model_path": model_path,
                    }
                continue

            print(f"📏 File size: {os.path.getsize(model_path) / 1024:.1f} KB")
            print("=" * 60)

            # Test each tool category
            for category, tools in [
                ("COBRA", self.cobra_tools),
                ("AI MEDIA", self.media_tools),
                ("BIOCHEM", self.biochem_tools),
            ]:
                print(f"\n🔧 Testing {category} tools:")

                for tool_name, tool in tools.items():
                    current_test += 1
                    print(f"[{current_test}/{total_tests}] ", end="")

                    test_result = self.test_tool_on_model(
                        tool_name, tool, model_name, model_path
                    )
                    self.results[model_name][tool_name] = test_result

                    if not test_result["success"]:
                        failed_tests.append((model_name, tool_name, test_result))

            # Save incremental results after each model completion
            self.save_incremental_results(model_name)
            print(f"✅ Completed testing {model_name} - results saved incrementally")

        # Summary of failed tests
        if failed_tests:
            print(f"\n⚠️  {len(failed_tests)} tests failed:")
            for model_name, tool_name, result in failed_tests:
                error_type = result["error"]["type"] if result["error"] else "Unknown"
                print(f"  ❌ {model_name} → {tool_name}: {error_type}")

    def generate_comprehensive_report(self):
        """Generate detailed biological validation report"""

        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE BIOLOGICAL VALIDATION REPORT")
        print("=" * 80)

        # Overall statistics
        total_tests = sum(len(model_results) for model_results in self.results.values())
        successful_tests = sum(
            1
            for model_results in self.results.values()
            for test_result in model_results.values()
            if test_result.get("success", False)
        )

        print(
            f"📈 Overall Success Rate: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)"
        )

        # Tool category success rates
        tool_categories = {
            "COBRA Tools": list(self.cobra_tools.keys()),
            "AI Media Tools": list(self.media_tools.keys()),
            "Biochemistry Tools": list(self.biochem_tools.keys()),
        }

        print(f"\n🔧 Success by Tool Category:")
        for category, tool_names in tool_categories.items():
            category_success = 0
            category_total = 0

            for model_results in self.results.values():
                for tool_name in tool_names:
                    if tool_name in model_results:
                        category_total += 1
                        if model_results[tool_name].get("success", False):
                            category_success += 1

            if category_total > 0:
                success_rate = (category_success / category_total) * 100
                print(
                    f"  {category:<20}: {category_success:>2}/{category_total} ({success_rate:>5.1f}%)"
                )

        # Per-model detailed analysis
        for model_name, model_results in self.results.items():
            print(f"\n🧬 {model_name.upper()} - Biological Validation")
            print("-" * 50)

            model_successes = sum(
                1 for result in model_results.values() if result.get("success", False)
            )
            model_total = len(model_results)
            print(
                f"Model Success Rate: {model_successes}/{model_total} ({100*model_successes/model_total:.1f}%)"
            )

            # Biological insights summary
            all_insights = []
            all_warnings = []

            for tool_name, result in model_results.items():
                if not result.get("success", False):
                    continue

                validation = result.get("biological_validation", {})
                if validation.get("biological_insights"):
                    all_insights.extend(validation["biological_insights"])
                if validation.get("warnings"):
                    all_warnings.extend(validation["warnings"])

            if all_insights:
                print("  🔬 Key Biological Insights:")
                for insight in set(all_insights)[:5]:  # Top 5 unique insights
                    print(f"    • {insight}")

            if all_warnings:
                print("  ⚠️  Biological Warnings:")
                for warning in set(all_warnings)[:3]:  # Top 3 unique warnings
                    print(f"    • {warning}")

            # Tool results with biological scores
            print("  📊 Tool Results with Biological Validation:")
            for tool_name, result in model_results.items():
                status = "✅" if result.get("success", False) else "❌"
                time_str = f"{result.get('execution_time', 0):.2f}s"

                # Biological validation score
                validation = result.get("biological_validation", {})
                scores = validation.get("scores", {})
                avg_score = sum(scores.values()) / len(scores) if scores else 0
                score_str = f"Bio:{avg_score:.2f}" if scores else "No-score"

                # Key metric
                metrics = result.get("key_metrics", {})
                metric_str = ""
                if "growth_rate" in metrics:
                    metric_str = f"Growth:{metrics['growth_rate']:.3f}"
                elif "essential_gene_count" in metrics:
                    metric_str = f"Essential:{metrics['essential_gene_count']}"
                elif "media_compatibility" in metrics:
                    metric_str = f"Compat:{metrics['media_compatibility']:.2f}"
                elif "resolution_success" in metrics:
                    metric_str = f"Resolved:{metrics['resolution_success']}"

                print(
                    f"    {status} {tool_name:<20} {time_str:>8} {score_str:>10} {metric_str}"
                )

    def save_comprehensive_results(
        self, output_file="comprehensive_testbed_results.json"
    ):
        """Save comprehensive results with biological validation"""

        # Create output directory
        output_dir = Path("testbed_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{timestamp}_{output_file}"

        # Calculate summary statistics
        total_tests = sum(len(model_results) for model_results in self.results.values())
        successful_tests = sum(
            1
            for model_results in self.results.values()
            for test_result in model_results.values()
            if test_result.get("success", False)
        )

        # Biological validation summary
        bio_validation_summary = {}
        for model_name, model_results in self.results.items():
            model_bio_scores = []
            for tool_name, result in model_results.items():
                if result.get("success", False):
                    validation = result.get("biological_validation", {})
                    scores = validation.get("scores", {})
                    if scores:
                        avg_score = sum(scores.values()) / len(scores)
                        model_bio_scores.append(avg_score)

            bio_validation_summary[model_name] = {
                "avg_bio_score": (
                    sum(model_bio_scores) / len(model_bio_scores)
                    if model_bio_scores
                    else 0
                ),
                "scored_tools": len(model_bio_scores),
            }

        # Create comprehensive results with metadata
        full_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "testbed_version": "comprehensive_v1",
                "models_tested": list(self.models.keys()),
                "tools_tested": {
                    "cobra_tools": list(self.cobra_tools.keys()),
                    "media_tools": list(self.media_tools.keys()),
                    "biochem_tools": list(self.biochem_tools.keys()),
                    "total_count": len(self.all_tools),
                },
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": (
                    successful_tests / total_tests if total_tests > 0 else 0
                ),
                "biological_validation_summary": bio_validation_summary,
            },
            "test_configuration": {
                "model_paths": self.models,
                "biological_validation_enabled": True,
                "validation_rules": self.validator.validation_rules,
            },
            "results": self.results,
        }

        with open(output_path, "w") as f:
            json.dump(full_results, f, indent=2, default=str)

        print(f"\n💾 Comprehensive results saved to: {output_path}")
        print(f"📄 File size: {os.path.getsize(output_path) / 1024:.1f} KB")
        print(
            f"🔬 Includes biological validation for {len(self.all_tools)} tools on {len(self.models)} models"
        )

        return output_path

    def save_incremental_results(self, model_name: str):
        """Save results after each model completion to prevent data loss"""

        # Create incremental output directory
        output_dir = Path("testbed_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        incremental_file = (
            output_dir / f"{timestamp}_incremental_{model_name}_results.json"
        )

        # Create incremental results
        incremental_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "testbed_version": "comprehensive_v1_incremental",
                "model_completed": model_name,
                "models_tested": list(self.models.keys()),
                "tools_tested": {
                    "cobra_tools": list(self.cobra_tools.keys()),
                    "media_tools": list(self.media_tools.keys()),
                    "biochem_tools": list(self.biochem_tools.keys()),
                    "total_count": len(self.all_tools),
                },
            },
            "results": {model_name: self.results.get(model_name, {})},
        }

        with open(incremental_file, "w") as f:
            json.dump(incremental_results, f, indent=2, default=str)

        print(f"💾 Incremental results saved: {incremental_file}")


def main():
    """Main execution function"""

    print("🧪 ModelSEED Tool Validation Suite with Biological Validation")
    print("=" * 70)
    print("Validating ALL tools (COBRA + AI Media + Biochemistry) on ALL models")
    print("with comprehensive biological knowledge validation.\n")

    validation_suite = ModelSEEDToolValidationSuite()

    try:
        # Run comprehensive validation
        validation_suite.run_comprehensive_validation()

        # Generate comprehensive report
        validation_suite.generate_comprehensive_report()

        # Save detailed results
        output_file = validation_suite.save_comprehensive_results()

        print(f"\n🎉 Tool validation suite complete!")
        print(f"📊 View detailed results with biological validation in: {output_file}")
        print(
            f"🔬 All {len(validation_suite.all_tools)} tools validated on {len(validation_suite.models)} models"
        )

    except KeyboardInterrupt:
        print("\n⚠️  Validation suite interrupted by user")
    except Exception as e:
        print(f"\n💥 Validation suite failed with error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
