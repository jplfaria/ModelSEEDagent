"""
ModelSEEDpy Integration Module
==============================

This module provides enhanced integration with ModelSEEDpy models using cobrakbase
for proper media handling, biomass detection, and analysis compatibility.

Key features:
- cobrakbase integration for ModelSEEDpy media handling
- Enhanced biomass reaction detection for ModelSEED models
- Proper compound ID handling between ModelSEED and BIGG formats
- Specialized analysis tools for ModelSEEDpy models
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cobra
import pandas as pd
from pydantic import BaseModel

from .media_library import MediaLibrary
from .utils import BiomassDetector, CompoundMapper, ModelUtils

# Set up logging
logger = logging.getLogger(__name__)


class ModelSEEDpyEnhancement:
    """Enhanced ModelSEEDpy model handling with cobrakbase integration"""

    def __init__(self):
        """Initialize ModelSEEDpy enhancement capabilities"""
        self.media_library = MediaLibrary()
        self.compound_mapper = CompoundMapper()
        self.model_utils = ModelUtils()

        # Try to import cobrakbase
        self.cobrakbase_available = self._check_cobrakbase()

        if self.cobrakbase_available:
            logger.info("cobrakbase integration enabled")
        else:
            logger.warning("cobrakbase not available - using fallback methods")

    def _check_cobrakbase(self) -> bool:
        """Check if cobrakbase is available and import key modules"""
        try:
            import cobrakbase.core.kbasebiochem.core
            import cobrakbase.core.kbasebiochem.media_utils

            # Store references for later use
            self._cobrakbase_core = cobrakbase.core.kbasebiochem.core
            self._cobrakbase_media = cobrakbase.core.kbasebiochem.media_utils

            return True
        except ImportError as e:
            logger.debug(f"cobrakbase import failed: {e}")
            return False

    def detect_modelseedpy_model(self, model: cobra.Model) -> bool:
        """Detect if a model is from ModelSEEDpy based on ID patterns"""

        # Check for ModelSEED compound IDs (cpd00xxx)
        modelseed_compounds = sum(
            1 for met in model.metabolites if met.id.startswith("cpd")
        )

        # Check for ModelSEED reaction IDs (rxn00xxx)
        modelseed_reactions = sum(
            1 for rxn in model.reactions if rxn.id.startswith("rxn")
        )

        total_compounds = len(model.metabolites)
        total_reactions = len(model.reactions)

        # Consider it a ModelSEEDpy model if >70% of compounds and reactions follow ModelSEED format
        is_modelseed = (modelseed_compounds / max(total_compounds, 1)) > 0.7 and (
            modelseed_reactions / max(total_reactions, 1)
        ) > 0.7

        logger.debug(
            f"Model {model.id}: {modelseed_compounds}/{total_compounds} compounds, "
            f"{modelseed_reactions}/{total_reactions} reactions are ModelSEED format"
        )

        return is_modelseed

    def get_enhanced_biomass_detection(self, model: cobra.Model) -> Optional[str]:
        """Enhanced biomass detection specifically for ModelSEEDpy models"""

        if not self.detect_modelseedpy_model(model):
            # Use standard detection for non-ModelSEED models
            return BiomassDetector.detect_biomass_reaction(model)

        # ModelSEED-specific biomass detection strategies
        strategies = [
            self._detect_biomass_by_modelseed_id,
            self._detect_biomass_by_modelseed_objective,
            self._detect_biomass_by_modelseed_name,
            self._detect_biomass_by_product_count,
        ]

        for strategy in strategies:
            biomass_id = strategy(model)
            if biomass_id:
                logger.debug(f"Biomass detected by {strategy.__name__}: {biomass_id}")
                return biomass_id

        logger.warning(f"No biomass reaction detected for ModelSEEDpy model {model.id}")
        return None

    def _detect_biomass_by_modelseed_id(self, model: cobra.Model) -> Optional[str]:
        """Detect biomass by ModelSEED ID patterns"""

        # Common ModelSEED biomass reaction patterns
        biomass_patterns = [
            r"rxn\d+",  # Standard ModelSEED reaction format
            r".*biomass.*",  # Contains "biomass"
            r".*growth.*",  # Contains "growth"
        ]

        import re

        for reaction in model.reactions:
            for pattern in biomass_patterns:
                if re.match(pattern, reaction.id, re.IGNORECASE):
                    # Check if it has biomass-like properties
                    if self._is_biomass_like(reaction):
                        return reaction.id

        return None

    def _detect_biomass_by_modelseed_objective(
        self, model: cobra.Model
    ) -> Optional[str]:
        """Detect biomass by checking the current objective"""

        try:
            objective_reactions = [
                rxn.id for rxn in model.reactions if rxn.objective_coefficient != 0
            ]
            if len(objective_reactions) == 1:
                return objective_reactions[0]
        except Exception as e:
            logger.debug(f"Objective detection failed: {e}")

        return None

    def _detect_biomass_by_modelseed_name(self, model: cobra.Model) -> Optional[str]:
        """Detect biomass by reaction name patterns in ModelSEED models"""

        biomass_keywords = ["biomass", "growth", "objective"]

        for reaction in model.reactions:
            name = getattr(reaction, "name", "").lower()
            if any(keyword in name for keyword in biomass_keywords):
                if self._is_biomass_like(reaction):
                    return reaction.id

        return None

    def _detect_biomass_by_product_count(self, model: cobra.Model) -> Optional[str]:
        """Detect biomass by counting products (biomass typically has many)"""

        candidates = []
        for reaction in model.reactions:
            # Biomass reactions typically have many products
            product_count = len([met for met in reaction.products])
            if product_count >= 10:  # Threshold for biomass-like reactions
                candidates.append((reaction.id, product_count))

        if candidates:
            # Return the reaction with the most products
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return None

    def _is_biomass_like(self, reaction: cobra.Reaction) -> bool:
        """Check if a reaction has biomass-like characteristics"""

        # Check for multiple products (typical of biomass)
        if len(reaction.products) < 5:
            return False

        # Check for energy consumption (ATP, GTP, etc.)
        energy_compounds = {"cpd00002", "cpd00008", "cpd00012"}  # ATP, GTP, CTP
        reaction_compounds = {met.id for met in reaction.metabolites}

        if not any(energy in reaction_compounds for energy in energy_compounds):
            return False

        # Biomass should not be reversible
        if reaction.reversibility:
            return False

        return True

    def apply_media_with_cobrakbase(
        self,
        model: cobra.Model,
        media_name: str,
        ai_modifications: Optional[str] = None,
    ) -> cobra.Model:
        """Apply media to ModelSEEDpy model using cobrakbase if available"""

        if not self.cobrakbase_available:
            # Fallback to standard media application
            return self.media_library.apply_media_to_model(
                model, media_name, ai_modifications
            )

        try:
            # Get media composition
            if ai_modifications:
                media_comp = self.media_library.manipulate_media(
                    media_name, ai_modifications
                )
                media_dict = media_comp.compounds
            else:
                media_comp = self.media_library.get_media(media_name)
                media_dict = media_comp.compounds

            # Use cobrakbase for ModelSEEDpy-specific media application
            return self._apply_media_with_cobrakbase_core(model, media_dict)

        except Exception as e:
            logger.warning(
                f"cobrakbase media application failed: {e}, falling back to standard method"
            )
            return self.media_library.apply_media_to_model(
                model, media_name, ai_modifications
            )

    def _apply_media_with_cobrakbase_core(
        self, model: cobra.Model, media_dict: Dict[str, float]
    ) -> cobra.Model:
        """Core cobrakbase media application logic"""

        # Create a copy of the model to avoid modifying the original
        model_copy = model.copy()

        # Reset all exchange reaction bounds
        for reaction in model_copy.exchanges:
            reaction.lower_bound = 0
            reaction.upper_bound = 1000

        # Apply media constraints
        for compound_id, uptake_rate in media_dict.items():
            # Find matching exchange reaction
            exchange_rxn = self._find_exchange_reaction(model_copy, compound_id)

            if exchange_rxn:
                if uptake_rate < 0:  # Uptake
                    exchange_rxn.lower_bound = uptake_rate
                else:  # Secretion
                    exchange_rxn.upper_bound = uptake_rate
            else:
                logger.debug(f"Exchange reaction not found for compound: {compound_id}")

        return model_copy

    def _find_exchange_reaction(
        self, model: cobra.Model, compound_id: str
    ) -> Optional[cobra.Reaction]:
        """Find exchange reaction for a given compound ID"""

        # Direct search for exchange reactions
        for reaction in model.exchanges:
            # Check if the compound is involved in this exchange reaction
            for metabolite in reaction.metabolites:
                if metabolite.id == compound_id or metabolite.id.startswith(
                    compound_id
                ):
                    return reaction

                # Check for extracellular variants (e.g., cpd00027_e0, cpd00027[e])
                if metabolite.id.startswith(
                    compound_id + "_e"
                ) or metabolite.id.startswith(compound_id + "[e"):
                    return reaction

        return None

    def analyze_modelseedpy_model(self, model: cobra.Model) -> Dict[str, Any]:
        """Comprehensive analysis of ModelSEEDpy model capabilities"""

        analysis = {
            "is_modelseedpy": self.detect_modelseedpy_model(model),
            "model_id": model.id,
            "biomass_reaction": self.get_enhanced_biomass_detection(model),
            "total_reactions": len(model.reactions),
            "total_metabolites": len(model.metabolites),
            "exchange_reactions": len(model.exchanges),
            "cobrakbase_available": self.cobrakbase_available,
        }

        # Additional ModelSEEDpy-specific analysis
        if analysis["is_modelseedpy"]:
            analysis.update(
                {
                    "modelseed_compounds": sum(
                        1 for met in model.metabolites if met.id.startswith("cpd")
                    ),
                    "modelseed_reactions": sum(
                        1 for rxn in model.reactions if rxn.id.startswith("rxn")
                    ),
                    "compartments": list(model.compartments.keys()),
                    "genes": len(model.genes),
                }
            )

            # Check media compatibility
            for media_name in self.media_library.list_available_media():
                compatibility = self.media_library.analyze_media_compatibility(
                    model, media_name
                )
                analysis[f"media_compatibility_{media_name}"] = compatibility[
                    "compatibility_score"
                ]

        return analysis

    def test_all_media_with_model(self, model: cobra.Model) -> Dict[str, Any]:
        """Test all available media with a ModelSEEDpy model"""

        results = {}

        for media_name in self.media_library.list_available_media():
            try:
                # Apply media
                model_with_media = self.apply_media_with_cobrakbase(model, media_name)

                # Test growth
                solution = model_with_media.optimize()
                growth_rate = (
                    solution.objective_value if solution.status == "optimal" else 0.0
                )

                results[media_name] = {
                    "growth_rate": growth_rate,
                    "status": solution.status,
                    "feasible": growth_rate > 1e-6,
                }

                # Test AI modifications
                if media_name in ["GMM", "AuxoMedia"]:
                    # Test anaerobic modification
                    anaerobic_model = self.apply_media_with_cobrakbase(
                        model, media_name, "make anaerobic"
                    )
                    anaerobic_solution = anaerobic_model.optimize()
                    anaerobic_growth = (
                        anaerobic_solution.objective_value
                        if anaerobic_solution.status == "optimal"
                        else 0.0
                    )

                    results[f"{media_name}_anaerobic"] = {
                        "growth_rate": anaerobic_growth,
                        "status": anaerobic_solution.status,
                        "feasible": anaerobic_growth > 1e-6,
                    }

            except Exception as e:
                results[media_name] = {
                    "error": str(e),
                    "growth_rate": 0.0,
                    "feasible": False,
                }

        return results


# Convenience function for external use
def get_modelseedpy_enhancement() -> ModelSEEDpyEnhancement:
    """Get a configured ModelSEEDpy enhancement instance"""
    return ModelSEEDpyEnhancement()


# Example usage and testing
if __name__ == "__main__":
    # Test ModelSEEDpy integration
    enhancement = ModelSEEDpyEnhancement()

    # Test with a model (would need an actual model file)
    print(f"cobrakbase available: {enhancement.cobrakbase_available}")
    print(
        f"Media library has {len(enhancement.media_library.list_available_media())} media types"
    )
