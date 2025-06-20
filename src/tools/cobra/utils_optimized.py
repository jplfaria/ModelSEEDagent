"""
Optimized COBRA Model Utilities

Performance-optimized version of model utilities with:
- Model caching to eliminate redundant file I/O
- Fixed GLPK tolerance configuration for COBRA >= 0.29
- Session-level model reuse
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cobra
from cobra.io import read_sbml_model, write_sbml_model

from .model_cache import CachedModelUtils, get_model_cache

logger = logging.getLogger(__name__)


class OptimizedModelUtils:
    """Performance-optimized utility functions for working with COBRA models"""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        if use_cache:
            self._cache = get_model_cache()
            self._cached_utils = CachedModelUtils(self._cache)

    def load_model(self, model_path: str) -> cobra.Model:
        """
        Load a metabolic model from a file with caching support.

        Args:
            model_path: Path to the SBML model file

        Returns:
            cobra.Model: The loaded metabolic model

        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If the model file is invalid
        """
        if self.use_cache:
            return self._cached_utils.load_model(model_path)
        else:
            # Fall back to direct loading
            return self._load_model_direct(model_path)

    def _load_model_direct(self, model_path: str) -> cobra.Model:
        """Direct model loading without caching (fallback)"""
        try:
            resolved_path = self._resolve_model_path(model_path)

            if not os.path.exists(resolved_path):
                raise FileNotFoundError(f'Model file not found: "{model_path}"')

            model = read_sbml_model(resolved_path)
            logger.info(f"Successfully loaded model: {model.id}")
            return model

        except Exception as e:
            logger.error(f'Error loading model from "{model_path}": {str(e)}')
            raise ValueError(f"Failed to load model: {str(e)}")

    def configure_solver_optimally(
        self, model: cobra.Model, tolerance: float = 1e-9
    ) -> cobra.Model:
        """
        Configure model solver with optimal settings and fixed tolerance configuration.

        Args:
            model: The COBRA model to configure
            tolerance: Optimality tolerance for solver

        Returns:
            Configured model
        """
        try:
            # Use the correct tolerance setting for COBRA >= 0.29
            if hasattr(model.solver, "configuration") and hasattr(
                model.solver.configuration, "tolerances"
            ):
                model.solver.configuration.tolerances.optimality = tolerance
                logger.debug(f"Set solver optimality tolerance to {tolerance}")
            else:
                # Fallback for older COBRA versions
                if hasattr(model, "tolerance"):
                    model.tolerance = tolerance
                    logger.debug(f"Set model tolerance to {tolerance} (legacy)")
                else:
                    logger.warning(
                        "Could not set solver tolerance - unsupported COBRA version"
                    )

        except Exception as e:
            logger.warning(f"Failed to set solver tolerance: {e}")

        return model

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        """
        Resolve model path relative to project root.

        Args:
            model_path: Original model path

        Returns:
            str: Resolved absolute path
        """
        # If already absolute, return as-is
        if os.path.isabs(model_path):
            return model_path

        # Find project root (directory containing this file's parent's parent)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent

        # Try relative to project root first
        project_relative = project_root / model_path
        if project_relative.exists():
            return str(project_relative)

        # Try relative to current working directory
        if os.path.exists(model_path):
            return os.path.abspath(model_path)

        # Return project-relative path for error reporting
        return str(project_relative)

    @staticmethod
    def save_model(model: cobra.Model, output_path: str) -> str:
        """
        Save a metabolic model to a file.

        Args:
            model: The metabolic model to save
            output_path: Path where to save the model

        Returns:
            str: Path to the saved model file

        Raises:
            ValueError: If the model cannot be saved
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save model
            write_sbml_model(model, output_path)
            logger.info(f"Successfully saved model to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving model to {output_path}: {str(e)}")
            raise ValueError(f"Failed to save model: {str(e)}")

    @staticmethod
    def verify_model(model: cobra.Model) -> Dict[str, Any]:
        """
        Verify model integrity and return basic statistics.

        Args:
            model: The metabolic model to verify

        Returns:
            Dict containing model verification results
        """
        verification = {
            "is_valid": True,
            "issues": [],
            "statistics": {
                "reactions": len(model.reactions),
                "metabolites": len(model.metabolites),
                "genes": len(model.genes),
            },
        }

        # Check for isolated metabolites
        isolated = []
        for met in model.metabolites:
            if len(met.reactions) == 0:
                isolated.append(met.id)
        if isolated:
            verification["issues"].append(
                {
                    "type": "isolated_metabolites",
                    "description": "Metabolites not connected to any reactions",
                    "items": isolated,
                }
            )

        # Check for reactions without genes
        no_genes = []
        for rxn in model.reactions:
            if len(rxn.genes) == 0 and not rxn.id.startswith("EX_"):
                no_genes.append(rxn.id)
        if no_genes:
            verification["issues"].append(
                {
                    "type": "no_gene_association",
                    "description": "Non-exchange reactions without gene associations",
                    "items": no_genes,
                }
            )

        # Check mass balance
        unbalanced = []
        for reaction in model.reactions:
            if not reaction.id.startswith("EX_") and not reaction.check_mass_balance():
                unbalanced.append(reaction.id)
        if unbalanced:
            verification["issues"].append(
                {
                    "type": "unbalanced_reactions",
                    "description": "Reactions with unbalanced mass",
                    "items": unbalanced,
                }
            )

        # Update validity flag if issues were found
        if verification["issues"]:
            verification["is_valid"] = False

        return verification

    @staticmethod
    def get_reaction_info(model: cobra.Model, reaction_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific reaction.

        Args:
            model: The metabolic model
            reaction_id: ID of the reaction to analyze

        Returns:
            Dict containing reaction information

        Raises:
            ValueError: If reaction is not found in the model
        """
        try:
            reaction = model.reactions.get_by_id(reaction_id)
            return {
                "id": reaction.id,
                "name": reaction.name,
                "subsystem": reaction.subsystem,
                "reaction_string": reaction.build_reaction_string(),
                "metabolites": {
                    met.id: coef for met, coef in reaction.metabolites.items()
                },
                "genes": [gene.id for gene in reaction.genes],
                "gene_reaction_rule": reaction.gene_reaction_rule,
                "bounds": reaction.bounds,
                "objective_coefficient": reaction.objective_coefficient,
                "is_exchange": reaction.id.startswith("EX_"),
                "is_balanced": reaction.check_mass_balance(),
            }
        except KeyError:
            raise ValueError(f"Reaction not found in model: {reaction_id}")

    @staticmethod
    def get_metabolite_info(model: cobra.Model, metabolite_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific metabolite.

        Args:
            model: The metabolic model
            metabolite_id: ID of the metabolite to analyze

        Returns:
            Dict containing metabolite information

        Raises:
            ValueError: If metabolite is not found in the model
        """
        try:
            metabolite = model.metabolites.get_by_id(metabolite_id)
            return {
                "id": metabolite.id,
                "name": metabolite.name,
                "formula": (
                    metabolite.formula if hasattr(metabolite, "formula") else None
                ),
                "charge": metabolite.charge if hasattr(metabolite, "charge") else None,
                "compartment": (
                    metabolite.compartment
                    if hasattr(metabolite, "compartment")
                    else None
                ),
                "reactions": [rxn.id for rxn in metabolite.reactions],
                "producing_reactions": [
                    rxn.id for rxn in metabolite.reactions if metabolite in rxn.products
                ],
                "consuming_reactions": [
                    rxn.id
                    for rxn in metabolite.reactions
                    if metabolite in rxn.reactants
                ],
            }
        except KeyError:
            raise ValueError(f"Metabolite not found in model: {metabolite_id}")

    @staticmethod
    def find_deadend_metabolites(model: cobra.Model) -> Dict[str, list]:
        """
        Find dead-end metabolites in the model.

        Args:
            model: The metabolic model

        Returns:
            Dict containing lists of dead-end metabolites
        """
        no_production = []
        no_consumption = []
        disconnected = []

        for metabolite in model.metabolites:
            producers = sum(
                1 for rxn in metabolite.reactions if metabolite in rxn.products
            )
            consumers = sum(
                1 for rxn in metabolite.reactions if metabolite in rxn.reactants
            )

            if producers == 0 and consumers == 0:
                disconnected.append(metabolite.id)
            elif producers == 0:
                no_production.append(metabolite.id)
            elif consumers == 0:
                no_consumption.append(metabolite.id)

        return {
            "no_production": no_production,
            "no_consumption": no_consumption,
            "disconnected": disconnected,
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get model cache statistics"""
        if self.use_cache:
            return self._cache.get_cache_stats()
        else:
            return {"caching": "disabled"}

    def clear_cache(self):
        """Clear the model cache"""
        if self.use_cache:
            self._cache.clear_cache()


# Copy other classes from original utils.py unchanged
from .utils import BiomassDetector, CompoundMapper, MediaManager

# Create a global optimized instance for easy drop-in replacement
_optimized_utils = OptimizedModelUtils(use_cache=True)


def load_model_optimized(model_path: str) -> cobra.Model:
    """Optimized model loading with caching"""
    return _optimized_utils.load_model(model_path)


def configure_solver_optimized(
    model: cobra.Model, tolerance: float = 1e-9
) -> cobra.Model:
    """Configure solver with fixed tolerance setting"""
    return _optimized_utils.configure_solver_optimally(model, tolerance)
