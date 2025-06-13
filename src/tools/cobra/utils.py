import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cobra
from cobra.io import read_sbml_model, write_sbml_model

logger = logging.getLogger(__name__)


class ModelUtils:
    """Utility functions for working with COBRA models"""

    @staticmethod
    def load_model(model_path: str) -> cobra.Model:
        """
        Load a metabolic model from a file.

        Args:
            model_path: Path to the SBML model file

        Returns:
            cobra.Model: The loaded metabolic model

        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If the model file is invalid
        """
        try:
            # Resolve path relative to project root if not absolute
            resolved_path = ModelUtils._resolve_model_path(model_path)

            if not os.path.exists(resolved_path):
                raise FileNotFoundError(f'Model file not found: "{model_path}"')

            model = read_sbml_model(resolved_path)
            logger.info(f"Successfully loaded model: {model.id}")
            return model

        except Exception as e:
            logger.error(f'Error loading model from "{model_path}": {str(e)}')
            raise ValueError(f"Failed to load model: {str(e)}")

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
        project_root = current_file.parent.parent.parent

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


class BiomassDetector:
    """Robust biomass reaction identification for any model type"""

    @staticmethod
    def detect_biomass_reaction(model: cobra.Model) -> Optional[str]:
        """
        Auto-detect biomass reaction using multiple strategies.

        Args:
            model: The metabolic model

        Returns:
            String ID of biomass reaction, or None if not found
        """
        # Strategy 1: Check current objective
        if model.objective and len(model.objective.variables) > 0:
            for var in model.objective.variables:
                if var.name:  # This is the reaction ID
                    logger.info(f"Found biomass reaction from objective: {var.name}")
                    return var.name

        # Strategy 2: Search by reaction ID patterns
        biomass_patterns = [
            r".*bio.*",
            r".*biomass.*",
            r".*growth.*",
            r".*BIOMASS.*",
            r".*BIO.*",
            r".*Growth.*",
            r"bio\d+",
            r"meta_R_bio.*",
        ]

        import re

        for pattern in biomass_patterns:
            for reaction in model.reactions:
                if re.search(pattern, reaction.id, re.IGNORECASE):
                    logger.info(
                        f"Found biomass reaction by ID pattern '{pattern}': {reaction.id}"
                    )
                    return reaction.id

        # Strategy 3: Search by reaction name patterns
        for pattern in biomass_patterns:
            for reaction in model.reactions:
                if reaction.name and re.search(pattern, reaction.name, re.IGNORECASE):
                    logger.info(
                        f"Found biomass reaction by name pattern '{pattern}': {reaction.id}"
                    )
                    return reaction.id

        # Strategy 4: Look for reactions with high product count (biomass typically has many products)
        high_product_reactions = []
        for reaction in model.reactions:
            if (
                len(reaction.products) > 10
            ):  # Biomass reactions typically have many products
                high_product_reactions.append((reaction.id, len(reaction.products)))

        if high_product_reactions:
            # Sort by product count and take the highest
            high_product_reactions.sort(key=lambda x: x[1], reverse=True)
            best_candidate = high_product_reactions[0][0]
            logger.info(f"Found biomass reaction by product count: {best_candidate}")
            return best_candidate

        logger.warning("Could not detect biomass reaction automatically")
        return None

    @staticmethod
    def set_biomass_objective(
        model: cobra.Model, biomass_reaction_id: Optional[str] = None
    ) -> bool:
        """
        Set the biomass reaction as the model objective.

        Args:
            model: The metabolic model
            biomass_reaction_id: Optional specific biomass reaction ID

        Returns:
            True if successful, False otherwise
        """
        if not biomass_reaction_id:
            biomass_reaction_id = BiomassDetector.detect_biomass_reaction(model)

        if not biomass_reaction_id:
            logger.error("No biomass reaction found to set as objective")
            return False

        try:
            model.objective = biomass_reaction_id
            logger.info(f"Set biomass objective to: {biomass_reaction_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to set biomass objective: {e}")
            return False


class CompoundMapper:
    """Map between different compound ID systems (ModelSEED <-> BIGG <-> Model-specific)"""

    # Core mapping from ModelSEED compound IDs to BIGG IDs
    MODELSEED_TO_BIGG = {
        "cpd00001": "h2o",  # H2O
        "cpd00007": "o2",  # O2
        "cpd00009": "pi",  # Phosphate
        "cpd00013": "nh4",  # NH3/NH4+
        "cpd00027": "glc__D",  # D-Glucose
        "cpd00030": "mn2",  # Mn2+
        "cpd00034": "zn2",  # Zn2+
        "cpd00048": "so4",  # Sulfate
        "cpd00058": "cu2",  # Cu2+
        "cpd00063": "ca2",  # Ca2+
        "cpd00067": "h",  # H+
        "cpd00099": "cl",  # Cl-
        "cpd00149": "cobalt2",  # Co2+
        "cpd00205": "k",  # K+
        "cpd00244": "ni2",  # Ni2+
        "cpd00254": "mg2",  # Mg2+
        "cpd00971": "na1",  # Na+
        "cpd10515": "fe2",  # Fe+2
        "cpd10516": "fe3",  # Fe+3
        "cpd11574": "mobd",  # Molybdate
    }

    # Reverse mapping from BIGG to ModelSEED
    BIGG_TO_MODELSEED = {v: k for k, v in MODELSEED_TO_BIGG.items()}

    @staticmethod
    def _detect_model_type(model: cobra.Model) -> str:
        """
        Detect if model uses ModelSEED or BIGG naming conventions.

        Args:
            model: The metabolic model

        Returns:
            "modelseed" or "bigg"
        """
        # Check a few exchange reactions to determine naming pattern
        modelseed_count = 0
        bigg_count = 0

        for reaction in list(model.exchanges)[:10]:  # Check first 10 exchange reactions
            if "_e0" in reaction.id:
                modelseed_count += 1
            elif "_e" in reaction.id and "_e0" not in reaction.id:
                bigg_count += 1

        return "modelseed" if modelseed_count > bigg_count else "bigg"

    @staticmethod
    def _bigg_to_modelseed(bigg_id: str) -> Optional[str]:
        """
        Convert BIGG compound ID to ModelSEED compound ID.

        Args:
            bigg_id: BIGG compound ID

        Returns:
            ModelSEED compound ID if found, None otherwise
        """
        return CompoundMapper.BIGG_TO_MODELSEED.get(bigg_id)

    @staticmethod
    def find_exchange_reaction(model: cobra.Model, compound_id: str) -> Optional[str]:
        """
        Find the exchange reaction ID for a given compound in the model.

        Args:
            model: The metabolic model
            compound_id: Compound ID (ModelSEED, BIGG, or model-specific)

        Returns:
            Exchange reaction ID if found, None otherwise
        """
        # Determine model type by checking a few exchange reactions
        model_type = CompoundMapper._detect_model_type(model)

        if model_type == "modelseed":
            # ModelSEED model: try direct compound ID first, then convert if needed
            search_patterns = [
                f"EX_{compound_id}_e0",  # Direct ModelSEED format
                f"EX_{compound_id}_e",  # Alternative compartment
                f"EX_{compound_id}",  # Without compartment
            ]

            # If compound_id looks like BIGG, also try converting to ModelSEED
            if not compound_id.startswith("cpd"):
                modelseed_id = CompoundMapper._bigg_to_modelseed(compound_id)
                if modelseed_id:
                    search_patterns.insert(0, f"EX_{modelseed_id}_e0")

        else:
            # BIGG model: try direct compound ID first, then convert if needed
            search_patterns = [
                f"EX_{compound_id}_e",  # Direct BIGG format
                f"EX_{compound_id}",  # Without compartment
                f"{compound_id}_e",  # Just compound with compartment
            ]

            # If compound_id looks like ModelSEED, convert to BIGG
            if compound_id.startswith("cpd"):
                bigg_id = CompoundMapper.MODELSEED_TO_BIGG.get(compound_id, compound_id)
                if bigg_id != compound_id:
                    search_patterns.insert(0, f"EX_{bigg_id}_e")

        # Search in model reactions
        for pattern in search_patterns:
            if pattern in [rxn.id for rxn in model.reactions]:
                logger.debug(
                    f"Found exchange reaction: {pattern} for compound {compound_id}"
                )
                return pattern

        # Fuzzy search in reaction IDs and names
        for reaction in model.reactions:
            if reaction.id.startswith("EX_"):
                # Check if compound appears in reaction ID
                if compound_id.lower() in reaction.id.lower():
                    logger.debug(
                        f"Found exchange reaction by fuzzy search: {reaction.id} for compound {compound_id}"
                    )
                    return reaction.id

        logger.warning(f"Could not find exchange reaction for compound: {compound_id}")
        return None

    @staticmethod
    def map_media_to_model(media_dict: Dict, model: cobra.Model) -> Dict[str, float]:
        """
        Convert media specification to model-specific exchange reaction bounds.

        Args:
            media_dict: Media specification with compound IDs as keys
            model: The metabolic model

        Returns:
            Dict mapping exchange reaction IDs to uptake rates
        """
        model_media = {}

        for compound_id, uptake_rate in media_dict.items():
            exchange_rxn_id = CompoundMapper.find_exchange_reaction(model, compound_id)
            if exchange_rxn_id:
                model_media[exchange_rxn_id] = uptake_rate
            else:
                logger.warning(
                    f"Could not map compound {compound_id} to model exchange reaction"
                )

        return model_media


class MediaManager:
    """Universal media handling for all model types"""

    @staticmethod
    def load_media_from_file(media_path: str) -> Dict[str, float]:
        """
        Load media from TSV or JSON file.

        Args:
            media_path: Path to media file (TSV or JSON)

        Returns:
            Dict mapping compound IDs to uptake rates (negative = uptake)
        """
        import json
        from pathlib import Path

        import pandas as pd

        media_path = Path(media_path)

        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        try:
            if media_path.suffix.lower() == ".json":
                # Load JSON format (ModelSEED style)
                with open(media_path, "r") as f:
                    media_data = json.load(f)

                media_dict = {}
                if "mediacompounds" in media_data:
                    for compound in media_data["mediacompounds"]:
                        # Extract compound ID from reference
                        compound_ref = compound["compound_ref"]
                        compound_id = compound_ref.split("/")[
                            -1
                        ]  # Get last part after '/'

                        # Use minFlux as uptake rate (should be negative for uptake)
                        uptake_rate = compound["minFlux"]
                        media_dict[compound_id] = uptake_rate

                logger.info(
                    f"Loaded {len(media_dict)} compounds from JSON media file: {media_path}"
                )
                return media_dict

            elif media_path.suffix.lower() in [".tsv", ".txt"]:
                # Load TSV format
                df = pd.read_csv(media_path, sep="\t")

                # Expected columns: compounds, name, formula, minFlux, maxFlux
                media_dict = {}
                for _, row in df.iterrows():
                    compound_id = row["compounds"]
                    uptake_rate = row["minFlux"]
                    media_dict[compound_id] = uptake_rate

                logger.info(
                    f"Loaded {len(media_dict)} compounds from TSV media file: {media_path}"
                )
                return media_dict

            else:
                raise ValueError(f"Unsupported media file format: {media_path.suffix}")

        except Exception as e:
            logger.error(f"Error loading media from {media_path}: {e}")
            raise ValueError(f"Failed to load media file: {e}")

    @staticmethod
    def apply_media_to_model(
        model: cobra.Model, media_dict: Dict[str, float]
    ) -> cobra.Model:
        """
        Apply media composition to a model by setting exchange reaction bounds.

        Args:
            model: The metabolic model
            media_dict: Dict mapping compound IDs to uptake rates

        Returns:
            Modified model with media applied
        """
        applied_count = 0

        # First, close all exchange reactions (set to 0 uptake)
        for reaction in model.exchanges:
            reaction.lower_bound = 0
            reaction.upper_bound = 1000  # Allow unlimited secretion

        # Apply media constraints
        for compound_id, uptake_rate in media_dict.items():
            exchange_rxn_id = CompoundMapper.find_exchange_reaction(model, compound_id)

            if exchange_rxn_id:
                try:
                    reaction = model.reactions.get_by_id(exchange_rxn_id)
                    # Set lower bound to uptake rate (negative values)
                    reaction.lower_bound = min(uptake_rate, 0)  # Ensure non-positive
                    # Keep upper bound high for secretion
                    reaction.upper_bound = 1000
                    applied_count += 1
                    logger.debug(
                        f"Applied media constraint: {exchange_rxn_id} = {uptake_rate}"
                    )
                except KeyError:
                    logger.warning(
                        f"Exchange reaction not found in model: {exchange_rxn_id}"
                    )
            else:
                logger.warning(
                    f"Could not find exchange reaction for compound: {compound_id}"
                )

        logger.info(
            f"Applied {applied_count}/{len(media_dict)} media constraints to model"
        )
        return model

    @staticmethod
    def get_current_media(model: cobra.Model) -> Dict[str, float]:
        """
        Extract current media composition from model exchange reactions.

        Args:
            model: The metabolic model

        Returns:
            Dict mapping exchange reaction IDs to current uptake rates
        """
        current_media = {}

        for reaction in model.exchanges:
            if reaction.lower_bound < 0:  # Has uptake allowed
                current_media[reaction.id] = reaction.lower_bound

        logger.info(
            f"Extracted current media with {len(current_media)} uptake constraints"
        )
        return current_media

    @staticmethod
    def test_growth_with_media(
        model: cobra.Model, media_dict: Dict[str, float]
    ) -> float:
        """
        Test if model can grow with given media composition.

        Args:
            model: The metabolic model
            media_dict: Dict mapping compound IDs to uptake rates

        Returns:
            Growth rate (objective value)
        """
        # Save original bounds
        original_bounds = {}
        for reaction in model.exchanges:
            original_bounds[reaction.id] = reaction.bounds

        try:
            # Apply media
            MediaManager.apply_media_to_model(model, media_dict)

            # Set biomass objective if needed
            BiomassDetector.set_biomass_objective(model)

            # Test growth
            solution = model.optimize()
            growth_rate = (
                solution.objective_value if solution.status == "optimal" else 0.0
            )

            logger.info(f"Growth rate with media: {growth_rate}")
            return growth_rate

        finally:
            # Restore original bounds
            for reaction_id, bounds in original_bounds.items():
                try:
                    model.reactions.get_by_id(reaction_id).bounds = bounds
                except KeyError:
                    pass
