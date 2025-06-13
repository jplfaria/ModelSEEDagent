"""
Dynamic AI-Driven Media Management System
==========================================

This module provides a comprehensive media library and AI-driven manipulation
capabilities for metabolic modeling with both COBRApy and ModelSEEDpy models.

Features:
- Pre-defined media library (GMM, AuxoMedia, PyruvateMinimalMedia, etc.)
- AI-driven media manipulation ("make anaerobic", "add vitamins", etc.)
- TSV format support with automatic format detection
- cobrakbase integration for ModelSEEDpy compatibility
- Dynamic media composition based on model requirements
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cobra
import pandas as pd
from pydantic import BaseModel, Field

from .utils import CompoundMapper, MediaManager, ModelUtils


class MediaComposition(BaseModel):
    """Data model for media composition"""

    name: str
    description: str
    compounds: Dict[str, float]  # compound_id -> uptake_rate
    conditions: Dict[str, Any] = Field(default_factory=dict)  # aerobic, pH, etc.
    format_type: str = "modelseed"  # "modelseed" or "bigg"


class AIMediaManipulator:
    """AI-driven media manipulation capabilities"""

    # Define compound groups for AI manipulation
    OXYGEN_COMPOUNDS = {"cpd00007", "o2"}  # O2
    ALTERNATIVE_ELECTRON_ACCEPTORS = {
        "cpd00209": -10.0,  # NO3- (nitrate)
        "cpd00048": -10.0,  # SO4-2 (sulfate)
        "cpd00058": -5.0,  # Cu2+
        "cpd00149": -5.0,  # Co2+
    }

    CARBON_SOURCES = {
        "cpd00027": "glucose",  # D-Glucose
        "cpd00020": "pyruvate",  # Pyruvate
        "cpd00036": "succinate",  # Succinate
        "cpd00039": "lysine",  # L-Lysine
        "cpd00041": "aspartate",  # L-Aspartate
    }

    AMINO_ACIDS = {
        "cpd00035": -1.0,  # L-Alanine
        "cpd00039": -1.0,  # L-Lysine
        "cpd00041": -1.0,  # L-Aspartate
        "cpd00051": -1.0,  # L-Arginine
        "cpd00054": -1.0,  # L-Serine
        "cpd00060": -1.0,  # L-Methionine
        "cpd00065": -1.0,  # L-Tryptophan
        "cpd00066": -1.0,  # L-Phenylalanine
        "cpd00069": -1.0,  # L-Tyrosine
        "cpd00084": -1.0,  # L-Cysteine
        "cpd00107": -1.0,  # L-Leucine
        "cpd00119": -1.0,  # L-Histidine
        "cpd00129": -1.0,  # L-Proline
        "cpd00156": -1.0,  # L-Valine
        "cpd00161": -1.0,  # L-Threonine
        "cpd00322": -1.0,  # L-Isoleucine
    }

    VITAMINS = {
        "cpd00220": -0.01,  # Riboflavin
        "cpd00305": -0.01,  # Thiamine
        "cpd00393": -0.01,  # Folate
        "cpd00541": -0.01,  # Lipoate
        "cpd01914": -0.01,  # Cobalamin
        "cpd00218": -0.01,  # Niacin
        "cpd00644": -0.01,  # Pantothenate
        "cpd00263": -0.01,  # Pyridoxine
        "cpd00133": -0.01,  # Biotin
    }

    @classmethod
    def make_anaerobic(cls, media: Dict[str, float]) -> Dict[str, float]:
        """Remove oxygen and add alternative electron acceptors"""
        modified_media = media.copy()

        # Remove oxygen
        for o2_compound in cls.OXYGEN_COMPOUNDS:
            if o2_compound in modified_media:
                del modified_media[o2_compound]

        # Add alternative electron acceptors
        modified_media.update(cls.ALTERNATIVE_ELECTRON_ACCEPTORS)

        return modified_media

    @classmethod
    def add_vitamins(cls, media: Dict[str, float]) -> Dict[str, float]:
        """Add vitamin supplementation to media"""
        modified_media = media.copy()
        modified_media.update(cls.VITAMINS)
        return modified_media

    @classmethod
    def add_amino_acids(cls, media: Dict[str, float]) -> Dict[str, float]:
        """Add amino acid supplementation to media"""
        modified_media = media.copy()
        modified_media.update(cls.AMINO_ACIDS)
        return modified_media

    @classmethod
    def change_carbon_source(
        cls, media: Dict[str, float], new_carbon_source: str, uptake_rate: float = -10.0
    ) -> Dict[str, float]:
        """Change the primary carbon source"""
        modified_media = media.copy()

        # Remove existing carbon sources
        for carbon_id in cls.CARBON_SOURCES.keys():
            if carbon_id in modified_media:
                del modified_media[carbon_id]

        # Add new carbon source
        modified_media[new_carbon_source] = uptake_rate

        return modified_media

    @classmethod
    def interpret_ai_command(
        cls, command: str, media: Dict[str, float]
    ) -> Dict[str, float]:
        """Interpret natural language commands and modify media accordingly"""
        command = command.lower().strip()

        if "anaerobic" in command or "no oxygen" in command or "without o2" in command:
            return cls.make_anaerobic(media)
        elif "vitamin" in command:
            return cls.add_vitamins(media)
        elif "amino acid" in command:
            return cls.add_amino_acids(media)
        elif "pyruvate" in command and ("carbon" in command or "switch" in command):
            return cls.change_carbon_source(media, "cpd00020", -10.0)
        elif "glucose" in command and ("carbon" in command or "switch" in command):
            return cls.change_carbon_source(media, "cpd00027", -10.0)
        else:
            # Return original media if command not recognized
            return media


class MediaLibrary:
    """Comprehensive media library with AI-driven manipulation capabilities"""

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize media library with data directory"""
        if data_dir is None:
            # Default to project data/examples directory
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            data_dir = project_root / "data" / "examples"

        self.data_dir = Path(data_dir)
        self.compound_mapper = CompoundMapper()
        self.media_manager = MediaManager()
        self.ai_manipulator = AIMediaManipulator()

        # Core media library
        self._media_library: Dict[str, MediaComposition] = {}
        self._initialize_core_library()

    def _initialize_core_library(self):
        """Initialize the core media library with 5 required media types"""

        # 1. GMM (Glucose Minimal Media)
        gmm_media = self._load_media_from_file("GMM.tsv")
        self._media_library["GMM"] = MediaComposition(
            name="GMM",
            description="Glucose Minimal Media - minimal components for growth with glucose",
            compounds=gmm_media,
            conditions={"aerobic": True, "carbon_source": "glucose"},
        )

        # 2. GMM without O2 (Anaerobic GMM)
        gmm_anaerobic = self.ai_manipulator.make_anaerobic(gmm_media)
        self._media_library["GMM_anaerobic"] = MediaComposition(
            name="GMM_anaerobic",
            description="Glucose Minimal Media without O2 - anaerobic growth with alternative electron acceptors",
            compounds=gmm_anaerobic,
            conditions={"aerobic": False, "carbon_source": "glucose"},
        )

        # 3. AuxoMedia (Rich media)
        auxo_media = self._load_media_from_file("AuxoMedia.tsv")
        self._media_library["AuxoMedia"] = MediaComposition(
            name="AuxoMedia",
            description="Rich media with glucose, amino acids, and vitamins",
            compounds=auxo_media,
            conditions={"aerobic": True, "carbon_source": "glucose", "rich": True},
        )

        # 4. PyruvateMinimalMedia
        pyruvate_media = self.ai_manipulator.change_carbon_source(
            gmm_media, "cpd00020", -10.0
        )
        self._media_library["PyruvateMinimalMedia"] = MediaComposition(
            name="PyruvateMinimalMedia",
            description="Minimal media with pyruvate as carbon source",
            compounds=pyruvate_media,
            conditions={"aerobic": True, "carbon_source": "pyruvate"},
        )

        # 5. PyruvateMinimalMedia without O2
        pyruvate_anaerobic = self.ai_manipulator.make_anaerobic(pyruvate_media)
        self._media_library["PyruvateMinimalMedia_anaerobic"] = MediaComposition(
            name="PyruvateMinimalMedia_anaerobic",
            description="Minimal media with pyruvate as carbon source - anaerobic",
            compounds=pyruvate_anaerobic,
            conditions={"aerobic": False, "carbon_source": "pyruvate"},
        )

    def _load_media_from_file(self, filename: str) -> Dict[str, float]:
        """Load media from TSV or JSON file"""
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Media file not found: {file_path}")

        if filename.endswith(".tsv"):
            # Load TSV format
            df = pd.read_csv(file_path, sep="\t")

            # Handle different TSV formats
            if "compound" in df.columns and "uptake_rate" in df.columns:
                return dict(zip(df["compound"], df["uptake_rate"]))
            elif "compound_id" in df.columns and "flux" in df.columns:
                return dict(zip(df["compound_id"], df["flux"]))
            elif "compounds" in df.columns and "minFlux" in df.columns:
                # ModelSEED TSV format with compounds and minFlux
                # Use maxFlux for uptake limits (positive values become negative for uptake)
                if "maxFlux" in df.columns:
                    # Use maxFlux as the uptake limit (convert positive to negative)
                    media_dict = {}
                    for _, row in df.iterrows():
                        compound = row["compounds"]
                        max_flux = row["maxFlux"]
                        # Convert positive maxFlux to negative uptake rate
                        uptake_rate = -abs(max_flux) if max_flux > 0 else max_flux
                        media_dict[compound] = uptake_rate
                    return media_dict
                else:
                    return dict(zip(df["compounds"], df["minFlux"]))
            elif len(df.columns) >= 2:
                # Default: first column is compound, check for numeric columns
                compound_col = df.columns[0]

                # Look for a numeric column to use as uptake rate
                for col in df.columns[1:]:
                    try:
                        # Try to convert to numeric, use first numeric column found
                        numeric_vals = pd.to_numeric(df[col], errors="coerce")
                        if not numeric_vals.isna().all():
                            return dict(zip(df[compound_col], numeric_vals))
                    except:
                        continue

                # If no numeric column found, assume second column and try to convert
                return dict(
                    zip(
                        df.iloc[:, 0],
                        pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(-10.0),
                    )
                )
            else:
                raise ValueError(f"Unsupported TSV format in {filename}")

        elif filename.endswith(".json"):
            # Load JSON format
            with open(file_path) as f:
                return json.load(f)

        else:
            raise ValueError(f"Unsupported media file format: {filename}")

    def get_media(self, media_name: str) -> MediaComposition:
        """Get media composition by name"""
        if media_name not in self._media_library:
            available = list(self._media_library.keys())
            raise ValueError(f"Media '{media_name}' not found. Available: {available}")

        return self._media_library[media_name]

    def list_available_media(self) -> List[str]:
        """List all available media in the library"""
        return list(self._media_library.keys())

    def add_custom_media(self, media_comp: MediaComposition):
        """Add custom media to the library"""
        self._media_library[media_comp.name] = media_comp

    def manipulate_media(self, media_name: str, ai_command: str) -> MediaComposition:
        """Use AI to manipulate existing media based on natural language command"""
        base_media = self.get_media(media_name)
        modified_compounds = self.ai_manipulator.interpret_ai_command(
            ai_command, base_media.compounds
        )

        # Create new media composition
        modified_name = f"{media_name}_{ai_command.replace(' ', '_')}"
        modified_description = f"{base_media.description} (Modified: {ai_command})"

        return MediaComposition(
            name=modified_name,
            description=modified_description,
            compounds=modified_compounds,
            conditions=base_media.conditions.copy(),
            format_type=base_media.format_type,
        )

    def apply_media_to_model(
        self,
        model: cobra.Model,
        media_name: str,
        ai_modifications: Optional[str] = None,
    ) -> cobra.Model:
        """Apply media to a COBRApy model with optional AI modifications"""

        # Get base media
        if ai_modifications:
            media_comp = self.manipulate_media(media_name, ai_modifications)
        else:
            media_comp = self.get_media(media_name)

        # Apply media using the MediaManager
        return self.media_manager.apply_media_to_model(model, media_comp.compounds)

    def get_media_for_modelseedpy(
        self, media_name: str, ai_modifications: Optional[str] = None
    ) -> Dict[str, float]:
        """Get media composition formatted for ModelSEEDpy models"""

        if ai_modifications:
            media_comp = self.manipulate_media(media_name, ai_modifications)
        else:
            media_comp = self.get_media(media_name)

        return media_comp.compounds

    def save_media_to_file(
        self, media_comp: MediaComposition, output_path: str, format_type: str = "tsv"
    ):
        """Save media composition to file"""

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format_type.lower() == "tsv":
            # Save as TSV
            df = pd.DataFrame(
                [
                    {"compound": compound, "uptake_rate": rate}
                    for compound, rate in media_comp.compounds.items()
                ]
            )
            df.to_csv(output_file, sep="\t", index=False)

        elif format_type.lower() == "json":
            # Save as JSON
            media_data = {
                "name": media_comp.name,
                "description": media_comp.description,
                "compounds": media_comp.compounds,
                "conditions": media_comp.conditions,
                "format_type": media_comp.format_type,
            }
            with open(output_file, "w") as f:
                json.dump(media_data, f, indent=2)

        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def analyze_media_compatibility(
        self, model: cobra.Model, media_name: str
    ) -> Dict[str, Any]:
        """Analyze media compatibility with a given model"""

        media_comp = self.get_media(media_name)

        # Detect model format
        model_format = (
            "bigg" if any("_" in rxn.id for rxn in model.exchanges[:5]) else "modelseed"
        )

        # Check compound mapping
        mapped_compounds = 0
        unmapped_compounds = []

        for compound in media_comp.compounds:
            if model_format == "bigg" and compound.startswith("cpd"):
                # Need to map ModelSEED to BIGG
                bigg_id = self.compound_mapper.MODELSEED_TO_BIGG.get(compound)
                if bigg_id:
                    mapped_compounds += 1
                else:
                    unmapped_compounds.append(compound)
            elif model_format == "modelseed" and not compound.startswith("cpd"):
                # Need to map BIGG to ModelSEED
                modelseed_id = self.compound_mapper.BIGG_TO_MODELSEED.get(compound)
                if modelseed_id:
                    mapped_compounds += 1
                else:
                    unmapped_compounds.append(compound)
            else:
                # Direct compatibility
                mapped_compounds += 1

        compatibility_score = mapped_compounds / len(media_comp.compounds)

        return {
            "model_format": model_format,
            "media_format": media_comp.format_type,
            "total_compounds": len(media_comp.compounds),
            "mapped_compounds": mapped_compounds,
            "unmapped_compounds": unmapped_compounds,
            "compatibility_score": compatibility_score,
            "recommended_action": (
                "Direct application"
                if compatibility_score > 0.9
                else (
                    "Requires compound mapping"
                    if compatibility_score > 0.5
                    else "Poor compatibility - consider alternative media"
                )
            ),
        }


# Convenience function for external use
def get_media_library() -> MediaLibrary:
    """Get a configured media library instance"""
    return MediaLibrary()


# Example usage and testing
if __name__ == "__main__":
    # Initialize media library
    library = MediaLibrary()

    # List available media
    print("Available media:")
    for media_name in library.list_available_media():
        media = library.get_media(media_name)
        print(f"  {media_name}: {media.description}")

    # Test AI manipulation
    print("\nTesting AI manipulation:")
    anaerobic_gmm = library.manipulate_media("GMM", "make anaerobic")
    print(f"Anaerobic GMM compounds: {len(anaerobic_gmm.compounds)}")

    vitamin_gmm = library.manipulate_media("GMM", "add vitamins")
    print(f"Vitamin-supplemented GMM compounds: {len(vitamin_gmm.compounds)}")
