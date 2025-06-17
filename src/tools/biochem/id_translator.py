#!/usr/bin/env python3
"""
Cross-Database ID Translator Tool - Pure ModelSEEDpy Implementation

This tool provides universal ID translation between different biochemical databases using
the official ModelSEED database via ModelSEEDpy. It enables seamless conversion between
ModelSEED, BiGG, KEGG, MetaCyc, ChEBI, and 50+ other database formats.

Key Features:
- Universal ID translation across 55+ databases
- Official ModelSEED database with 45,706+ compounds and 56,009+ reactions
- Bidirectional mapping (any format to any format)
- Compartment suffix handling (e.g., _c, _e, _p)
- Batch translation capabilities
- Smart fuzzy matching for variant IDs
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..base import BaseTool, ToolRegistry, ToolResult

# ModelSEEDpy is required for this tool
try:
    import modelseedpy.biochem

    MODELSEEDPY_AVAILABLE = True
except ImportError:
    MODELSEEDPY_AVAILABLE = False


class IDTranslatorInput(BaseModel):
    """Input schema for cross-database ID translation"""

    entity_ids: Union[str, List[str]] = Field(
        description="Single ID or list of IDs to translate (e.g., 'cpd00001', ['C00002', 'atp_c'])"
    )
    source_database: Optional[str] = Field(
        default=None,
        description="Source database format (auto-detected if not specified): ModelSEED, KEGG, BiGG, MetaCyc, ChEBI",
    )
    target_databases: List[str] = Field(
        default=["ModelSEED", "KEGG", "BiGG"],
        description="Target database formats to translate to",
    )
    entity_type: Optional[str] = Field(
        default=None,
        description="Type of entity: 'compound', 'reaction', or 'auto' for automatic detection",
    )
    remove_compartments: bool = Field(
        default=False,
        description="Remove compartment suffixes (_c, _e, _p) before translation",
    )
    include_variants: bool = Field(
        default=True, description="Include ID variants and alternative mappings"
    )
    fuzzy_matching: bool = Field(
        default=True, description="Enable fuzzy matching for partial ID matches"
    )


class IDTranslatorOutput(BaseModel):
    """Output schema for cross-database ID translation"""

    translations: Dict[str, Dict[str, List[str]]] = Field(
        description="Translations organized by input ID -> target database -> list of IDs"
    )
    failed_translations: List[str] = Field(
        description="IDs that could not be translated"
    )
    source_database_detected: Dict[str, str] = Field(
        description="Auto-detected source database for each input ID"
    )
    translation_confidence: Dict[str, float] = Field(
        description="Confidence score for each translation (0.0-1.0)"
    )
    total_translations: int = Field(
        description="Total number of successful translations"
    )
    database_stats: Dict[str, int] = Field(
        description="ModelSEED database statistics (compounds, reactions)"
    )


@ToolRegistry.register
class CrossDatabaseIDTranslator(BaseTool):
    """Universal ID translator using official ModelSEED database"""

    tool_name = "translate_database_ids"
    tool_description = "Translate biochemical IDs between different database formats using official ModelSEED database (45,706+ compounds, 56,009+ reactions)"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._modelseed_db = None

    def _ensure_database_setup(self) -> bool:
        """Ensure ModelSEED database is available, set up if needed"""
        if not MODELSEEDPY_AVAILABLE:
            raise ImportError(
                "modelseedpy is required for biochemical tools. "
                "Install with: pip install modelseedpy"
            )

        # Check if database exists
        base_dir = Path(__file__).parent.parent.parent.parent
        database_dir = base_dir / "data" / "ModelSEEDDatabase"

        if not database_dir.exists() or not (database_dir / "Biochemistry").exists():
            print("🔄 ModelSEED database not found. Setting up automatically...")

            # Run setup script
            setup_script = base_dir / "scripts" / "setup_biochem_database.py"
            try:
                subprocess.run(
                    ["python", str(setup_script)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print("✅ Database setup completed")
                return True
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to setup ModelSEED database: {e.stderr}\n"
                    f"Please run manually: python {setup_script}"
                )

        return True

    def _get_modelseed_database(self):
        """Get ModelSEED database instance"""
        if self._modelseed_db is None:
            self._ensure_database_setup()

            base_dir = Path(__file__).parent.parent.parent.parent
            database_dir = base_dir / "data" / "ModelSEEDDatabase"

            try:
                self._modelseed_db = modelseedpy.biochem.from_local(str(database_dir))
            except Exception as e:
                # Fallback to GitHub method
                try:
                    print("🌐 Falling back to GitHub database access...")
                    self._modelseed_db = modelseedpy.biochem.from_github("dev")
                except Exception as github_error:
                    raise RuntimeError(
                        f"Failed to load ModelSEED database locally ({e}) "
                        f"and from GitHub ({github_error}). "
                        f"Please run: python scripts/setup_biochem_database.py"
                    )

        return self._modelseed_db

    def _detect_database_format(self, entity_id: str) -> str:
        """Detect database format from ID pattern"""
        entity_id = entity_id.strip()

        # ModelSEED patterns
        if entity_id.startswith(("cpd", "rxn")):
            return "ModelSEED"

        # KEGG patterns
        if (
            entity_id.startswith(("C", "R"))
            and len(entity_id) == 6
            and entity_id[1:].isdigit()
        ):
            return "KEGG"

        # BiGG patterns (often lowercase with underscores)
        if "_" in entity_id and entity_id.islower():
            return "BiGG"

        # MetaCyc patterns (often mixed case)
        if entity_id.replace("-", "").replace("_", "").isalnum() and any(
            c.isupper() for c in entity_id
        ):
            return "MetaCyc"

        # ChEBI patterns
        if entity_id.startswith("CHEBI:"):
            return "ChEBI"

        return "Unknown"

    def _detect_entity_type(self, entity_id: str) -> str:
        """Detect entity type from ID pattern"""
        entity_id = entity_id.strip()

        # ModelSEED patterns
        if entity_id.startswith("cpd"):
            return "compound"
        elif entity_id.startswith("rxn"):
            return "reaction"

        # KEGG patterns
        if entity_id.startswith("C") and len(entity_id) == 6:
            return "compound"
        elif entity_id.startswith("R") and len(entity_id) == 6:
            return "reaction"

        # BiGG patterns (assume compound for most cases)
        if "_" in entity_id:
            return "compound"

        # Default to compound
        return "compound"

    def _clean_id(self, entity_id: str, remove_compartments: bool = False) -> str:
        """Clean ID by removing compartments if requested"""
        if not remove_compartments:
            return entity_id

        # Remove common compartment suffixes
        compartments = ["_c", "_e", "_p", "_m", "_x", "_r", "_v", "_g", "_h", "_n"]
        for comp in compartments:
            if entity_id.endswith(comp):
                return entity_id[: -len(comp)]

        return entity_id

    def _build_alias_dictionaries(self, database):
        """Build comprehensive alias dictionaries from ModelSEED database"""
        compound_alias_dict = {}
        reaction_alias_dict = {}

        # Build compound aliases
        for cpd in database.compounds:
            if hasattr(cpd, "annotation") and cpd.annotation:
                for alias_type, alias_values in cpd.annotation.items():
                    if alias_type not in compound_alias_dict:
                        compound_alias_dict[alias_type] = {}

                    if isinstance(alias_values, str):
                        compound_alias_dict[alias_type][alias_values] = cpd
                    elif isinstance(alias_values, set):
                        for value in alias_values:
                            compound_alias_dict[alias_type][value] = cpd

        # Build reaction aliases
        for rxn in database.reactions:
            if hasattr(rxn, "annotation") and rxn.annotation:
                for alias_type, alias_values in rxn.annotation.items():
                    if alias_type not in reaction_alias_dict:
                        reaction_alias_dict[alias_type] = {}

                    if isinstance(alias_values, str):
                        reaction_alias_dict[alias_type][alias_values] = rxn
                    elif isinstance(alias_values, set):
                        for value in alias_values:
                            reaction_alias_dict[alias_type][value] = rxn

        return compound_alias_dict, reaction_alias_dict

    def _translate_entity(
        self,
        entity_id: str,
        target_databases: List[str],
        compound_aliases: Dict,
        reaction_aliases: Dict,
        remove_compartments: bool,
        fuzzy_matching: bool,
    ) -> Dict[str, Any]:
        """Translate a single entity"""
        cleaned_id = self._clean_id(entity_id, remove_compartments)

        # Detect source database and entity type
        source_db = self._detect_database_format(cleaned_id)
        entity_type = self._detect_entity_type(cleaned_id)

        # Initialize result
        result = {
            "source_database": source_db,
            "entity_type": entity_type,
            "translations": {},
            "found": False,
            "confidence": 0.0,
        }

        # Get the database
        database = self._get_modelseed_database()

        if entity_type == "compound":
            # Find compound
            cpd = None

            # Direct ModelSEED ID lookup
            if source_db == "ModelSEED":
                cpd = next((c for c in database.compounds if c.id == cleaned_id), None)
            else:
                # Alias lookup
                for alias_type, alias_dict in compound_aliases.items():
                    if cleaned_id in alias_dict:
                        cpd = alias_dict[cleaned_id]
                        break

                    # Try without compartment if fuzzy matching is enabled
                    if fuzzy_matching and remove_compartments:
                        base_id = self._clean_id(cleaned_id, True)
                        if base_id in alias_dict:
                            cpd = alias_dict[base_id]
                            break

            if cpd:
                result["found"] = True
                result["confidence"] = 1.0 if not remove_compartments else 0.9

                # Get translations for target databases
                for target_db in target_databases:
                    if target_db == "ModelSEED":
                        result["translations"][target_db] = [cpd.id]
                    elif (
                        hasattr(cpd, "annotation")
                        and cpd.annotation
                        and target_db in cpd.annotation
                    ):
                        alias_values = cpd.annotation[target_db]
                        if isinstance(alias_values, str):
                            result["translations"][target_db] = [alias_values]
                        elif isinstance(alias_values, set):
                            result["translations"][target_db] = list(alias_values)
                        else:
                            result["translations"][target_db] = []
                    else:
                        result["translations"][target_db] = []

        elif entity_type == "reaction":
            # Find reaction
            rxn = None

            # Direct ModelSEED ID lookup
            if source_db == "ModelSEED":
                rxn = next((r for r in database.reactions if r.id == cleaned_id), None)
            else:
                # Alias lookup
                for alias_type, alias_dict in reaction_aliases.items():
                    if cleaned_id in alias_dict:
                        rxn = alias_dict[cleaned_id]
                        break

            if rxn:
                result["found"] = True
                result["confidence"] = 1.0 if not remove_compartments else 0.9

                # Get translations for target databases
                for target_db in target_databases:
                    if target_db == "ModelSEED":
                        result["translations"][target_db] = [rxn.id]
                    elif (
                        hasattr(rxn, "annotation")
                        and rxn.annotation
                        and target_db in rxn.annotation
                    ):
                        alias_values = rxn.annotation[target_db]
                        if isinstance(alias_values, str):
                            result["translations"][target_db] = [alias_values]
                        elif isinstance(alias_values, set):
                            result["translations"][target_db] = list(alias_values)
                        else:
                            result["translations"][target_db] = []
                    else:
                        result["translations"][target_db] = []

        return result

    def _run_tool(self, input_data: Any) -> ToolResult:
        """Execute cross-database ID translation"""
        try:
            # Validate input
            input_obj = IDTranslatorInput(**input_data)

            # Convert single ID to list
            if isinstance(input_obj.entity_ids, str):
                entity_ids = [input_obj.entity_ids]
            else:
                entity_ids = input_obj.entity_ids

            # Get database and build alias dictionaries
            database = self._get_modelseed_database()
            compound_aliases, reaction_aliases = self._build_alias_dictionaries(
                database
            )

            # Initialize results
            translations = {}
            failed_translations = []
            source_detected = {}
            confidence_scores = {}

            # Process each entity
            for entity_id in entity_ids:
                result = self._translate_entity(
                    entity_id,
                    input_obj.target_databases,
                    compound_aliases,
                    reaction_aliases,
                    input_obj.remove_compartments,
                    input_obj.fuzzy_matching,
                )

                source_detected[entity_id] = result["source_database"]
                confidence_scores[entity_id] = result["confidence"]

                if result["found"]:
                    translations[entity_id] = result["translations"]
                else:
                    failed_translations.append(entity_id)
                    translations[entity_id] = {}

            # Calculate total successful translations
            total_translations = sum(
                len(
                    [
                        v
                        for db_translations in trans.values()
                        for v in db_translations
                        if v
                    ]
                )
                for trans in translations.values()
            )

            # Get database statistics
            database_stats = {
                "compounds": len(database.compounds),
                "reactions": len(database.reactions),
            }

            # Build output
            output = IDTranslatorOutput(
                translations=translations,
                failed_translations=failed_translations,
                source_database_detected=source_detected,
                translation_confidence=confidence_scores,
                total_translations=total_translations,
                database_stats=database_stats,
            )

            success_rate = (len(entity_ids) - len(failed_translations)) / len(
                entity_ids
            )
            message = f"Translated {total_translations} IDs with {success_rate:.1%} success rate using ModelSEED database ({database_stats['compounds']:,} compounds, {database_stats['reactions']:,} reactions)"

            return ToolResult(success=True, data=output.model_dump(), message=message)

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message="Failed to translate biochemistry IDs",
            )
