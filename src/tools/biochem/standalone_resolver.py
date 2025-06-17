#!/usr/bin/env python3
"""
Standalone Biochemistry Resolution Functions - Pure ModelSEEDpy Implementation

This module provides biochemistry resolution without LangChain dependencies
for immediate use and testing, using the official ModelSEED database.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

# ModelSEEDpy is required for this module
try:
    import modelseedpy.biochem

    MODELSEEDPY_AVAILABLE = True
except ImportError:
    MODELSEEDPY_AVAILABLE = False


class BiochemDatabase:
    """Standalone biochemistry database helper using ModelSEEDpy"""

    def __init__(self):
        if not MODELSEEDPY_AVAILABLE:
            raise ImportError(
                "modelseedpy is required for biochemical tools. "
                "Install with: pip install modelseedpy"
            )
        self._database = None

    def _ensure_database_setup(self) -> bool:
        """Ensure ModelSEED database is available, set up if needed"""
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

    def _get_database(self):
        """Get ModelSEED database instance"""
        if self._database is None:
            self._ensure_database_setup()

            base_dir = Path(__file__).parent.parent.parent.parent
            database_dir = base_dir / "data" / "ModelSEEDDatabase"

            try:
                self._database = modelseedpy.biochem.from_local(str(database_dir))
            except Exception as e:
                # Fallback to GitHub method
                try:
                    print("🌐 Falling back to GitHub database access...")
                    self._database = modelseedpy.biochem.from_github("dev")
                except Exception as github_error:
                    raise RuntimeError(
                        f"Failed to load ModelSEED database locally ({e}) "
                        f"and from GitHub ({github_error}). "
                        f"Please run: python scripts/setup_biochem_database.py"
                    )

        return self._database

    def _clean_id(self, entity_id: str, remove_compartments: bool = True) -> str:
        """Clean ID by removing compartments if requested"""
        if not remove_compartments:
            return entity_id

        # Remove common compartment suffixes
        compartments = ["_c", "_e", "_p", "_m", "_x", "_r", "_v", "_g", "_h", "_n"]
        for comp in compartments:
            if entity_id.endswith(comp):
                return entity_id[: -len(comp)]

        return entity_id

    def _find_entity_by_alias(self, entity_id: str, entity_type: str):
        """Find entity by searching through aliases"""
        database = self._get_database()
        cleaned_id = self._clean_id(entity_id, True)

        if entity_type == "compound":
            # Search through compounds
            for cpd in database.compounds:
                # Direct ID match
                if cpd.id == entity_id or cpd.id == cleaned_id:
                    return cpd

                # Alias search in annotation
                if hasattr(cpd, "annotation") and cpd.annotation:
                    for alias_type, alias_values in cpd.annotation.items():
                        if isinstance(alias_values, str):
                            if alias_values == entity_id or alias_values == cleaned_id:
                                return cpd
                        elif isinstance(alias_values, set):
                            if entity_id in alias_values or cleaned_id in alias_values:
                                return cpd

        elif entity_type == "reaction":
            # Search through reactions
            for rxn in database.reactions:
                # Direct ID match
                if rxn.id == entity_id or rxn.id == cleaned_id:
                    return rxn

                # Alias search in annotation
                if hasattr(rxn, "annotation") and rxn.annotation:
                    for alias_type, alias_values in rxn.annotation.items():
                        if isinstance(alias_values, str):
                            if alias_values == entity_id or alias_values == cleaned_id:
                                return rxn
                        elif isinstance(alias_values, set):
                            if entity_id in alias_values or cleaned_id in alias_values:
                                return rxn

        return None

    def _detect_entity_type(self, entity_id: str) -> str:
        """Detect entity type from ID pattern"""
        entity_id = entity_id.strip()

        # ModelSEED pattern detection
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
        if "__" in entity_id or "_" in entity_id:
            return "compound"

        # Default to compound for most cases
        return "compound"

    def resolve_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Resolve entity ID to information"""
        # Detect entity type
        entity_type = self._detect_entity_type(entity_id)

        # Find entity
        entity = self._find_entity_by_alias(entity_id, entity_type)

        if not entity:
            # Try the other type
            other_type = "reaction" if entity_type == "compound" else "compound"
            entity = self._find_entity_by_alias(entity_id, other_type)
            if entity:
                entity_type = other_type

        if entity:
            result = {
                "id": entity.id,
                "name": getattr(entity, "name", entity.id),
                "type": entity_type,
                "source_db": "ModelSEED",
            }

            if entity_type == "compound":
                result.update(
                    {
                        "formula": getattr(entity, "formula", None),
                        "charge": getattr(entity, "charge", None),
                        "mass": getattr(entity, "mass", None),
                    }
                )
            elif entity_type == "reaction":
                result.update(
                    {
                        "equation": getattr(entity, "equation", None),
                        "direction": getattr(entity, "direction", None),
                    }
                )

            # Add cross-references
            cross_refs = []
            if hasattr(entity, "annotation") and entity.annotation:
                for alias_type, alias_values in entity.annotation.items():
                    if isinstance(alias_values, str):
                        cross_refs.append(f"{alias_type}:{alias_values}")
                    elif isinstance(alias_values, set):
                        for alias in alias_values:
                            cross_refs.append(f"{alias_type}:{alias}")

            result["cross_refs"] = cross_refs
            return result

        return None

    def search_entities(
        self, query: str, entity_type: Optional[str] = None, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for entities by name or keyword"""
        database = self._get_database()
        results = []
        query_lower = query.lower()

        # Search compounds
        if entity_type is None or entity_type == "compound":
            for cpd in database.compounds:
                match_score = 0

                # Check ID match
                if cpd.id.lower() == query_lower:
                    match_score = 100
                elif query_lower in cpd.id.lower():
                    match_score = 80

                # Check name match
                name = getattr(cpd, "name", "")
                if name:
                    if name.lower() == query_lower:
                        match_score = max(match_score, 95)
                    elif query_lower in name.lower():
                        match_score = max(match_score, 70)

                # Check formula match
                formula = getattr(cpd, "formula", "")
                if formula and (
                    formula.lower() == query_lower or query_lower in formula.lower()
                ):
                    match_score = max(match_score, 60)

                # Check aliases
                if hasattr(cpd, "annotation") and cpd.annotation:
                    for alias_type, alias_values in cpd.annotation.items():
                        if isinstance(alias_values, str):
                            if alias_values.lower() == query_lower:
                                match_score = max(match_score, 85)
                            elif query_lower in alias_values.lower():
                                match_score = max(match_score, 60)
                        elif isinstance(alias_values, set):
                            for alias in alias_values:
                                if alias.lower() == query_lower:
                                    match_score = max(match_score, 85)
                                elif query_lower in alias.lower():
                                    match_score = max(match_score, 60)

                if match_score > 0:
                    # Add cross-references
                    cross_refs = []
                    if hasattr(cpd, "annotation") and cpd.annotation:
                        for alias_type, alias_values in cpd.annotation.items():
                            if isinstance(alias_values, str):
                                cross_refs.append(f"{alias_type}:{alias_values}")
                            elif isinstance(alias_values, set):
                                for alias in alias_values:
                                    cross_refs.append(f"{alias_type}:{alias}")

                    result = {
                        "id": cpd.id,
                        "name": getattr(cpd, "name", cpd.id),
                        "type": "compound",
                        "formula": getattr(cpd, "formula", None),
                        "charge": getattr(cpd, "charge", None),
                        "mass": getattr(cpd, "mass", None),
                        "source_db": "ModelSEED",
                        "cross_refs": cross_refs,
                        "match_score": match_score,
                    }
                    results.append(result)

                if len(results) >= max_results * 2:  # Get extra for sorting
                    break

        # Search reactions
        if entity_type is None or entity_type == "reaction":
            remaining = max_results - len(
                [r for r in results if r["type"] == "compound"]
            )
            if remaining > 0:
                for rxn in database.reactions:
                    match_score = 0

                    # Check ID match
                    if rxn.id.lower() == query_lower:
                        match_score = 100
                    elif query_lower in rxn.id.lower():
                        match_score = 80

                    # Check name match
                    name = getattr(rxn, "name", "")
                    if name:
                        if name.lower() == query_lower:
                            match_score = max(match_score, 95)
                        elif query_lower in name.lower():
                            match_score = max(match_score, 70)

                    # Check equation match
                    equation = getattr(rxn, "equation", "")
                    if equation and query_lower in equation.lower():
                        match_score = max(match_score, 60)

                    # Check aliases
                    if hasattr(rxn, "annotation") and rxn.annotation:
                        for alias_type, alias_values in rxn.annotation.items():
                            if isinstance(alias_values, str):
                                if alias_values.lower() == query_lower:
                                    match_score = max(match_score, 85)
                                elif query_lower in alias_values.lower():
                                    match_score = max(match_score, 60)
                            elif isinstance(alias_values, set):
                                for alias in alias_values:
                                    if alias.lower() == query_lower:
                                        match_score = max(match_score, 85)
                                    elif query_lower in alias.lower():
                                        match_score = max(match_score, 60)

                    if match_score > 0:
                        # Add cross-references
                        cross_refs = []
                        if hasattr(rxn, "annotation") and rxn.annotation:
                            for alias_type, alias_values in rxn.annotation.items():
                                if isinstance(alias_values, str):
                                    cross_refs.append(f"{alias_type}:{alias_values}")
                                elif isinstance(alias_values, set):
                                    for alias in alias_values:
                                        cross_refs.append(f"{alias_type}:{alias}")

                        result = {
                            "id": rxn.id,
                            "name": getattr(rxn, "name", rxn.id),
                            "type": "reaction",
                            "equation": getattr(rxn, "equation", None),
                            "direction": getattr(rxn, "direction", None),
                            "source_db": "ModelSEED",
                            "cross_refs": cross_refs,
                            "match_score": match_score,
                        }
                        results.append(result)

                    if (
                        len([r for r in results if r["type"] == "reaction"])
                        >= remaining
                    ):
                        break

        # Sort by match score and return top results
        results.sort(key=lambda x: x["match_score"], reverse=True)

        # Remove match_score from final results
        for result in results:
            result.pop("match_score", None)

        return results[:max_results]


def resolve_entity_id(entity_id: str) -> Optional[str]:
    """Quick resolve entity ID to human-readable name"""
    try:
        db = BiochemDatabase()
        entity = db.resolve_entity(entity_id)
        return entity["name"] if entity else None
    except Exception:
        return None


def enhance_tool_output_with_names(tool_output: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance tool output by adding human-readable names for biochemistry IDs"""
    try:
        db = BiochemDatabase()
        enhanced_output = tool_output.copy()

        # Recursively search for entity IDs and enhance them
        def enhance_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and (
                        value.startswith(("cpd", "rxn"))
                        or "cpd" in key.lower()
                        or "rxn" in key.lower()
                        or "reaction" in key.lower()
                        or "compound" in key.lower()
                        or "metabolite" in key.lower()
                    ):
                        # Try to resolve this as an entity ID
                        entity = db.resolve_entity(value)
                        if entity:
                            # Add human-readable name alongside the ID
                            name_key = f"{key}_name"
                            obj[name_key] = entity["name"]
                    elif isinstance(value, list):
                        # Handle lists of entity IDs
                        names = []
                        for item in value:
                            if isinstance(item, str) and item.startswith(
                                ("cpd", "rxn")
                            ):
                                entity = db.resolve_entity(item)
                                if entity:
                                    names.append(entity["name"])
                        if names and len(names) == len(value):
                            name_key = (
                                f"{key}_names" if len(names) > 1 else f"{key}_name"
                            )
                            obj[name_key] = names if len(names) > 1 else names[0]
                    else:
                        enhance_recursive(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    enhance_recursive(item, f"{path}[{i}]")

        enhance_recursive(enhanced_output)
        return enhanced_output

    except Exception:
        # If enhancement fails, return original output
        return tool_output


def search_biochem_entities(
    query: str,
    entity_type: Optional[str] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Search biochemistry database"""
    try:
        db = BiochemDatabase()
        return db.search_entities(query, entity_type, max_results)
    except Exception:
        return []
