#!/usr/bin/env python3
"""
Biochemistry Entity Resolution Tools
This module provides tools for resolving biochemistry entity IDs to human-readable
names and searching the biochemistry database. These tools enable the AI agent
to reason about "Phosphoglycerate mutase" instead of "rxn10271".

Phase 3 Implementation:
- Universal ID resolution across ModelSEED, BiGG, and KEGG
- Human-readable name mapping for enhanced reasoning
- Search capabilities for compound and reaction discovery
"""

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ...tools.base import BaseTool, ToolRegistry, ToolResult


class BiochemEntityResolverInput(BaseModel):
    """Input schema for biochemistry entity resolution"""

    entity_id: str = Field(
        description="The biochemistry entity ID to resolve (e.g., 'cpd00001', 'rxn00001', 'ATP', 'glc__D')"
    )
    entity_type: Optional[str] = Field(
        default=None,
        description="Type of entity: 'compound', 'reaction', or 'auto' for automatic detection",
    )
    include_aliases: bool = Field(
        default=True, description="Whether to include alias mappings in the results"
    )
    max_results: int = Field(
        default=10, description="Maximum number of results to return"
    )


class BiochemEntityResolverOutput(BaseModel):
    """Output schema for biochemistry entity resolution"""

    entity_id: str = Field(description="The input entity ID")
    entity_type: str = Field(description="Detected entity type")
    resolved: bool = Field(description="Whether the entity was successfully resolved")
    primary_id: Optional[str] = Field(description="Primary ModelSEED ID if resolved")
    primary_name: Optional[str] = Field(description="Primary human-readable name")
    aliases: List[Dict[str, str]] = Field(description="List of alias mappings")
    names: List[Dict[str, str]] = Field(description="List of alternative names")
    suggestions: List[str] = Field(
        description="Suggested similar entities if not resolved"
    )


class BiochemSearchInput(BaseModel):
    """Input schema for biochemistry database search"""

    query: str = Field(description="Search query (name, alias, or partial match)")
    entity_type: Optional[str] = Field(
        default=None,
        description="Type of entity to search: 'compound', 'reaction', or 'both'",
    )
    source_filter: Optional[str] = Field(
        default=None,
        description="Filter by source database (e.g., 'BiGG', 'KEGG', 'MetaCyc')",
    )
    max_results: int = Field(
        default=20, description="Maximum number of results to return"
    )


class BiochemSearchOutput(BaseModel):
    """Output schema for biochemistry database search"""

    query: str = Field(description="The search query")
    total_results: int = Field(description="Total number of matches found")
    compounds: List[Dict[str, Any]] = Field(description="Matching compounds")
    reactions: List[Dict[str, Any]] = Field(description="Matching reactions")


@ToolRegistry.register
class BiochemEntityResolverTool(BaseTool):
    """Tool for resolving biochemistry entity IDs to human-readable information"""

    tool_name = "resolve_biochem_entity"
    tool_description = "Resolve biochemistry entity IDs (compounds/reactions) to human-readable names and aliases"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._db_path = None

    @property
    def db_path(self) -> Path:
        """Get the path to the biochemistry database"""
        if self._db_path is None:
            # Look for database in data directory
            base_dir = Path(__file__).parent.parent.parent.parent
            db_path = base_dir / "data" / "biochem.db"

            # Check for environment variable override
            env_db_path = os.getenv("BIOCHEM_DB_PATH")
            if env_db_path:
                db_path = Path(env_db_path)

            if not db_path.exists():
                raise FileNotFoundError(
                    f"Biochemistry database not found at {db_path}. "
                    "Please run 'python scripts/build_mvp_biochem_db.py' to create it, "
                    "or set BIOCHEM_DB_PATH environment variable to an existing database."
                )
            self._db_path = db_path
        return self._db_path

    def _detect_entity_type(self, entity_id: str) -> str:
        """Detect entity type from ID pattern"""
        entity_id = entity_id.strip()

        # ModelSEED pattern detection
        if entity_id.startswith("cpd"):
            return "compound"
        elif entity_id.startswith("rxn"):
            return "reaction"

        # BiGG patterns
        if "__" in entity_id:  # Common BiGG compound pattern
            return "compound"

        # Default to compound for most cases
        return "compound"

    def _resolve_compound(
        self,
        conn: sqlite3.Connection,
        entity_id: str,
        include_aliases: bool,
        max_results: int,
    ) -> Dict[str, Any]:
        """Resolve a compound entity"""
        cursor = conn.cursor()

        # Try direct ModelSEED ID lookup first
        cursor.execute(
            """
            SELECT modelseed_id, primary_name
            FROM compounds
            WHERE modelseed_id = ?
        """,
            (entity_id,),
        )

        result = cursor.fetchone()
        if result:
            return self._build_compound_result(
                conn, result[0], result[1], include_aliases
            )

        # Try alias lookup
        cursor.execute(
            """
            SELECT c.modelseed_id, c.primary_name, ca.source
            FROM compounds c
            JOIN compound_aliases ca ON c.modelseed_id = ca.modelseed_id
            WHERE ca.external_id = ?
            LIMIT ?
        """,
            (entity_id, max_results),
        )

        results = cursor.fetchall()
        if results:
            # Return first match (most relevant)
            return self._build_compound_result(
                conn, results[0][0], results[0][1], include_aliases
            )

        # Try name lookup
        cursor.execute(
            """
            SELECT c.modelseed_id, c.primary_name
            FROM compounds c
            JOIN compound_names cn ON c.modelseed_id = cn.modelseed_id
            WHERE cn.name LIKE ?
            LIMIT ?
        """,
            (f"%{entity_id}%", max_results),
        )

        results = cursor.fetchall()
        if results:
            return self._build_compound_result(
                conn, results[0][0], results[0][1], include_aliases
            )

        return {
            "resolved": False,
            "primary_id": None,
            "primary_name": None,
            "aliases": [],
            "names": [],
            "suggestions": self._get_compound_suggestions(conn, entity_id),
        }

    def _resolve_reaction(
        self,
        conn: sqlite3.Connection,
        entity_id: str,
        include_aliases: bool,
        max_results: int,
    ) -> Dict[str, Any]:
        """Resolve a reaction entity"""
        cursor = conn.cursor()

        # Try direct ModelSEED ID lookup first
        cursor.execute(
            """
            SELECT modelseed_id, primary_name
            FROM reactions
            WHERE modelseed_id = ?
        """,
            (entity_id,),
        )

        result = cursor.fetchone()
        if result:
            return self._build_reaction_result(
                conn, result[0], result[1], include_aliases
            )

        # Try alias lookup
        cursor.execute(
            """
            SELECT r.modelseed_id, r.primary_name, ra.source
            FROM reactions r
            JOIN reaction_aliases ra ON r.modelseed_id = ra.modelseed_id
            WHERE ra.external_id = ?
            LIMIT ?
        """,
            (entity_id, max_results),
        )

        results = cursor.fetchall()
        if results:
            return self._build_reaction_result(
                conn, results[0][0], results[0][1], include_aliases
            )

        return {
            "resolved": False,
            "primary_id": None,
            "primary_name": None,
            "aliases": [],
            "names": [],
            "suggestions": self._get_reaction_suggestions(conn, entity_id),
        }

    def _build_compound_result(
        self,
        conn: sqlite3.Connection,
        modelseed_id: str,
        primary_name: str,
        include_aliases: bool,
    ) -> Dict[str, Any]:
        """Build a complete compound resolution result"""
        cursor = conn.cursor()

        result = {
            "resolved": True,
            "primary_id": modelseed_id,
            "primary_name": primary_name,
            "aliases": [],
            "names": [],
        }

        if include_aliases:
            # Get aliases
            cursor.execute(
                """
                SELECT external_id, source
                FROM compound_aliases
                WHERE modelseed_id = ?
                ORDER BY source
                LIMIT 20
            """,
                (modelseed_id,),
            )

            result["aliases"] = [
                {"external_id": row[0], "source": row[1]} for row in cursor.fetchall()
            ]

            # Get names
            cursor.execute(
                """
                SELECT name, source
                FROM compound_names
                WHERE modelseed_id = ?
                ORDER BY source
                LIMIT 10
            """,
                (modelseed_id,),
            )

            result["names"] = [
                {"name": row[0], "source": row[1]} for row in cursor.fetchall()
            ]

        return result

    def _build_reaction_result(
        self,
        conn: sqlite3.Connection,
        modelseed_id: str,
        primary_name: str,
        include_aliases: bool,
    ) -> Dict[str, Any]:
        """Build a complete reaction resolution result"""
        cursor = conn.cursor()

        result = {
            "resolved": True,
            "primary_id": modelseed_id,
            "primary_name": primary_name,
            "aliases": [],
            "names": [],
        }

        if include_aliases:
            # Get aliases
            cursor.execute(
                """
                SELECT external_id, source
                FROM reaction_aliases
                WHERE modelseed_id = ?
                ORDER BY source
                LIMIT 20
            """,
                (modelseed_id,),
            )

            result["aliases"] = [
                {"external_id": row[0], "source": row[1]} for row in cursor.fetchall()
            ]

        return result

    def _get_compound_suggestions(
        self, conn: sqlite3.Connection, entity_id: str
    ) -> List[str]:
        """Get suggested compounds for failed lookups"""
        cursor = conn.cursor()

        # Look for partial matches in names
        cursor.execute(
            """
            SELECT DISTINCT c.modelseed_id
            FROM compounds c
            JOIN compound_names cn ON c.modelseed_id = cn.modelseed_id
            WHERE cn.name LIKE ?
            LIMIT 5
        """,
            (f"%{entity_id[:3]}%",),
        )

        return [row[0] for row in cursor.fetchall()]

    def _get_reaction_suggestions(
        self, conn: sqlite3.Connection, entity_id: str
    ) -> List[str]:
        """Get suggested reactions for failed lookups"""
        cursor = conn.cursor()

        # Look for partial matches in aliases
        cursor.execute(
            """
            SELECT DISTINCT modelseed_id
            FROM reaction_aliases
            WHERE external_id LIKE ?
            LIMIT 5
        """,
            (f"%{entity_id[:3]}%",),
        )

        return [row[0] for row in cursor.fetchall()]

    def _run_tool(self, input_data: Any) -> ToolResult:
        """Execute biochemistry entity resolution"""
        try:
            # Validate input
            input_obj = BiochemEntityResolverInput(**input_data)

            # Detect entity type if not specified
            entity_type = input_obj.entity_type or self._detect_entity_type(
                input_obj.entity_id
            )

            # Connect to database
            conn = sqlite3.connect(str(self.db_path))

            try:
                # Resolve based on entity type
                if entity_type == "compound":
                    resolution = self._resolve_compound(
                        conn,
                        input_obj.entity_id,
                        input_obj.include_aliases,
                        input_obj.max_results,
                    )
                elif entity_type == "reaction":
                    resolution = self._resolve_reaction(
                        conn,
                        input_obj.entity_id,
                        input_obj.include_aliases,
                        input_obj.max_results,
                    )
                else:
                    # Try both types
                    compound_result = self._resolve_compound(
                        conn,
                        input_obj.entity_id,
                        input_obj.include_aliases,
                        input_obj.max_results,
                    )
                    if compound_result["resolved"]:
                        resolution = compound_result
                        entity_type = "compound"
                    else:
                        resolution = self._resolve_reaction(
                            conn,
                            input_obj.entity_id,
                            input_obj.include_aliases,
                            input_obj.max_results,
                        )
                        entity_type = "reaction"

                # Build output
                output_data = {
                    "entity_id": input_obj.entity_id,
                    "entity_type": entity_type,
                    **resolution,
                }

                output = BiochemEntityResolverOutput(**output_data)

                return ToolResult(
                    success=True,
                    data=output.dict(),
                    message=f"Resolved {input_obj.entity_id} as {entity_type}",
                )

            finally:
                conn.close()

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message="Failed to resolve biochemistry entity",
            )


@ToolRegistry.register
class BiochemSearchTool(BaseTool):
    """Tool for searching the biochemistry database"""

    tool_name = "search_biochem"
    tool_description = (
        "Search the biochemistry database for compounds and reactions by name or alias"
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._db_path = None

    @property
    def db_path(self) -> Path:
        """Get the path to the biochemistry database"""
        if self._db_path is None:
            # Look for database in data directory
            base_dir = Path(__file__).parent.parent.parent.parent
            db_path = base_dir / "data" / "biochem.db"

            # Check for environment variable override
            env_db_path = os.getenv("BIOCHEM_DB_PATH")
            if env_db_path:
                db_path = Path(env_db_path)

            if not db_path.exists():
                raise FileNotFoundError(
                    f"Biochemistry database not found at {db_path}. "
                    "Please run 'python scripts/build_mvp_biochem_db.py' to create it, "
                    "or set BIOCHEM_DB_PATH environment variable to an existing database."
                )
            self._db_path = db_path
        return self._db_path

    def _search_compounds(
        self,
        conn: sqlite3.Connection,
        query: str,
        source_filter: Optional[str],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Search for compounds"""
        cursor = conn.cursor()

        # Search in compound names and aliases
        if source_filter:
            sql = """
                SELECT DISTINCT c.modelseed_id, c.primary_name,
                       cn.name as match_name, cn.source, 'name' as match_type
                FROM compounds c
                JOIN compound_names cn ON c.modelseed_id = cn.modelseed_id
                WHERE cn.name LIKE ? AND cn.source = ?
                UNION
                SELECT DISTINCT c.modelseed_id, c.primary_name,
                       ca.external_id as match_name, ca.source, 'alias' as match_type
                FROM compounds c
                JOIN compound_aliases ca ON c.modelseed_id = ca.modelseed_id
                WHERE ca.external_id LIKE ? AND ca.source = ?
                ORDER BY match_name
                LIMIT ?
            """
            params = (
                f"%{query}%",
                source_filter,
                f"%{query}%",
                source_filter,
                max_results,
            )
        else:
            sql = """
                SELECT DISTINCT c.modelseed_id, c.primary_name,
                       cn.name as match_name, cn.source, 'name' as match_type
                FROM compounds c
                JOIN compound_names cn ON c.modelseed_id = cn.modelseed_id
                WHERE cn.name LIKE ?
                UNION
                SELECT DISTINCT c.modelseed_id, c.primary_name,
                       ca.external_id as match_name, ca.source, 'alias' as match_type
                FROM compounds c
                JOIN compound_aliases ca ON c.modelseed_id = ca.modelseed_id
                WHERE ca.external_id LIKE ?
                ORDER BY match_name
                LIMIT ?
            """
            params = (f"%{query}%", f"%{query}%", max_results)

        cursor.execute(sql, params)

        return [
            {
                "modelseed_id": row[0],
                "primary_name": row[1],
                "match_name": row[2],
                "source": row[3],
                "match_type": row[4],
            }
            for row in cursor.fetchall()
        ]

    def _search_reactions(
        self,
        conn: sqlite3.Connection,
        query: str,
        source_filter: Optional[str],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Search for reactions"""
        cursor = conn.cursor()

        # Search in reaction aliases
        if source_filter:
            sql = """
                SELECT DISTINCT r.modelseed_id, r.primary_name,
                       ra.external_id as match_name, ra.source, 'alias' as match_type
                FROM reactions r
                JOIN reaction_aliases ra ON r.modelseed_id = ra.modelseed_id
                WHERE ra.external_id LIKE ? AND ra.source = ?
                ORDER BY match_name
                LIMIT ?
            """
            params = (f"%{query}%", source_filter, max_results)
        else:
            sql = """
                SELECT DISTINCT r.modelseed_id, r.primary_name,
                       ra.external_id as match_name, ra.source, 'alias' as match_type
                FROM reactions r
                JOIN reaction_aliases ra ON r.modelseed_id = ra.modelseed_id
                WHERE ra.external_id LIKE ?
                ORDER BY match_name
                LIMIT ?
            """
            params = (f"%{query}%", max_results)

        cursor.execute(sql, params)

        return [
            {
                "modelseed_id": row[0],
                "primary_name": row[1],
                "match_name": row[2],
                "source": row[3],
                "match_type": row[4],
            }
            for row in cursor.fetchall()
        ]

    def _run_tool(self, input_data: Any) -> ToolResult:
        """Execute biochemistry database search"""
        try:
            # Validate input
            input_obj = BiochemSearchInput(**input_data)

            # Connect to database
            conn = sqlite3.connect(str(self.db_path))

            try:
                compounds = []
                reactions = []

                # Search based on entity type
                if input_obj.entity_type == "compound":
                    compounds = self._search_compounds(
                        conn,
                        input_obj.query,
                        input_obj.source_filter,
                        input_obj.max_results,
                    )
                elif input_obj.entity_type == "reaction":
                    reactions = self._search_reactions(
                        conn,
                        input_obj.query,
                        input_obj.source_filter,
                        input_obj.max_results,
                    )
                else:
                    # Search both
                    half_results = input_obj.max_results // 2
                    compounds = self._search_compounds(
                        conn, input_obj.query, input_obj.source_filter, half_results
                    )
                    reactions = self._search_reactions(
                        conn, input_obj.query, input_obj.source_filter, half_results
                    )

                # Build output
                output = BiochemSearchOutput(
                    query=input_obj.query,
                    total_results=len(compounds) + len(reactions),
                    compounds=compounds,
                    reactions=reactions,
                )

                return ToolResult(
                    success=True,
                    data=output.dict(),
                    message=f"Found {output.total_results} matches for '{input_obj.query}'",
                )

            finally:
                conn.close()

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message="Failed to search biochemistry database",
            )


def resolve_entity_id(
    entity_id: str, db_path: str = "data/biochem.db"
) -> Optional[str]:
    """
    Convenience function to quickly resolve an entity ID to a human-readable name

    Args:
        entity_id: Entity ID to resolve
        db_path: Path to biochemistry database

    Returns:
        Human-readable name or None if not found
    """
    try:
        tool = BiochemEntityResolverTool()
        result = tool.run({"entity_id": entity_id, "include_aliases": False})
        if result.success and result.data.get("resolved"):
            return result.data.get("primary_name")
        return None
    except Exception:
        return None


def enhance_tool_output_with_names(
    tool_output: Dict[str, Any], db_path: str = "data/biochem.db"
) -> Dict[str, Any]:
    """
    Enhance tool output by adding human-readable names for biochemistry IDs

    Args:
        tool_output: Original tool output dictionary
        db_path: Path to biochemistry database

    Returns:
        Enhanced output with human-readable names
    """
    try:
        enhanced_output = tool_output.copy()

        # Recursively search for entity IDs and enhance them
        def enhance_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and (
                        value.startswith(("cpd", "rxn"))
                        or "cpd" in key.lower()
                        or "rxn" in key.lower()
                        or "reaction" in key.lower()
                        or "compound" in key.lower()
                    ):
                        # Try to resolve this as an entity ID
                        name = resolve_entity_id(value, db_path)
                        if name:
                            # Add human-readable name alongside the ID
                            name_key = f"{key}_name"
                            obj[name_key] = name
                    else:
                        enhance_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    enhance_recursive(item)

        enhance_recursive(enhanced_output)
        return enhanced_output

    except Exception:
        # If enhancement fails, return original output
        return tool_output
