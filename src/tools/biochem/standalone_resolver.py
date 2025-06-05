#!/usr/bin/env python3
"""
Standalone Biochemistry Resolution Functions

This module provides biochemistry resolution without LangChain dependencies
for immediate use and testing.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class BiochemDatabase:
    """Standalone biochemistry database helper"""
    
    def __init__(self, db_path: str = "data/biochem.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Biochemistry database not found at {db_path}. "
                "Run 'python scripts/build_simple_biochem_db.py' to create it."
            )
    
    def resolve_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Resolve entity ID to information"""
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Try compounds first
            cursor = conn.execute("""
                SELECT * FROM compound_search WHERE id = ?
            """, (entity_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "id": row["id"],
                    "name": row["name"],
                    "type": "compound",
                    "formula": row["formula"],
                    "charge": row["charge"],
                    "mass": row["mass"],
                    "source_db": row["source_db"],
                    "cross_refs": row["cross_refs"].split("; ") if row["cross_refs"] else []
                }
            
            # Try reactions if not found
            cursor = conn.execute("""
                SELECT * FROM reaction_search WHERE id = ?
            """, (entity_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "id": row["id"],
                    "name": row["name"],
                    "type": "reaction",
                    "equation": row["equation"],
                    "enzyme_ec": row["enzyme_ec"],
                    "direction": row["direction"],
                    "source_db": row["source_db"],
                    "cross_refs": row["cross_refs"].split("; ") if row["cross_refs"] else []
                }
            
            return None
            
        finally:
            conn.close()
    
    def search_entities(self, query: str, entity_type: Optional[str] = None, 
                       max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for entities by name or keyword"""
        
        entities = []
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Search compounds
            if entity_type is None or entity_type == "compound":
                cursor = conn.execute("""
                    SELECT * FROM compound_search 
                    WHERE name LIKE ? OR id LIKE ? OR formula LIKE ?
                    ORDER BY 
                        CASE WHEN name = ? THEN 1
                             WHEN name LIKE ? THEN 2
                             WHEN id = ? THEN 3
                             ELSE 4 END,
                        name
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", f"%{query}%", 
                     query, f"{query}%", query, max_results))
                
                for row in cursor.fetchall():
                    entities.append({
                        "id": row["id"],
                        "name": row["name"],
                        "type": "compound",
                        "formula": row["formula"],
                        "charge": row["charge"],
                        "mass": row["mass"],
                        "source_db": row["source_db"],
                        "cross_refs": row["cross_refs"].split("; ") if row["cross_refs"] else []
                    })
            
            # Search reactions
            if entity_type is None or entity_type == "reaction":
                remaining = max_results - len(entities)
                if remaining > 0:
                    cursor = conn.execute("""
                        SELECT * FROM reaction_search 
                        WHERE name LIKE ? OR id LIKE ? OR equation LIKE ? OR enzyme_ec LIKE ?
                        ORDER BY 
                            CASE WHEN name = ? THEN 1
                                 WHEN name LIKE ? THEN 2
                                 WHEN id = ? THEN 3
                                 ELSE 4 END,
                            name
                        LIMIT ?
                    """, (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%",
                         query, f"{query}%", query, remaining))
                    
                    for row in cursor.fetchall():
                        entities.append({
                            "id": row["id"],
                            "name": row["name"],
                            "type": "reaction",
                            "equation": row["equation"],
                            "enzyme_ec": row["enzyme_ec"],
                            "direction": row["direction"],
                            "source_db": row["source_db"],
                            "cross_refs": row["cross_refs"].split("; ") if row["cross_refs"] else []
                        })
            
        finally:
            conn.close()
        
        return entities[:max_results]


def resolve_entity_id(entity_id: str, db_path: str = "data/biochem.db") -> Optional[str]:
    """Quick resolve entity ID to human-readable name"""
    try:
        db = BiochemDatabase(db_path)
        entity = db.resolve_entity(entity_id)
        return entity["name"] if entity else None
    except Exception:
        return None


def enhance_tool_output_with_names(tool_output: Dict[str, Any], 
                                  db_path: str = "data/biochem.db") -> Dict[str, Any]:
    """Enhance tool output by adding human-readable names for biochemistry IDs"""
    try:
        db = BiochemDatabase(db_path)
        enhanced_output = tool_output.copy()
        
        # Recursively search for entity IDs and enhance them
        def enhance_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and (
                        value.startswith(("cpd", "rxn")) or 
                        "cpd" in key.lower() or "rxn" in key.lower() or
                        "reaction" in key.lower() or "compound" in key.lower() or
                        "metabolite" in key.lower()
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
                            if isinstance(item, str) and item.startswith(("cpd", "rxn")):
                                entity = db.resolve_entity(item)
                                if entity:
                                    names.append(entity["name"])
                        if names and len(names) == len(value):
                            name_key = f"{key}_names" if len(names) > 1 else f"{key}_name"
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


def search_biochem_entities(query: str, entity_type: Optional[str] = None, 
                           max_results: int = 10, db_path: str = "data/biochem.db") -> List[Dict[str, Any]]:
    """Search biochemistry database"""
    try:
        db = BiochemDatabase(db_path)
        return db.search_entities(query, entity_type, max_results)
    except Exception:
        return []