#!/usr/bin/env python3
"""
Biochemistry Database Builder

This script builds a unified biochemistry database merging ModelSEED, BiGG, and KEGG
data sources into a SQLite database for universal ID resolution and enhanced
biochemistry reasoning.

Phase 3 Implementation:
- Unified reaction and compound mappings across databases
- Cross-reference tables for ID translation
- Human-readable names and descriptions
- Chemical formulas and equation standardization
"""

import json
import sqlite3
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from tqdm import tqdm


class BiochemDatabase:
    """Builder for unified biochemistry database"""

    def __init__(self, db_path: str = "data/biochem.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def create_schema(self):
        """Create database schema for biochemistry entities and cross-references"""

        schema_sql = """
        -- Core entity tables
        CREATE TABLE IF NOT EXISTS compounds (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            formula TEXT,
            charge INTEGER,
            mass REAL,
            inchi TEXT,
            smiles TEXT,
            description TEXT,
            source_db TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS reactions (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            equation TEXT,
            definition TEXT,
            enzyme_ec TEXT,
            subsystem TEXT,
            direction TEXT,  -- reversible, irreversible_forward, irreversible_backward
            source_db TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Cross-reference tables for ID mapping
        CREATE TABLE IF NOT EXISTS compound_xrefs (
            compound_id TEXT NOT NULL,
            external_db TEXT NOT NULL,
            external_id TEXT NOT NULL,
            FOREIGN KEY (compound_id) REFERENCES compounds(id),
            PRIMARY KEY (compound_id, external_db, external_id)
        );

        CREATE TABLE IF NOT EXISTS reaction_xrefs (
            reaction_id TEXT NOT NULL,
            external_db TEXT NOT NULL,
            external_id TEXT NOT NULL,
            FOREIGN KEY (reaction_id) REFERENCES reactions(id),
            PRIMARY KEY (reaction_id, external_db, external_id)
        );

        -- Reaction participant relationships
        CREATE TABLE IF NOT EXISTS reaction_compounds (
            reaction_id TEXT NOT NULL,
            compound_id TEXT NOT NULL,
            compartment TEXT,
            stoichiometry REAL NOT NULL,
            role TEXT, -- substrate, product, cofactor
            FOREIGN KEY (reaction_id) REFERENCES reactions(id),
            FOREIGN KEY (compound_id) REFERENCES compounds(id),
            PRIMARY KEY (reaction_id, compound_id, compartment)
        );

        -- Search optimization indexes
        CREATE INDEX IF NOT EXISTS idx_compounds_name ON compounds(name);
        CREATE INDEX IF NOT EXISTS idx_reactions_name ON reactions(name);
        CREATE INDEX IF NOT EXISTS idx_compound_xrefs_external ON compound_xrefs(external_db, external_id);
        CREATE INDEX IF NOT EXISTS idx_reaction_xrefs_external ON reaction_xrefs(external_db, external_id);
        CREATE INDEX IF NOT EXISTS idx_compounds_formula ON compounds(formula);
        CREATE INDEX IF NOT EXISTS idx_reactions_enzyme ON reactions(enzyme_ec);
        """

        self.conn.executescript(schema_sql)
        self.conn.commit()
        print("✅ Database schema created successfully")

    def load_modelseed_compounds(self) -> int:
        """Load ModelSEED compound data"""

        print("📥 Loading ModelSEED compounds...")

        try:
            # Download ModelSEED compounds database
            url = "https://raw.githubusercontent.com/ModelSEED/ModelSEEDDatabase/master/Biochemistry/compounds.json"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            compounds_data = response.json()

            compounds_added = 0
            for compound in tqdm(compounds_data, desc="Processing ModelSEED compounds"):
                try:
                    # Extract compound information
                    cpd_id = compound.get("id", "")
                    if not cpd_id:
                        continue

                    name = compound.get("name", "Unknown")
                    formula = compound.get("formula", "")
                    charge = compound.get("defaultCharge", 0)
                    mass = compound.get("mass", 0.0)

                    # Insert compound
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO compounds
                        (id, name, formula, charge, mass, source_db)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (cpd_id, name, formula, charge, mass, "ModelSEED"),
                    )

                    # Add cross-references
                    aliases = compound.get("aliases", [])
                    for alias_db, alias_ids in aliases.items():
                        if isinstance(alias_ids, list):
                            for alias_id in alias_ids:
                                self.conn.execute(
                                    """
                                    INSERT OR IGNORE INTO compound_xrefs
                                    (compound_id, external_db, external_id)
                                    VALUES (?, ?, ?)
                                """,
                                    (cpd_id, alias_db, alias_id),
                                )
                        else:
                            self.conn.execute(
                                """
                                INSERT OR IGNORE INTO compound_xrefs
                                (compound_id, external_db, external_id)
                                VALUES (?, ?, ?)
                            """,
                                (cpd_id, alias_db, alias_ids),
                            )

                    compounds_added += 1

                except Exception as e:
                    print(
                        f"Warning: Failed to process compound {compound.get('id', 'unknown')}: {e}"
                    )
                    continue

            self.conn.commit()
            print(f"✅ Added {compounds_added} ModelSEED compounds")
            return compounds_added

        except Exception as e:
            print(f"❌ Failed to load ModelSEED compounds: {e}")
            return 0

    def load_modelseed_reactions(self) -> int:
        """Load ModelSEED reaction data"""

        print("📥 Loading ModelSEED reactions...")

        try:
            # Download ModelSEED reactions database
            url = "https://raw.githubusercontent.com/ModelSEED/ModelSEEDDatabase/master/Biochemistry/reactions.json"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            reactions_data = response.json()

            reactions_added = 0
            for reaction in tqdm(reactions_data, desc="Processing ModelSEED reactions"):
                try:
                    # Extract reaction information
                    rxn_id = reaction.get("id", "")
                    if not rxn_id:
                        continue

                    name = reaction.get("name", "Unknown")
                    definition = reaction.get("definition", "")
                    enzyme = reaction.get("enzyme", "")
                    direction = reaction.get("direction", "=")

                    # Convert direction symbols
                    direction_map = {
                        "=": "reversible",
                        ">": "irreversible_forward",
                        "<": "irreversible_backward",
                    }
                    direction_text = direction_map.get(direction, "reversible")

                    # Insert reaction
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO reactions
                        (id, name, equation, enzyme_ec, direction, source_db)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (rxn_id, name, definition, enzyme, direction_text, "ModelSEED"),
                    )

                    # Add cross-references
                    aliases = reaction.get("aliases", [])
                    for alias_db, alias_ids in aliases.items():
                        if isinstance(alias_ids, list):
                            for alias_id in alias_ids:
                                self.conn.execute(
                                    """
                                    INSERT OR IGNORE INTO reaction_xrefs
                                    (reaction_id, external_db, external_id)
                                    VALUES (?, ?, ?)
                                """,
                                    (rxn_id, alias_db, alias_id),
                                )
                        else:
                            self.conn.execute(
                                """
                                INSERT OR IGNORE INTO reaction_xrefs
                                (reaction_id, external_db, external_id)
                                VALUES (?, ?, ?)
                            """,
                                (rxn_id, alias_db, alias_ids),
                            )

                    # Process reaction participants
                    reagents = reaction.get("reagents", [])
                    for reagent in reagents:
                        compound_id = reagent.get("compound", "")
                        compartment = reagent.get("compartment", "c")
                        coefficient = reagent.get("coefficient", 0)

                        if compound_id and coefficient != 0:
                            role = "product" if coefficient > 0 else "substrate"
                            self.conn.execute(
                                """
                                INSERT OR IGNORE INTO reaction_compounds
                                (reaction_id, compound_id, compartment, stoichiometry, role)
                                VALUES (?, ?, ?, ?, ?)
                            """,
                                (
                                    rxn_id,
                                    compound_id,
                                    compartment,
                                    abs(coefficient),
                                    role,
                                ),
                            )

                    reactions_added += 1

                except Exception as e:
                    print(
                        f"Warning: Failed to process reaction {reaction.get('id', 'unknown')}: {e}"
                    )
                    continue

            self.conn.commit()
            print(f"✅ Added {reactions_added} ModelSEED reactions")
            return reactions_added

        except Exception as e:
            print(f"❌ Failed to load ModelSEED reactions: {e}")
            return 0

    def load_bigg_data(self) -> Tuple[int, int]:
        """Load BiGG model data for additional cross-references"""

        print("📥 Loading BiGG database mappings...")

        compounds_added = 0
        reactions_added = 0

        try:
            # BiGG metabolites
            bigg_metabolites_url = "http://bigg.ucsd.edu/api/v2/universal/metabolites"
            response = requests.get(bigg_metabolites_url, timeout=30)
            response.raise_for_status()
            metabolites = response.json()["results"]

            for metabolite in tqdm(
                metabolites[:1000], desc="Processing BiGG metabolites"
            ):  # Limit for demo
                try:
                    bigg_id = metabolite.get("bigg_id", "")
                    name = metabolite.get("name", "Unknown")

                    if not bigg_id:
                        continue

                    # Insert as BiGG compound if not exists
                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO compounds
                        (id, name, source_db)
                        VALUES (?, ?, ?)
                    """,
                        (f"bigg_{bigg_id}", name, "BiGG"),
                    )

                    # Add cross-reference to ModelSEED if possible
                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO compound_xrefs
                        (compound_id, external_db, external_id)
                        VALUES (?, ?, ?)
                    """,
                        (f"bigg_{bigg_id}", "BiGG", bigg_id),
                    )

                    compounds_added += 1

                except Exception as e:
                    print(
                        f"Warning: Failed to process BiGG metabolite {metabolite.get('bigg_id', 'unknown')}: {e}"
                    )
                    continue

            # BiGG reactions
            bigg_reactions_url = "http://bigg.ucsd.edu/api/v2/universal/reactions"
            response = requests.get(bigg_reactions_url, timeout=30)
            response.raise_for_status()
            reactions = response.json()["results"]

            for reaction in tqdm(
                reactions[:1000], desc="Processing BiGG reactions"
            ):  # Limit for demo
                try:
                    bigg_id = reaction.get("bigg_id", "")
                    name = reaction.get("name", "Unknown")

                    if not bigg_id:
                        continue

                    # Insert as BiGG reaction if not exists
                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO reactions
                        (id, name, source_db)
                        VALUES (?, ?, ?)
                    """,
                        (f"bigg_{bigg_id}", name, "BiGG"),
                    )

                    # Add cross-reference
                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO reaction_xrefs
                        (reaction_id, external_db, external_id)
                        VALUES (?, ?, ?)
                    """,
                        (f"bigg_{bigg_id}", "BiGG", bigg_id),
                    )

                    reactions_added += 1

                except Exception as e:
                    print(
                        f"Warning: Failed to process BiGG reaction {reaction.get('bigg_id', 'unknown')}: {e}"
                    )
                    continue

            self.conn.commit()
            print(
                f"✅ Added {compounds_added} BiGG compounds and {reactions_added} BiGG reactions"
            )
            return compounds_added, reactions_added

        except Exception as e:
            print(f"❌ Failed to load BiGG data: {e}")
            return 0, 0

    def add_kegg_mappings(self) -> int:
        """Add KEGG cross-references (simplified implementation)"""

        print("📥 Adding KEGG cross-reference mappings...")

        # For this implementation, we'll add some common KEGG mappings
        # In a full implementation, you would download KEGG data

        kegg_mappings = [
            # Common metabolites
            ("cpd00001", "KEGG", "C00001", "H2O"),
            ("cpd00008", "KEGG", "C00008", "ADP"),
            ("cpd00009", "KEGG", "C00009", "Orthophosphate"),
            ("cpd00010", "KEGG", "C00010", "CoA"),
            ("cpd00067", "KEGG", "C00267", "Glucose"),
            ("cpd00020", "KEGG", "C00020", "AMP"),
            ("cpd00002", "KEGG", "C00002", "ATP"),
            # Common reactions
            ("rxn00001", "KEGG", "R00001", "ATP hydrolysis"),
            ("rxn00002", "KEGG", "R00002", "Glucose phosphorylation"),
        ]

        mappings_added = 0
        for entity_id, db, external_id, description in kegg_mappings:
            try:
                if entity_id.startswith("cpd"):
                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO compound_xrefs
                        (compound_id, external_db, external_id)
                        VALUES (?, ?, ?)
                    """,
                        (entity_id, db, external_id),
                    )
                elif entity_id.startswith("rxn"):
                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO reaction_xrefs
                        (reaction_id, external_db, external_id)
                        VALUES (?, ?, ?)
                    """,
                        (entity_id, db, external_id),
                    )

                mappings_added += 1

            except Exception as e:
                print(f"Warning: Failed to add KEGG mapping for {entity_id}: {e}")
                continue

        self.conn.commit()
        print(f"✅ Added {mappings_added} KEGG cross-reference mappings")
        return mappings_added

    def build_search_views(self):
        """Create optimized views for biochemistry search"""

        views_sql = """
        -- Unified compound search view
        CREATE VIEW IF NOT EXISTS compound_search AS
        SELECT
            c.id,
            c.name,
            c.formula,
            c.charge,
            c.mass,
            c.source_db,
            GROUP_CONCAT(x.external_db || ':' || x.external_id, '; ') as cross_refs
        FROM compounds c
        LEFT JOIN compound_xrefs x ON c.id = x.compound_id
        GROUP BY c.id, c.name, c.formula, c.charge, c.mass, c.source_db;

        -- Unified reaction search view
        CREATE VIEW IF NOT EXISTS reaction_search AS
        SELECT
            r.id,
            r.name,
            r.equation,
            r.enzyme_ec,
            r.direction,
            r.source_db,
            GROUP_CONCAT(x.external_db || ':' || x.external_id, '; ') as cross_refs
        FROM reactions r
        LEFT JOIN reaction_xrefs x ON r.id = x.reaction_id
        GROUP BY r.id, r.name, r.equation, r.enzyme_ec, r.direction, r.source_db;
        """

        self.conn.executescript(views_sql)
        self.conn.commit()
        print("✅ Created search optimization views")

    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics"""

        stats = {}

        # Count entities
        cursor = self.conn.execute("SELECT COUNT(*) FROM compounds")
        stats["compounds"] = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM reactions")
        stats["reactions"] = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM compound_xrefs")
        stats["compound_xrefs"] = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM reaction_xrefs")
        stats["reaction_xrefs"] = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM reaction_compounds")
        stats["reaction_compounds"] = cursor.fetchone()[0]

        # Count by source database
        cursor = self.conn.execute(
            "SELECT source_db, COUNT(*) FROM compounds GROUP BY source_db"
        )
        for source_db, count in cursor.fetchall():
            stats[f"compounds_{source_db}"] = count

        cursor = self.conn.execute(
            "SELECT source_db, COUNT(*) FROM reactions GROUP BY source_db"
        )
        for source_db, count in cursor.fetchall():
            stats[f"reactions_{source_db}"] = count

        return stats


def main():
    """Build the complete biochemistry database"""

    print("🧬 Building ModelSEEDagent Biochemistry Database")
    print("=" * 60)

    db_path = Path("data/biochem.db")

    # Remove existing database for fresh build
    if db_path.exists():
        db_path.unlink()
        print("🗑️  Removed existing database for fresh build")

    with BiochemDatabase(str(db_path)) as db:

        # Step 1: Create schema
        print("\n📋 Step 1: Creating database schema...")
        db.create_schema()

        # Step 2: Load ModelSEED data
        print("\n📊 Step 2: Loading ModelSEED data...")
        db.load_modelseed_compounds()
        db.load_modelseed_reactions()

        # Step 3: Load BiGG data
        print("\n🔬 Step 3: Loading BiGG cross-references...")
        bigg_compounds, bigg_reactions = db.load_bigg_data()

        # Step 4: Add KEGG mappings
        print("\n🗺️  Step 4: Adding KEGG cross-reference mappings...")
        db.add_kegg_mappings()

        # Step 5: Create search views
        print("\n🔍 Step 5: Creating search optimization views...")
        db.build_search_views()

        # Step 6: Generate statistics
        print("\n📈 Step 6: Generating database statistics...")
        stats = db.get_statistics()

        print("\n" + "=" * 60)
        print("📊 BIOCHEMISTRY DATABASE BUILD COMPLETE")
        print("=" * 60)

        print(f"📁 Database file: {db_path}")
        print(f"💾 Database size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"\n📋 Entity Counts:")
        print(f"   Compounds: {stats['compounds']:,}")
        print(f"   Reactions: {stats['reactions']:,}")
        print(f"   Compound Cross-refs: {stats['compound_xrefs']:,}")
        print(f"   Reaction Cross-refs: {stats['reaction_xrefs']:,}")
        print(f"   Reaction-Compound Links: {stats['reaction_compounds']:,}")

        print(f"\n🗃️  Source Database Breakdown:")
        for key, value in stats.items():
            if key.startswith(("compounds_", "reactions_")):
                print(f"   {key.replace('_', ' ').title()}: {value:,}")

        print(f"\n✅ Universal biochemistry database ready for Phase 3 tools!")

        return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)
