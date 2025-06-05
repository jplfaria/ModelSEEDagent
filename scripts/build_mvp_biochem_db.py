#!/usr/bin/env python3
"""
MVP Biochemistry Database Builder
Builds a biochemistry database using the existing ModelSEED Database dev branch
alias and name files. This creates an MVP for Phase 3 that provides universal
ID resolution without trying to recreate complex database integration.

Based on ModelSEED Database dev branch:
https://github.com/ModelSEED/ModelSEEDDatabase/tree/dev/Biochemistry/Aliases

Files used:
- Unique_ModelSEED_Compound_Aliases.txt
- Unique_ModelSEED_Compound_Names.txt
- Unique_ModelSEED_Reaction_Aliases.txt
"""

import csv
import os
import sqlite3
from collections import defaultdict
from pathlib import Path


def create_database_schema(conn):
    """Create the biochemistry database schema"""
    cursor = conn.cursor()

    # Compounds table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS compounds (
            modelseed_id TEXT PRIMARY KEY,
            primary_name TEXT,
            formula TEXT,
            charge INTEGER,
            mass REAL
        )
    """
    )

    # Compound aliases table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS compound_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            modelseed_id TEXT,
            external_id TEXT,
            source TEXT,
            FOREIGN KEY (modelseed_id) REFERENCES compounds (modelseed_id)
        )
    """
    )

    # Compound names table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS compound_names (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            modelseed_id TEXT,
            name TEXT,
            source TEXT,
            FOREIGN KEY (modelseed_id) REFERENCES compounds (modelseed_id)
        )
    """
    )

    # Reactions table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reactions (
            modelseed_id TEXT PRIMARY KEY,
            primary_name TEXT,
            equation TEXT,
            reversibility TEXT,
            direction TEXT
        )
    """
    )

    # Reaction aliases table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reaction_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            modelseed_id TEXT,
            external_id TEXT,
            source TEXT,
            FOREIGN KEY (modelseed_id) REFERENCES reactions (modelseed_id)
        )
    """
    )

    # Create indexes for fast lookups
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_compound_aliases_external ON compound_aliases(external_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_compound_aliases_source ON compound_aliases(source)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_compound_names_name ON compound_names(name)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_reaction_aliases_external ON reaction_aliases(external_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_reaction_aliases_source ON reaction_aliases(source)"
    )

    conn.commit()


def load_compound_names(conn, file_path):
    """Load compound names from ModelSEED names file"""
    cursor = conn.cursor()

    print(f"📝 Loading compound names from {file_path}...")

    # Track unique compounds and their primary names
    compounds = {}
    names_data = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            modelseed_id = row["ModelSEED ID"].strip()
            name = row["External ID"].strip()
            source = row["Source"].strip()

            # Use first name encountered as primary name
            if modelseed_id not in compounds:
                compounds[modelseed_id] = name

            names_data.append((modelseed_id, name, source))

    # Insert compounds
    for modelseed_id, primary_name in compounds.items():
        cursor.execute(
            """
            INSERT OR IGNORE INTO compounds (modelseed_id, primary_name)
            VALUES (?, ?)
        """,
            (modelseed_id, primary_name),
        )

    # Insert names
    cursor.executemany(
        """
        INSERT INTO compound_names (modelseed_id, name, source)
        VALUES (?, ?, ?)
    """,
        names_data,
    )

    conn.commit()
    print(f"✅ Loaded {len(compounds)} compounds with {len(names_data)} name entries")


def load_compound_aliases(conn, file_path):
    """Load compound aliases from ModelSEED aliases file"""
    cursor = conn.cursor()

    print(f"🔗 Loading compound aliases from {file_path}...")

    aliases_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            modelseed_id = row["ModelSEED ID"].strip()
            external_id = row["External ID"].strip()
            source = row["Source"].strip()

            aliases_data.append((modelseed_id, external_id, source))

    cursor.executemany(
        """
        INSERT INTO compound_aliases (modelseed_id, external_id, source)
        VALUES (?, ?, ?)
    """,
        aliases_data,
    )

    conn.commit()
    print(f"✅ Loaded {len(aliases_data)} compound aliases")


def load_reaction_aliases(conn, file_path):
    """Load reaction aliases from ModelSEED aliases file"""
    cursor = conn.cursor()

    print(f"⚡ Loading reaction aliases from {file_path}...")

    # Track unique reactions
    reactions = set()
    aliases_data = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            modelseed_id = row["ModelSEED ID"].strip()
            external_id = row["External ID"].strip()
            source = row["Source"].strip()

            reactions.add(modelseed_id)
            aliases_data.append((modelseed_id, external_id, source))

    # Insert reactions (basic entries)
    for modelseed_id in reactions:
        cursor.execute(
            """
            INSERT OR IGNORE INTO reactions (modelseed_id, primary_name)
            VALUES (?, ?)
        """,
            (modelseed_id, f"Reaction {modelseed_id}"),
        )

    # Insert aliases
    cursor.executemany(
        """
        INSERT INTO reaction_aliases (modelseed_id, external_id, source)
        VALUES (?, ?, ?)
    """,
        aliases_data,
    )

    conn.commit()
    print(f"✅ Loaded {len(reactions)} reactions with {len(aliases_data)} aliases")


def create_database_stats(conn):
    """Generate and print database statistics"""
    cursor = conn.cursor()

    print("\n📊 Database Statistics:")
    print("=" * 50)

    # Compounds
    cursor.execute("SELECT COUNT(*) FROM compounds")
    compound_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM compound_names")
    compound_names_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM compound_aliases")
    compound_aliases_count = cursor.fetchone()[0]

    # Reactions
    cursor.execute("SELECT COUNT(*) FROM reactions")
    reaction_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM reaction_aliases")
    reaction_aliases_count = cursor.fetchone()[0]

    print(f"🧪 Compounds: {compound_count:,}")
    print(f"📝 Compound Names: {compound_names_count:,}")
    print(f"🔗 Compound Aliases: {compound_aliases_count:,}")
    print(f"⚡ Reactions: {reaction_count:,}")
    print(f"🔗 Reaction Aliases: {reaction_aliases_count:,}")

    # Source breakdown
    print(f"\n🗂️ Source Distribution:")
    cursor.execute(
        """
        SELECT source, COUNT(*)
        FROM compound_aliases
        GROUP BY source
        ORDER BY COUNT(*) DESC
        LIMIT 10
    """
    )
    for source, count in cursor.fetchall():
        print(f"   {source}: {count:,} compound aliases")

    cursor.execute(
        """
        SELECT source, COUNT(*)
        FROM reaction_aliases
        GROUP BY source
        ORDER BY COUNT(*) DESC
        LIMIT 10
    """
    )
    print(f"\n   Top reaction alias sources:")
    for source, count in cursor.fetchall():
        print(f"   {source}: {count:,} reaction aliases")


def test_database_queries(conn):
    """Test some basic database queries"""
    cursor = conn.cursor()

    print(f"\n🧪 Testing Database Queries:")
    print("=" * 50)

    # Test compound resolution
    test_compounds = ["cpd00001", "cpd00002", "cpd00067"]  # Water, ATP, H+
    for cpd_id in test_compounds:
        cursor.execute(
            """
            SELECT c.modelseed_id, c.primary_name,
                   GROUP_CONCAT(cn.name, '; ') as all_names
            FROM compounds c
            LEFT JOIN compound_names cn ON c.modelseed_id = cn.modelseed_id
            WHERE c.modelseed_id = ?
            GROUP BY c.modelseed_id
        """,
            (cpd_id,),
        )

        result = cursor.fetchone()
        if result:
            print(f"🔍 {result[0]}: {result[1]}")
            if result[2]:
                names = result[2].split("; ")[:3]  # Show first 3 names
                print(f"   Names: {', '.join(names)}...")
        else:
            print(f"❌ {cpd_id}: Not found")

    # Test alias lookups
    print(f"\n🔗 Testing Alias Lookups:")
    test_aliases = ["ATP", "WATER", "glc__D"]
    for alias in test_aliases:
        cursor.execute(
            """
            SELECT ca.modelseed_id, c.primary_name, ca.source
            FROM compound_aliases ca
            JOIN compounds c ON ca.modelseed_id = c.modelseed_id
            WHERE ca.external_id = ?
            LIMIT 3
        """,
            (alias,),
        )

        results = cursor.fetchall()
        if results:
            print(f"🔍 '{alias}' maps to:")
            for result in results:
                print(f"   {result[0]} ({result[1]}) via {result[2]}")
        else:
            print(f"❌ '{alias}': No mappings found")


def main():
    """Main function to build the MVP biochemistry database"""
    print("🧬 Building ModelSEEDagent MVP Biochemistry Database")
    print("=" * 60)

    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    sources_dir = data_dir / "biochem_sources"
    db_path = data_dir / "biochem.db"

    # Check source files exist
    compound_names_file = sources_dir / "compound_names.txt"
    compound_aliases_file = sources_dir / "compound_aliases.txt"
    reaction_aliases_file = sources_dir / "reaction_aliases.txt"

    missing_files = []
    for file_path in [
        compound_names_file,
        compound_aliases_file,
        reaction_aliases_file,
    ]:
        if not file_path.exists():
            missing_files.append(str(file_path))

    if missing_files:
        print("❌ Missing source files:")
        for file_path in missing_files:
            print(f"   {file_path}")
        print("\nPlease run the download commands first to get the source files.")
        return

    # Remove existing database
    if db_path.exists():
        os.remove(db_path)
        print(f"🗑️ Removed existing database: {db_path}")

    # Create database
    print(f"🏗️ Creating database: {db_path}")
    conn = sqlite3.connect(str(db_path))

    try:
        # Create schema
        create_database_schema(conn)

        # Load data
        load_compound_names(conn, compound_names_file)
        load_compound_aliases(conn, compound_aliases_file)
        load_reaction_aliases(conn, reaction_aliases_file)

        # Generate stats
        create_database_stats(conn)

        # Test queries
        test_database_queries(conn)

        print(f"\n✅ MVP Biochemistry Database created successfully!")
        print(f"📁 Database file: {db_path}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
