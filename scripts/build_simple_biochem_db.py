#!/usr/bin/env python3
"""
Simple Biochemistry Database Builder

Creates a demonstration biochemistry database with essential compounds and reactions
for Phase 3 testing. This version uses hardcoded data instead of downloading
from external APIs to ensure reliability.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List


def create_demo_biochem_db(db_path: str = "data/biochem.db"):
    """Create a demonstration biochemistry database"""

    print("🧬 Building Demonstration Biochemistry Database...")

    # Ensure data directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Remove existing database
    if Path(db_path).exists():
        Path(db_path).unlink()

    # Create database and schema
    conn = sqlite3.connect(db_path)

    schema_sql = """
    -- Core entity tables
    CREATE TABLE compounds (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        formula TEXT,
        charge INTEGER,
        mass REAL,
        source_db TEXT NOT NULL
    );

    CREATE TABLE reactions (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        equation TEXT,
        enzyme_ec TEXT,
        direction TEXT,
        source_db TEXT NOT NULL
    );

    -- Cross-reference tables
    CREATE TABLE compound_xrefs (
        compound_id TEXT NOT NULL,
        external_db TEXT NOT NULL,
        external_id TEXT NOT NULL,
        FOREIGN KEY (compound_id) REFERENCES compounds(id),
        PRIMARY KEY (compound_id, external_db, external_id)
    );

    CREATE TABLE reaction_xrefs (
        reaction_id TEXT NOT NULL,
        external_db TEXT NOT NULL,
        external_id TEXT NOT NULL,
        FOREIGN KEY (reaction_id) REFERENCES reactions(id),
        PRIMARY KEY (reaction_id, external_db, external_id)
    );

    -- Search optimization views
    CREATE VIEW compound_search AS
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

    CREATE VIEW reaction_search AS
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

    -- Search indexes
    CREATE INDEX idx_compounds_name ON compounds(name);
    CREATE INDEX idx_reactions_name ON reactions(name);
    CREATE INDEX idx_compounds_formula ON compounds(formula);
    """

    conn.executescript(schema_sql)

    # Essential compounds for demonstration
    compounds = [
        # Core metabolites
        ("cpd00001", "H2O", "H2O", 0, 18.015, "ModelSEED"),
        ("cpd00002", "ATP", "C10H12N5O13P3", -4, 507.181, "ModelSEED"),
        ("cpd00008", "ADP", "C10H12N5O10P2", -3, 427.201, "ModelSEED"),
        ("cpd00009", "Orthophosphate", "HO4P", -2, 95.979, "ModelSEED"),
        ("cpd00010", "CoA", "C21H32N7O16P3S", -4, 767.535, "ModelSEED"),
        ("cpd00020", "AMP", "C10H12N5O7P", -2, 347.221, "ModelSEED"),
        ("cpd00067", "Glucose", "C6H12O6", 0, 180.156, "ModelSEED"),
        ("cpd00023", "L-Glutamate", "C5H8NO4", -1, 147.053, "ModelSEED"),
        ("cpd00033", "Acetate", "C2H3O2", -1, 59.044, "ModelSEED"),
        ("cpd00041", "L-Aspartate", "C4H6NO4", -1, 133.037, "ModelSEED"),
        ("cpd00051", "L-Arginine", "C6H15N4O2", 1, 175.209, "ModelSEED"),
        ("cpd00054", "L-Serine", "C3H7NO3", 0, 105.093, "ModelSEED"),
        ("cpd00060", "L-Methionine", "C5H11NO2S", 0, 149.211, "ModelSEED"),
        ("cpd00066", "L-Phenylalanine", "C9H11NO2", 0, 165.19, "ModelSEED"),
        ("cpd00069", "L-Tyrosine", "C9H11NO3", 0, 181.189, "ModelSEED"),
        ("cpd00084", "L-Cysteine", "C3H7NO2S", 0, 121.159, "ModelSEED"),
        ("cpd00107", "L-Leucine", "C6H13NO2", 0, 131.173, "ModelSEED"),
        ("cpd00119", "L-Histidine", "C6H10N3O2", 1, 155.155, "ModelSEED"),
        ("cpd00129", "L-Proline", "C5H9NO2", 0, 115.131, "ModelSEED"),
        ("cpd00156", "L-Valine", "C5H11NO2", 0, 117.147, "ModelSEED"),
        ("cpd00161", "L-Threonine", "C4H9NO3", 0, 119.119, "ModelSEED"),
        ("cpd00322", "L-Isoleucine", "C6H13NO2", 0, 131.173, "ModelSEED"),
        # Bioenergetics
        ("cpd00003", "NAD", "C21H26N7O14P2", -1, 663.425, "ModelSEED"),
        ("cpd00004", "NADH", "C21H27N7O14P2", -2, 664.433, "ModelSEED"),
        ("cpd00006", "NADP", "C21H25N7O17P3", -4, 743.405, "ModelSEED"),
        ("cpd00005", "NADPH", "C21H26N7O17P3", -5, 744.413, "ModelSEED"),
        ("cpd00011", "CO2", "CO2", 0, 44.01, "ModelSEED"),
        ("cpd00013", "NH3", "H4N", 1, 18.039, "ModelSEED"),
        ("cpd00007", "O2", "O2", 0, 31.998, "ModelSEED"),
        ("cpd00012", "PPi", "HO7P2", -3, 177.975, "ModelSEED"),
    ]

    # Essential reactions
    reactions = [
        # Central carbon metabolism
        (
            "rxn00001",
            "ATP hydrolysis",
            "ATP + H2O -> ADP + Orthophosphate",
            "3.6.1.3",
            "irreversible_forward",
            "ModelSEED",
        ),
        (
            "rxn00002",
            "Glucose phosphorylation",
            "Glucose + ATP -> Glucose-6-phosphate + ADP",
            "2.7.1.1",
            "irreversible_forward",
            "ModelSEED",
        ),
        (
            "rxn00781",
            "Phosphoglycerate mutase",
            "3-Phosphoglycerate -> 2-Phosphoglycerate",
            "5.4.2.11",
            "reversible",
            "ModelSEED",
        ),
        (
            "rxn00148",
            "Enolase",
            "2-Phosphoglycerate -> Phosphoenolpyruvate + H2O",
            "4.2.1.11",
            "reversible",
            "ModelSEED",
        ),
        (
            "rxn00200",
            "Pyruvate kinase",
            "Phosphoenolpyruvate + ADP -> Pyruvate + ATP",
            "2.7.1.40",
            "irreversible_forward",
            "ModelSEED",
        ),
        (
            "rxn00259",
            "Lactate dehydrogenase",
            "Pyruvate + NADH -> Lactate + NAD",
            "1.1.1.27",
            "reversible",
            "ModelSEED",
        ),
        (
            "rxn00351",
            "Glucose-6-phosphate dehydrogenase",
            "Glucose-6-phosphate + NADP -> 6-Phosphogluconolactone + NADPH",
            "1.1.1.49",
            "irreversible_forward",
            "ModelSEED",
        ),
        (
            "rxn05938",
            "Citrate synthase",
            "Acetyl-CoA + Oxaloacetate + H2O -> Citrate + CoA",
            "2.3.3.1",
            "irreversible_forward",
            "ModelSEED",
        ),
        (
            "rxn00257",
            "Aconitase",
            "Citrate -> Isocitrate",
            "4.2.1.3",
            "reversible",
            "ModelSEED",
        ),
        (
            "rxn00158",
            "Isocitrate dehydrogenase",
            "Isocitrate + NADP -> 2-Oxoglutarate + CO2 + NADPH",
            "1.1.1.42",
            "irreversible_forward",
            "ModelSEED",
        ),
        # Amino acid metabolism
        (
            "rxn00480",
            "Glutamate dehydrogenase",
            "2-Oxoglutarate + NH3 + NADH -> L-Glutamate + NAD + H2O",
            "1.4.1.2",
            "reversible",
            "ModelSEED",
        ),
        (
            "rxn00274",
            "Aspartate aminotransferase",
            "L-Aspartate + 2-Oxoglutarate -> Oxaloacetate + L-Glutamate",
            "2.6.1.1",
            "reversible",
            "ModelSEED",
        ),
        (
            "rxn00832",
            "Serine hydroxymethyltransferase",
            "L-Serine + Tetrahydrofolate -> Glycine + 5,10-Methylenetetrahydrofolate + H2O",
            "2.1.2.1",
            "reversible",
            "ModelSEED",
        ),
        # Transport reactions
        (
            "rxn05467",
            "Glucose transport",
            "Glucose[extracellular] -> Glucose[cytoplasm]",
            "",
            "reversible",
            "ModelSEED",
        ),
        (
            "rxn09037",
            "Acetate transport",
            "Acetate[extracellular] -> Acetate[cytoplasm]",
            "",
            "reversible",
            "ModelSEED",
        ),
        (
            "rxn08172",
            "Ammonia transport",
            "NH3[extracellular] -> NH3[cytoplasm]",
            "",
            "reversible",
            "ModelSEED",
        ),
    ]

    # Insert compounds
    for compound in compounds:
        conn.execute(
            """
            INSERT INTO compounds (id, name, formula, charge, mass, source_db)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            compound,
        )

    # Insert reactions
    for reaction in reactions:
        conn.execute(
            """
            INSERT INTO reactions (id, name, equation, enzyme_ec, direction, source_db)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            reaction,
        )

    # Add cross-references
    cross_refs = [
        # Compound cross-references
        ("cpd00001", "BiGG", "h2o"),
        ("cpd00001", "KEGG", "C00001"),
        ("cpd00002", "BiGG", "atp"),
        ("cpd00002", "KEGG", "C00002"),
        ("cpd00008", "BiGG", "adp"),
        ("cpd00008", "KEGG", "C00008"),
        ("cpd00009", "BiGG", "pi"),
        ("cpd00009", "KEGG", "C00009"),
        ("cpd00067", "BiGG", "glc__D"),
        ("cpd00067", "KEGG", "C00031"),
        ("cpd00023", "BiGG", "glu__L"),
        ("cpd00023", "KEGG", "C00025"),
        ("cpd00033", "BiGG", "ac"),
        ("cpd00033", "KEGG", "C00033"),
        # Reaction cross-references
        ("rxn00001", "BiGG", "ATPS4r"),
        ("rxn00001", "KEGG", "R00086"),
        ("rxn00002", "BiGG", "HEX1"),
        ("rxn00002", "KEGG", "R00299"),
        ("rxn00781", "BiGG", "PGM"),
        ("rxn00781", "KEGG", "R01518"),
        ("rxn00200", "BiGG", "PYK"),
        ("rxn00200", "KEGG", "R00200"),
    ]

    for cpd_id, db, external_id in cross_refs:
        if cpd_id.startswith("cpd"):
            conn.execute(
                """
                INSERT INTO compound_xrefs (compound_id, external_db, external_id)
                VALUES (?, ?, ?)
            """,
                (cpd_id, db, external_id),
            )
        elif cpd_id.startswith("rxn"):
            conn.execute(
                """
                INSERT INTO reaction_xrefs (reaction_id, external_db, external_id)
                VALUES (?, ?, ?)
            """,
                (cpd_id, db, external_id),
            )

    conn.commit()

    # Get statistics
    cursor = conn.execute("SELECT COUNT(*) FROM compounds")
    compound_count = cursor.fetchone()[0]

    cursor = conn.execute("SELECT COUNT(*) FROM reactions")
    reaction_count = cursor.fetchone()[0]

    cursor = conn.execute("SELECT COUNT(*) FROM compound_xrefs")
    compound_xref_count = cursor.fetchone()[0]

    cursor = conn.execute("SELECT COUNT(*) FROM reaction_xrefs")
    reaction_xref_count = cursor.fetchone()[0]

    conn.close()

    print(f"✅ Demonstration biochemistry database created successfully!")
    print(f"📁 Database file: {db_path}")
    print(f"📊 Statistics:")
    print(f"   Compounds: {compound_count}")
    print(f"   Reactions: {reaction_count}")
    print(f"   Compound cross-references: {compound_xref_count}")
    print(f"   Reaction cross-references: {reaction_xref_count}")

    return True


if __name__ == "__main__":
    create_demo_biochem_db()
