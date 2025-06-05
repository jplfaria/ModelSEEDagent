#!/usr/bin/env python3
"""
Test Phase 3: Simple Biochemistry Resolution
This test validates the biochemistry database and resolution functions
without LangChain dependencies.
"""

import sqlite3
import sys
from pathlib import Path


def test_database_structure():
    """Test the database structure and content"""
    print("🗄️ Testing Database Structure")
    
    db_path = Path("data/biochem.db")
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = ['compounds', 'compound_names', 'compound_aliases', 'reactions', 'reaction_aliases']
        
        missing_tables = set(expected_tables) - set(tables)
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            return False
        
        print(f"✅ All expected tables present: {tables}")
        
        # Check table counts
        for table in expected_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   {table}: {count:,} entries")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database structure test failed: {e}")
        return False


def test_compound_resolution():
    """Test compound ID resolution"""
    print("🧪 Testing Compound Resolution")
    
    db_path = Path("data/biochem.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Test cases: [entity_id, should_resolve]
    test_cases = [
        ("cpd00001", True),   # Water
        ("cpd00002", True),   # ATP
        ("cpd00067", True),   # H+
        ("ATP", True),        # ATP alias
        ("WATER", True),      # Water alias
        ("glc__D", True),     # BiGG glucose
        ("invalid123", False) # Should not resolve
    ]
    
    success_count = 0
    for entity_id, should_resolve in test_cases:
        try:
            # Try direct ModelSEED ID lookup
            cursor.execute("""
                SELECT modelseed_id, primary_name
                FROM compounds
                WHERE modelseed_id = ?
            """, (entity_id,))
            
            result = cursor.fetchone()
            if result:
                print(f"✅ {entity_id} → {result[0]} ({result[1]})")
                success_count += 1
                continue
            
            # Try alias lookup
            cursor.execute("""
                SELECT c.modelseed_id, c.primary_name, ca.source
                FROM compounds c
                JOIN compound_aliases ca ON c.modelseed_id = ca.modelseed_id
                WHERE ca.external_id = ?
                LIMIT 1
            """, (entity_id,))
            
            result = cursor.fetchone()
            if result:
                print(f"✅ {entity_id} → {result[0]} ({result[1]}) via {result[2]}")
                success_count += 1
            elif not should_resolve:
                print(f"✅ {entity_id} → Correctly not resolved")
                success_count += 1
            else:
                print(f"❌ {entity_id} → Should resolve but didn't")
        
        except Exception as e:
            print(f"❌ {entity_id} → Error: {e}")
    
    conn.close()
    print(f"📊 Compound Resolution: {success_count}/{len(test_cases)} passed")
    return success_count == len(test_cases)


def test_reaction_resolution():
    """Test reaction ID resolution"""
    print("⚡ Testing Reaction Resolution")
    
    db_path = Path("data/biochem.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Test cases
    test_cases = [
        ("rxn00001", True),   # Inorganic pyrophosphatase
        ("PPA", True),        # BiGG alias for pyrophosphatase
        ("invalid_rxn", False)
    ]
    
    success_count = 0
    for entity_id, should_resolve in test_cases:
        try:
            # Try direct ModelSEED ID lookup
            cursor.execute("""
                SELECT modelseed_id, primary_name
                FROM reactions
                WHERE modelseed_id = ?
            """, (entity_id,))
            
            result = cursor.fetchone()
            if result:
                print(f"✅ {entity_id} → {result[0]} ({result[1]})")
                success_count += 1
                continue
            
            # Try alias lookup
            cursor.execute("""
                SELECT r.modelseed_id, r.primary_name, ra.source
                FROM reactions r
                JOIN reaction_aliases ra ON r.modelseed_id = ra.modelseed_id
                WHERE ra.external_id = ?
                LIMIT 1
            """, (entity_id,))
            
            result = cursor.fetchone()
            if result:
                print(f"✅ {entity_id} → {result[0]} ({result[1]}) via {result[2]}")
                success_count += 1
            elif not should_resolve:
                print(f"✅ {entity_id} → Correctly not resolved")
                success_count += 1
            else:
                print(f"❌ {entity_id} → Should resolve but didn't")
        
        except Exception as e:
            print(f"❌ {entity_id} → Error: {e}")
    
    conn.close()
    print(f"📊 Reaction Resolution: {success_count}/{len(test_cases)} passed")
    return success_count == len(test_cases)


def test_search_functionality():
    """Test search functionality"""
    print("🔍 Testing Search Functionality")
    
    db_path = Path("data/biochem.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Test compound search
    search_queries = ["glucose", "ATP", "water"]
    
    success_count = 0
    for query in search_queries:
        try:
            # Search in compound names
            cursor.execute("""
                SELECT DISTINCT c.modelseed_id, c.primary_name, cn.name
                FROM compounds c
                JOIN compound_names cn ON c.modelseed_id = cn.modelseed_id
                WHERE cn.name LIKE ?
                LIMIT 5
            """, (f"%{query}%",))
            
            name_results = cursor.fetchall()
            
            # Search in compound aliases
            cursor.execute("""
                SELECT DISTINCT c.modelseed_id, c.primary_name, ca.external_id, ca.source
                FROM compounds c
                JOIN compound_aliases ca ON c.modelseed_id = ca.modelseed_id
                WHERE ca.external_id LIKE ?
                LIMIT 5
            """, (f"%{query}%",))
            
            alias_results = cursor.fetchall()
            
            total_results = len(name_results) + len(alias_results)
            
            if total_results > 0:
                print(f"✅ '{query}': {total_results} results")
                if name_results:
                    print(f"   Name match: {name_results[0][1]} ({name_results[0][0]})")
                if alias_results:
                    print(f"   Alias match: {alias_results[0][1]} ({alias_results[0][0]}) via {alias_results[0][3]}")
                success_count += 1
            else:
                print(f"❌ '{query}': No results found")
        
        except Exception as e:
            print(f"❌ '{query}': Error: {e}")
    
    conn.close()
    print(f"📊 Search Tests: {success_count}/{len(search_queries)} passed")
    return success_count == len(search_queries)


def test_source_coverage():
    """Test coverage of different database sources"""
    print("🌐 Testing Source Database Coverage")
    
    db_path = Path("data/biochem.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Check compound sources
        cursor.execute("""
            SELECT source, COUNT(DISTINCT modelseed_id) as compound_count
            FROM compound_aliases
            GROUP BY source
            ORDER BY compound_count DESC
            LIMIT 10
        """)
        
        print("Top compound alias sources:")
        compound_sources = cursor.fetchall()
        for source, count in compound_sources:
            print(f"   {source}: {count:,} compounds")
        
        # Check reaction sources
        cursor.execute("""
            SELECT source, COUNT(DISTINCT modelseed_id) as reaction_count
            FROM reaction_aliases
            GROUP BY source
            ORDER BY reaction_count DESC
            LIMIT 10
        """)
        
        print(f"\nTop reaction alias sources:")
        reaction_sources = cursor.fetchall()
        for source, count in reaction_sources:
            print(f"   {source}: {count:,} reactions")
        
        # Check if key sources are present
        key_sources = {'BiGG', 'KEGG', 'MetaCyc'}
        compound_source_names = {source for source, _ in compound_sources}
        reaction_source_names = {source for source, _ in reaction_sources}
        
        all_sources = compound_source_names | reaction_source_names
        found_key_sources = key_sources & all_sources
        
        print(f"\nKey sources found: {found_key_sources}")
        print(f"Total unique sources: {len(all_sources)}")
        
        conn.close()
        return len(found_key_sources) >= 2  # At least 2 key sources
        
    except Exception as e:
        print(f"❌ Source coverage test failed: {e}")
        conn.close()
        return False


def test_performance():
    """Test basic performance"""
    print("⚡ Testing Basic Performance")
    
    import time
    
    db_path = Path("data/biochem.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Test lookup performance
        test_ids = ["cpd00001", "cpd00002", "ATP", "WATER", "glc__D"]
        
        start_time = time.time()
        
        for entity_id in test_ids:
            # Simple lookup
            cursor.execute("""
                SELECT modelseed_id, primary_name
                FROM compounds
                WHERE modelseed_id = ?
                UNION
                SELECT c.modelseed_id, c.primary_name
                FROM compounds c
                JOIN compound_aliases ca ON c.modelseed_id = ca.modelseed_id
                WHERE ca.external_id = ?
                LIMIT 1
            """, (entity_id, entity_id))
            
            result = cursor.fetchone()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Lookup performance: {len(test_ids)} queries in {duration:.3f} seconds")
        print(f"   Average: {duration/len(test_ids):.3f} seconds per query")
        
        # Test search performance
        start_time = time.time()
        cursor.execute("""
            SELECT DISTINCT c.modelseed_id, c.primary_name
            FROM compounds c
            JOIN compound_names cn ON c.modelseed_id = cn.modelseed_id
            WHERE cn.name LIKE ?
            LIMIT 20
        """, ("%glucose%",))
        
        results = cursor.fetchall()
        end_time = time.time()
        
        print(f"✅ Search performance: Found {len(results)} glucose matches in {end_time - start_time:.3f} seconds")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        conn.close()
        return False


def demonstrate_before_after():
    """Demonstrate the improvement from IDs to names"""
    print("🎭 Demonstrating Before/After Enhancement")
    
    db_path = Path("data/biochem.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Before: Raw tool output with cryptic IDs
        print("❌ BEFORE - Cryptic IDs (hard for AI to reason about):")
        print("   Key reactions: ['rxn00001', 'rxn00200', 'rxn00781']")
        print("   Bottleneck metabolites: ['cpd00002', 'cpd00009']")
        
        print("\n✅ AFTER - Human-readable names (easy for AI to reason about):")
        
        # Show individual resolutions
        for entity_id in ["rxn00001", "cpd00002", "cpd00009"]:
            if entity_id.startswith('rxn'):
                cursor.execute("""
                    SELECT modelseed_id, primary_name
                    FROM reactions
                    WHERE modelseed_id = ?
                """, (entity_id,))
            else:
                cursor.execute("""
                    SELECT modelseed_id, primary_name
                    FROM compounds
                    WHERE modelseed_id = ?
                """, (entity_id,))
            
            result = cursor.fetchone()
            if result:
                print(f"   {entity_id} → '{result[1]}'")
        
        print("\n🧠 Impact: AI can now reason about:")
        print("   - 'Inorganic pyrophosphatase' instead of 'rxn00001'")
        print("   - 'ATP' instead of 'cpd00002'")
        print("   - 'Orthophosphate' instead of 'cpd00009'")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Before/after demonstration failed: {e}")
        conn.close()
        return False


def main():
    """Run all Phase 3 simple biochemistry tests"""
    
    print("🧬 ModelSEEDagent Phase 3: Simple Biochemistry Testing")
    print("=" * 70)
    
    # Check if database exists
    db_path = Path("data/biochem.db")
    if not db_path.exists():
        print(f"❌ Biochemistry database not found at {db_path}")
        print("Please run: python scripts/build_mvp_biochem_db.py")
        return False
    
    print(f"✅ Using biochemistry database: {db_path}")
    print(f"   Database size: {db_path.stat().st_size / (1024*1024):.1f} MB")
    
    tests = [
        ("Database Structure", test_database_structure),
        ("Compound Resolution", test_compound_resolution),
        ("Reaction Resolution", test_reaction_resolution),
        ("Search Functionality", test_search_functionality),
        ("Source Coverage", test_source_coverage),
        ("Performance", test_performance),
        ("Before/After Demo", demonstrate_before_after),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   Result: {status}")
        except Exception as e:
            print(f"   Result: ❌ ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 PHASE 3 SIMPLE BIOCHEMISTRY TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All Phase 3 simple biochemistry tests PASSED!")
        print("✅ Universal ID resolution system functional")
        print("✅ Agent can now reason about biochemistry names instead of IDs")
        print("✅ Tool outputs can be enhanced with human-readable information")
        return True
    else:
        print("\n⚠️  Some biochemistry resolution issues detected")
        print("❌ Review failed tests and address resolution gaps")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)