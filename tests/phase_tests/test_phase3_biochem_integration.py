#!/usr/bin/env python3
"""
Test Phase 3: Biochemistry Integration Tools
This test validates the complete biochemistry resolution system and ensures
it provides universal ID resolution capabilities for enhanced reasoning.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tools.biochem.resolver import (
    BiochemEntityResolverTool,
    BiochemSearchTool,
    enhance_tool_output_with_names,
    resolve_entity_id,
)


def test_biochem_entity_resolver():
    """Test the biochemistry entity resolver tool"""
    print("\n🧪 Testing Biochemistry Entity Resolver Tool")
    print("=" * 60)

    tool = BiochemEntityResolverTool()

    # Test cases: [entity_id, expected_resolved, expected_type]
    test_cases = [
        # ModelSEED IDs
        ("cpd00001", True, "compound"),  # Water
        ("cpd00002", True, "compound"),  # ATP
        ("cpd00067", True, "compound"),  # H+
        ("rxn00001", True, "reaction"),  # Inorganic pyrophosphatase
        # BiGG aliases
        ("ATP", True, "compound"),  # Should resolve to cpd00002
        ("WATER", True, "compound"),  # Should resolve to cpd00001
        ("glc__D", True, "compound"),  # Should resolve to cpd00027
        ("PPA", True, "reaction"),  # Should resolve to rxn00001
        # Non-existent entities
        ("cpd99999", False, "compound"),
        ("invalid_id", False, "compound"),
    ]

    success_count = 0
    for entity_id, expected_resolved, expected_type in test_cases:
        try:
            result = tool.run({"entity_id": entity_id})

            if result.success:
                data = result.data
                resolved = data.get("resolved", False)
                entity_type = data.get("entity_type")
                primary_name = data.get("primary_name")

                if resolved == expected_resolved and entity_type == expected_type:
                    print(f"✅ {entity_id}: {primary_name} ({entity_type})")
                    success_count += 1
                else:
                    print(
                        f"❌ {entity_id}: Expected resolved={expected_resolved}, type={expected_type}, "
                        f"got resolved={resolved}, type={entity_type}"
                    )
            else:
                if not expected_resolved:
                    print(f"✅ {entity_id}: Correctly not resolved")
                    success_count += 1
                else:
                    print(
                        f"❌ {entity_id}: Expected to resolve but failed: {result.error}"
                    )

        except Exception as e:
            print(f"❌ {entity_id}: Exception occurred: {e}")

    print(f"\n📊 Entity Resolution Tests: {success_count}/{len(test_cases)} passed")
    return success_count == len(test_cases)


def test_biochem_search():
    """Test the biochemistry search tool"""
    print("\n🔍 Testing Biochemistry Search Tool")
    print("=" * 60)

    tool = BiochemSearchTool()

    # Test cases: [query, entity_type, min_expected_results]
    test_cases = [
        ("ATP", None, 1),  # Should find ATP compounds
        ("glucose", None, 1),  # Should find glucose variants
        ("kinase", None, 1),  # Should find kinase reactions
        ("water", "compound", 1),  # Compound-specific search
        ("PPA", "reaction", 1),  # Reaction-specific search
        ("xyz123", None, 0),  # Non-existent query
    ]

    success_count = 0
    for query, entity_type, min_expected in test_cases:
        try:
            search_input = {"query": query, "max_results": 5}
            if entity_type:
                search_input["entity_type"] = entity_type

            result = tool.run(search_input)

            if result.success:
                data = result.data
                total_results = data.get("total_results", 0)
                compounds = data.get("compounds", [])
                reactions = data.get("reactions", [])

                if total_results >= min_expected:
                    print(
                        f"✅ '{query}': Found {total_results} results "
                        f"({len(compounds)} compounds, {len(reactions)} reactions)"
                    )

                    # Show top results
                    if compounds:
                        print(
                            f"   Top compound: {compounds[0]['primary_name']} ({compounds[0]['modelseed_id']})"
                        )
                    if reactions:
                        print(
                            f"   Top reaction: {reactions[0]['primary_name']} ({reactions[0]['modelseed_id']})"
                        )

                    success_count += 1
                else:
                    print(
                        f"❌ '{query}': Expected ≥{min_expected} results, got {total_results}"
                    )
            else:
                if min_expected == 0:
                    print(f"✅ '{query}': Correctly found no results")
                    success_count += 1
                else:
                    print(f"❌ '{query}': Search failed: {result.error}")

        except Exception as e:
            print(f"❌ '{query}': Exception occurred: {e}")

    print(f"\n📊 Search Tests: {success_count}/{len(test_cases)} passed")
    return success_count == len(test_cases)


def test_convenience_functions():
    """Test convenience functions for integration"""
    print("\n🔧 Testing Convenience Functions")
    print("=" * 60)

    # Test resolve_entity_id function
    print("Testing resolve_entity_id function:")
    test_ids = ["cpd00001", "cpd00002", "ATP", "invalid_id"]

    success_count = 0
    for entity_id in test_ids:
        try:
            name = resolve_entity_id(entity_id)
            if name:
                print(f"✅ {entity_id} → {name}")
                success_count += 1
            else:
                print(f"⚠️  {entity_id} → Not resolved")
                if entity_id == "invalid_id":
                    success_count += 1  # Expected for invalid ID
        except Exception as e:
            print(f"❌ {entity_id}: Exception occurred: {e}")

    # Test enhance_tool_output_with_names function
    print(f"\nTesting enhance_tool_output_with_names function:")
    sample_output = {
        "compound_id": "cpd00001",
        "reaction_id": "rxn00001",
        "other_data": "some value",
        "nested": {"compound": "cpd00002"},
    }

    try:
        enhanced = enhance_tool_output_with_names(sample_output)

        # Check if names were added
        if "compound_id_name" in enhanced and "reaction_id_name" in enhanced:
            print(f"✅ Enhancement successful:")
            print(f"   {enhanced['compound_id']} → {enhanced['compound_id_name']}")
            print(f"   {enhanced['reaction_id']} → {enhanced['reaction_id_name']}")
            success_count += 1
        else:
            print(f"❌ Enhancement failed: Names not added")

    except Exception as e:
        print(f"❌ Enhancement failed: {e}")

    print(
        f"\n📊 Convenience Function Tests: {success_count}/{len(test_ids) + 1} passed"
    )
    return success_count == len(test_ids) + 1


def test_database_performance():
    """Test database query performance"""
    print("\n⚡ Testing Database Performance")
    print("=" * 60)

    import time

    tool = BiochemEntityResolverTool()

    # Test batch resolution performance
    test_entities = ["cpd00001", "cpd00002", "cpd00067", "ATP", "WATER", "glc__D"]

    start_time = time.time()

    resolved_count = 0
    for entity_id in test_entities:
        result = tool.run({"entity_id": entity_id})
        if result.success and result.data.get("resolved"):
            resolved_count += 1

    end_time = time.time()
    duration = end_time - start_time

    print(
        f"✅ Resolved {resolved_count}/{len(test_entities)} entities in {duration:.3f} seconds"
    )
    print(f"   Average: {duration/len(test_entities):.3f} seconds per entity")

    # Test search performance
    search_tool = BiochemSearchTool()

    start_time = time.time()
    result = search_tool.run({"query": "glucose", "max_results": 20})
    end_time = time.time()

    if result.success:
        total_results = result.data.get("total_results", 0)
        print(
            f"✅ Search for 'glucose' found {total_results} results in {end_time - start_time:.3f} seconds"
        )
    else:
        print(f"❌ Search performance test failed: {result.error}")

    return True


def test_alias_mappings():
    """Test alias mappings across different databases"""
    print("\n🔗 Testing Cross-Database Alias Mappings")
    print("=" * 60)

    tool = BiochemEntityResolverTool()

    # Test cases with known cross-database aliases
    alias_tests = [
        ("ATP", ["BiGG", "MetaCyc", "KEGG"]),  # ATP should have many aliases
        ("glc__D", ["BiGG"]),  # BiGG-style glucose
        ("WATER", ["MetaCyc", "BiGG"]),  # Water aliases
    ]

    success_count = 0
    for alias, expected_sources in alias_tests:
        try:
            result = tool.run({"entity_id": alias, "include_aliases": True})

            if result.success and result.data.get("resolved"):
                aliases = result.data.get("aliases", [])
                sources_found = set(alias_info.get("source") for alias_info in aliases)

                expected_sources_found = any(
                    source in sources_found for source in expected_sources
                )

                if expected_sources_found:
                    print(
                        f"✅ {alias}: Found aliases from {len(sources_found)} sources"
                    )
                    print(f"   Sources: {', '.join(sorted(sources_found)[:5])}...")
                    success_count += 1
                else:
                    print(
                        f"❌ {alias}: Expected sources {expected_sources}, found {sources_found}"
                    )
            else:
                print(f"❌ {alias}: Failed to resolve")

        except Exception as e:
            print(f"❌ {alias}: Exception occurred: {e}")

    print(f"\n📊 Alias Mapping Tests: {success_count}/{len(alias_tests)} passed")
    return success_count == len(alias_tests)


def main():
    """Run all Phase 3 biochemistry integration tests"""
    print("🧬 ModelSEEDagent Phase 3: Biochemistry Integration Testing")
    print("=" * 70)

    # Check if database exists
    db_path = Path("data/biochem.db")
    if not db_path.exists():
        print(f"❌ Biochemistry database not found at {db_path}")
        print("Please run: python scripts/build_mvp_biochem_db.py")
        return False

    print(f"✅ Using biochemistry database: {db_path}")
    print(f"   Database size: {db_path.stat().st_size / (1024*1024):.1f} MB")

    # Run all tests
    all_tests_passed = True

    test_results = [
        ("Entity Resolution", test_biochem_entity_resolver()),
        ("Database Search", test_biochem_search()),
        ("Convenience Functions", test_convenience_functions()),
        ("Database Performance", test_database_performance()),
        ("Alias Mappings", test_alias_mappings()),
    ]

    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)

    print(f"\n🎯 Final Results")
    print("=" * 70)

    for test_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_tests_passed = False

    print(f"\n📊 Overall: {passed_tests}/{total_tests} test suites passed")

    if all_tests_passed:
        print("🎉 Phase 3 Biochemistry Integration: ALL TESTS PASSED!")
        print("\n✅ Ready for CLI integration and documentation update")
    else:
        print("❌ Some tests failed. Please review the issues above.")

    return all_tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
