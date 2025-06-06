#!/usr/bin/env python3
"""
Test Phase 3: Biochemistry Resolution Tools

This test validates the biochemistry entity resolution system and ensures
the tools can resolve IDs to human-readable names effectively.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.tools.biochem.resolver import (
        BiochemDatabase,
        BiochemEntityResolverTool,
        BiochemSearchTool,
        enhance_tool_output_with_names,
        resolve_entity_id,
    )

    print("✅ All imports successful")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_database_connection():
    """Test basic database connectivity"""

    print("\n🔗 Testing database connection...")

    try:
        db = BiochemDatabase("data/biochem.db")

        with db.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM compounds")
            compound_count = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM reactions")
            reaction_count = cursor.fetchone()[0]

        print(f"✅ Database connected successfully")
        print(f"   Compounds: {compound_count}")
        print(f"   Reactions: {reaction_count}")

        return True

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def test_entity_resolution():
    """Test resolving specific entity IDs"""

    print("\n🔍 Testing entity resolution...")

    try:
        tool = BiochemEntityResolverTool({})

        # Test cases: (entity_id, expected_name_contains)
        test_cases = [
            ("cpd00001", "H2O"),
            ("cpd00002", "ATP"),
            ("cpd00067", "Glucose"),
            ("rxn00001", "ATP hydrolysis"),
            ("rxn00781", "Phosphoglycerate mutase"),
            ("rxn00200", "Pyruvate kinase"),
        ]

        all_passed = True
        for entity_id, expected in test_cases:
            try:
                result = tool._run({"entity_id": entity_id})

                if result.success:
                    resolved_name = result.data["human_readable_name"]
                    if expected.lower() in resolved_name.lower():
                        print(
                            f"✅ {entity_id} → '{resolved_name}' (contains '{expected}')"
                        )
                    else:
                        print(
                            f"❌ {entity_id} → '{resolved_name}' (expected '{expected}')"
                        )
                        all_passed = False
                else:
                    print(f"❌ {entity_id} → Failed: {result.message}")
                    all_passed = False

            except Exception as e:
                print(f"❌ {entity_id} → Error: {e}")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"❌ Entity resolution test failed: {e}")
        return False


def test_biochem_search():
    """Test biochemistry search functionality"""

    print("\n🔎 Testing biochemistry search...")

    try:
        tool = BiochemSearchTool({})

        # Test cases: (query, entity_type, min_expected_results)
        test_cases = [
            ("glucose", None, 1),
            ("ATP", "compound", 1),
            ("kinase", "reaction", 1),
            ("phospho", None, 2),
            ("amino", None, 1),
        ]

        all_passed = True
        for query, entity_type, min_results in test_cases:
            try:
                input_data = {"query": query}
                if entity_type:
                    input_data["entity_type"] = entity_type

                result = tool._run(input_data)

                if result.success:
                    result_count = result.data["result_count"]
                    if result_count >= min_results:
                        print(
                            f"✅ Search '{query}' → {result_count} results (≥{min_results})"
                        )

                        # Show first result as example
                        if result.data["search_results"]:
                            first_result = result.data["search_results"][0]
                            print(
                                f"   Example: {first_result['id']} → '{first_result['name']}'"
                            )
                    else:
                        print(
                            f"❌ Search '{query}' → {result_count} results (expected ≥{min_results})"
                        )
                        all_passed = False
                else:
                    print(f"❌ Search '{query}' → Failed: {result.message}")
                    all_passed = False

            except Exception as e:
                print(f"❌ Search '{query}' → Error: {e}")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"❌ Biochemistry search test failed: {e}")
        return False


def test_convenience_functions():
    """Test convenience functions for quick resolution"""

    print("\n⚡ Testing convenience functions...")

    try:
        # Test quick resolve function
        test_cases = [
            ("cpd00001", "H2O"),
            ("cpd00002", "ATP"),
            ("rxn00781", "Phosphoglycerate"),
            ("nonexistent_id", None),
        ]

        all_passed = True
        for entity_id, expected in test_cases:
            try:
                resolved_name = resolve_entity_id(entity_id)

                if expected is None:
                    if resolved_name is None:
                        print(f"✅ {entity_id} → None (expected for nonexistent ID)")
                    else:
                        print(f"❌ {entity_id} → '{resolved_name}' (expected None)")
                        all_passed = False
                else:
                    if resolved_name and expected.lower() in resolved_name.lower():
                        print(
                            f"✅ {entity_id} → '{resolved_name}' (contains '{expected}')"
                        )
                    else:
                        print(
                            f"❌ {entity_id} → '{resolved_name}' (expected '{expected}')"
                        )
                        all_passed = False

            except Exception as e:
                print(f"❌ {entity_id} → Error: {e}")
                all_passed = False

        # Test tool output enhancement
        try:
            mock_output = {
                "reaction_id": "rxn00781",
                "compound_ids": ["cpd00001", "cpd00002"],
                "other_data": "some_value",
            }

            enhanced = enhance_tool_output_with_names(mock_output)

            # Check if names were added
            has_enhancements = any(key.endswith("_name") for key in enhanced.keys())
            if has_enhancements:
                print("✅ Tool output enhancement working")
                for key, value in enhanced.items():
                    if key.endswith("_name"):
                        print(f"   Added: {key} = '{value}'")
            else:
                print("❌ Tool output enhancement not working")
                all_passed = False

        except Exception as e:
            print(f"❌ Tool output enhancement failed: {e}")
            all_passed = False

        return all_passed

    except Exception as e:
        print(f"❌ Convenience functions test failed: {e}")
        return False


def test_cross_references():
    """Test cross-reference functionality"""

    print("\n🔗 Testing cross-reference lookup...")

    try:
        tool = BiochemEntityResolverTool({})

        # Test with entities that should have cross-references
        test_cases = ["cpd00001", "cpd00002", "rxn00001"]

        all_passed = True
        for entity_id in test_cases:
            try:
                result = tool._run({"entity_id": entity_id, "include_cross_refs": True})

                if result.success:
                    cross_refs = result.data["resolved_entity"]["cross_references"]
                    if cross_refs:
                        print(f"✅ {entity_id} has cross-references:")
                        for db, ids in cross_refs.items():
                            print(f"   {db}: {ids}")
                    else:
                        print(
                            f"⚠️  {entity_id} has no cross-references (may be expected)"
                        )
                else:
                    print(f"❌ {entity_id} → Failed: {result.message}")
                    all_passed = False

            except Exception as e:
                print(f"❌ {entity_id} → Error: {e}")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"❌ Cross-reference test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling"""

    print("\n🚨 Testing edge cases and error handling...")

    try:
        resolver_tool = BiochemEntityResolverTool({})
        search_tool = BiochemSearchTool({})

        test_cases = [
            # Empty inputs
            ("resolver", {}, False, "empty entity_id"),
            ("resolver", {"entity_id": ""}, False, "empty string entity_id"),
            ("search", {}, False, "empty query"),
            ("search", {"query": ""}, False, "empty string query"),
            # Invalid inputs
            (
                "resolver",
                {"entity_id": "nonexistent_12345"},
                False,
                "nonexistent entity",
            ),
            ("search", {"query": "xyzzylakjsdf"}, False, "nonsense query"),
            (
                "search",
                {"query": "ATP", "entity_type": "invalid"},
                False,
                "invalid entity_type",
            ),
            # Valid edge cases
            ("search", {"query": "ATP", "max_results": 1}, True, "limited results"),
            (
                "search",
                {"query": "ATP", "entity_type": "compound"},
                True,
                "compound filter",
            ),
            (
                "search",
                {"query": "kinase", "entity_type": "reaction"},
                True,
                "reaction filter",
            ),
        ]

        all_passed = True
        for tool_type, input_data, should_succeed, description in test_cases:
            try:
                if tool_type == "resolver":
                    result = resolver_tool._run(input_data)
                else:
                    result = search_tool._run(input_data)

                if should_succeed:
                    if result.success:
                        print(f"✅ {description} → Success as expected")
                    else:
                        print(
                            f"❌ {description} → Failed unexpectedly: {result.message}"
                        )
                        all_passed = False
                else:
                    if not result.success:
                        print(
                            f"✅ {description} → Failed as expected: {result.message}"
                        )
                    else:
                        print(f"❌ {description} → Succeeded unexpectedly")
                        all_passed = False

            except Exception as e:
                print(f"❌ {description} → Exception: {e}")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        return False


def main():
    """Run all Phase 3 biochemistry resolution tests"""

    print("🧬 ModelSEEDagent Phase 3: Biochemistry Resolution Testing")
    print("=" * 70)

    tests = [
        ("Database Connection", test_database_connection),
        ("Entity Resolution", test_entity_resolution),
        ("Biochemistry Search", test_biochem_search),
        ("Convenience Functions", test_convenience_functions),
        ("Cross-References", test_cross_references),
        ("Edge Cases & Error Handling", test_edge_cases),
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
    print("📊 PHASE 3 BIOCHEMISTRY RESOLUTION TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All Phase 3 biochemistry resolution tests PASSED!")
        print("✅ Universal ID resolution system working correctly")
        print("✅ Agent can now reason about biochemistry names instead of IDs")
        return True
    else:
        print("\n⚠️  Some biochemistry resolution issues detected")
        print("❌ Review failed tests and address resolution gaps")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
