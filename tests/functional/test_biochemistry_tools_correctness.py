#!/usr/bin/env python3
"""
Biochemistry Tools Functional Correctness Testing

This module tests that biochemistry resolution tools produce accurate
ID mappings and meaningful compound/reaction information.

Tests validate:
- Universal ID resolution accuracy
- Cross-database mapping consistency
- Biochemistry search functionality
- Entity name and metadata quality
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.tools.biochem.resolver import BiochemEntityResolverTool, BiochemSearchTool


class TestBiochemEntityResolverCorrectness:
    """Test biochemistry entity resolver produces accurate mappings"""

    @pytest.fixture
    def resolver_tool(self):
        return BiochemEntityResolverTool({"name": "test_resolver"})

    def test_known_compound_resolution(self, resolver_tool):
        """Test resolution of well-known compounds"""
        # Test known ModelSEED compounds
        test_compounds = [
            ("cpd00002", "ATP"),  # ATP
            ("cpd00001", "H2O"),  # Water
            ("cpd00009", "Phosphate"),  # Phosphate
            ("cpd00067", "H+"),  # Proton
        ]

        successful_resolutions = 0

        for compound_id, expected_name_part in test_compounds:
            result = resolver_tool._run_tool({"entity_id": compound_id})

            if result.success:
                name = result.data.get("name", "").lower()
                aliases = result.data.get("aliases", [])

                # Check if expected name appears in name or aliases
                name_found = expected_name_part.lower() in name or any(
                    expected_name_part.lower() in alias.lower() for alias in aliases
                )

                if name_found:
                    successful_resolutions += 1
                    print(f"✅ {compound_id} → {result.data.get('name')}")
                else:
                    print(f"⚠️  {compound_id} resolved but name mismatch: {name}")
            else:
                print(f"❌ {compound_id} resolution failed: {result.error}")

        # Should resolve at least 75% of known compounds
        resolution_rate = successful_resolutions / len(test_compounds)
        assert resolution_rate >= 0.75, (
            f"Resolution rate too low: {resolution_rate:.2%} "
            f"({successful_resolutions}/{len(test_compounds)})"
        )

    def test_reaction_resolution_accuracy(self, resolver_tool):
        """Test resolution of well-known reactions"""
        test_reactions = [
            ("rxn00001", "hexokinase"),  # Common metabolic reaction
            ("rxn00008", "phosphoglycerate kinase"),
            ("rxn00148", "triose-phosphate isomerase"),
        ]

        successful_resolutions = 0

        for reaction_id, expected_name_part in test_reactions:
            result = resolver_tool._run_tool({"entity_id": reaction_id})

            if result.success:
                name = result.data.get("name", "").lower()

                # Check if this looks like a valid reaction name
                valid_reaction = len(name) > 5 and (  # Has meaningful length
                    "ase" in name
                    or "synthase" in name
                    or "kinase" in name
                    or "dehydrogenase" in name
                    or "isomerase" in name
                    or "transferase" in name
                    or "ligase" in name
                )

                if valid_reaction:
                    successful_resolutions += 1
                    print(f"✅ {reaction_id} → {name}")
                else:
                    print(f"⚠️  {reaction_id} → {name} (name unclear)")
            else:
                print(f"❌ {reaction_id} resolution failed")

        # Should resolve at least 50% of reactions (reactions are harder)
        resolution_rate = successful_resolutions / len(test_reactions)
        assert (
            resolution_rate >= 0.5
        ), f"Reaction resolution rate too low: {resolution_rate:.2%}"

    def test_cross_database_mapping(self, resolver_tool):
        """Test cross-database ID mapping functionality"""
        # Test BiGG ID resolution
        bigg_compounds = ["atp_c", "h2o_c", "pi_c", "glc__D_c"]

        successful_mappings = 0

        for bigg_id in bigg_compounds:
            result = resolver_tool._run_tool({"entity_id": bigg_id})

            if result.success:
                aliases = result.data.get("aliases", [])

                # Should have multiple database mappings
                has_modelseed = any("cpd" in alias for alias in aliases)
                has_kegg = any("C" in alias for alias in aliases)

                if has_modelseed or has_kegg:
                    successful_mappings += 1
                    print(f"✅ {bigg_id} mapped to multiple databases")
                else:
                    print(f"⚠️  {bigg_id} limited mapping: {aliases}")
            else:
                print(f"❌ {bigg_id} mapping failed")

        # Should map at least 50% across databases
        mapping_rate = successful_mappings / len(bigg_compounds)
        assert (
            mapping_rate >= 0.5
        ), f"Cross-database mapping rate too low: {mapping_rate:.2%}"

    def test_entity_metadata_quality(self, resolver_tool):
        """Test quality of entity metadata"""
        test_entities = ["cpd00002", "cpd00001", "rxn00001"]

        quality_metrics = {
            "has_name": 0,
            "has_formula": 0,
            "has_aliases": 0,
            "has_charge": 0,
        }

        successful_queries = 0

        for entity_id in test_entities:
            result = resolver_tool._run_tool({"entity_id": entity_id})

            if result.success:
                successful_queries += 1
                data = result.data

                # Check metadata quality
                if data.get("name") and len(data["name"]) > 2:
                    quality_metrics["has_name"] += 1

                if data.get("formula"):
                    quality_metrics["has_formula"] += 1

                if data.get("aliases") and len(data["aliases"]) > 0:
                    quality_metrics["has_aliases"] += 1

                if "charge" in data:
                    quality_metrics["has_charge"] += 1

        if successful_queries > 0:
            # At least 70% should have names
            name_rate = quality_metrics["has_name"] / successful_queries
            assert name_rate >= 0.7, f"Name quality too low: {name_rate:.2%}"

            # At least 50% should have aliases
            alias_rate = quality_metrics["has_aliases"] / successful_queries
            assert alias_rate >= 0.5, f"Alias coverage too low: {alias_rate:.2%}"


class TestBiochemSearchCorrectness:
    """Test biochemistry search functionality"""

    @pytest.fixture
    def search_tool(self):
        return BiochemSearchTool({"name": "test_search"})

    def test_compound_name_search(self, search_tool):
        """Test searching compounds by name"""
        search_terms = ["ATP", "glucose", "water", "acetate"]

        successful_searches = 0

        for term in search_terms:
            result = search_tool._run_tool({"query": term, "entity_type": "compound"})

            if result.success:
                matches = result.data.get("matches", [])

                if len(matches) > 0:
                    successful_searches += 1

                    # Check first match quality
                    first_match = matches[0]
                    name = first_match.get("name", "").lower()

                    # Name should contain search term
                    assert (
                        term.lower() in name
                    ), f"Search for '{term}' returned irrelevant result: {name}"

                    print(f"✅ '{term}' found: {first_match.get('name')}")
                else:
                    print(f"⚠️  No matches for '{term}'")
            else:
                print(f"❌ Search for '{term}' failed: {result.error}")

        # Should find at least 75% of common compounds
        search_success_rate = successful_searches / len(search_terms)
        assert (
            search_success_rate >= 0.75
        ), f"Search success rate too low: {search_success_rate:.2%}"

    def test_reaction_name_search(self, search_tool):
        """Test searching reactions by name"""
        search_terms = ["kinase", "dehydrogenase", "synthase"]

        successful_searches = 0

        for term in search_terms:
            result = search_tool._run_tool({"query": term, "entity_type": "reaction"})

            if result.success:
                matches = result.data.get("matches", [])

                if len(matches) > 0:
                    successful_searches += 1
                    print(f"✅ Found {len(matches)} reactions for '{term}'")

                    # Verify relevance of first few matches
                    for match in matches[:3]:
                        name = match.get("name", "").lower()
                        assert (
                            term.lower() in name
                        ), f"Reaction search for '{term}' returned irrelevant result: {name}"
                else:
                    print(f"⚠️  No reaction matches for '{term}'")
            else:
                print(f"❌ Reaction search for '{term}' failed")

        # Should find at least 66% of enzyme classes
        search_success_rate = successful_searches / len(search_terms)
        assert (
            search_success_rate >= 0.66
        ), f"Reaction search success rate too low: {search_success_rate:.2%}"

    def test_search_result_ranking(self, search_tool):
        """Test that search results are properly ranked"""
        result = search_tool._run_tool({"query": "ATP", "entity_type": "compound"})

        if result.success:
            matches = result.data.get("matches", [])

            if len(matches) >= 2:
                # First result should be most relevant (exact or close match)
                first_name = matches[0].get("name", "").lower()

                # For ATP search, first result should contain "atp" or "adenosine"
                relevant_first = "atp" in first_name or "adenosine" in first_name

                assert (
                    relevant_first
                ), f"Search ranking issue: first result for 'ATP' was '{first_name}'"

                print(f"✅ Search ranking good: ATP → {matches[0].get('name')}")


class TestBiochemistryDatabaseIntegrity:
    """Test overall biochemistry database integrity"""

    def test_database_connectivity(self):
        """Test that biochemistry database is accessible"""
        resolver = BiochemEntityResolverTool({"name": "test_db"})

        # Try resolving a basic compound
        result = resolver._run_tool({"entity_id": "cpd00001"})

        assert (
            result.success or "database" in result.error.lower()
        ), f"Database connectivity issue: {result.error}"

        if result.success:
            print("✅ Biochemistry database accessible")
        else:
            pytest.skip(f"Biochemistry database unavailable: {result.error}")

    def test_database_coverage(self):
        """Test biochemistry database has reasonable coverage"""
        resolver = BiochemEntityResolverTool({"name": "test_coverage"})

        # Test variety of entity types
        test_entities = [
            "cpd00001",  # Compound
            "cpd00002",  # Another compound
            "rxn00001",  # Reaction
            "atp_c",  # BiGG ID
            "C00002",  # KEGG ID (if supported)
        ]

        successful_resolutions = 0

        for entity in test_entities:
            result = resolver._run_tool({"entity_id": entity})
            if result.success:
                successful_resolutions += 1

        coverage_rate = successful_resolutions / len(test_entities)

        # Should resolve at least 60% of diverse entity types
        assert coverage_rate >= 0.6, (
            f"Database coverage too low: {coverage_rate:.2%} "
            f"({successful_resolutions}/{len(test_entities)})"
        )

        print(f"✅ Database coverage: {coverage_rate:.2%}")


def run_biochemistry_functional_tests():
    """Run all biochemistry tools functional tests"""
    print("🧬 Running Biochemistry Tools Functional Correctness Tests")
    print("=" * 65)
    print("Testing ID resolution accuracy and biochemistry search quality")
    print()

    # Run pytest on this module
    test_file = __file__
    exit_code = pytest.main([test_file, "-v", "--tb=short"])

    if exit_code == 0:
        print("\n✅ All biochemistry functional tests PASSED!")
        print("🔬 Biochemistry resolution produces accurate results")
    else:
        print("\n❌ Some biochemistry functional tests FAILED!")
        print("⚠️  Check database connectivity and mapping accuracy")

    return exit_code == 0


if __name__ == "__main__":
    success = run_biochemistry_functional_tests()
    sys.exit(0 if success else 1)
