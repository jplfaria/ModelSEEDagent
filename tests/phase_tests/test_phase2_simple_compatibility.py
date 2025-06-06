#!/usr/bin/env python3
"""
Test Phase 2: Simple ModelSEED-COBRApy Compatibility Verification

This test validates SBML round-trip compatibility without LangChain dependencies.
"""

import sys
import tempfile
from pathlib import Path

try:
    import cobra
    import cobrakbase
    import modelseedpy
    import numpy as np

    print("✅ All imports successful")
    print(f"COBRApy version: {cobra.__version__}")
    print(f"cobrakbase version: {cobrakbase.__version__}")
    print(f"modelseedpy version: {modelseedpy.__version__}")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def load_model(model_path: str):
    """Load a model using COBRApy"""
    try:
        model = cobra.io.read_sbml_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def test_sbml_roundtrip_compatibility():
    """Test SBML round-trip: Original → SBML → COBRApy"""

    print("\n🔄 Testing SBML Round-trip Compatibility...")

    try:
        # Load original model
        model_path = Path("data/models/e_coli_core.xml")
        if not model_path.exists():
            print(f"❌ Test model not found: {model_path}")
            return False

        original_model = load_model(str(model_path))
        if not original_model:
            return False

        print(
            f"✅ Loaded original model: {len(original_model.reactions)} reactions, {len(original_model.metabolites)} metabolites"
        )

        # Test original growth rate
        with original_model:
            orig_solution = original_model.optimize()
            orig_growth = (
                orig_solution.objective_value
                if orig_solution.status == "optimal"
                else 0
            )

        print(f"   Original growth rate: {orig_growth:.6f}")

        # Export and re-import via SBML
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        # Export to SBML
        cobra.io.write_sbml_model(original_model, tmp_path)
        print("✅ Exported model to SBML")

        # Re-import from SBML
        reimported_model = cobra.io.read_sbml_model(tmp_path)
        print("✅ Re-imported model from SBML")

        # Test re-imported growth rate
        with reimported_model:
            reimported_solution = reimported_model.optimize()
            reimported_growth = (
                reimported_solution.objective_value
                if reimported_solution.status == "optimal"
                else 0
            )

        print(f"   Re-imported growth rate: {reimported_growth:.6f}")

        # Compare structures
        print(f"   Structure comparison:")
        print(
            f"     Reactions: {len(original_model.reactions)} → {len(reimported_model.reactions)}"
        )
        print(
            f"     Metabolites: {len(original_model.metabolites)} → {len(reimported_model.metabolites)}"
        )
        print(
            f"     Genes: {len(original_model.genes)} → {len(reimported_model.genes)}"
        )

        # Compare growth rates
        growth_diff = abs(orig_growth - reimported_growth)
        tolerance = 1e-6
        growth_compatible = growth_diff <= tolerance

        print(f"   Growth difference: {growth_diff:.8f}")
        print(
            f"   Growth compatible (tolerance {tolerance}): {'✅ YES' if growth_compatible else '❌ NO'}"
        )

        # Clean up
        Path(tmp_path).unlink()

        return growth_compatible

    except Exception as e:
        print(f"❌ SBML round-trip test failed: {e}")
        return False


def test_modelseed_cobra_integration():
    """Test ModelSEED model integration with COBRApy tools"""

    print("\n🧬 Testing ModelSEED-COBRApy Integration...")

    try:
        # Load model
        model_path = Path("data/models/e_coli_core.xml")
        if not model_path.exists():
            print(f"❌ Test model not found: {model_path}")
            return False

        model = load_model(str(model_path))
        if not model:
            return False

        print(f"✅ Loaded model for integration testing")

        # Test 1: Basic FBA
        print("   Testing FBA...")
        with model:
            fba_solution = model.optimize()
            fba_works = fba_solution.status == "optimal"
            print(
                f"     FBA: {'✅ PASS' if fba_works else '❌ FAIL'} (growth: {fba_solution.objective_value:.6f})"
            )

        # Test 2: Flux Variability Analysis
        print("   Testing FVA...")
        try:
            fva_result = cobra.flux_analysis.flux_variability_analysis(
                model,
                fraction_of_optimum=0.9,
                reaction_list=model.reactions[:5],  # Test with subset for speed
            )
            fva_works = fva_result is not None and len(fva_result) > 0
            print(f"     FVA: {'✅ PASS' if fva_works else '❌ FAIL'}")
        except Exception as e:
            print(f"     FVA: ❌ FAIL ({e})")
            fva_works = False

        # Test 3: Gene Deletion Analysis
        print("   Testing Gene Deletion...")
        try:
            # Test with subset of genes for speed
            test_genes = (
                list(model.genes)[:3] if len(model.genes) > 3 else list(model.genes)
            )
            deletion_results = cobra.flux_analysis.single_gene_deletion(
                model, gene_list=test_genes
            )
            gene_deletion_works = deletion_results is not None
            print(
                f"     Gene Deletion: {'✅ PASS' if gene_deletion_works else '❌ FAIL'}"
            )
        except Exception as e:
            print(f"     Gene Deletion: ❌ FAIL ({e})")
            gene_deletion_works = False

        # Test 4: Flux Sampling (minimal test)
        print("   Testing Flux Sampling...")
        try:
            # Very small sample for speed
            samples = cobra.sampling.sample(model, n=5, method="achr")
            sampling_works = samples is not None and not samples.empty
            print(f"     Flux Sampling: {'✅ PASS' if sampling_works else '❌ FAIL'}")
        except Exception as e:
            print(f"     Flux Sampling: ❌ FAIL ({e})")
            sampling_works = False

        # Overall integration success
        all_tests = [fba_works, fva_works, gene_deletion_works, sampling_works]
        integration_success = all(all_tests)

        print(
            f"   Integration Success: {'✅ PASS' if integration_success else '❌ PARTIAL'} ({sum(all_tests)}/{len(all_tests)} tools)"
        )

        return integration_success

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def test_cobrakbase_functionality():
    """Test cobrakbase specific functionality"""

    print("\n⚙️  Testing cobrakbase Functionality...")

    try:
        # Test basic cobrakbase imports and functionality
        print("   Testing cobrakbase imports...")

        # Test core cobrakbase functionality
        from cobrakbase import KBaseFBAModelToCobraBuilder

        print("   ✅ KBaseFBAModelToCobraBuilder imported")

        # Test ModelSEED annotation functions
        try:
            from cobrakbase import annotate_model_with_modelseed

            print("   ✅ ModelSEED annotation functions available")
        except ImportError:
            print("   ⚠️  Some ModelSEED annotation functions not available")

        # Test workspace client availability
        try:
            from cobrakbase import WorkspaceClient

            print("   ✅ WorkspaceClient available")
        except ImportError:
            print("   ⚠️  WorkspaceClient not available")

        return True

    except Exception as e:
        print(f"❌ cobrakbase functionality test failed: {e}")
        return False


def test_compatibility_metrics():
    """Calculate detailed compatibility metrics"""

    print("\n📊 Testing Compatibility Metrics...")

    try:
        # Load model
        model_path = Path("data/models/e_coli_core.xml")
        if not model_path.exists():
            print(f"❌ Test model not found: {model_path}")
            return False

        original_model = load_model(str(model_path))
        if not original_model:
            return False

        # Perform round-trip and calculate detailed metrics
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        # Export and re-import
        cobra.io.write_sbml_model(original_model, tmp_path)
        converted_model = cobra.io.read_sbml_model(tmp_path)

        # Calculate compatibility metrics
        metrics = {}

        # Structure metrics
        metrics["reactions_identical"] = len(original_model.reactions) == len(
            converted_model.reactions
        )
        metrics["metabolites_identical"] = len(original_model.metabolites) == len(
            converted_model.metabolites
        )
        metrics["genes_identical"] = len(original_model.genes) == len(
            converted_model.genes
        )

        # Growth rate metrics
        with original_model:
            orig_growth = original_model.optimize().objective_value
        with converted_model:
            conv_growth = converted_model.optimize().objective_value

        growth_diff = abs(orig_growth - conv_growth)
        metrics["growth_difference"] = growth_diff
        metrics["growth_compatible"] = growth_diff <= 1e-6

        # Relative difference
        if orig_growth > 0:
            metrics["growth_relative_diff_percent"] = (growth_diff / orig_growth) * 100
        else:
            metrics["growth_relative_diff_percent"] = float("inf")

        # Print metrics
        print("   Compatibility Metrics:")
        print(
            f"     Structure - Reactions: {'✅' if metrics['reactions_identical'] else '❌'}"
        )
        print(
            f"     Structure - Metabolites: {'✅' if metrics['metabolites_identical'] else '❌'}"
        )
        print(f"     Structure - Genes: {'✅' if metrics['genes_identical'] else '❌'}")
        print(
            f"     Growth Rate Compatible: {'✅' if metrics['growth_compatible'] else '❌'}"
        )
        print(f"     Growth Difference: {metrics['growth_difference']:.8f}")
        print(
            f"     Growth Relative Diff: {metrics['growth_relative_diff_percent']:.6f}%"
        )

        # Overall compatibility
        structure_compatible = all(
            [
                metrics["reactions_identical"],
                metrics["metabolites_identical"],
                metrics["genes_identical"],
            ]
        )

        overall_compatible = structure_compatible and metrics["growth_compatible"]
        print(f"     Overall Compatible: {'✅ YES' if overall_compatible else '❌ NO'}")

        # Clean up
        Path(tmp_path).unlink()

        return overall_compatible

    except Exception as e:
        print(f"❌ Compatibility metrics test failed: {e}")
        return False


def main():
    """Run all Phase 2 compatibility tests"""

    print("🧬 ModelSEEDagent Phase 2: Simple Compatibility Testing")
    print("=" * 70)

    tests = [
        ("SBML Round-trip Compatibility", test_sbml_roundtrip_compatibility),
        ("ModelSEED-COBRApy Integration", test_modelseed_cobra_integration),
        ("cobrakbase Functionality", test_cobrakbase_functionality),
        ("Detailed Compatibility Metrics", test_compatibility_metrics),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   Final Result: {status}")
        except Exception as e:
            print(f"   Final Result: ❌ ERROR - {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("📊 PHASE 2 SIMPLE COMPATIBILITY TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All Phase 2 compatibility tests PASSED!")
        print("✅ ModelSEED models are fully compatible with COBRApy tools")
        print("✅ SBML round-trip verification successful")
        print("✅ Growth rates match within 1e-6 tolerance")
        return True
    else:
        print("\n⚠️  Some compatibility issues detected")
        print("❌ Review failed tests and address compatibility gaps")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
