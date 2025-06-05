#!/usr/bin/env python3
"""
Test Phase 2: ModelSEED-COBRApy Compatibility Verification

This test validates the SBML round-trip compatibility and ensures ModelSEED
models work seamlessly with existing COBRA tools.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import cobra
    import cobrakbase
    import modelseedpy
    from src.tools.modelseed.compatibility import (
        ModelCompatibilityTool,
        test_modelseed_cobra_pipeline,
        verify_cobra_tool_compatibility
    )
    from src.tools.cobra.utils import ModelUtils
    
    print("✅ All imports successful")
    print(f"COBRApy version: {cobra.__version__}")
    print(f"cobrakbase version: {cobrakbase.__version__}")
    print(f"modelseedpy version: {modelseedpy.__version__}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_compatibility_tool_basic():
    """Test basic functionality of ModelCompatibilityTool"""
    
    print("\n🧪 Testing ModelCompatibilityTool basic functionality...")
    
    try:
        # Load e_coli_core model for testing
        model_path = Path("data/models/e_coli_core.xml")
        if not model_path.exists():
            print(f"❌ Test model not found: {model_path}")
            return False
        
        model = ModelUtils().load_model(str(model_path))
        print(f"✅ Loaded test model: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites")
        
        # Initialize compatibility tool
        tool = ModelCompatibilityTool({})
        
        # Test compatibility
        result = tool._run({
            "modelseed_model": model,
            "media_condition": "Complete"
        })
        
        print(f"✅ Compatibility test completed")
        print(f"   Success: {result.success}")
        print(f"   Message: {result.message}")
        
        # Check specific metrics
        metrics = result.data["compatibility_metrics"]
        print(f"   SBML round-trip: {metrics['sbml_roundtrip_success']}")
        print(f"   Growth compatible: {metrics['growth_compatible']}")
        print(f"   Growth difference: {metrics['growth_difference']:.8f}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Compatibility tool test failed: {e}")
        return False


def test_sbml_roundtrip():
    """Test SBML round-trip conversion specifically"""
    
    print("\n🔄 Testing SBML round-trip conversion...")
    
    try:
        # Load model
        model_path = Path("data/models/e_coli_core.xml")
        if not model_path.exists():
            print(f"❌ Test model not found: {model_path}")
            return False
        
        original_model = ModelUtils().load_model(str(model_path))
        
        # Test round-trip: Model → SBML → COBRApy
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        # Export to SBML
        original_model.write_sbml_file(tmp_path)
        print("✅ Exported model to SBML")
        
        # Import with COBRApy
        cobra_model = cobra.io.read_sbml_model(tmp_path)
        print("✅ Imported SBML with COBRApy")
        
        # Compare structures
        print(f"   Original: {len(original_model.reactions)} rxns, {len(original_model.metabolites)} mets")
        print(f"   COBRApy:  {len(cobra_model.reactions)} rxns, {len(cobra_model.metabolites)} mets")
        
        # Compare growth rates
        with original_model:
            orig_solution = original_model.optimize()
            orig_growth = orig_solution.objective_value if orig_solution.status == 'optimal' else 0
        
        with cobra_model:
            cobra_solution = cobra_model.optimize()
            cobra_growth = cobra_solution.objective_value if cobra_solution.status == 'optimal' else 0
        
        growth_diff = abs(orig_growth - cobra_growth)
        print(f"   Growth rates - Original: {orig_growth:.6f}, COBRApy: {cobra_growth:.6f}")
        print(f"   Growth difference: {growth_diff:.8f}")
        
        # Clean up
        Path(tmp_path).unlink()
        
        # Success if growth rates match within tolerance
        tolerance = 1e-6
        success = growth_diff <= tolerance
        print(f"   Round-trip {'✅ PASSED' if success else '❌ FAILED'} (tolerance: {tolerance})")
        
        return success
        
    except Exception as e:
        print(f"❌ SBML round-trip test failed: {e}")
        return False


def test_cobra_tool_compatibility():
    """Test compatibility with specific COBRApy tools"""
    
    print("\n🔧 Testing COBRApy tool compatibility...")
    
    try:
        # Load model
        model_path = Path("data/models/e_coli_core.xml")
        if not model_path.exists():
            print(f"❌ Test model not found: {model_path}")
            return False
        
        model = ModelUtils().load_model(str(model_path))
        
        # Test specific COBRApy tools
        tools_to_test = ["fba", "fva", "gene_deletion", "flux_sampling"]
        compatibility = verify_cobra_tool_compatibility(model, tools_to_test)
        
        print("   COBRApy tool compatibility:")
        all_compatible = True
        for tool_name, is_compatible in compatibility.items():
            status = "✅ PASS" if is_compatible else "❌ FAIL"
            print(f"     {tool_name}: {status}")
            if not is_compatible:
                all_compatible = False
        
        return all_compatible
        
    except Exception as e:
        print(f"❌ COBRApy tool compatibility test failed: {e}")
        return False


def test_convenience_function():
    """Test the convenience pipeline function"""
    
    print("\n🚀 Testing convenience pipeline function...")
    
    try:
        model_path = Path("data/models/e_coli_core.xml")
        if not model_path.exists():
            print(f"❌ Test model not found: {model_path}")
            return False
        
        # Test pipeline function
        result = test_modelseed_cobra_pipeline(str(model_path), "Complete")
        
        print("✅ Pipeline function completed")
        print(f"   Compatibility metrics available: {len(result['compatibility_metrics'])} metrics")
        print(f"   Recommendations: {len(result['recommendations'])} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline function test failed: {e}")
        return False


def main():
    """Run all Phase 2 compatibility tests"""
    
    print("🧬 ModelSEEDagent Phase 2: Compatibility Testing")
    print("=" * 60)
    
    tests = [
        ("Basic Compatibility Tool", test_compatibility_tool_basic),
        ("SBML Round-trip", test_sbml_roundtrip),
        ("COBRApy Tool Compatibility", test_cobra_tool_compatibility),
        ("Convenience Pipeline", test_convenience_function),
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
    print("\n" + "=" * 60)
    print("📊 PHASE 2 COMPATIBILITY TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 2 compatibility tests PASSED!")
        print("✅ ModelSEED models are fully compatible with COBRApy tools")
        return True
    else:
        print("⚠️  Some compatibility issues detected")
        print("❌ Review failed tests and address compatibility gaps")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)