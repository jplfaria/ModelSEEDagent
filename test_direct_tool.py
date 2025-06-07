#!/usr/bin/env python3
"""
Direct tool test - bypasses LLM to test core functionality
"""

import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tools.cobra.fba import FBATool


def test_direct_fba():
    """Test FBA tool directly without agent"""
    print("🧪 Testing FBA tool directly...")
    
    fba_tool = FBATool({"name": "test_fba"})
    
    # Test with E. coli core model
    result = fba_tool._run_tool({"model_path": "data/examples/e_coli_core.xml"})
    
    if result.success:
        print("✅ FBA tool works!")
        print(f"📊 Growth rate: {result.data.get('objective_value', 'N/A')}")
        print(f"📝 Message: {result.message}")
        
        # Show some flux data
        sig_fluxes = result.data.get("significant_fluxes", {})
        if sig_fluxes:
            print(f"🔧 Found {len(sig_fluxes)} significant fluxes")
            
        return True
    else:
        print(f"❌ FBA failed: {result.error}")
        return False


def test_model_file():
    """Test if model file exists"""
    model_path = "data/examples/e_coli_core.xml"
    
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
        file_size = os.path.getsize(model_path)
        print(f"📊 File size: {file_size:,} bytes")
        return True
    else:
        print(f"❌ Model file missing: {model_path}")
        return False


def main():
    """Run direct tests"""
    print("🔧 ModelSEEDagent Direct Tool Test")
    print("=" * 50)
    print("Testing tools without LLM to isolate issues\n")
    
    # Test model file
    model_ok = test_model_file()
    print()
    
    if model_ok:
        # Test FBA tool
        fba_ok = test_direct_fba()
        print()
        
        if fba_ok:
            print("✅ Core functionality works!")
            print("🔍 Issue is likely with LLM integration")
            print("\nSuggestions:")
            print("1. Check network connectivity to Argo Gateway")
            print("2. Try with explicit model path in query")
            print("3. The timeout protection should now prevent hanging")
        else:
            print("❌ Core tools have issues")
    else:
        print("❌ Model file is missing")
        
    print(f"\nThe agent should now have 30-second timeouts on LLM calls")
    print("Try the interactive CLI again - it should not hang!")


if __name__ == "__main__":
    main()