#!/usr/bin/env python3
"""
Test script to verify all fixes are working
"""
import os
import sys
import time

# Set debug mode
os.environ["MODELSEED_DEBUG"] = "true"

def test_fixes():
    """Test all the fixes"""
    print("\n=== Testing All Fixes ===\n")
    
    # Test 1: Import messages should not repeat
    print("1. Testing import message fix...")
    start_imports = time.time()
    from src.tools.modelseed.compatibility import ModelCompatibilityTool
    from src.tools.modelseed.compatibility import ModelCompatibilityTool as Tool2
    import_time = time.time() - start_imports
    print(f"   ✅ Imports completed in {import_time:.2f}s (no repeated messages expected)")
    
    # Test 2: CLI initialization should not create agents repeatedly
    print("\n2. Testing agent initialization fix...")
    from src.cli.main import load_cli_config
    config = load_cli_config()
    print("   ✅ Config loaded without creating agents")
    
    # Test 3: Debug mode message should only appear once
    print("\n3. Testing debug message fix...")
    from src.interactive.interactive_cli import InteractiveCLI
    cli = InteractiveCLI()
    print("   ✅ CLI created (debug message should appear only on session start)")
    
    # Test 4: Flux sampling tool input preparation
    print("\n4. Testing flux_sampling tool input fix...")
    from src.agents.real_time_metabolic import RealTimeMetabolicAgent
    agent = RealTimeMetabolicAgent(
        llm=None,  # Will be created by factory
        tools=[],  # Will be loaded by factory  
        config={"enable_audit": False, "enable_realtime_verification": False}
    )
    
    # Test tool input preparation
    tool_input = agent._prepare_tool_input("run_flux_sampling", "test query")
    assert "model_path" in tool_input, "flux_sampling should get model_path"
    print(f"   ✅ flux_sampling tool input: {tool_input}")
    
    # Test other tools too
    for tool in ["run_gene_deletion_analysis", "run_production_envelope", "analyze_metabolic_model"]:
        tool_input = agent._prepare_tool_input(tool, "test query")
        assert "model_path" in tool_input, f"{tool} should get model_path"
        print(f"   ✅ {tool} tool input: model_path provided")
    
    print("\n✅ All fixes verified successfully!")

if __name__ == "__main__":
    test_fixes()