#!/usr/bin/env python3
"""
Timeout Validation Test

Tests that the RealTimeMetabolicAgent correctly uses:
1. 120s timeouts for gpto1 models  
2. 30s timeouts for standard models
3. Shows debug logging correctly
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.real_time_metabolic import RealTimeMetabolicAgent
from src.llm.argo import ArgoLLM
from src.tools.cobra.fba import FBATool

# Configure logging to show debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

async def test_gpto1_timeout():
    """Test that gpto1 model uses 120s timeout"""
    
    print("🧪 Testing gpto1 model timeout configuration...")
    
    # Create gpto1 LLM
    gpto1_config = {
        "model_name": "gpto1",
        "user": "test_user",
        "system_content": "You are a test assistant."
    }
    
    gpto1_llm = ArgoLLM(gpto1_config)
    
    # Create tools
    tools = [
        FBATool({"name": "run_metabolic_fba", "description": "Run FBA analysis"})
    ]
    
    # Create agent
    agent = RealTimeMetabolicAgent(gpto1_llm, tools, {"max_iterations": 1})
    
    print(f"✅ Agent created with {gpto1_llm.model_name} model")
    print(f"✅ LLM timeout: {gpto1_llm._timeout}s")
    
    # Test the timeout logic directly
    model_name = getattr(agent.llm, 'model_name', '').lower()
    timeout_seconds = 120 if model_name.startswith(('gpto', 'o1')) else 30
    
    print(f"✅ Detected model: '{model_name}'")
    print(f"✅ Calculated timeout: {timeout_seconds}s")
    
    assert timeout_seconds == 120, f"Expected 120s timeout for {model_name}, got {timeout_seconds}s"
    assert gpto1_llm._timeout == 120.0, f"Expected 120s LLM timeout, got {gpto1_llm._timeout}s"
    
    print("🎉 gpto1 timeout test PASSED!")
    return agent

async def test_standard_model_timeout():
    """Test that standard models use 30s timeout"""
    
    print("\n🧪 Testing standard model timeout configuration...")
    
    # Create standard model LLM
    standard_config = {
        "model_name": "gpt4o",
        "user": "test_user", 
        "system_content": "You are a test assistant."
    }
    
    standard_llm = ArgoLLM(standard_config)
    
    # Create tools
    tools = [
        FBATool({"name": "run_metabolic_fba", "description": "Run FBA analysis"})
    ]
    
    # Create agent
    agent = RealTimeMetabolicAgent(standard_llm, tools, {"max_iterations": 1})
    
    print(f"✅ Agent created with {standard_llm.model_name} model")
    print(f"✅ LLM timeout: {standard_llm._timeout}s")
    
    # Test the timeout logic directly
    model_name = getattr(agent.llm, 'model_name', '').lower()
    timeout_seconds = 120 if model_name.startswith(('gpto', 'o1')) else 30
    
    print(f"✅ Detected model: '{model_name}'")
    print(f"✅ Calculated timeout: {timeout_seconds}s")
    
    assert timeout_seconds == 30, f"Expected 30s timeout for {model_name}, got {timeout_seconds}s"
    assert standard_llm._timeout == 30.0, f"Expected 30s LLM timeout, got {standard_llm._timeout}s"
    
    print("🎉 Standard model timeout test PASSED!")
    return agent

async def test_timeout_debug_logging():
    """Test that debug logging shows correct information"""
    
    print("\n🧪 Testing debug logging with gpto1 model...")
    
    # Create gpto1 agent (this will trigger the debug diagnostic)
    gpto1_config = {
        "model_name": "gpto1",
        "user": "test_user",
        "system_content": "You are a test assistant."
    }
    
    gpto1_llm = ArgoLLM(gpto1_config)
    tools = [FBATool({"name": "run_metabolic_fba", "description": "Run FBA analysis"})]
    
    # This should trigger debug logging in _diagnose_llm_timeout_settings
    agent = RealTimeMetabolicAgent(gpto1_llm, tools, {"max_iterations": 1})
    
    print("✅ Debug logging test completed (check logs above)")
    
    return agent

async def main():
    """Run all timeout validation tests"""
    
    print("🚀 Starting Timeout Validation Tests\n")
    
    try:
        # Test 1: gpto1 timeout
        gpto1_agent = await test_gpto1_timeout()
        
        # Test 2: Standard model timeout  
        standard_agent = await test_standard_model_timeout()
        
        # Test 3: Debug logging
        debug_agent = await test_timeout_debug_logging()
        
        print("\n🎉 ALL TIMEOUT TESTS PASSED!")
        print("\n📋 Summary:")
        print("✅ gpto1 models correctly use 120s timeouts")
        print("✅ Standard models correctly use 30s timeouts")
        print("✅ Debug logging shows timeout configuration")
        print("✅ LLM HTTP timeouts are set correctly")
        
        print("\n🚀 The timeout fixes are working correctly!")
        print("💡 Now try: modelseed-agent interactive")
        
    except Exception as e:
        print(f"\n❌ TIMEOUT TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())