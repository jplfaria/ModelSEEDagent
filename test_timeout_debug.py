#!/usr/bin/env python3
"""
Debug script to test timeout and fallback logic
"""

import asyncio
import os
import sys
import time

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents import create_real_time_agent
from src.llm.factory import LLMFactory
from src.tools import ToolRegistry


async def test_timeout_fallback():
    """Test that timeout triggers fallback logic"""
    print("🧪 Testing timeout and fallback logic...")
    
    # Create a mock LLM that will always timeout
    class TimeoutLLM:
        def __init__(self, config):
            self.config = config
            
        def _generate_response(self, prompt):
            print("⏱️ LLM call started - will timeout in 30s...")
            # Sleep for longer than the timeout to trigger it
            time.sleep(35)  # This should never complete
            return type('obj', (), {'text': 'This should never be reached'})()
    
    # Set up tools
    tool_names = ['run_metabolic_fba', 'find_minimal_media', 'analyze_essentiality']
    tools = []
    for tool_name in tool_names:
        try:
            tool = ToolRegistry.create_tool(tool_name, {})
            tools.append(tool)
            print(f"✅ Loaded tool: {tool_name}")
        except Exception as e:
            print(f"⚠️ Could not load tool {tool_name}: {e}")
    
    if not tools:
        print("❌ No tools available, skipping test")
        return
    
    # Create LLM config 
    llm_config = {
        "model_name": "test-timeout",
        "system_content": "Test system",
        "temperature": 0.7,
        "max_tokens": 1000,
    }
    
    # Create timeout LLM
    llm = TimeoutLLM(llm_config)
    
    # Create agent
    config = {"max_iterations": 3}
    agent = create_real_time_agent(llm, tools, config)
    
    print(f"🤖 Agent created with {len(tools)} tools")
    
    # Test query
    query = "Analyze the E. coli core model"
    
    print(f"\n🔍 Testing query: '{query}'")
    print("Expected behavior:")
    print("1. LLM call times out after 30s")
    print("2. Fallback tool selection kicks in")
    print("3. Tool execution proceeds")
    
    start_time = time.time()
    
    try:
        # This should timeout and use fallback
        result = await agent.run({"query": query})
        
        elapsed = time.time() - start_time
        print(f"\n⏱️ Total time: {elapsed:.1f} seconds")
        
        if result.success:
            print("✅ Fallback logic worked!")
            print(f"📊 Message: {result.message[:200]}...")
            
            # Check if any tools were executed
            tools_executed = result.metadata.get("tools_executed", [])
            if tools_executed:
                print(f"🔧 Tools executed: {tools_executed}")
            else:
                print("⚠️ No tools were executed")
        else:
            print(f"❌ Agent failed: {result.error}")
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n💥 Exception after {elapsed:.1f} seconds: {e}")


def test_direct_fallback():
    """Test the fallback logic directly"""
    print("\n🧪 Testing direct fallback logic...")
    
    # Test fallback tool selection
    from src.agents.real_time_metabolic import RealTimeMetabolicAgent
    
    # Create minimal agent for testing fallback
    class MockLLM:
        def __init__(self, config):
            self.config = config
            
        def _generate_response(self, prompt):
            return type('obj', (), {'text': 'Mock response'})()
    
    tools = []
    try:
        tool = ToolRegistry.create_tool('run_metabolic_fba', {})
        tools.append(tool)
    except Exception as e:
        print(f"⚠️ Could not load FBA tool: {e}")
        return
    
    agent = RealTimeMetabolicAgent(MockLLM({}), tools, {})
    
    # Test fallback selection
    test_queries = [
        "comprehensive analysis of e coli",
        "analyze growth rate",
        "find essential genes",
        "unknown request"
    ]
    
    for query in test_queries:
        selected_tool = agent._fallback_tool_selection(query)
        print(f"📋 Query: '{query}' → Tool: {selected_tool}")


async def main():
    """Run all tests"""
    print("🔧 ModelSEEDagent Timeout & Fallback Debug")
    print("=" * 50)
    
    # Test direct fallback first
    test_direct_fallback()
    
    print("\n" + "=" * 50)
    print("⚠️ WARNING: Next test will take ~30+ seconds (testing timeout)")
    
    response = input("Continue with timeout test? (y/n): ")
    if response.lower() in ['y', 'yes']:
        await test_timeout_fallback()
    else:
        print("Skipping timeout test")
    
    print("\n✅ Debug complete!")


if __name__ == "__main__":
    asyncio.run(main())