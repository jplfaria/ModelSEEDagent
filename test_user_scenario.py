#!/usr/bin/env python3
"""
Test the exact user scenario without the interactive CLI to verify it works
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


async def test_user_scenarios():
    """Test the exact scenarios the user was trying"""
    print("🧪 Testing user scenarios...")
    
    # Get tools the user would have access to
    tool_names = ["run_metabolic_fba", "find_minimal_media", "analyze_essentiality", "identify_auxotrophies"]
    tools = []
    for tool_name in tool_names:
        try:
            tool = ToolRegistry.create_tool(tool_name, {})
            tools.append(tool)
            print(f"✅ Loaded tool: {tool_name}")
        except Exception as e:
            print(f"⚠️ Could not load tool {tool_name}: {e}")
    
    if not tools:
        print("❌ No tools available")
        return
    
    # Create LLM config 
    llm_config = {
        "model_name": "gpt-4o-mini",
        "system_content": "You are an expert metabolic modeling AI agent that makes real-time decisions based on data analysis.",
        "temperature": 0.7,
        "max_tokens": 4000,
    }
    
    # Try to create LLM
    llm = None
    try:
        llm = LLMFactory.create("argo", llm_config)
        print("✅ Argo LLM created")
    except Exception:
        try:
            llm = LLMFactory.create("openai", llm_config)
            print("✅ OpenAI LLM created")
        except Exception:
            print("❌ No LLM available, exiting")
            return
    
    # Create agent
    config = {"max_iterations": 4}
    agent = create_real_time_agent(llm, tools, config)
    
    print(f"🤖 Agent created with {len(tools)} tools")
    
    # Test the user's exact queries
    test_queries = [
        "for our e coli core model what I I need a comprehensive metabolic analysis",
        "Analyze the model at data/examples/e_coli_core.xml"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"🔍 Test {i}: '{query}'")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            result = await agent.run({"query": query})
            
            elapsed = time.time() - start_time
            print(f"\n⏱️ Completed in {elapsed:.1f} seconds")
            
            if result.success:
                print("✅ SUCCESS!")
                print(f"📝 Message: {result.message[:300]}...")
                
                tools_executed = result.metadata.get("tools_executed", [])
                if tools_executed:
                    print(f"🔧 Tools executed: {tools_executed}")
                    print("🎉 Multi-tool analysis working!")
                else:
                    print("⚠️ No tools were executed")
                    
                # Show confidence
                confidence = result.metadata.get("confidence_score", 0)
                print(f"🎯 Confidence: {confidence:.2f}")
                
            else:
                print(f"❌ FAILED: {result.error}")
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n💥 Exception after {elapsed:.1f} seconds: {e}")


if __name__ == "__main__":
    asyncio.run(test_user_scenarios())