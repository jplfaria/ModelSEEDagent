#!/usr/bin/env python3
"""
Simplified test to isolate the timeout and fallback issue
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


async def simple_test():
    """Simple test of the timeout behavior"""
    print("🧪 Simple timeout test...")

    # Get one tool
    tools = []
    try:
        tool = ToolRegistry.create_tool("run_metabolic_fba", {})
        tools.append(tool)
        print(f"✅ Loaded tool: run_metabolic_fba")
    except Exception as e:
        print(f"❌ Could not load tool: {e}")
        return

    # Create LLM config
    llm_config = {
        "model_name": "gpt-4o-mini",
        "system_content": "You are a metabolic modeling AI agent.",
        "temperature": 0.7,
        "max_tokens": 1000,
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
    config = {"max_iterations": 2}  # Shorter for testing
    agent = create_real_time_agent(llm, tools, config)

    print(f"🤖 Agent created with {len(tools)} tools")

    # Test simple query
    query = "analyze data/examples/e_coli_core.xml"

    print(f"\n🔍 Testing: '{query}'")
    print("Expected: 30s timeout → fallback tool selection → tool execution")

    start_time = time.time()

    try:
        result = await agent.run({"query": query})

        elapsed = time.time() - start_time
        print(f"\n⏱️ Completed in {elapsed:.1f} seconds")

        if result.success:
            print("✅ SUCCESS!")
            print(f"📝 Message: {result.message[:200]}...")

            tools_executed = result.metadata.get("tools_executed", [])
            if tools_executed:
                print(f"🔧 Tools executed: {tools_executed}")
                print("🎉 Fallback logic working!")
            else:
                print("⚠️ No tools were executed - fallback may not be working")
        else:
            print(f"❌ FAILED: {result.error}")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n💥 Exception after {elapsed:.1f} seconds: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(simple_test())
