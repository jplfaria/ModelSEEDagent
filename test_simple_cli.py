#!/usr/bin/env python3
"""
Simple test script to verify the CLI works with basic queries
"""

import asyncio
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.factory import create_real_time_agent
from src.llm.factory import LLMFactory


async def test_simple_query():
    """Test a simple FBA query"""
    print("\n🧪 Testing Simple FBA Query")
    print("=" * 50)
    
    # Create LLM
    try:
        llm = LLMFactory.create("argo")
        print("✅ Connected to Argo Gateway")
    except:
        try:
            llm = LLMFactory.create("openai")
            print("✅ Connected to OpenAI")
        except:
            print("❌ No LLM backend available")
            return
    
    # Create agent
    agent = create_real_time_agent(llm)
    print("✅ Agent initialized")
    
    # Simple query
    query = "Run FBA on the E. coli core model at data/examples/e_coli_core.xml"
    print(f"\n📝 Query: {query}")
    
    result = await agent.run({"query": query})
    
    if result.success:
        print("\n✅ Success!")
        print(f"📊 Result: {result.message}")
        
        # Show tools used
        tools = result.metadata.get("tools_executed", [])
        if tools:
            print(f"\n🔧 Tools used: {', '.join(tools)}")
    else:
        print(f"\n❌ Failed: {result.error}")


async def test_multi_tool_query():
    """Test a query that should trigger multiple tools"""
    print("\n\n🧪 Testing Multi-Tool Query")
    print("=" * 50)
    
    # Create components
    try:
        llm = LLMFactory.create("argo")
    except:
        try:
            llm = LLMFactory.create("openai")
        except:
            print("❌ No LLM backend available")
            return
    
    agent = create_real_time_agent(llm)
    
    # Multi-tool query
    query = "Check growth rate and find essential genes for E. coli core model at data/examples/e_coli_core.xml"
    print(f"\n📝 Query: {query}")
    
    result = await agent.run({"query": query})
    
    if result.success:
        print("\n✅ Success!")
        print(f"📊 Result: {result.message}")
        
        # Show tools used
        tools = result.metadata.get("tools_executed", [])
        if tools:
            print(f"\n🔧 Tools used ({len(tools)}): {', '.join(tools)}")
    else:
        print(f"\n❌ Failed: {result.error}")


def main():
    """Run tests"""
    print("🚀 ModelSEEDagent CLI Test")
    
    # Run tests
    asyncio.run(test_simple_query())
    asyncio.run(test_multi_tool_query())
    
    print("\n\n✨ Tests complete!")
    print("\nIf these worked, the interactive CLI should also work.")
    print("Run: python -m src.cli.main interactive")


if __name__ == "__main__":
    main()