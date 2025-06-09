#!/usr/bin/env python3
"""
Test script to demonstrate multi-tool analysis functionality

This script shows how the dynamic AI agent selects and chains multiple tools
based on the analysis request and discovered results.
"""

import asyncio
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Disable unnecessary warnings
import warnings

from src.agents.factory import create_real_time_agent
from src.llm.factory import LLMFactory

warnings.filterwarnings("ignore", category=RuntimeWarning)


async def test_comprehensive_analysis():
    """Test comprehensive metabolic analysis with multiple tools"""
    print("\n🧬 ModelSEEDagent Multi-Tool Analysis Demo")
    print("=" * 60)

    # Create LLM client (will try multiple backends)
    print("\n1️⃣ Initializing LLM backend...")
    llm = None
    backends_to_try = ["argo", "openai", "local"]

    for backend in backends_to_try:
        try:
            llm = LLMFactory.create(backend)
            print(f"✅ Using {backend} backend")
            break
        except Exception as e:
            print(f"⚠️  {backend} backend unavailable: {e}")
            continue

    if not llm:
        print("❌ No LLM backend available!")
        return

    # Create real-time agent
    print("\n2️⃣ Creating dynamic AI agent...")
    agent = create_real_time_agent(llm)
    print("✅ Agent initialized with 19 tools")

    # Test comprehensive analysis
    print("\n3️⃣ Running comprehensive metabolic analysis...")
    print(
        "   Query: 'Perform a comprehensive metabolic analysis of E. coli core model'"
    )
    print("   Model: data/examples/e_coli_core.xml")
    print("\n" + "-" * 60)

    query = (
        "Perform a comprehensive metabolic analysis of the E. coli core model at "
        "data/examples/e_coli_core.xml - analyze growth, find essential genes, "
        "determine minimal media requirements, and identify any auxotrophies"
    )

    result = await agent.run({"query": query})

    if result.success:
        print("\n✅ Analysis Complete!")
        print("\n📊 Results Summary:")
        print(result.message)

        # Show tools used
        tools_used = result.metadata.get("tools_executed", [])
        if tools_used:
            print(f"\n🔧 Tools Executed ({len(tools_used)}):")
            for i, tool in enumerate(tools_used, 1):
                print(f"   {i}. {tool}")

        # Show AI reasoning steps
        reasoning_steps = result.metadata.get("reasoning_steps", 0)
        if reasoning_steps:
            print(f"\n🧠 AI Reasoning Steps: {reasoning_steps}")

    else:
        print(f"\n❌ Analysis failed: {result.error}")

    # Test growth investigation
    print("\n" + "=" * 60)
    print("\n4️⃣ Testing growth investigation workflow...")
    print("   Query: 'Why might this model be growing slowly?'")
    print("\n" + "-" * 60)

    query2 = (
        "Why might the model at data/examples/e_coli_core.xml be growing slowly? "
        "Investigate potential bottlenecks by checking growth rate, essential nutrients, "
        "and metabolic efficiency"
    )

    result2 = await agent.run({"query": query2})

    if result2.success:
        print("\n✅ Investigation Complete!")
        print("\n📊 Findings:")
        print(result2.message)

        tools_used = result2.metadata.get("tools_executed", [])
        if tools_used:
            print(f"\n🔧 Tools Used for Investigation ({len(tools_used)}):")
            for i, tool in enumerate(tools_used, 1):
                print(f"   {i}. {tool}")
    else:
        print(f"\n❌ Investigation failed: {result2.error}")


async def test_tool_chaining():
    """Demonstrate how AI chains tools based on results"""
    print("\n🔗 Tool Chaining Demo")
    print("=" * 60)

    # Create components
    llm = None
    for backend in ["argo", "openai"]:
        try:
            llm = LLMFactory.create(backend)
            print(f"✅ Using {backend} backend")
            break
        except:
            continue

    if not llm:
        print("❌ No LLM available")
        return

    agent = create_real_time_agent(llm)

    # Simple query that should trigger tool chaining
    query = (
        "Check if the E. coli model at data/examples/e_coli_core.xml can grow, "
        "and if so, what are its essential nutrients?"
    )

    print(f"\n📝 Query: {query}")
    print("\n🤔 Expected tool chain:")
    print("   1. run_metabolic_fba → Check growth capability")
    print("   2. find_minimal_media → If growing, find nutrients")
    print("   3. identify_auxotrophies → Check biosynthetic gaps")
    print("\n" + "-" * 60)

    result = await agent.run({"query": query})

    if result.success:
        print("\n✅ Analysis complete!")
        print(f"\n📊 Result: {result.message}")

        # Show actual tool chain
        tools = result.metadata.get("tools_executed", [])
        if tools:
            print(f"\n🔧 Actual tool chain ({len(tools)} tools):")
            for i, tool in enumerate(tools, 1):
                print(f"   {i}. {tool}")
    else:
        print(f"\n❌ Failed: {result.error}")


def main():
    """Run all demonstrations"""
    print("🚀 ModelSEEDagent Multi-Tool Analysis Test")
    print("This demonstrates how the AI agent chains multiple tools dynamically")

    # Run async tests
    asyncio.run(test_comprehensive_analysis())
    print("\n" + "=" * 80 + "\n")
    asyncio.run(test_tool_chaining())

    print("\n✨ Demo complete!")
    print("\nTo use this in the interactive CLI, run:")
    print("  python -m src.cli.main interactive")
    print("\nThen try queries like:")
    print("  - 'Perform comprehensive analysis of E. coli core model'")
    print("  - 'Why is this model growing slowly?'")
    print("  - 'What nutrients does this organism need?'")


if __name__ == "__main__":
    main()
