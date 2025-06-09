#!/usr/bin/env python3
"""
Test script to verify improved tool selection logic.
This script tests the AI's tool selection without the full interactive CLI.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.real_time_metabolic import RealTimeMetabolicAgent
from src.llm.argo import ArgoLLM
from src.tools.cobra.essentiality import EssentialityAnalysisTool
from src.tools.cobra.fba import FBATool
from src.tools.cobra.minimal_media import MinimalMediaTool
from src.tools.modelseed.builder import ModelBuildTool


def test_tool_selection():
    """Test that the AI selects appropriate tools for analysis queries."""
    print("🧪 Testing Improved Tool Selection Logic")
    print("=" * 50)

    # Create minimal LLM config for testing
    llm_config = {
        "model_name": "gpt4o",
        "user": "test_user",
        "system_content": "You are an expert metabolic modeling assistant.",
        "temperature": 0.1,
        "max_tokens": 1000,
    }

    try:
        llm = ArgoLLM(llm_config)
        print("✅ LLM created successfully")
    except Exception as e:
        print(f"❌ LLM creation failed: {e}")
        print("💡 This test requires Argo Gateway access")
        return

    # Create tools including both analysis and build tools
    tools = [
        FBATool({"name": "run_metabolic_fba", "description": "Run FBA analysis"}),
        MinimalMediaTool(
            {
                "name": "find_minimal_media",
                "description": "Find minimal media requirements",
            }
        ),
        EssentialityAnalysisTool(
            {"name": "analyze_essentiality", "description": "Analyze gene essentiality"}
        ),
        ModelBuildTool(
            {
                "name": "build_metabolic_model",
                "description": "Build metabolic model from genome",
            }
        ),
    ]

    print(f"✅ Created {len(tools)} tools (mix of analysis and build tools)")

    # Create agent
    config = {"max_iterations": 1}  # Just test tool selection, don't run full analysis
    agent = RealTimeMetabolicAgent(llm, tools, config)
    print("✅ Agent created successfully")

    # Test queries that should select analysis tools
    test_queries = [
        "I need a comprehensive metabolic analysis of E. coli",
        "Analyze the growth capabilities of the E. coli core model",
        "Characterize the metabolic properties of this model",
        "What are the nutritional requirements for growth?",
        "Perform FBA analysis on the model",
    ]

    print("\n🔍 Testing Tool Selection for Analysis Queries:")
    print("-" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        try:
            selected_tool, reasoning = agent._ai_analyze_query_for_first_tool(query)
            print(f"   🎯 Selected Tool: {selected_tool}")
            print(f"   💭 AI Reasoning: {reasoning[:100]}...")

            # Check if appropriate tool was selected
            analysis_tools = [
                "run_metabolic_fba",
                "find_minimal_media",
                "analyze_essentiality",
            ]
            build_tools = ["build_metabolic_model"]

            if selected_tool in analysis_tools:
                print("   ✅ GOOD: Selected an analysis tool")
            elif selected_tool in build_tools:
                print("   ❌ ISSUE: Selected a build tool for analysis query")
            else:
                print(f"   ⚠️  UNEXPECTED: Selected unexpected tool: {selected_tool}")

        except Exception as e:
            print(f"   ❌ ERROR: Tool selection failed: {e}")

    # Test a query that should select build tools
    build_query = "Build a new metabolic model from my E. coli genome annotation file"
    print(f"\n🏗️  Testing Build Query: '{build_query}'")
    try:
        selected_tool, reasoning = agent._ai_analyze_query_for_first_tool(build_query)
        print(f"   🎯 Selected Tool: {selected_tool}")
        print(f"   💭 AI Reasoning: {reasoning[:100]}...")

        if selected_tool == "build_metabolic_model":
            print("   ✅ GOOD: Correctly selected build tool for build query")
        else:
            print(f"   ⚠️  UNEXPECTED: Selected {selected_tool} instead of build tool")
    except Exception as e:
        print(f"   ❌ ERROR: Tool selection failed: {e}")

    print("\n🎉 Tool Selection Test Complete!")
    print("\nℹ️  This test verifies that the AI selects appropriate tools.")
    print(
        "   Analysis queries should select analysis tools (FBA, minimal media, essentiality)"
    )
    print("   Build queries should select build tools (build_metabolic_model)")


if __name__ == "__main__":
    test_tool_selection()
