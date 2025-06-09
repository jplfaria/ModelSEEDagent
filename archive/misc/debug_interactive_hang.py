#!/usr/bin/env python3
"""
Debug script to identify where the interactive CLI hangs.
This script tests the full agent workflow step by step.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from src.agents.real_time_metabolic import RealTimeMetabolicAgent
from src.llm.argo import ArgoLLM
from src.tools import ToolRegistry


async def debug_agent_execution():
    """Debug the agent execution step by step."""
    print("🔍 DEBUG: Starting agent execution debugging")
    print("=" * 60)

    # Create minimal LLM config
    llm_config = {
        "model_name": "gpt4o",  # Use faster model for debugging
        "user": "test_user",
        "system_content": "You are an expert metabolic modeling assistant.",
        "temperature": 0.1,
        "max_tokens": 1000,
    }

    try:
        print("🔍 DEBUG: Creating LLM...")
        llm = ArgoLLM(llm_config)
        print("✅ DEBUG: LLM created successfully")
    except Exception as e:
        print(f"❌ DEBUG: LLM creation failed: {e}")
        return

    try:
        print("🔍 DEBUG: Loading tools...")
        tool_names = ToolRegistry.list_tools()
        print(f"🔍 DEBUG: Found {len(tool_names)} tools: {tool_names}")

        tools = []
        for tool_name in tool_names[:5]:  # Only load first 5 tools for debugging
            try:
                tool = ToolRegistry.create_tool(tool_name, {})
                tools.append(tool)
                print(f"✅ DEBUG: Loaded tool: {tool_name}")
            except Exception as e:
                print(f"❌ DEBUG: Failed to load tool {tool_name}: {e}")

        print(f"✅ DEBUG: Successfully loaded {len(tools)} tools")
    except Exception as e:
        print(f"❌ DEBUG: Tool loading failed: {e}")
        return

    try:
        print("🔍 DEBUG: Creating agent...")
        config = {"max_iterations": 2}  # Limit iterations for debugging
        agent = RealTimeMetabolicAgent(llm, tools, config)
        print("✅ DEBUG: Agent created successfully")
    except Exception as e:
        print(f"❌ DEBUG: Agent creation failed: {e}")
        return

    # Test the workflow step by step
    query = "I need a comprehensive metabolic analysis of E. coli"

    try:
        print(f"🔍 DEBUG: Testing first tool selection for query: '{query}'")
        selected_tool, reasoning = agent._ai_analyze_query_for_first_tool(query)
        print(f"✅ DEBUG: Selected tool: {selected_tool}")
        print(f"✅ DEBUG: Reasoning: {reasoning[:100]}...")
    except Exception as e:
        print(f"❌ DEBUG: First tool selection failed: {e}")
        import traceback

        traceback.print_exc()
        return

    try:
        print(f"🔍 DEBUG: Testing tool input preparation for: {selected_tool}")
        tool_input = agent._prepare_tool_input(selected_tool, query)
        print(f"✅ DEBUG: Tool input prepared: {tool_input}")
    except Exception as e:
        print(f"❌ DEBUG: Tool input preparation failed: {e}")
        import traceback

        traceback.print_exc()
        return

    try:
        print(f"🔍 DEBUG: Testing tool execution for: {selected_tool}")
        result = await agent._execute_tool_with_audit(selected_tool, query)
        print(f"✅ DEBUG: Tool execution result: success={result.success}")
        if not result.success:
            print(f"❌ DEBUG: Tool error: {result.error}")
        else:
            print(
                f"✅ DEBUG: Tool data keys: {list(result.data.keys()) if isinstance(result.data, dict) else 'non-dict data'}"
            )
    except Exception as e:
        print(f"❌ DEBUG: Tool execution failed: {e}")
        import traceback

        traceback.print_exc()
        return

    try:
        print("🔍 DEBUG: Testing next step decision...")
        knowledge_base = {selected_tool: result.data} if result.success else {}
        next_tool, should_continue, reasoning = (
            agent._ai_analyze_results_and_decide_next_step(query, knowledge_base)
        )
        print(
            f"✅ DEBUG: Next decision - tool: {next_tool}, continue: {should_continue}"
        )
        print(f"✅ DEBUG: Decision reasoning: {reasoning[:100]}...")
    except Exception as e:
        print(f"❌ DEBUG: Next step decision failed: {e}")
        import traceback

        traceback.print_exc()
        return

    print("🎉 DEBUG: All individual steps completed successfully!")
    print("🔍 DEBUG: Now testing full agent.run()...")

    try:
        print("🔍 DEBUG: Running full agent workflow...")
        full_result = await agent.run({"query": query})
        print(f"✅ DEBUG: Full workflow result: success={full_result.success}")
        if not full_result.success:
            print(f"❌ DEBUG: Full workflow error: {full_result.error}")
        else:
            print(
                f"✅ DEBUG: Full workflow completed with message: {full_result.message[:100]}..."
            )
    except Exception as e:
        print(f"❌ DEBUG: Full workflow failed: {e}")
        import traceback

        traceback.print_exc()
        return

    print("🎉 DEBUG: All tests completed!")


if __name__ == "__main__":
    asyncio.run(debug_agent_execution())
