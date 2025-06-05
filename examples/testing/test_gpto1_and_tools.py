#!/usr/bin/env python3
"""
Test GPT-o1 Model and Debug Tool Execution

This script:
1. Tests the new gpto1 model configuration
2. Investigates the tool execution bug we discovered
3. Validates that tools are actually being called
"""

import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents import AgentFactory

# Import and configure
from src.config.settings import load_config
from src.llm import LLMFactory
from src.tools import ToolRegistry

# Import all tools to ensure registration
from src.tools.cobra import (
    analysis,
    auxotrophy,
    fba,
    minimal_media,
    missing_media,
    reaction_expression,
)

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_gpto1_model():
    """Test that the new gpto1 model works"""
    print("🔧 Testing GPT-o1 Model Configuration")
    print("=" * 50)

    try:
        config = load_config()
        print(f"✅ Config loaded successfully")
        print(f"📋 Available models: {list(config.argo.models.keys())}")
        print(f"🎯 Default model: {config.argo.default_model}")

        # Test creating LLM with gpto1
        argo_config = {
            "llm_name": "gpto1",
            "api_base": config.argo.models["gpto1"]["api_base"],
            "user": config.argo.user,
            "system_content": "You are a test assistant.",
            "temperature": 0.0,
            "max_tokens": 100,
        }

        llm = LLMFactory.create(config.llm.llm_backend, argo_config)
        print(f"✅ GPT-o1 LLM created successfully: {llm}")

        return True

    except Exception as e:
        print(f"❌ Error testing GPT-o1: {e}")
        return False


def test_single_tool_directly():
    """Test calling a single tool directly to see if it works"""
    print("\n🔧 Testing Direct Tool Execution")
    print("=" * 50)

    try:
        # Create a single tool and test it directly
        tool = ToolRegistry.create_tool(
            "analyze_metabolic_model",
            {
                "name": "analyze_metabolic_model",
                "description": "Test tool",
                "analysis_config": {
                    "flux_threshold": 1e-6,
                    "include_subsystems": True,
                    "track_metabolites": True,
                },
            },
        )

        print(f"✅ Tool created: {tool}")

        # Test with correct model path
        result = tool._run("data/models/e_coli_core.xml")
        print(f"📊 Tool result type: {type(result)}")
        print(f"📊 Tool success: {result.success}")
        print(f"📊 Tool message: {result.message}")

        if result.success:
            print(
                f"📊 Tool data keys: {list(result.data.keys()) if result.data else 'No data'}"
            )
        else:
            print(f"❌ Tool error: {result.error}")

        return result.success

    except Exception as e:
        print(f"❌ Error testing direct tool: {e}")
        logger.exception("Full error details:")
        return False


def create_debugging_agent():
    """Create an agent with extensive debugging"""
    config = load_config()

    # Use gpto1 with debugging
    argo_config = {
        "llm_name": "gpto1",
        "api_base": config.argo.models["gpto1"]["api_base"],
        "user": config.argo.user,
        "system_content": """You are a metabolic modeling expert.

DEBUGGING MODE: I want to see exactly what happens when you use tools.

CRITICAL: Use EXACTLY this file path: "data/models/e_coli_core.xml"

When you use a tool, I will monitor:
1. Whether the tool actually gets called
2. What the real result is
3. Whether you get real data or hallucinate

Please analyze the E. coli model and tell me its growth rate using FBA.""",
        "temperature": 0.0,
        "max_tokens": 2000,
    }

    llm = LLMFactory.create(config.llm.llm_backend, argo_config)

    # Create a single tool for focused testing
    tools = [
        ToolRegistry.create_tool(
            "run_metabolic_fba",
            {
                "name": "run_metabolic_fba",
                "description": "Calculate growth rates using FBA. Input should be model file path as string.",
                "fba_config": {
                    "default_objective": "biomass_reaction",
                    "solver": "glpk",
                    "tolerance": 1e-6,
                },
            },
        )
    ]

    # Create agent with verbose logging
    agent = AgentFactory.create_agent(
        agent_type="metabolic",
        llm=llm,
        tools=tools,
        config={
            "name": "debugging_agent",
            "max_iterations": 3,
            "verbose": True,
            "handle_parsing_errors": True,
        },
    )

    return agent


def test_agent_with_debugging():
    """Test agent execution with detailed debugging"""
    print("\n🔧 Testing Agent with GPT-o1 and Debugging")
    print("=" * 50)

    agent = create_debugging_agent()

    # Simple focused question
    question = (
        "What is the growth rate of the E. coli model at data/models/e_coli_core.xml?"
    )

    print(f"❓ Question: {question}")
    print("🔍 Monitoring tool execution...")

    input_data = {"input": question}
    result = agent.run(input_data)

    print(f"\n📊 DEBUGGING RESULTS:")
    print(f"✅ Success: {result.success}")
    print(f"🔧 Tools used (metadata): {result.metadata.get('tools_used', {})}")
    print(f"🔄 Iterations: {result.metadata.get('iterations', 0)}")
    print(f"📝 Final answer: {result.data.get('final_answer', 'No answer')}")
    print(f"🔬 Tool results: {len(result.data.get('tool_results', []))}")
    print(f"📁 Files created: {len(result.data.get('files', {}))}")

    # Check if any actual tool results were captured
    if result.data.get("tool_results"):
        print("\n📋 Tool Results Details:")
        for i, tool_result in enumerate(result.data["tool_results"]):
            print(f"  Tool {i+1}: {tool_result.get('action', 'Unknown')}")
    else:
        print("\n⚠️  WARNING: No tool results captured - possible hallucination!")

    return result


def main():
    """Run all tests"""
    print("🧪 COMPREHENSIVE TESTING: GPT-o1 + Tool Execution Debug")
    print("=" * 70)

    # Test 1: Model configuration
    model_ok = test_gpto1_model()

    # Test 2: Direct tool execution
    tool_ok = test_single_tool_directly()

    # Test 3: Agent with debugging
    agent_result = test_agent_with_debugging()

    # Summary
    print(f"\n🎯 TEST SUMMARY:")
    print(f"  GPT-o1 Model: {'✅ Working' if model_ok else '❌ Failed'}")
    print(f"  Direct Tool: {'✅ Working' if tool_ok else '❌ Failed'}")
    print(f"  Agent Execution: {'✅ Working' if agent_result.success else '❌ Failed'}")

    if not tool_ok:
        print(
            "\n⚠️  CRITICAL: Direct tool execution failed - this explains the hallucination issue!"
        )
    elif agent_result.success and not agent_result.data.get("tool_results"):
        print(
            "\n⚠️  CRITICAL: Agent completes but no tool results captured - agent pipeline bug!"
        )

    return model_ok, tool_ok, agent_result


if __name__ == "__main__":
    main()
