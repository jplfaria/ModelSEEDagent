#!/usr/bin/env python3
"""
IMPROVED Intelligent Metabolic Analysis Script

Fixes identified issues:
1. Proper metadata tracking of tool usage
2. Better error handling and recovery
3. Correct file paths for models
4. More robust tool input formatting
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_improved_agent(model_name="gpto1preview"):
    """Create an agent with improved reasoning and error handling"""
    config = load_config()

    # Enhanced system prompt with better instructions
    argo_config = {
        "llm_name": model_name,
        "api_base": config.argo.models[model_name]["api_base"],
        "user": config.argo.user,
        "system_content": """You are an expert metabolic modeling scientist.

CRITICAL INSTRUCTIONS:
1. ALWAYS use the EXACT file path: "data/models/e_coli_core.xml" for the E. coli model
2. When a tool fails, try alternative approaches or acknowledge the limitation
3. Use tools strategically based on what information you need
4. For tool inputs, use simple strings like "data/models/e_coli_core.xml", not complex JSON
5. If you can't run a tool successfully, explain what you would need to do the analysis

Be scientific, methodical, and honest about limitations.""",
        "temperature": 0.0,  # More deterministic for consistent file paths
        "max_tokens": 8000,
    }

    llm = LLMFactory.create(config.llm.llm_backend, argo_config)

    # Create ALL tools with better descriptions
    tools = [
        ToolRegistry.create_tool(
            "analyze_metabolic_model",
            {
                "name": "analyze_metabolic_model",
                "description": "Analyze model structure, network connectivity, reactions, metabolites, and identify issues. Input: model file path as string.",
                "analysis_config": {
                    "flux_threshold": 1e-6,
                    "include_subsystems": True,
                    "track_metabolites": True,
                },
            },
        ),
        ToolRegistry.create_tool(
            "run_metabolic_fba",
            {
                "name": "run_metabolic_fba",
                "description": "Calculate growth rates and flux distributions. Input: model file path as string.",
                "fba_config": {
                    "default_objective": "biomass_reaction",
                    "solver": "glpk",
                    "tolerance": 1e-6,
                },
            },
        ),
        ToolRegistry.create_tool(
            "check_missing_media",
            {
                "name": "check_missing_media",
                "description": "Identify missing nutrients that might limit growth. Input: model file path as string.",
            },
        ),
        ToolRegistry.create_tool(
            "find_minimal_media",
            {
                "name": "find_minimal_media",
                "description": "Determine minimal nutrients needed for growth. Input: model file path as string.",
            },
        ),
        ToolRegistry.create_tool(
            "identify_auxotrophies",
            {
                "name": "identify_auxotrophies",
                "description": "Test which nutrients organism cannot synthesize. Input: model file path as string.",
            },
        ),
        ToolRegistry.create_tool(
            "analyze_reaction_expression",
            {
                "name": "analyze_reaction_expression",
                "description": "Analyze flux patterns and reaction activity. Input: model file path as string.",
            },
        ),
    ]

    # Create agent with better configuration
    agent = AgentFactory.create_agent(
        agent_type="metabolic",
        llm=llm,
        tools=tools,
        config={
            "name": "improved_intelligent_agent",
            "max_iterations": 8,
            "verbose": True,
            "handle_parsing_errors": True,
        },
    )

    return agent


def test_specific_question():
    """Test with a specific question that should demonstrate good reasoning"""

    agent = create_improved_agent()

    # A focused question that requires strategic tool selection
    question = """
    I have the E. coli core model at data/models/e_coli_core.xml and I'm curious about its growth capabilities.
    Can you tell me what the maximum growth rate is and what are the major metabolic bottlenecks that might limit growth even further?
    """

    print("🧠 IMPROVED INTELLIGENT ANALYSIS")
    print("=" * 70)
    print(f"Question: {question.strip()}")
    print("=" * 70)
    print("🔍 Watching for:")
    print("  - Correct file path usage")
    print("  - Strategic tool selection")
    print("  - Proper error handling")
    print("  - Accurate metadata tracking")
    print()

    input_data = {"input": question}
    result = agent.run(input_data)

    print(f"\n📊 RESULTS:")
    print(f"✅ Success: {result.success}")
    print(f"🔧 Tools used: {list(result.metadata.get('tools_used', {}).keys())}")
    print(f"🔄 Iterations: {result.metadata.get('iterations', 0)}")
    print(f"📁 Files created: {len(result.data.get('files', {}))}")

    if result.success:
        print(f"\n📝 Final Answer:\n{result.data.get('final_answer', 'No answer')}")
    else:
        print(f"\n❌ Error: {result.error}")

    return result


def test_multiple_questions():
    """Test multiple questions to see reasoning patterns"""

    agent = create_improved_agent()

    questions = [
        "What is the growth rate of E. coli in data/models/e_coli_core.xml?",
        "What nutrients are absolutely essential for E. coli growth?",
        "How many reactions and metabolites are in the E. coli core model?",
        "Can E. coli make all amino acids internally or does it need some from the environment?",
    ]

    results = []

    for i, question in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"🧠 TEST {i+1}: {question}")
        print("=" * 60)

        input_data = {"input": question}
        result = agent.run(input_data)

        tools_used = list(result.metadata.get("tools_used", {}).keys())

        print(f"✅ Success: {result.success}")
        print(f"🔧 Tools: {tools_used}")
        print(f"📝 Answer: {result.data.get('final_answer', 'No answer')[:150]}...")

        results.append(
            {
                "question": question,
                "success": result.success,
                "tools_used": tools_used,
                "iterations": result.metadata.get("iterations", 0),
            }
        )

    # Summary
    print(f"\n🎯 SUMMARY:")
    for i, r in enumerate(results):
        tools_str = (
            f"{len(r['tools_used'])} tools: {r['tools_used']}"
            if r["tools_used"]
            else "No tools used"
        )
        print(f"  Test {i+1}: {'✅' if r['success'] else '❌'} - {tools_str}")

    return results


if __name__ == "__main__":
    print("🔧 IMPROVED Intelligent Agent Analysis")
    print("This version fixes metadata tracking, error handling, and file paths.\n")

    choice = input(
        "Choose test:\n1. Specific growth analysis question\n2. Multiple reasoning tests\nChoice (1 or 2): "
    ).strip()

    if choice == "1":
        result = test_specific_question()
    else:
        results = test_multiple_questions()

    print(f"\n✨ Analysis complete!")
