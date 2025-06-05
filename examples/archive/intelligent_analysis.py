#!/usr/bin/env python3
"""
Intelligent Metabolic Analysis Script

This script demonstrates TRUE intelligent agent behavior where the agent:
1. Reasons about what information it needs based on the question
2. Dynamically chooses appropriate tools
3. Analyzes results and decides what to do next
4. Continues until it has sufficient information to answer

No explicit tool instructions - just natural questions!
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


def create_intelligent_agent(model_name="gpto1preview"):
    """Create an agent that reasons intelligently about tool usage"""
    config = load_config()

    # Create LLM with enhanced reasoning instructions
    argo_config = {
        "llm_name": model_name,
        "api_base": config.argo.models[model_name]["api_base"],
        "user": config.argo.user,
        "system_content": """You are an expert metabolic modeling scientist.

IMPORTANT: You must REASON intelligently about what information you need to answer questions.
- Analyze the question to understand what's being asked
- Choose tools strategically based on what you need to know
- Look at results and decide if you need more information
- Only use tools that help answer the specific question
- Don't use all tools just because they're available

Be scientific, methodical, and efficient in your analysis.""",
        "temperature": 0.1,  # Low temperature for more focused reasoning
        "max_tokens": 8000,
    }

    llm = LLMFactory.create(config.llm.llm_backend, argo_config)

    # Create ALL tools (but agent will choose which to use)
    tools = [
        ToolRegistry.create_tool(
            "analyze_metabolic_model",
            {
                "name": "analyze_metabolic_model",
                "description": "Analyze model structure, network connectivity, and identify potential issues",
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
                "description": "Calculate growth rates and flux distributions through the metabolic network",
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
                "description": "Identify missing nutrients that might be limiting growth",
            },
        ),
        ToolRegistry.create_tool(
            "find_minimal_media",
            {
                "name": "find_minimal_media",
                "description": "Determine the minimal set of nutrients required for growth",
            },
        ),
        ToolRegistry.create_tool(
            "identify_auxotrophies",
            {
                "name": "identify_auxotrophies",
                "description": "Test which nutrients the organism cannot synthesize internally",
            },
        ),
        ToolRegistry.create_tool(
            "analyze_reaction_expression",
            {
                "name": "analyze_reaction_expression",
                "description": "Analyze which reactions carry flux and their activity levels",
            },
        ),
    ]

    # Create agent with focus on reasoning
    agent = AgentFactory.create_agent(
        agent_type="metabolic",
        llm=llm,
        tools=tools,
        config={
            "name": "intelligent_agent",
            "max_iterations": 8,  # Allow more iterations for reasoning
            "verbose": True,
            "handle_parsing_errors": True,
        },
    )

    return agent


def test_intelligent_reasoning():
    """Test the agent with various questions that require different tools"""

    agent = create_intelligent_agent()

    # Test cases that should trigger different reasoning patterns
    test_questions = [
        {
            "question": "What's wrong with the E. coli core model? Why might it not be growing optimally?",
            "expected_reasoning": "Should analyze model structure first, then check growth, then investigate issues",
        },
        {
            "question": "Is E. coli auxotrophic for any amino acids?",
            "expected_reasoning": "Should directly use auxotrophy tool, maybe FBA for context",
        },
        {
            "question": "What's the minimum I need to feed E. coli to keep it alive?",
            "expected_reasoning": "Should use minimal media tool, maybe check growth rate",
        },
        {
            "question": "How well connected is the E. coli metabolic network?",
            "expected_reasoning": "Should use model analysis tool for network properties",
        },
        {
            "question": "Can E. coli grow without oxygen and what happens to its metabolism?",
            "expected_reasoning": "Should use FBA to test growth, then analyze flux patterns under different conditions",
        },
    ]

    results = []

    for i, test in enumerate(test_questions):
        print(f"\n{'='*80}")
        print(f"🧠 INTELLIGENT REASONING TEST {i+1}")
        print(f"{'='*80}")
        print(f"❓ Question: {test['question']}")
        print(f"🤔 Expected reasoning: {test['expected_reasoning']}")
        print(f"{'='*80}")

        # Run the analysis
        input_data = {"input": test["question"]}
        result = agent.run(input_data)

        results.append(
            {
                "question": test["question"],
                "result": result,
                "tools_used": result.metadata.get("tools_used", {}),
                "success": result.success,
            }
        )

        print(f"✅ Success: {result.success}")
        print(f"🔧 Tools used: {list(result.metadata.get('tools_used', {}).keys())}")
        print(f"📝 Answer: {result.data.get('final_answer', 'No answer')[:200]}...")

    return results


def run_focused_analysis():
    """Run analysis with a specific focused question"""
    agent = create_intelligent_agent()

    # A question that requires reasoning about what tools to use
    question = """
    I'm trying to optimize E. coli for bioethanol production. Can you tell me:
    1. What are the current metabolic capabilities and limitations?
    2. Where are the bottlenecks in the system?
    3. What would I need to change to improve production?

    Please analyze the E. coli core model at data/models/e_coli_core.xml.
    """

    print("🔬 FOCUSED INTELLIGENT ANALYSIS")
    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60)
    print("🧠 Let's see how the agent reasons about this...")
    print()

    input_data = {"input": question}
    result = agent.run(input_data)

    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"Success: {result.success}")
    print(f"Tools used: {list(result.metadata.get('tools_used', {}).keys())}")
    print(f"Iterations: {result.metadata.get('iterations', 0)}")
    print(f"\n📝 Final Answer:\n{result.data.get('final_answer', 'No answer')}")

    return result


if __name__ == "__main__":
    print("🧠 Testing Intelligent Agent Reasoning")
    print("This demonstrates an agent that chooses tools based on the question")
    print("rather than being told explicitly what tools to use.\n")

    # Choose which test to run
    test_choice = input(
        "Choose test:\n1. Multiple reasoning tests\n2. Focused bioethanol analysis\nChoice (1 or 2): "
    ).strip()

    if test_choice == "1":
        results = test_intelligent_reasoning()
        print(f"\n🎯 SUMMARY: Completed {len(results)} reasoning tests")
        for i, r in enumerate(results):
            print(
                f"  Test {i+1}: {'✅' if r['success'] else '❌'} - Used {len(r['tools_used'])} tools"
            )
    else:
        result = run_focused_analysis()
        print(
            f"\n🎯 Analysis completed with {len(result.metadata.get('tools_used', {}))} tools"
        )
