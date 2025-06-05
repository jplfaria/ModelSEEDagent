#!/usr/bin/env python3
"""
Metabolic Analysis Workflow Example

This script demonstrates the same type of analysis workflow that was shown
in the user's original script, but adapted to work with our refactored codebase.

It shows how to:
1. Create an interactive agent for metabolic modeling
2. Run different types of analyses on E. coli core model
3. Get results similar to the original workflow

Usage:
    python examples/metabolic_analysis_workflow.py
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents import AgentFactory

# Import required modules from our refactored codebase
from src.config.settings import load_config
from src.llm import LLMFactory
from src.tools import ToolRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


def create_interactive_agent():
    """Create an interactive agent for metabolic modeling"""
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Create LLM instance
        if config.llm.llm_backend == "argo":
            # Argo configuration
            argo_config = {
                "llm_name": "gpt4o",
                "api_base": config.argo.models["gpt4o"]["api_base"],
                "user": config.argo.user,
                "system_content": """You are a metabolic modeling expert. Analyze metabolic models using the available tools.
                When asked to analyze a model:
                1. First run analyze_metabolic_model to get basic stats
                2. Then run run_metabolic_fba to get growth rates and fluxes
                3. Provide a clear summary of findings

                Format your response as:
                Thought: your reasoning
                Action: tool to use
                Action Input: input for tool
                Observation: result from tool
                ... (repeat as needed)
                Final Answer: summary of findings""",
                "temperature": 0.0,
                "max_tokens": 5000,
            }
            llm = LLMFactory.create(config.llm.llm_backend, argo_config)
        else:
            # Use local or other backend
            llm = LLMFactory.create(config.llm.llm_backend)

        logger.info(f"Created LLM instance using {config.llm.llm_backend} backend")

        # Create tools
        tools = [
            ToolRegistry.create_tool(
                "run_metabolic_fba",
                {
                    "name": "run_metabolic_fba",
                    "description": "Calculate growth rates and fluxes",
                    "fba_config": {
                        "default_objective": "biomass_reaction",
                        "solver": "glpk",
                        "tolerance": 1e-6,
                    },
                },
            ),
            ToolRegistry.create_tool(
                "analyze_metabolic_model",
                {
                    "name": "analyze_metabolic_model",
                    "description": "Get model statistics and analysis",
                    "analysis_config": {
                        "flux_threshold": 1e-6,
                        "include_subsystems": True,
                        "track_metabolites": True,
                    },
                },
            ),
        ]
        logger.info("Created tool instances")

        # Create agent
        agent = AgentFactory.create_agent(
            agent_type="metabolic",
            llm=llm,
            tools=tools,
            config={
                "name": "metabolic_agent",
                "description": "Natural language metabolic model analysis",
                "max_iterations": 3,
                "verbose": True,
                "handle_parsing_errors": True,
            },
        )
        logger.info("Created agent instance")

        return agent

    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise


def analyze_model(query: str, model_path: str):
    """Handle natural language analysis requests"""
    logger.info(f"Starting analysis - Query: {query}")

    try:
        agent = create_interactive_agent()
        full_query = f"{query} Model path: {model_path}"

        # Use the analyze_model method from the metabolic agent
        result = agent.analyze_model(full_query)

        if result.success:
            print("\nAnalysis Results:")
            print("-" * 50)

            if "final_answer" in result.data:
                print(result.data["final_answer"])
                logger.info("Analysis completed successfully")
            else:
                print("Raw results:")
                if "tool_results" in result.data:
                    for step in result.data["tool_results"]:
                        print(f"\nTool: {step['action']}")
                        print(f"Observation: {step['observation']}")
                logger.info("Analysis completed with tool results only")
        else:
            print("\nError: Unable to complete analysis")
            if result.error:
                print(f"Details: {result.error}")
            logger.error(f"Analysis failed: {result.error}")

    except Exception as e:
        error_msg = f"\nError during analysis: {str(e)}"
        print(error_msg)
        logger.exception("Unexpected error during analysis")


def test_model_analysis():
    """Run a series of test analyses - same as the original workflow"""
    model_path = project_root / "data" / "models" / "e_coli_core.xml"

    # Verify model file exists
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Using model: {model_path}")

    # Same tests as the original workflow
    tests = [
        (
            "Basic Model Characteristics",
            "What are the key characteristics of this model?",
        ),
        ("Growth Analysis", "What is the growth rate and what are the key reactions?"),
        ("Pathway Analysis", "Analyze the central carbon metabolism in this model"),
    ]

    for test_name, query in tests:
        print(f"\nTest: {test_name}")
        print("-" * 50)
        analyze_model(query, str(model_path))
        time.sleep(1)  # Add small delay between tests


def main():
    """Main execution function"""
    print("=" * 80)
    print("ModelSEEDagent - Metabolic Analysis Workflow")
    print("=" * 80)

    try:
        # Check configuration
        config = load_config()
        print(f"Using LLM backend: {config.llm.llm_backend}")

        if config.llm.llm_backend == "argo":
            print("⚠️  Note: Argo backend requires ANL network access")
            print("If you get connection errors, switch to local backend:")
            print("python -m src.cli.main switch local")
            print()

        # Run the test workflow
        test_model_analysis()

    except Exception as e:
        logger.exception("Error in main execution")
        print(f"Critical error: {str(e)}")

        # Suggest fallback
        print("\n" + "=" * 50)
        print("If you're getting errors, try switching to local backend:")
        print("python -m src.cli.main switch local")
        print("Then run this script again.")


if __name__ == "__main__":
    main()
