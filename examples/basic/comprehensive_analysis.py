#!/usr/bin/env python3
"""
Comprehensive Metabolic Analysis Script

This script demonstrates a complete metabolic characterization using ALL available tools.
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


def run_comprehensive_analysis():
    """Run comprehensive analysis with all tools"""
    config = load_config()

    # Create LLM with GPT-o1 Preview
    argo_config = {
        "llm_name": "gpto1preview",
        "api_base": config.argo.models["gpto1preview"]["api_base"],
        "user": config.argo.user,
        "system_content": "You are a comprehensive metabolic modeling expert. Use ALL available tools for complete analysis.",
        "temperature": 0.0,
        "max_tokens": 8000,
    }

    llm = LLMFactory.create(config.llm.llm_backend, argo_config)

    # Create ALL tools
    tools = [
        ToolRegistry.create_tool(
            "analyze_metabolic_model",
            {
                "name": "analyze_metabolic_model",
                "description": "Analyze model structure and network properties",
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
                "description": "Perform FBA for growth and flux analysis",
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
                "description": "Identify missing media components",
            },
        ),
        ToolRegistry.create_tool(
            "find_minimal_media",
            {
                "name": "find_minimal_media",
                "description": "Find minimal media requirements",
            },
        ),
        ToolRegistry.create_tool(
            "identify_auxotrophies",
            {"name": "identify_auxotrophies", "description": "Identify auxotrophies"},
        ),
        ToolRegistry.create_tool(
            "analyze_reaction_expression",
            {
                "name": "analyze_reaction_expression",
                "description": "Analyze reaction flux patterns",
            },
        ),
    ]

    # Create agent
    agent = AgentFactory.create_agent(
        agent_type="metabolic",
        llm=llm,
        tools=tools,
        config={"name": "comprehensive_agent", "max_iterations": 10, "verbose": True},
    )

    # Run comprehensive query with proper input format
    query = """Perform complete metabolic characterization of data/models/e_coli_core.xml:
    1) Analyze model structure and connectivity
    2) Run flux balance analysis for growth rates
    3) Check for missing media components
    4) Find minimal media requirements
    5) Identify auxotrophies
    6) Analyze reaction expression patterns

    Use ALL available tools and provide detailed insights."""

    input_data = {"input": query}

    print("🔬 Running comprehensive analysis...")
    result = agent.run(input_data)
    print(f"📊 Result: {result}")
    return result


if __name__ == "__main__":
    run_comprehensive_analysis()
