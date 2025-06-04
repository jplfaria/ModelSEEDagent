#!/usr/bin/env python3
"""
Quick test script for the new LangGraph Metabolic Agent
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.langgraph_metabolic import LangGraphMetabolicAgent
from src.llm.argo import ArgoLLM
from src.llm.base import LLMConfig
from src.tools.base import BaseTool, ToolResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Mock tool for testing
class MockFBATool(BaseTool):
    tool_name = "run_metabolic_fba"
    tool_description = "Run FBA analysis on metabolic model"

    def __init__(self, config):
        super().__init__(config)

    def _run(self, input_data):
        logger.info(f"Mock FBA tool called with: {input_data}")
        return ToolResult(
            success=True,
            message="Mock FBA analysis completed successfully. Growth rate: 0.873/hr",
            data={
                "growth_rate": 0.873,
                "objective_value": 0.873,
                "active_reactions": [
                    "GLCpts",
                    "PFK",
                    "GAPD",
                    "PYK",
                    "BIOMASS_Ecoli_core",
                ],
            },
        )


class MockAnalysisTool(BaseTool):
    tool_name = "analyze_metabolic_model"
    tool_description = "Analyze metabolic model structure"

    def __init__(self, config):
        super().__init__(config)

    def _run(self, input_data):
        logger.info(f"Mock Analysis tool called with: {input_data}")
        return ToolResult(
            success=True,
            message="Model analysis completed. Found 95 reactions and 72 metabolites.",
            data={
                "num_reactions": 95,
                "num_metabolites": 72,
                "num_genes": 137,
                "compartments": ["c", "e"],
            },
        )


def test_langgraph_agent():
    """Test the LangGraph agent with mock components"""

    print("🚀 Testing LangGraph Metabolic Agent...")

    # Create mock LLM config (test mode)
    llm_config = {
        "model_name": "test-model",
        "api_base": "https://test.api/",
        "user": "test_user",
        "system_content": "You are a helpful metabolic modeling assistant.",
        "max_tokens": 1000,
        "temperature": 0.1,
    }

    # Create mock LLM
    try:
        llm = ArgoLLM(llm_config)
        print("✅ LLM created successfully")
    except Exception as e:
        print(f"❌ LLM creation failed: {e}")
        return False

    # Create mock tools
    tools = [
        MockFBATool({"name": "run_metabolic_fba", "description": "FBA tool"}),
        MockAnalysisTool(
            {"name": "analyze_metabolic_model", "description": "Analysis tool"}
        ),
    ]
    print(f"✅ Created {len(tools)} mock tools")

    # Create LangGraph agent
    try:
        agent_config = {
            "name": "test_langgraph_agent",
            "description": "Test LangGraph metabolic agent",
        }

        agent = LangGraphMetabolicAgent(llm, tools, agent_config)
        print("✅ LangGraph agent created successfully")
        print(f"   Run directory: {agent.run_dir}")

    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

    # Test simple query
    try:
        print("\n🧪 Testing simple analysis query...")

        result = agent.run(
            {
                "query": "Analyze the basic characteristics of this metabolic model",
                "max_iterations": 3,
            }
        )

        print(f"✅ Query completed!")
        print(f"   Success: {result.success}")
        print(f"   Message: {result.message[:200]}...")
        print(f"   Tools used: {result.metadata.get('tools_used', [])}")
        print(f"   Iterations: {result.metadata.get('iterations', 0)}")

        if result.data:
            print(f"   Data keys: {list(result.data.keys())}")

        return True

    except Exception as e:
        print(f"❌ Query execution failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_langgraph_agent()

    if success:
        print("\n🎉 LangGraph agent test completed successfully!")
        print("✅ Phase 2.1: LangGraph Migration - READY FOR INTEGRATION")
    else:
        print("\n❌ LangGraph agent test failed")
        sys.exit(1)
