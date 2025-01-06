from pathlib import Path
from typing import List, Optional
from src.config.settings import load_config
from src.llm import LLMFactory
from src.tools import ToolRegistry
from src.agents import AgentFactory

def setup_metabolic_agent(
    model_name: str = "gpt4",
    tool_names: Optional[List[str]] = None
):
    """
    Set up a metabolic agent with common configurations.
    
    Args:
        model_name: Name of the LLM model to use
        tool_names: List of tool names to initialize
        
    Returns:
        tuple: (agent, config)
    """
    # Load configuration
    config = load_config()
    
    # Set up LLM
    config["argo"]["default_model"] = model_name
    llm = LLMFactory.create("argo", config["argo"])
    
    # Set up tools
    if tool_names is None:
        tool_names = ["run_metabolic_fba", "analyze_metabolic_model"]
    
    tools = [
        ToolRegistry.create_tool(name, config["tools"].get(name, {}))
        for name in tool_names
    ]
    
    # Create agent
    agent = AgentFactory.create_agent(
        agent_type="metabolic",
        llm=llm,
        tools=tools,
        config=config["agent"]
    )
    
    return agent, config

# Example usage in notebooks:
# from utils import setup_metabolic_agent
# agent, config = setup_metabolic_agent(model_name="gpt4")