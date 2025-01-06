from typing import Dict, Any, List, Type
from .base import BaseAgent
from .metabolic import MetabolicAgent
from ..llm.base import BaseLLM
from ..tools.base import BaseTool

class AgentFactory:
    """Factory class for creating agent instances"""
    
    _agent_registry: Dict[str, Type[BaseAgent]] = {
        "metabolic": MetabolicAgent,
        # Add more agent types here as they are implemented
    }
    
    @classmethod
    def register_agent(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register a new agent class.
        
        Args:
            name: Name identifier for the agent type
            agent_class: Agent class to register
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"Agent class must inherit from BaseAgent: {agent_class}")
        cls._agent_registry[name] = agent_class
    
    @classmethod
    def create_agent(
        cls,
        agent_type: str,
        llm: BaseLLM,
        tools: List[BaseTool],
        config: Dict[str, Any]
    ) -> BaseAgent:
        """
        Create an agent instance.
        
        Args:
            agent_type: Type of agent to create
            llm: Language model instance
            tools: List of tools for the agent
            config: Agent configuration
            
        Returns:
            Configured agent instance
            
        Raises:
            ValueError: If agent type is not registered
        """
        if agent_type not in cls._agent_registry:
            raise ValueError(
                f"Unknown agent type: {agent_type}. "
                f"Available types: {list(cls._agent_registry.keys())}"
            )
        
        agent_class = cls._agent_registry[agent_type]
        return agent_class(llm=llm, tools=tools, config=config)
    
    @classmethod
    def list_available_agents(cls) -> List[str]:
        """List all registered agent types"""
        return list(cls._agent_registry.keys())

# Example usage of the factory
def create_metabolic_agent(
    llm: BaseLLM,
    tools: List[BaseTool],
    config: Dict[str, Any]
) -> MetabolicAgent:
    """
    Convenience function to create a metabolic modeling agent.
    
    Args:
        llm: Language model instance
        tools: List of tools for the agent
        config: Agent configuration
        
    Returns:
        Configured MetabolicAgent instance
    """
    return AgentFactory.create_agent(
        agent_type="metabolic",
        llm=llm,
        tools=tools,
        config=config
    )