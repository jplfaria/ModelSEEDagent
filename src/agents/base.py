from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.llm.base import BaseLLM
from src.tools.base import BaseTool


class AgentResult(BaseModel):
    """Standardized result format for agent operations"""

    model_config = {"protected_namespaces": ()}
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    intermediate_steps: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Base configuration for agents"""

    model_config = {"protected_namespaces": ()}
    name: str = "default"
    description: str = ""
    max_iterations: int = 5
    verbose: bool = False
    handle_parsing_errors: bool = True
    default_type: str = "metabolic"
    additional_config: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(
        self, llm: BaseLLM, tools: List[BaseTool], config: Dict[str, Any] | AgentConfig
    ):
        self.config = (
            config if isinstance(config, AgentConfig) else AgentConfig(**dict(config))
        )
        self.llm = llm
        self.tools = tools
        self._agent_executor = None
        self._setup_agent()

    @abstractmethod
    def _create_prompt(self) -> PromptTemplate:
        """Create the prompt template for the agent"""
        pass

    @abstractmethod
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent executor"""
        pass

    def _setup_agent(self) -> None:
        """Set up the agent executor"""
        self._agent_executor = self._create_agent()

    def _format_result(self, result: Dict[str, Any]) -> AgentResult:
        """Format the raw agent result into standardized format"""
        # Handle the case where result might be None
        if not result:
            return AgentResult(
                success=False,
                message="No result produced",
                error="Agent execution produced no result",
                data={},
                intermediate_steps=[],
            )

        # Extract intermediate steps and parse them
        steps = result.get("intermediate_steps", [])
        formatted_steps = []
        for step in steps:
            if isinstance(step, tuple) and len(step) == 2:
                action, observation = step
                formatted_step = {
                    "action": action.tool if hasattr(action, "tool") else str(action),
                    "action_input": (
                        action.tool_input
                        if hasattr(action, "tool_input")
                        else str(action)
                    ),
                    "observation": str(observation),
                }
                formatted_steps.append(formatted_step)

        # Create the result
        return AgentResult(
            success="error" not in result,
            message=result.get("output", "No output provided"),
            data={
                "final_answer": result.get("output", ""),
                "tool_results": formatted_steps,
            },
            intermediate_steps=formatted_steps,
            error=result.get("error"),
            metadata={
                "iterations": len(formatted_steps),
                "tools_used": self._get_tools_used(formatted_steps),
            },
        )

    def _get_tools_used(self, steps: List[Dict[str, Any]]) -> Dict[str, int]:
        """Track which tools were used and how many times"""
        tool_usage = {}
        for step in steps:
            tool_name = step.get("action", "unknown")
            if isinstance(tool_name, str):
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        return tool_usage

    def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """Run the agent on the input data"""
        try:
            result = self._agent_executor.invoke(input_data)
            return self._format_result(result)
        except Exception as e:
            return AgentResult(
                success=False, message="Agent execution failed", error=str(e), data={}
            )

    async def arun(self, input_data: Dict[str, Any]) -> AgentResult:
        """Run the agent asynchronously"""
        try:
            result = await self._agent_executor.ainvoke(input_data)
            return self._format_result(result)
        except Exception as e:
            return AgentResult(
                success=False, message="Agent execution failed", error=str(e), data={}
            )

    def add_tool(self, tool: BaseTool) -> None:
        """Add a new tool to the agent"""
        self.tools.append(tool)
        self._setup_agent()

    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from the agent"""
        self.tools = [t for t in self.tools if t.name != tool_name]
        self._setup_agent()
