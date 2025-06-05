from typing import Any, ClassVar, Dict, List, Optional

from langchain_core.tools import BaseTool as LangChainBaseTool
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    model_config = {"protected_namespaces": ()}
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class ToolConfig(BaseModel):
    """Configuration object for tools"""

    name: str
    description: str
    additional_config: Dict[str, Any] = Field(default_factory=dict)


class BaseTool(LangChainBaseTool):
    """Base class for all tools with integrated execution auditing"""

    tool_name: ClassVar[str] = ""
    tool_description: ClassVar[str] = ""

    def __init__(self, config: Dict[str, Any]):
        # Initialize the parent class first
        super().__init__(
            name=config.get("name", self.tool_name),
            description=config.get("description", self.tool_description),
        )
        # Store the configuration
        self._config = config
        self._tool_config = ToolConfig(
            name=config.get("name", self.tool_name),
            description=config.get("description", self.tool_description),
            additional_config={
                k: v for k, v in config.items() if k not in ["name", "description"]
            },
        )

        # Audit configuration
        self._audit_enabled = config.get("audit_enabled", True)
        self._audit_watch_dirs = config.get("audit_watch_dirs", ["."])

    @property
    def config(self) -> ToolConfig:
        """Get the tool configuration object for test compatibility"""
        return self._tool_config

    @property
    def tool_config(self) -> Dict[str, Any]:
        """Get the raw tool configuration"""
        return self._config

    def _run(self, input_data: Any) -> Any:
        """
        Main execution method that handles auditing automatically.

        This method wraps the actual tool execution (_run_tool) with
        comprehensive auditing capabilities for hallucination detection.
        """
        if not self._audit_enabled:
            # Direct execution without auditing
            return self._run_tool(input_data)

        # Import here to avoid circular imports
        from .audit import get_auditor

        auditor = get_auditor()

        # Execute with full auditing
        result, audit_file = auditor.audit_tool_execution(
            tool_name=self.name,
            input_data=input_data,
            execution_func=self._run_tool,
            watch_dirs=self._audit_watch_dirs,
        )

        # Add audit information to result metadata if it's a ToolResult
        if hasattr(result, "metadata") and isinstance(result.metadata, dict):
            result.metadata["audit_file"] = str(audit_file)
            result.metadata["audit_enabled"] = True

        return result

    def _run_tool(self, input_data: Any) -> Any:
        """
        Actual tool implementation - override this in subclasses.

        This method should contain the core tool logic without any
        auditing concerns. The auditing is handled transparently
        by the _run method above.
        """
        raise NotImplementedError("Subclasses must implement _run_tool method")


class ToolRegistry:
    _tools: Dict[str, type] = {}

    @classmethod
    def register(cls, tool_class: type):
        if not issubclass(tool_class, BaseTool):
            raise ValueError(f"Tool class must inherit from BaseTool: {tool_class}")
        cls._tools[tool_class.tool_name] = tool_class
        return tool_class

    @classmethod
    def get_tool(cls, name: str) -> Optional[type]:
        return cls._tools.get(name)

    @classmethod
    def create_tool(cls, name: str, config: Dict[str, Any]) -> BaseTool:
        tool_class = cls.get_tool(name)
        if tool_class is None:
            raise ValueError(f"Tool not found: {name}")
        return tool_class(config)

    @classmethod
    def list_tools(cls) -> List[str]:
        return list(cls._tools.keys())
