from typing import Optional, Dict, Any, List, ClassVar
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool as LangChainBaseTool

class ToolResult(BaseModel):
    model_config = {"protected_namespaces": ()}
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

class BaseTool(LangChainBaseTool):
    """Base class for all tools"""
    tool_name: ClassVar[str] = ""
    tool_description: ClassVar[str] = ""
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize the parent class first
        super().__init__(
            name=config.get("name", self.tool_name),
            description=config.get("description", self.tool_description)
        )
        # Store the configuration
        self._config = config
    
    @property
    def tool_config(self) -> Dict[str, Any]:
        """Get the tool configuration"""
        return self._config

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