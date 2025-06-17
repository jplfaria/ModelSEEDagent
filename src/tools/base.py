import json
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

from langchain_core.tools import BaseTool as LangChainBaseTool
from pydantic import BaseModel, Field, field_validator


class ToolResult(BaseModel):
    """Enhanced ToolResult with Smart Summarization Framework support

    Three-tier information hierarchy:
    1. key_findings: Critical insights for LLM (≤2KB)
    2. summary_dict: Structured data for analysis (≤5KB)
    3. full_data_path: Complete raw data on disk (unlimited)
    """

    model_config = {"protected_namespaces": ()}

    # Core result fields
    success: bool
    message: str
    error: Optional[str] = None

    # Smart Summarization Framework fields
    full_data_path: Optional[str] = None  # Path to raw artifact on disk
    summary_dict: Optional[Dict[str, Any]] = None  # Compressed stats (≤5KB)
    key_findings: Optional[List[str]] = None  # Critical bullets (≤2KB)
    schema_version: str = "1.0"  # Framework version
    tool_name: Optional[str] = None  # For summarizer registry
    model_stats: Optional[Dict[str, Union[str, int]]] = (
        None  # Model metadata (id, reactions, genes, etc.)
    )

    # Legacy compatibility
    data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("key_findings")
    @classmethod
    def validate_key_findings_size(cls, v):
        """Ensure key_findings ≤ 2KB for LLM efficiency"""
        if v is not None:
            size = len(json.dumps(v))
            if size > 2000:
                raise ValueError(f"key_findings too large: {size}B > 2KB limit")
        return v

    @field_validator("summary_dict")
    @classmethod
    def validate_summary_dict_size(cls, v):
        """Ensure summary_dict ≤ 5KB for structured analysis"""
        if v is not None:
            size = len(json.dumps(v))
            if size > 5000:
                raise ValueError(f"summary_dict too large: {size}B > 5KB limit")
        return v

    def get_llm_summary(self) -> str:
        """Get LLM-optimized summary for agent consumption"""
        if self.key_findings:
            return "\n".join(f"• {finding}" for finding in self.key_findings)
        elif self.summary_dict:
            # Fallback to summary_dict if no key_findings
            return f"Analysis completed: {self.message}"
        else:
            # Legacy fallback
            return self.message

    def has_smart_summarization(self) -> bool:
        """Check if result uses smart summarization framework"""
        return self.key_findings is not None or self.summary_dict is not None

    def get_artifact_size(self) -> Optional[int]:
        """Get size of stored artifact in bytes"""
        if self.full_data_path and os.path.exists(self.full_data_path):
            return os.path.getsize(self.full_data_path)
        return None


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

        # Smart Summarization Framework configuration
        self._smart_summarization_enabled = config.get(
            "smart_summarization_enabled", False
        )
        self._force_summarization = config.get("force_summarization", False)

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

        # Apply smart summarization if enabled
        if self._smart_summarization_enabled and isinstance(result, ToolResult):
            result = self._apply_smart_summarization(result, input_data)

        return result

    def _apply_smart_summarization(
        self, result: ToolResult, input_data: Any
    ) -> ToolResult:
        """Apply smart summarization framework to tool result

        Args:
            result: Original tool result
            input_data: Original input to extract model information

        Returns:
            Enhanced result with smart summarization
        """
        try:
            # Import here to avoid circular imports
            from .smart_summarization import enable_smart_summarization

            # Extract model stats if available
            model_stats = self._extract_model_stats(result, input_data)

            # Enable smart summarization
            enhanced_result = enable_smart_summarization(
                tool_result=result,
                tool_name=self.tool_name,
                raw_data=result.data,
                model_stats=model_stats,
            )

            # Preserve original metadata and add summarization info
            if enhanced_result.metadata is None:
                enhanced_result.metadata = {}
            enhanced_result.metadata.update(result.metadata or {})
            enhanced_result.metadata["smart_summarization_enabled"] = True
            enhanced_result.metadata["schema_version"] = enhanced_result.schema_version

            return enhanced_result

        except Exception as e:
            # If summarization fails, return original result with error info
            if result.metadata is None:
                result.metadata = {}
            result.metadata["smart_summarization_error"] = str(e)
            result.metadata["smart_summarization_enabled"] = False
            return result

    def _extract_model_stats(
        self, result: ToolResult, input_data: Any
    ) -> Optional[Dict[str, Union[str, int]]]:
        """Extract model statistics from input or result for summarization context

        Args:
            result: Tool result
            input_data: Original input data

        Returns:
            Dictionary with model statistics or None
        """
        model_stats = {}

        # Try to extract model_id from input
        if isinstance(input_data, dict):
            model_path = input_data.get("model_path")
            if model_path:
                # Extract model name from path
                model_id = Path(model_path).stem
                model_stats["model_id"] = model_id
        elif isinstance(input_data, str) and input_data.endswith(
            (".xml", ".json", ".sbml")
        ):
            # Input is a model path
            model_id = Path(input_data).stem
            model_stats["model_id"] = model_id

        # Try to extract model statistics from result metadata
        if result.metadata:
            for key in ["model_id", "num_reactions", "num_genes", "num_metabolites"]:
                if key in result.metadata:
                    model_stats[key] = result.metadata[key]

        return model_stats if model_stats else None

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
