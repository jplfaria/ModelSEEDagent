"""FetchArtifact Tool for Smart Summarization Framework

Allows agents to retrieve complete raw data from Smart Summarization artifacts
when detailed analysis is needed beyond the key findings and summary.
"""

from typing import Any, Dict

from pydantic import BaseModel

from .base import BaseTool, ToolRegistry, ToolResult
from .smart_summarization import artifact_storage


class FetchArtifactInput(BaseModel):
    """Input for fetching artifact data"""

    artifact_path: str
    format: str = "json"


@ToolRegistry.register
class FetchArtifactTool(BaseTool):
    """Tool for retrieving full data from Smart Summarization artifacts"""

    tool_name = "fetch_artifact_data"
    tool_description = """Retrieve complete raw data from a Smart Summarization artifact
    when detailed analysis is needed beyond the key findings and summary.

    Use this tool when:
    - User asks for "detailed analysis" or "complete results"
    - Statistical analysis beyond summary_dict scope is needed
    - Debugging or troubleshooting scenarios require full data
    - Cross-model comparisons requiring raw data

    Input: artifact_path (string) - Path to the stored artifact
    """

    def _run_tool(self, input_data: Any) -> ToolResult:
        """Fetch and return complete artifact data

        Args:
            input_data: Either artifact_path string or dict with artifact_path

        Returns:
            ToolResult with complete raw data
        """
        try:
            # Handle both string and dict inputs
            if isinstance(input_data, str):
                artifact_path = input_data
                format_type = "json"
            elif isinstance(input_data, dict):
                artifact_path = input_data.get("artifact_path")
                format_type = input_data.get("format", "json")
            else:
                return ToolResult(
                    success=False,
                    message="Invalid input: expected artifact_path string or dict",
                    error="Invalid input format",
                )

            if not artifact_path:
                return ToolResult(
                    success=False,
                    message="Missing required parameter: artifact_path",
                    error="artifact_path is required",
                )

            # Load the artifact data
            try:
                raw_data = artifact_storage.load_artifact(artifact_path, format_type)
            except FileNotFoundError:
                return ToolResult(
                    success=False,
                    message=f"Artifact not found: {artifact_path}",
                    error="Artifact file does not exist",
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    message="Failed to load artifact data",
                    error=f"Loading error: {str(e)}",
                )

            # Calculate data size for metrics
            data_size = len(str(raw_data))

            return ToolResult(
                success=True,
                message=f"Successfully retrieved artifact data ({data_size:,} bytes)",
                data=raw_data,
                metadata={
                    "artifact_path": artifact_path,
                    "format": format_type,
                    "data_size_bytes": data_size,
                    "fetch_timestamp": str(artifact_storage.pd.Timestamp.now()),
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message="Error fetching artifact data",
                error=str(e),
            )
