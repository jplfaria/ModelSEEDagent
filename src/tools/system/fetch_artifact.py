"""FetchArtifact Tool for Smart Summarization Framework

This tool provides access to full raw data artifacts stored by the Smart Summarization
Framework. When tools produce large outputs, the framework stores complete data as
JSON artifacts while providing compressed summaries to the LLM. This tool allows
retrieval of the full data when detailed analysis is needed.

Example usage:
    result = fetch_artifact_tool.run({
        "artifact_path": "/tmp/modelseed_artifacts/flux_sampling_e_coli_20250617_123456.json"
    })
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..base import BaseTool, ToolConfig, ToolRegistry, ToolResult


@ToolRegistry.register
class FetchArtifactTool(BaseTool):
    """Tool for retrieving full data artifacts from Smart Summarization storage"""

    tool_name = "fetch_artifact"
    tool_description = (
        "Retrieve full raw data from Smart Summarization artifacts when detailed "
        "analysis is needed beyond the compressed summaries"
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Store artifact directory in config for access in _run_tool
        self._artifact_dir = Path(
            config.get(
                "artifact_dir",
                os.environ.get("MODELSEED_ARTIFACTS_DIR", "/tmp/modelseed_artifacts"),
            )
        )

    def _run_tool(self, input_data: Union[str, Dict[str, Any]]) -> ToolResult:
        """Fetch and return artifact data

        Args:
            input_data: Either artifact path string or dict with 'artifact_path' key

        Returns:
            ToolResult with the loaded artifact data
        """
        try:
            # Handle input formats
            if isinstance(input_data, str):
                artifact_path = input_data
            elif isinstance(input_data, dict):
                artifact_path = input_data.get("artifact_path")
                if not artifact_path:
                    return ToolResult(
                        success=False,
                        message="No artifact_path provided",
                        error="Missing required parameter: artifact_path",
                    )
            else:
                return ToolResult(
                    success=False,
                    message="Invalid input format",
                    error=f"Expected string or dict, got {type(input_data)}",
                )

            # Convert to Path object
            artifact_path = Path(artifact_path)

            # Security check - ensure path is within artifact directory
            try:
                artifact_path = artifact_path.resolve()
                self._artifact_dir.resolve()

                # Check if path is under artifact directory (security measure)
                if not str(artifact_path).startswith(str(self._artifact_dir)):
                    return ToolResult(
                        success=False,
                        message="Access denied: artifact path outside allowed directory",
                        error=f"Path {artifact_path} is not within {self._artifact_dir}",
                    )
            except Exception as e:
                return ToolResult(
                    success=False, message="Invalid artifact path", error=str(e)
                )

            # Check if file exists
            if not artifact_path.exists():
                return ToolResult(
                    success=False,
                    message=f"Artifact not found: {artifact_path}",
                    error="File does not exist",
                )

            # Load the artifact
            try:
                with open(artifact_path, "r") as f:
                    artifact_data = json.load(f)
            except json.JSONDecodeError as e:
                return ToolResult(
                    success=False, message="Failed to parse artifact JSON", error=str(e)
                )
            except Exception as e:
                return ToolResult(
                    success=False, message="Failed to read artifact file", error=str(e)
                )

            # Extract metadata from filename if possible
            filename = artifact_path.name
            metadata = {
                "artifact_path": str(artifact_path),
                "file_size_bytes": artifact_path.stat().st_size,
                "filename": filename,
            }

            # Try to parse tool name and model from filename
            # Expected format: toolname_modelid_timestamp_uuid.json
            parts = filename.replace(".json", "").split("_")
            if len(parts) >= 2:
                metadata["tool_name"] = parts[0]
                metadata["model_id"] = parts[1]

            # Determine data type and size
            if isinstance(artifact_data, dict):
                metadata["data_type"] = "dictionary"
                metadata["num_keys"] = len(artifact_data)

                # Special handling for common tool outputs
                if "fva_results" in artifact_data:
                    metadata["tool_type"] = "flux_variability"
                    metadata["num_reactions"] = len(
                        artifact_data.get("fva_results", {}).get("minimum", {})
                    )
                elif "samples" in artifact_data:
                    metadata["tool_type"] = "flux_sampling"
                    metadata["num_samples"] = len(artifact_data.get("samples", []))
                elif "gene_deletions" in artifact_data:
                    metadata["tool_type"] = "gene_deletion"
                    metadata["num_genes"] = len(artifact_data.get("gene_deletions", {}))
                elif (
                    "objective_value" in artifact_data
                    and "significant_fluxes" in artifact_data
                ):
                    metadata["tool_type"] = "fba"
                    metadata["growth_rate"] = artifact_data.get("objective_value", 0)

            elif isinstance(artifact_data, list):
                metadata["data_type"] = "list"
                metadata["num_items"] = len(artifact_data)
            else:
                metadata["data_type"] = type(artifact_data).__name__

            return ToolResult(
                success=True,
                message=f"Successfully loaded artifact from {filename}",
                data=artifact_data,
                metadata=metadata,
                key_findings=[
                    f"Loaded {metadata.get('data_type', 'unknown')} artifact with {metadata.get('file_size_bytes', 0):,} bytes",
                    f"Tool: {metadata.get('tool_name', 'unknown')}, Model: {metadata.get('model_id', 'unknown')}",
                    f"Data contains {metadata.get('num_keys', metadata.get('num_items', 'unknown'))} entries",
                ],
            )

        except Exception as e:
            return ToolResult(
                success=False,
                message="Unexpected error fetching artifact",
                error=str(e),
            )

    def validate_input(self, input_data: Any) -> tuple[bool, Optional[str]]:
        """Validate input parameters

        Args:
            input_data: Input to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if input_data is None:
            return False, "Input data is required"

        if isinstance(input_data, str):
            return True, None

        if isinstance(input_data, dict):
            if "artifact_path" not in input_data:
                return False, "artifact_path is required in input dictionary"
            return True, None

        return False, f"Invalid input type: {type(input_data)}"
