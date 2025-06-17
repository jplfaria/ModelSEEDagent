"""Smart Summarization Framework for ModelSEED Agent

Implements three-tier information hierarchy to optimize LLM context usage:
1. key_findings: Critical insights for LLM consumption (≤2KB)
2. summary_dict: Structured data for follow-up analysis (≤5KB)  
3. full_data_path: Complete raw data stored on disk (unlimited)
"""

from typing import Any, Dict, Callable, Optional, List, Union
import json
import os
import tempfile
import uuid
from pathlib import Path
from abc import ABC, abstractmethod

from .base import ToolResult


class BaseSummarizer(ABC):
    """Base class for tool-specific summarizers"""
    
    @abstractmethod
    def summarize(self, raw_output: Any, artifact_path: str, model_stats: Optional[Dict[str, Union[str, int]]] = None) -> ToolResult:
        """Transform raw tool output into three-tier summarized result
        
        Args:
            raw_output: Original tool output data
            artifact_path: Path where full data should be stored
            model_stats: Model metadata (reactions, genes, metabolites counts)
            
        Returns:
            ToolResult with three-tier summarization
        """
        pass
    
    @abstractmethod 
    def get_tool_name(self) -> str:
        """Return the tool name this summarizer handles"""
        pass
    
    def validate_size_limits(self, key_findings: List[str], summary_dict: Dict[str, Any]) -> None:
        """Validate that summarization respects size limits"""
        key_findings_size = len(json.dumps(key_findings))
        summary_size = len(json.dumps(summary_dict))
        
        if key_findings_size > 2000:
            raise ValueError(f"key_findings too large: {key_findings_size}B > 2KB limit")
        if summary_size > 5000:
            raise ValueError(f"summary_dict too large: {summary_size}B > 5KB limit")


class SummarizerRegistry:
    """Central registry for tool summarizers"""
    
    def __init__(self):
        self._summarizers: Dict[str, BaseSummarizer] = {}
        
    def register(self, summarizer: BaseSummarizer) -> None:
        """Register a summarizer for a specific tool"""
        tool_name = summarizer.get_tool_name()
        self._summarizers[tool_name] = summarizer
        
    def get_summarizer(self, tool_name: str) -> Optional[BaseSummarizer]:
        """Get summarizer for a specific tool"""
        return self._summarizers.get(tool_name)
    
    def has_summarizer(self, tool_name: str) -> bool:
        """Check if a summarizer exists for the tool"""
        return tool_name in self._summarizers
    
    def list_tools(self) -> List[str]:
        """List all tools with registered summarizers"""
        return list(self._summarizers.keys())


# Global registry instance
summarizer_registry = SummarizerRegistry()


class ArtifactStorage:
    """Utility for storing and managing full data artifacts"""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.environ.get('MODELSEED_ARTIFACTS_DIR', '/tmp/modelseed_artifacts')
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def store_artifact(self, data: Any, tool_name: str, model_id: str = "unknown", 
                      format: str = "json") -> str:
        """Store full data artifact and return path
        
        Args:
            data: Raw data to store
            tool_name: Name of the tool that generated the data
            model_id: Identifier of the model analyzed
            format: Storage format (json, csv, pickle, etc.)
            
        Returns:
            Path to stored artifact
        """
        # Generate unique filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{tool_name}_{model_id}_{timestamp}_{unique_id}.{format}"
        
        artifact_path = self.base_dir / filename
        
        # Store data based on format
        if format == "json":
            with open(artifact_path, 'w') as f:
                if hasattr(data, 'to_dict'):
                    json.dump(data.to_dict(), f, indent=2)
                else:
                    json.dump(data, f, indent=2)
        elif format == "csv":
            if hasattr(data, 'to_csv'):
                data.to_csv(artifact_path, index=True)
            else:
                raise ValueError(f"Data type {type(data)} not compatible with CSV format")
        elif format == "pickle":
            import pickle
            with open(artifact_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return str(artifact_path)
    
    def load_artifact(self, artifact_path: str, format: str = "json") -> Any:
        """Load artifact from storage
        
        Args:
            artifact_path: Path to the artifact
            format: Storage format used
            
        Returns:
            Loaded data
        """
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
            
        if format == "json":
            with open(artifact_path, 'r') as f:
                return json.load(f)
        elif format == "csv":
            import pandas as pd
            return pd.read_csv(artifact_path, index_col=0)
        elif format == "pickle":
            import pickle
            with open(artifact_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def cleanup_old_artifacts(self, days_old: int = 7) -> int:
        """Clean up artifacts older than specified days
        
        Args:
            days_old: Remove artifacts older than this many days
            
        Returns:
            Number of artifacts removed
        """
        import time
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        removed_count = 0
        
        for artifact_path in self.base_dir.glob("*"):
            if artifact_path.is_file() and artifact_path.stat().st_mtime < cutoff_time:
                artifact_path.unlink()
                removed_count += 1
                
        return removed_count


# Global artifact storage instance
artifact_storage = ArtifactStorage()


def enable_smart_summarization(tool_result: ToolResult, tool_name: str, 
                               raw_data: Any = None, model_stats: Optional[Dict[str, Union[str, int]]] = None) -> ToolResult:
    """Enable smart summarization for a tool result
    
    Args:
        tool_result: Original tool result (legacy format)
        tool_name: Name of the tool
        raw_data: Raw data to store as artifact (optional)
        model_stats: Model metadata (optional)
        
    Returns:
        Enhanced ToolResult with smart summarization
    """
    summarizer = summarizer_registry.get_summarizer(tool_name)
    
    if summarizer is None:
        # No summarizer available - return original result with metadata
        if raw_data is not None:
            # Store raw data as artifact
            model_id = model_stats.get('model_id', 'unknown') if model_stats else 'unknown'
            artifact_path = artifact_storage.store_artifact(raw_data, tool_name, model_id)
            tool_result.full_data_path = artifact_path
            
        tool_result.tool_name = tool_name
        tool_result.model_stats = model_stats
        return tool_result
    
    # Use summarizer to create three-tier result
    if raw_data is None:
        raw_data = tool_result.data
        
    model_id = model_stats.get('model_id', 'unknown') if model_stats else 'unknown'
    artifact_path = artifact_storage.store_artifact(raw_data, tool_name, model_id)
    
    return summarizer.summarize(raw_data, artifact_path, model_stats)


# Import pandas for timestamp generation
try:
    import pandas as pd
except ImportError:
    # Fallback to datetime if pandas not available
    import datetime
    class pd:
        class Timestamp:
            @staticmethod
            def now():
                return datetime.datetime.now()