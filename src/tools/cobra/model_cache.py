"""
COBRA Model Cache

Provides session-level caching for COBRA models to eliminate redundant
SBML file loading and improve performance.
"""

import hashlib
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import cobra
from cobra.io import read_sbml_model

logger = logging.getLogger(__name__)


class ModelCache:
    """Thread-safe cache for COBRA models with file modification tracking"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._cache: Dict[str, cobra.Model] = {}
            self._file_info: Dict[str, Dict] = (
                {}
            )  # Store file size, mtime for invalidation
            self._cache_lock = threading.Lock()
            self._initialized = True
            logger.debug("Model Cache initialized")

    def get_model(self, model_path: str) -> cobra.Model:
        """
        Get model from cache or load from file if not cached or file changed.

        Args:
            model_path: Path to the SBML model file

        Returns:
            cobra.Model: The loaded model (potentially from cache)
        """
        # Resolve path to absolute
        resolved_path = self._resolve_path(model_path)
        cache_key = resolved_path

        with self._cache_lock:
            # Check if we have the model cached and file hasn't changed
            if self._is_cache_valid(cache_key, resolved_path):
                logger.debug(f"🔥 Using cached model: {model_path}")
                # Return a copy to prevent modifications affecting cache
                return self._cache[cache_key].copy()

            # Load model from file
            logger.debug(f"📁 Loading model from disk: {model_path}")
            model = self._load_model_from_file(resolved_path)

            # Cache the model and file info
            self._cache[cache_key] = model.copy()  # Store a copy
            self._file_info[cache_key] = self._get_file_info(resolved_path)

            logger.debug(f"✅ Cached model: {model_path} (ID: {model.id})")
            return model

    def _resolve_path(self, model_path: str) -> str:
        """Resolve model path to absolute path"""
        if os.path.isabs(model_path):
            return model_path

        # Try relative to project root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        project_relative = project_root / model_path

        if project_relative.exists():
            return str(project_relative.resolve())

        # Return absolute path from current directory
        return str(Path(model_path).resolve())

    def _is_cache_valid(self, cache_key: str, file_path: str) -> bool:
        """Check if cached model is still valid (file hasn't changed)"""
        if cache_key not in self._cache:
            return False

        if cache_key not in self._file_info:
            return False

        # Check if file still exists
        if not os.path.exists(file_path):
            logger.warning(f"Cached model file no longer exists: {file_path}")
            return False

        # Check if file has been modified
        current_info = self._get_file_info(file_path)
        cached_info = self._file_info[cache_key]

        if (
            current_info["size"] != cached_info["size"]
            or current_info["mtime"] != cached_info["mtime"]
        ):
            logger.debug(f"Model file changed, invalidating cache: {file_path}")
            return False

        return True

    def _get_file_info(self, file_path: str) -> Dict:
        """Get file information for cache invalidation"""
        stat = os.stat(file_path)
        return {"size": stat.st_size, "mtime": stat.st_mtime}

    def _load_model_from_file(self, file_path: str) -> cobra.Model:
        """Load model from SBML file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Model file not found: "{file_path}"')

        try:
            model = read_sbml_model(file_path)
            logger.info(f"Successfully loaded model: {model.id}")
            return model
        except Exception as e:
            logger.error(f'Error loading model from "{file_path}": {str(e)}')
            raise ValueError(f"Failed to load model: {str(e)}")

    def clear_cache(self):
        """Clear all cached models"""
        with self._cache_lock:
            self._cache.clear()
            self._file_info.clear()
            logger.debug("Model cache cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self._cache_lock:
            return {
                "cached_models": len(self._cache),
                "cache_keys": list(self._cache.keys()),
                "memory_usage_mb": self._estimate_memory_usage(),
            }

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of cached models (rough approximation)"""
        total_size = 0
        for model in self._cache.values():
            # Rough estimate: reactions + metabolites + genes
            model_size = (
                len(model.reactions) + len(model.metabolites) + len(model.genes)
            )
            total_size += model_size * 100  # Rough bytes per entity estimate

        return total_size / (1024 * 1024)  # Convert to MB


class CachedModelUtils:
    """Drop-in replacement for ModelUtils that uses caching"""

    def __init__(self, cache: Optional[ModelCache] = None):
        self._cache = cache or get_model_cache()

    def load_model(self, model_path: str) -> cobra.Model:
        """
        Load a metabolic model from a file with caching.

        Args:
            model_path: Path to the SBML model file

        Returns:
            cobra.Model: The loaded metabolic model

        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If the model file is invalid
        """
        return self._cache.get_model(model_path)


# Global cache instance
_model_cache = None


def get_model_cache() -> ModelCache:
    """Get the global model cache instance"""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache


def clear_model_cache():
    """Clear the global model cache"""
    cache = get_model_cache()
    cache.clear_cache()


def get_cache_stats() -> Dict:
    """Get statistics about the model cache"""
    cache = get_model_cache()
    return cache.get_cache_stats()


@lru_cache(maxsize=32)
def load_model_cached(model_path: str) -> cobra.Model:
    """
    Simple LRU cache wrapper for model loading.
    Alternative approach using functools.lru_cache.
    """
    cache = get_model_cache()
    return cache.get_model(model_path)
