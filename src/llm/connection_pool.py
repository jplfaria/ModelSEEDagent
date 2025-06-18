"""
LLM Connection Pool Manager

Provides session-level connection pooling for LLM instances to eliminate
redundant HTTP client initialization and SSL context creation.
"""

import logging
import os
import threading
from typing import Any, Dict, Optional
from weakref import WeakValueDictionary

import httpx

from .argo import ArgoLLM
from .base import BaseLLM

logger = logging.getLogger(__name__)


class LLMConnectionPool:
    """Manages reusable LLM connections with connection pooling"""

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
            self._llm_instances: WeakValueDictionary = WeakValueDictionary()
            self._http_clients: Dict[str, httpx.Client] = {}
            self._clients_lock = threading.Lock()
            self._initialized = True
            logger.debug("LLM Connection Pool initialized")

    def get_http_client(self, config_key: str, timeout: float = 30.0) -> httpx.Client:
        """Get or create a reusable HTTP client for the given configuration"""
        with self._clients_lock:
            if config_key not in self._http_clients:
                self._http_clients[config_key] = httpx.Client(
                    timeout=timeout, follow_redirects=True
                )
                logger.debug(f"Created new HTTP client for config: {config_key}")
            else:
                logger.debug(f"Reusing existing HTTP client for config: {config_key}")

            return self._http_clients[config_key]

    def get_llm_instance(self, backend: str, config: Dict[str, Any]) -> BaseLLM:
        """Get or create a reusable LLM instance with connection pooling"""

        # Create a stable cache key from config
        cache_key = self._create_cache_key(backend, config)

        # Check if we already have this LLM instance
        if cache_key in self._llm_instances:
            logger.debug(f"Reusing existing LLM instance: {cache_key}")
            return self._llm_instances[cache_key]

        # Create new LLM instance with connection pooling
        if backend == "argo":
            llm = self._create_pooled_argo_llm(config, cache_key)
        else:
            # For other backends, use standard factory
            from .factory import LLMFactory

            llm = LLMFactory.create(backend, config)

        # Cache the instance
        self._llm_instances[cache_key] = llm
        logger.debug(f"Created and cached new LLM instance: {cache_key}")

        return llm

    def _create_cache_key(self, backend: str, config: Dict[str, Any]) -> str:
        """Create a stable cache key from LLM configuration"""
        # Extract key configuration elements that affect connection
        key_elements = [
            backend,
            config.get("model_name", "unknown"),
            config.get("env", "prod"),
            config.get("user", "default"),
            str(config.get("timeout", 30.0)),
        ]
        return "|".join(str(e) for e in key_elements)

    def _create_pooled_argo_llm(
        self, config: Dict[str, Any], cache_key: str
    ) -> ArgoLLM:
        """Create ArgoLLM instance with connection pooling"""

        # Create the ArgoLLM instance
        llm = PooledArgoLLM(config, self, cache_key)
        return llm

    def cleanup(self):
        """Clean up connection pool resources"""
        with self._clients_lock:
            for client in self._http_clients.values():
                try:
                    client.close()
                except Exception as e:
                    logger.warning(f"Error closing HTTP client: {e}")
            self._http_clients.clear()

        self._llm_instances.clear()
        logger.debug("LLM Connection Pool cleaned up")


class PooledArgoLLM(ArgoLLM):
    """ArgoLLM with connection pooling support"""

    def __init__(self, config: Dict[str, Any], pool: LLMConnectionPool, cache_key: str):
        # Initialize without calling parent __init__ to avoid creating HTTP client
        from .base import BaseLLM

        BaseLLM.__init__(self, config)

        # Store pool reference and cache key
        self._pool = pool
        self._cache_key = cache_key

        # Extract Argo-specific configuration (copy from parent __init__)
        self._model_name = self.config.llm_name
        self._user = (
            config.get("user")
            or os.getenv("ARGO_USER")
            or os.getenv("USER")
            or os.getlogin()
        )
        self._api_key = config.get("api_key") or os.getenv("ARGO_API_KEY")
        self._env = config.get("env") or self._determine_environment()
        self._retries = config.get("retries", 5)
        self._debug = config.get("debug", False) or os.getenv(
            "ARGO_DEBUG", ""
        ).lower() in ("1", "true")

        # Compute timeout
        self._timeout = self._compute_timeout(config.get("timeout"))

        # Build URLs
        self._base_url = self._build_base_url(self._env)
        self._is_test = config.get("api_base", "").startswith("https://test.")

        # Streaming configuration
        stream_capable = {"gpto3mini", "o1-mini"}
        self._supports_streaming = self._model_name in stream_capable
        self._use_streaming = config.get("stream", False) and self._supports_streaming

        # Build endpoint URL
        if self._is_test:
            self._url = config["api_base"]
        else:
            self._url = self._base_url + (
                "streamchat/" if self._use_streaming else "chat/"
            )

        # Get pooled HTTP client instead of creating new one
        client_key = f"{self._env}_{self._timeout}"
        self._client = self._pool.get_http_client(client_key, self._timeout)

        # Set up headers
        self._headers = {"Content-Type": "application/json"}
        if self._api_key:
            self._headers["x-api-key"] = self._api_key

        # Configure logging
        if self._debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled for PooledArgoLLM")

        logger.info(
            "PooledArgoLLM initialized | model=%s env=%s timeout=%.1fs url=%s cached=%s",
            self._model_name,
            self._env,
            self._timeout,
            self._url,
            cache_key,
        )

        # Test connectivity for dual-env models (skip in test mode)
        if not self._is_test:
            self._test_dual_env_connectivity()


# Global connection pool instance
_connection_pool = None


def get_connection_pool() -> LLMConnectionPool:
    """Get the global LLM connection pool instance"""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = LLMConnectionPool()
    return _connection_pool


def create_pooled_llm(backend: str, config: Dict[str, Any]) -> BaseLLM:
    """Create an LLM instance with connection pooling"""
    pool = get_connection_pool()
    return pool.get_llm_instance(backend, config)


def cleanup_connection_pool():
    """Clean up the global connection pool"""
    global _connection_pool
    if _connection_pool:
        _connection_pool.cleanup()
        _connection_pool = None
