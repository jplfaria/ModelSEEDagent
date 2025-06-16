# src/llm/factory.py
from typing import Any, Dict, Type

from .argo import ArgoLLM
from .base import BaseLLM
from .openai_llm import OpenAILLM


class LLMFactory:
    @classmethod
    def _get_llm_class(cls, backend: str) -> Type[BaseLLM]:
        """Get the LLM class based on backend type with lazy loading"""
        if backend == "argo":
            return ArgoLLM
        elif backend == "openai":
            return OpenAILLM
        elif backend == "local":
            # Import LocalLLM only when needed to avoid import errors
            from .local_llm import LocalLLM

            return LocalLLM
        else:
            raise ValueError(
                f"Unsupported LLM backend: {backend}. "
                f"Available backends: {cls.list_backends()}"
            )

    @classmethod
    def create(cls, backend: str, config: Dict[str, Any]) -> BaseLLM:
        """Create an LLM instance based on the specified backend with connection pooling"""
        try:
            # Use connection pooling for supported backends
            if backend == "argo":
                from .connection_pool import create_pooled_llm

                return create_pooled_llm(backend, config)
            else:
                # Use direct creation for other backends
                llm_class = cls._get_llm_class(backend)
                return llm_class(config)
        except Exception as e:
            raise ValueError(f"Failed to create {backend} LLM: {str(e)}")

    @classmethod
    def list_backends(cls) -> list:
        """List all registered LLM backends"""
        return ["argo", "openai", "local"]
