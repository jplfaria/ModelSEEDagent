from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM as LangchainBaseLLM
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import Generation, LLMResult
from pydantic import BaseModel, ConfigDict, Field


@dataclass
class LLMConfig:
    llm_name: str
    system_content: str
    max_tokens: Optional[int]
    temperature: float = 0.7
    stop_sequences: Optional[List[str]] = None
    api_base: Optional[str] = None
    user: Optional[str] = None
    safety_settings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "max_api_calls": 100,
            "max_tokens": 50000,
        }
    )

    @property
    def model_name(self) -> str:
        """Alias for llm_name for backwards compatibility"""
        return self.llm_name


class LLMResponse(BaseModel):
    text: str
    tokens_used: int
    llm_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(protected_namespaces=())

    @property
    def model_name(self) -> str:
        """Alias for llm_name for backwards compatibility"""
        return self.llm_name


class BaseLLM(LangchainBaseLLM):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self._tokens = 0
        self._calls = 0
        self._config = self._create_config(config)

    def _create_config(self, config: Dict[str, Any]) -> LLMConfig:
        safety_settings = config.get(
            "safety_settings",
            {"enabled": True, "max_api_calls": 100, "max_tokens": 50000},
        )

        # Support both 'model_name' (for tests) and 'llm_name' (legacy)
        llm_name = config.get("model_name") or config.get("llm_name")
        if not llm_name:
            raise ValueError(
                "Either 'model_name' or 'llm_name' must be provided in config"
            )

        return LLMConfig(
            llm_name=llm_name,
            system_content=config["system_content"],
            max_tokens=config.get("max_tokens"),
            temperature=config.get("temperature", 0.7),
            stop_sequences=config.get("stop_sequences"),
            api_base=config.get("api_base"),
            user=config.get("user"),
            safety_settings=safety_settings,
        )

    @property
    def config(self):
        return self._config

    @property
    def total_tokens(self):
        return self._tokens

    @total_tokens.setter
    def total_tokens(self, value):
        self._tokens = value

    @property
    def total_calls(self):
        return self._calls

    def update_usage(self, tokens: int):
        self._tokens += tokens
        self._calls += 1

    def check_limits(self, estimated_tokens: int):
        if self.config.safety_settings["enabled"]:
            max_tokens = self.config.safety_settings["max_tokens"]
            max_calls = self.config.safety_settings["max_api_calls"]

            if self._tokens + estimated_tokens > max_tokens:
                raise ValueError(
                    f"Token limit exceeded. Current: {self._tokens}, Requested: {estimated_tokens}"
                )
            if self._calls + 1 > max_calls:
                raise ValueError(f"API call limit exceeded. Current: {self._calls}")

    @abstractmethod
    def _generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        pass

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            response = self._generate_response(prompt, stop_sequences=stop, **kwargs)
            generations.append([Generation(text=response.text)])
        return LLMResult(generations=generations)

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _format_messages_as_text(self, messages: Sequence[BaseMessage]) -> str:
        message_text = []
        for message in messages:
            role = message.__class__.__name__.replace("Message", "")
            message_text.append(f"{role}: {message.content}")
        return "\n".join(message_text)

    def predict(self, text: str, **kwargs: Any) -> str:
        response = self._generate_response(text, **kwargs)
        return response.text

    def predict_messages(
        self, messages: List[BaseMessage], **kwargs: Any
    ) -> BaseMessage:
        text = self._format_messages_as_text(messages)
        response = self._generate_response(text, **kwargs)
        return AIMessage(content=response.text)

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs) -> AIMessage:
        if isinstance(input, dict):
            text = input.get("input", "")
            if isinstance(text, (list, tuple)):
                text = self._format_messages_as_text(text)
            response = self.predict(str(text), **kwargs)
            return AIMessage(content=response)

        if isinstance(input, (list, tuple)):
            text = self._format_messages_as_text(input)
        else:
            text = str(input)
        response = self.predict(text, **kwargs)
        return AIMessage(content=response)

    async def ainvoke(
        self, input: Any, config: Optional[dict] = None, **kwargs
    ) -> AIMessage:
        return self.invoke(input, config, **kwargs)

    async def apredict(self, text: str, **kwargs) -> str:
        return self.predict(text, **kwargs)

    async def apredict_messages(
        self, messages: List[BaseMessage], **kwargs
    ) -> BaseMessage:
        return self.predict_messages(messages, **kwargs)

    @property
    def _llm_type(self) -> str:
        return self.__class__.__name__.lower().replace("llm", "")
