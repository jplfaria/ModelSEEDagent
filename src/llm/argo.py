# src/llm/argo.py
import requests
import httpx
import asyncio
from typing import Optional, List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.outputs import LLMResult
from .base import BaseLLM, LLMResponse

class ArgoLLM(BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._api_base = self._config.api_base
        self._user = self._config.user
        # Optional streaming API endpoint (if provided in the config)
        self._streaming_api = getattr(self._config, "streaming_api", None)
        
        if not self._api_base or not self._user:
            raise ValueError("ArgoLLM requires api_base and user to be set in config")
    
    @property
    def api_base(self):
        return self._api_base
        
    @property
    def user(self):
        return self._user
    
    def _format_messages(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        prompt = []
        system_message = None
        
        for message in messages:
            if isinstance(message, SystemMessage):
                system_message = message.content
            elif isinstance(message, HumanMessage):
                prompt.append(message.content)
            elif isinstance(message, AIMessage):
                prompt.append(message.content)
        
        return {
            "prompt": prompt,
            "system": system_message or self.config.system_content
        }
    
    def _get_prompt_str(self, prompt: Any) -> str:
        if hasattr(prompt, "to_string"):
            return prompt.to_string()
        elif hasattr(prompt, "text"):
            return prompt.text
        elif isinstance(prompt, str):
            return prompt
        elif isinstance(prompt, list):
            return " ".join(str(p) for p in prompt)
        else:
            return str(prompt)
    
    def _generate_response(self,
                           prompt: Any,
                           max_tokens: Optional[int] = None,
                           temperature: Optional[float] = None,
                           stop_sequences: Optional[List[str]] = None,
                           stream: bool = False,
                           **kwargs) -> LLMResponse:
        # Convert prompt to string format
        if isinstance(prompt, list) and all(isinstance(m, BaseMessage) for m in prompt):
            formatted = self._format_messages(prompt)
            prompt_text = formatted["prompt"]
            system = formatted["system"]
        else:
            prompt_text = [self._get_prompt_str(prompt)]
            system = self.config.system_content
            
        estimated_tokens = self.estimate_tokens(str(prompt_text))
        self.check_limits(estimated_tokens)
        
        payload = {
            "user": self.user,
            "model": self.config.llm_name,
            "prompt": prompt_text,
            "system": system,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens or 1000,
            "stop": stop_sequences or self.config.stop_sequences or []
        }
        
        # Special handling for o1 models: only use 'prompt' and 'max_completion_tokens'
        if self.config.llm_name in ["gpto1preview", "gpto1mini"]:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            payload.pop("system", None)
            payload.pop("stop", None)
            payload.pop("temperature", None)
            payload.pop("top_p", None)
        
        if not stream:
            # Synchronous call using requests
            response = requests.post(self.api_base, json=payload, headers={'Content-Type': 'application/json'})
            response.raise_for_status()
            data = response.json()
            reply = data.get('response', '')
            if not reply:
                raise ValueError("No response received from Argo API")
            tokens_used = data.get('usage', {}).get('total_tokens', estimated_tokens)
            self.update_usage(tokens_used)
            return LLMResponse(
                text=reply,
                tokens_used=tokens_used,
                llm_name=self.config.llm_name,
                metadata={"api_response": data}
            )
        else:
            # Streaming call using httpx asynchronous client
            if not self._streaming_api:
                raise ValueError("Streaming API endpoint not configured in the config")
            async def stream_response() -> LLMResponse:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    async with client.stream("POST", self._streaming_api, json=payload, headers={'Content-Type': 'application/json'}) as resp:
                        resp.raise_for_status()
                        collected_response = ""
                        async for chunk in resp.aiter_text():
                            collected_response += chunk
                        # A simple token estimation based on word count; adjust as needed
                        tokens_used = len(collected_response.split())
                        self.update_usage(tokens_used)
                        return LLMResponse(
                            text=collected_response,
                            tokens_used=tokens_used,
                            llm_name=self.config.llm_name,
                            metadata={"streamed": True}
                        )
            return asyncio.run(stream_response())
    
    # Asynchronous version of _generate_response for async workflows
    async def a_generate_response(self,
                                  prompt: Any,
                                  max_tokens: Optional[int] = None,
                                  temperature: Optional[float] = None,
                                  stop_sequences: Optional[List[str]] = None,
                                  stream: bool = False,
                                  **kwargs) -> LLMResponse:
        if isinstance(prompt, list) and all(isinstance(m, BaseMessage) for m in prompt):
            formatted = self._format_messages(prompt)
            prompt_text = formatted["prompt"]
            system = formatted["system"]
        else:
            prompt_text = [self._get_prompt_str(prompt)]
            system = self.config.system_content
            
        estimated_tokens = self.estimate_tokens(str(prompt_text))
        self.check_limits(estimated_tokens)
        
        payload = {
            "user": self.user,
            "model": self.config.llm_name,
            "prompt": prompt_text,
            "system": system,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens or 1000,
            "stop": stop_sequences or self.config.stop_sequences or []
        }
        
        if self.config.llm_name in ["gpto1preview", "gpto1mini"]:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            payload.pop("system", None)
            payload.pop("stop", None)
            payload.pop("temperature", None)
            payload.pop("top_p", None)
        
        if not stream:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(self.api_base, json=payload, headers={'Content-Type': 'application/json'})
                resp.raise_for_status()
                data = resp.json()
                reply = data.get('response', '')
                if not reply:
                    raise ValueError("No response received from Argo API")
                tokens_used = data.get('usage', {}).get('total_tokens', estimated_tokens)
                self.update_usage(tokens_used)
                return LLMResponse(
                    text=reply,
                    tokens_used=tokens_used,
                    llm_name=self.config.llm_name,
                    metadata={"api_response": data}
                )
        else:
            if not self._streaming_api:
                raise ValueError("Streaming API endpoint not configured in the config")
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", self._streaming_api, json=payload, headers={'Content-Type': 'application/json'}) as resp:
                    resp.raise_for_status()
                    collected_response = ""
                    async for chunk in resp.aiter_text():
                        collected_response += chunk
                    tokens_used = len(collected_response.split())
                    self.update_usage(tokens_used)
                    return LLMResponse(
                        text=collected_response,
                        tokens_used=tokens_used,
                        llm_name=self.config.llm_name,
                        metadata={"streamed": True}
                    )