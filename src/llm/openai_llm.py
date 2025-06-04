# src/llm/openai_llm.py
import openai
from typing import Optional, List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import LLMResult
from .base import BaseLLM, LLMResponse, LLMConfig

class OpenAILLM(BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        # Extract API key before calling super().__init__
        api_key = config.get('api_key')
        if not api_key:
            # For testing, allow missing API key
            api_key = 'test-api-key'
        
        super().__init__(config)
        
        # Set OpenAI API key globally
        openai.api_key = api_key
    
    def _generate_response(self,
                          prompt: str,
                          max_tokens: Optional[int] = None,
                          temperature: Optional[float] = None,
                          stop_sequences: Optional[List[str]] = None) -> LLMResponse:
        estimated_tokens = self.estimate_tokens(prompt)
        self.check_limits(estimated_tokens)
        
        messages = [
            {"role": "system", "content": self.config.system_content},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = openai.ChatCompletion.create(
                model=self.config.llm_name,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stop=stop_sequences or self.config.stop_sequences
            )
            
            reply = response["choices"][0]["message"]["content"]
            tokens_used = response['usage']['total_tokens']
            
            self.update_usage(tokens_used)
            
            return LLMResponse(
                text=reply,
                tokens_used=tokens_used,
                llm_name=self.config.llm_name,
                metadata={"api_response": response}
            )
            
        except Exception as e:
            raise ValueError(f"OpenAI API Error: {str(e)}")

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs) -> AIMessage:
        if isinstance(input, dict):
            text = input.get('input', '')
            if isinstance(text, (list, tuple)):
                text = self._format_messages_as_text(text)
            response = self.predict(str(text), **kwargs)
            return AIMessage(content=response)
        
        text = self._format_messages_as_text(input)
        response = self.predict(text, **kwargs)
        return AIMessage(content=response)
    
    async def agenerate_prompt(self, prompts: List[str], **kwargs) -> LLMResult:
        return self.generate_prompt(prompts, **kwargs)
    
    async def apredict(self, text: str, **kwargs) -> str:
        return self.predict(text, **kwargs)
    
    async def apredict_messages(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        return self.predict_messages(messages, **kwargs)
    
    async def ainvoke(self, input: Any, config: Optional[dict] = None, **kwargs) -> AIMessage:
        return await self.invoke(input, config, **kwargs)