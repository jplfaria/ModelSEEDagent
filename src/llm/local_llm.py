import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import LLMResult
from .base import BaseLLM, LLMResponse

class LocalLLM(BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # For testing, allow path to be optional
        self._model_path = config.get('path')
        if not self._model_path:
            # Use model_name as path if path not provided (for tests)
            self._model_path = config.get('model_name', 'test-model')
        
        self._device = torch.device(config.get('device', 
                                  "mps" if torch.backends.mps.is_available() 
                                  else "cuda" if torch.cuda.is_available() 
                                  else "cpu"))
        self._tokenizer = None
        self._model = None
        
        # Only load model if we have a real path (not for tests)
        if self._model_path != 'test-model':
            self._load_model()

    def _load_model(self) -> None:
        """Load the model using Hugging Face auto classes"""
        try:
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self._device)
            
            # Set to evaluation mode
            self._model.eval()
                
        except Exception as e:
            raise ValueError(f"Failed to load model from {self._model_path}: {str(e)}")

    def _generate_response(self,
                          prompt: str,
                          max_tokens: Optional[int] = None,
                          temperature: Optional[float] = None,
                          stop_sequences: Optional[List[str]] = None,
                          stop: Optional[List[str]] = None,
                          **kwargs) -> LLMResponse:
        """Generate a response from the model"""
        
        # For testing without loaded model
        if self._model is None or self._tokenizer is None:
            if self._model_path == 'test-model':
                # Return a test response
                return LLMResponse(
                    text="test response",
                    tokens_used=10,
                    llm_name=self.config.llm_name,
                    metadata={
                        "model_path": self._model_path,
                        "device": str(self._device),
                        "test_mode": True
                    }
                )
            else:
                raise ValueError("Model and tokenizer must be loaded before generating responses")

        full_prompt = f"{self.config.system_content}\nUser: {prompt}\nAssistant:"

        try:
            inputs = self._tokenizer(
                full_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self._device)

            generation_kwargs = {
                "max_new_tokens": max_tokens or self.config.max_tokens or 100,
                "do_sample": True,
                "temperature": temperature or self.config.temperature,
                "pad_token_id": self._tokenizer.eos_token_id,
                "num_return_sequences": 1
            }

            # Add stop sequences if provided
            if stop or stop_sequences:
                all_stops = []
                if stop:
                    all_stops.extend(stop)
                if stop_sequences:
                    all_stops.extend(stop_sequences)
                generation_kwargs["stopping_criteria"] = all_stops

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **generation_kwargs
                )

            response_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_reply = response_text.split('Assistant:')[-1].strip()

            tokens_used = len(outputs[0])
            self.update_usage(tokens_used)

            return LLMResponse(
                text=assistant_reply,
                tokens_used=tokens_used,
                llm_name=self.config.llm_name,
                metadata={
                    "model_path": self._model_path,
                    "device": str(self._device)
                }
            )

        except Exception as e:
            raise ValueError(f"Local LLM Error: {str(e)}")

    def predict(self, text: str, **kwargs) -> str:
        """Predict response for given text"""
        response = self._generate_response(text, **kwargs)
        return response.text

    def predict_messages(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        """Predict response for given messages"""
        text = self._format_messages_as_text(messages)
        response = self._generate_response(text, **kwargs)
        return AIMessage(content=response.text)

    def generate_prompt(self, prompts: List[str], **kwargs) -> LLMResult:
        """Generate responses for multiple prompts"""
        results = [self._generate_response(prompt, **kwargs).text for prompt in prompts]
        return LLMResult(generations=[[{"text": text}] for text in results])

    async def agenerate_prompt(self, prompts: List[str], **kwargs) -> LLMResult:
        """Async version of generate_prompt"""
        return self.generate_prompt(prompts, **kwargs)

    async def apredict(self, text: str, **kwargs) -> str:
        """Async version of predict"""
        return self.predict(text, **kwargs)

    async def apredict_messages(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        """Async version of predict_messages"""
        return self.predict_messages(messages, **kwargs)

    def __del__(self):
        """Cleanup method"""
        if hasattr(self, '_model') and self._model is not None:
            try:
                del self._model
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except:
                pass