from unittest.mock import Mock, patch

import pytest

from src.llm import ArgoLLM, BaseLLM, LLMFactory, LocalLLM, OpenAILLM
from src.llm.base import LLMConfig, LLMResponse


@pytest.fixture
def mock_config():
    return {
        "model_name": "test-model",
        "system_content": "test system content",
        "max_tokens": 100,
        "temperature": 0.7,
        "safety_settings": {"enabled": True, "max_api_calls": 10, "max_tokens": 1000},
    }


@pytest.fixture
def mock_argo_config(mock_config):
    return {**mock_config, "api_base": "https://test.api/", "user": "test_user"}


class TestBaseLLM:
    def test_init(self, mock_config):
        # Create a concrete implementation for testing
        class ConcreteLLM(BaseLLM):
            def _generate_response(self, prompt: str) -> LLMResponse:
                return LLMResponse(
                    text="test response",
                    tokens_used=10,
                    model_name=self.config.model_name,
                )

        llm = ConcreteLLM(mock_config)
        assert llm.config.model_name == "test-model"
        assert llm.total_tokens == 0
        assert llm.total_calls == 0

    def test_check_limits_success(self, mock_config):
        # Create a concrete implementation for testing
        class ConcreteLLM(BaseLLM):
            def _generate_response(self, prompt: str) -> LLMResponse:
                return LLMResponse(
                    text="test response",
                    tokens_used=10,
                    model_name=self.config.model_name,
                )

        llm = ConcreteLLM(mock_config)
        llm.check_limits(100)  # Should not raise exception

    def test_check_limits_exceeded(self, mock_config):
        # Create a concrete implementation for testing
        class ConcreteLLM(BaseLLM):
            def _generate_response(self, prompt: str) -> LLMResponse:
                return LLMResponse(
                    text="test response",
                    tokens_used=10,
                    model_name=self.config.model_name,
                )

        llm = ConcreteLLM(mock_config)
        llm.total_tokens = 950
        with pytest.raises(ValueError, match="Token limit exceeded"):
            llm.check_limits(100)


class TestArgoLLM:
    @patch("requests.post")
    def test_generate_response(self, mock_post, mock_argo_config):
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "test response",
            "usage": {"total_tokens": 50},
        }
        mock_post.return_value = mock_response

        llm = ArgoLLM(mock_argo_config)
        response = llm._generate_response("test prompt")

        assert isinstance(response, LLMResponse)
        assert response.text == "test response"
        assert response.tokens_used == 50
        assert response.model_name == "test-model"

    @patch("requests.post")
    def test_api_error(self, mock_post, mock_argo_config):
        # Mock API error
        mock_post.side_effect = Exception("API Error")

        llm = ArgoLLM(mock_argo_config)
        with pytest.raises(ValueError, match="API Error"):
            llm._generate_response("test prompt")


class TestOpenAILLM:
    @patch("openai.ChatCompletion.create")
    def test_generate_response(self, mock_create, mock_config):
        # Mock OpenAI API response
        mock_create.return_value = {
            "choices": [{"message": {"content": "test response"}}],
            "usage": {"total_tokens": 50},
        }

        llm = OpenAILLM(mock_config)
        response = llm._generate_response("test prompt")

        assert isinstance(response, LLMResponse)
        assert response.text == "test response"
        assert response.tokens_used == 50


class TestLocalLLM:
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_generate_response(self, mock_tokenizer, mock_model, mock_config):
        # Mock tokenizer and model
        mock_tokenizer.return_value.encode.return_value = [[1, 2, 3]]
        mock_tokenizer.return_value.decode.return_value = "test response"
        mock_model.return_value.generate.return_value = [[1, 2, 3, 4]]

        llm = LocalLLM(mock_config)
        response = llm._generate_response("test prompt")

        assert isinstance(response, LLMResponse)
        assert response.text == "test response"


class TestLLMFactory:
    def test_create_argo(self, mock_argo_config):
        llm = LLMFactory.create("argo", mock_argo_config)
        assert isinstance(llm, ArgoLLM)

    def test_create_openai(self, mock_config):
        llm = LLMFactory.create("openai", mock_config)
        assert isinstance(llm, OpenAILLM)

    def test_create_local(self, mock_config):
        llm = LLMFactory.create("local", mock_config)
        assert isinstance(llm, LocalLLM)

    def test_invalid_backend(self, mock_config):
        with pytest.raises(ValueError, match="Unsupported LLM backend"):
            LLMFactory.create("invalid", mock_config)
