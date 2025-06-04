from unittest.mock import Mock, patch

import pytest
from langchain.agents import AgentExecutor

from src.agents.base import AgentResult, BaseAgent
from src.agents.factory import AgentFactory
from src.agents.metabolic import MetabolicAgent
from src.llm.base import BaseLLM
from src.tools.base import BaseTool, ToolResult


@pytest.fixture
def mock_llm():
    llm = Mock(spec=BaseLLM)
    llm._generate_response.return_value.text = "Test response"
    return llm


@pytest.fixture
def mock_tool():
    tool = Mock(spec=BaseTool)
    tool.name = "test_tool"
    tool.description = "Test tool"
    tool._run.return_value = ToolResult(
        success=True, message="Tool executed successfully", data={"test": "data"}
    )
    return tool


@pytest.fixture
def mock_agent_config():
    return {
        "name": "test_agent",
        "description": "Test agent",
        "max_iterations": 3,
        "verbose": True,
        "handle_parsing_errors": True,
    }


class TestBaseAgent:
    def test_init(self, mock_llm, mock_tool, mock_agent_config):
        class TestAgent(BaseAgent):
            def _create_prompt(self):
                return Mock()

            def _create_agent(self):
                return Mock()

        agent = TestAgent(mock_llm, [mock_tool], mock_agent_config)
        assert agent.llm == mock_llm
        assert len(agent.tools) == 1
        assert agent.config.name == "test_agent"

    def test_format_result(self, mock_llm, mock_tool, mock_agent_config):
        class TestAgent(BaseAgent):
            def _create_prompt(self):
                return Mock()

            def _create_agent(self):
                return Mock()

        agent = TestAgent(mock_llm, [mock_tool], mock_agent_config)

        # Create mock action objects that match LangChain's format
        mock_action1 = Mock()
        mock_action1.tool = "test_tool"
        mock_action1.tool_input = "input1"

        mock_action2 = Mock()
        mock_action2.tool = "test_tool"
        mock_action2.tool_input = "input2"

        result = agent._format_result(
            {
                "output": "Test output",
                "intermediate_steps": [
                    (mock_action1, "observation1"),
                    (mock_action2, "observation2"),
                ],
            }
        )

        assert isinstance(result, AgentResult)
        assert result.success
        assert result.message == "Test output"
        assert result.metadata["tools_used"]["test_tool"] == 2

    def test_add_remove_tool(self, mock_llm, mock_tool, mock_agent_config):
        class TestAgent(BaseAgent):
            def _create_prompt(self):
                return Mock()

            def _create_agent(self):
                return Mock()

        agent = TestAgent(mock_llm, [], mock_agent_config)
        assert len(agent.tools) == 0

        # Add tool
        agent.add_tool(mock_tool)
        assert len(agent.tools) == 1

        # Remove tool
        agent.remove_tool("test_tool")
        assert len(agent.tools) == 0


class TestMetabolicAgent:
    def test_init(self, mock_llm, mock_tool, mock_agent_config):
        agent = MetabolicAgent(mock_llm, [mock_tool], mock_agent_config)
        assert agent.llm == mock_llm
        assert len(agent.tools) == 1

    def test_create_prompt(self, mock_llm, mock_tool, mock_agent_config):
        agent = MetabolicAgent(mock_llm, [mock_tool], mock_agent_config)
        prompt = agent._create_prompt()
        assert "metabolic modeling" in prompt.template.lower()

    @patch("langchain.agents.AgentExecutor")
    def test_analyze_model(self, mock_executor, mock_llm, mock_tool, mock_agent_config):
        # Mock successful execution
        mock_executor_instance = Mock()
        mock_executor_instance.invoke.return_value = {
            "output": "Analysis complete",
            "intermediate_steps": [],
        }
        mock_executor.return_value = mock_executor_instance

        agent = MetabolicAgent(mock_llm, [mock_tool], mock_agent_config)
        agent._agent_executor = mock_executor_instance

        result = agent.analyze_model("test_model.xml")
        assert result.success
        assert result.message == "Analysis completed successfully"

        # Check that the executor was called with correct input
        mock_executor_instance.invoke.assert_called_once()
        call_args = mock_executor_instance.invoke.call_args[0][0]
        assert "test_model.xml" in call_args["input"]

    @patch("langchain.agents.AgentExecutor")
    def test_analyze_model_with_type(
        self, mock_executor, mock_llm, mock_tool, mock_agent_config
    ):
        mock_executor_instance = Mock()
        mock_executor_instance.invoke.return_value = {
            "output": "Pathway analysis complete",
            "intermediate_steps": [],
        }
        mock_executor.return_value = mock_executor_instance

        agent = MetabolicAgent(mock_llm, [mock_tool], mock_agent_config)
        agent._agent_executor = mock_executor_instance

        result = agent.analyze_model("test_model.xml", analysis_type="pathway")
        assert result.success
        assert result.message == "Analysis completed successfully"
        assert "pathway" in mock_executor_instance.invoke.call_args[0][0]["input"]

    @patch("langchain.agents.AgentExecutor")
    def test_suggest_improvements(
        self, mock_executor, mock_llm, mock_tool, mock_agent_config
    ):
        mock_executor_instance = Mock()
        mock_executor_instance.invoke.return_value = {
            "output": "Improvement suggestions ready",
            "intermediate_steps": [],
        }
        mock_executor.return_value = mock_executor_instance

        agent = MetabolicAgent(mock_llm, [mock_tool], mock_agent_config)
        agent._agent_executor = mock_executor_instance

        result = agent.suggest_improvements("test_model.xml")
        assert result.success
        assert result.message == "Analysis completed successfully"
        assert (
            "improvements"
            in mock_executor_instance.invoke.call_args[0][0]["input"].lower()
        )

    @patch("langchain.agents.AgentExecutor")
    def test_compare_models(
        self, mock_executor, mock_llm, mock_tool, mock_agent_config
    ):
        mock_executor_instance = Mock()
        mock_executor_instance.invoke.return_value = {
            "output": "Model comparison complete",
            "intermediate_steps": [],
        }
        mock_executor.return_value = mock_executor_instance

        agent = MetabolicAgent(mock_llm, [mock_tool], mock_agent_config)
        agent._agent_executor = mock_executor_instance

        model_paths = ["model1.xml", "model2.xml"]
        result = agent.compare_models(model_paths)
        assert result.success
        assert result.message == "Analysis completed successfully"

        # Verify that both model paths are in the input
        input_text = mock_executor_instance.invoke.call_args[0][0]["input"]
        assert all(path in input_text for path in model_paths)


class TestAgentFactory:
    def test_register_agent(self):
        class CustomAgent(BaseAgent):
            def _create_prompt(self):
                return Mock()

            def _create_agent(self):
                return Mock()

        AgentFactory.register_agent("custom", CustomAgent)
        assert "custom" in AgentFactory._agent_registry

    def test_create_agent(self, mock_llm, mock_tool, mock_agent_config):
        agent = AgentFactory.create_agent(
            "metabolic", mock_llm, [mock_tool], mock_agent_config
        )
        assert isinstance(agent, MetabolicAgent)

    def test_create_invalid_agent(self, mock_llm, mock_tool, mock_agent_config):
        with pytest.raises(ValueError):
            AgentFactory.create_agent(
                "invalid_agent_type", mock_llm, [mock_tool], mock_agent_config
            )

    def test_list_available_agents(self):
        agents = AgentFactory.list_available_agents()
        assert "metabolic" in agents
        assert isinstance(agents, list)


@pytest.mark.asyncio
class TestAsyncAgent:
    async def test_arun(self, mock_llm, mock_tool, mock_agent_config):
        class TestAgent(BaseAgent):
            def _create_prompt(self):
                return Mock()

            def _create_agent(self):
                mock_executor = Mock(spec=AgentExecutor)
                mock_executor.ainvoke.return_value = {
                    "output": "Async test complete",
                    "intermediate_steps": [],
                }
                return mock_executor

        agent = TestAgent(mock_llm, [mock_tool], mock_agent_config)
        result = await agent.arun({"input": "test"})
        assert result.success
        assert result.message == "Async test complete"

    async def test_arun_error(self, mock_llm, mock_tool, mock_agent_config):
        class TestAgent(BaseAgent):
            def _create_prompt(self):
                return Mock()

            def _create_agent(self):
                mock_executor = Mock(spec=AgentExecutor)
                mock_executor.ainvoke.side_effect = Exception("Async error")
                return mock_executor

        agent = TestAgent(mock_llm, [mock_tool], mock_agent_config)
        result = await agent.arun({"input": "test"})
        assert not result.success
        assert "Async error" in result.error
