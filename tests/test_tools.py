import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import cobra
from src.tools.base import BaseTool, ToolResult, ToolRegistry
from src.tools.cobra.fba import FBATool
from src.tools.cobra.analysis import ModelAnalysisTool, PathwayAnalysisTool
from src.tools.cobra.utils import ModelUtils

@pytest.fixture
def test_model_path(tmp_path):
    # Create a simple test model
    model = cobra.Model('test_model')
    reaction = cobra.Reaction('R1')
    reaction.name = 'Test Reaction'
    reaction.lower_bound = -1000
    reaction.upper_bound = 1000
    model.add_reactions([reaction])
    
    # Save the model
    model_path = tmp_path / "test_model.xml"
    cobra.io.write_sbml_model(model, str(model_path))
    return str(model_path)

@pytest.fixture
def mock_tool_config():
    return {
        "name": "test_tool",
        "description": "Test tool description",
        "additional_config": {}
    }

class TestBaseTool:
    def test_init(self, mock_tool_config):
        class TestTool(BaseTool):
            def _run(self, input_data):
                return ToolResult(success=True, message="Test")
        
        tool = TestTool(mock_tool_config)
        assert tool.config.name == "test_tool"
        assert tool.config.description == "Test tool description"

    def test_run_success(self, mock_tool_config):
        class TestTool(BaseTool):
            def _run(self, input_data):
                return ToolResult(success=True, message="Success")
        
        tool = TestTool(mock_tool_config)
        result = tool.run("test input")
        assert result.success
        assert result.message == "Success"

class TestFBATool:
    def test_init(self):
        config = {
            "name": "fba_tool",
            "description": "FBA tool",
            "fba_config": {
                "default_objective": "BIOMASS",
                "solver": "glpk"
            }
        }
        tool = FBATool(config)
        assert tool.config.name == "fba_tool"
        assert tool.fba_config.default_objective == "BIOMASS"

    def test_run_fba(self, test_model_path):
        config = {
            "name": "fba_tool",
            "description": "FBA tool",
            "fba_config": {
                "default_objective": "R1",
                "solver": "glpk"
            }
        }
        tool = FBATool(config)
        result = tool._run(test_model_path)
        
        assert result.success
        assert "objective_value" in result.data
        assert "significant_fluxes" in result.data

    def test_invalid_model_path(self):
        tool = FBATool({"name": "fba_tool", "description": "FBA tool"})
        result = tool._run("nonexistent_model.xml")
        assert not result.success
        assert "Error" in result.message

class TestModelAnalysisTool:
    def test_init(self):
        config = {
            "name": "analysis_tool",
            "description": "Analysis tool",
            "analysis_config": {
                "flux_threshold": 1e-6
            }
        }
        tool = ModelAnalysisTool(config)
        assert tool.config.name == "analysis_tool"
        assert tool.analysis_config.flux_threshold == 1e-6

    def test_model_analysis(self, test_model_path):
        tool = ModelAnalysisTool({
            "name": "analysis_tool",
            "description": "Analysis tool"
        })
        result = tool._run(test_model_path)
        
        assert result.success
        assert "basic_statistics" in result.data
        assert "pathway_coverage" in result.data
        assert "potential_gaps" in result.data

class TestPathwayAnalysisTool:
    def test_pathway_analysis(self, test_model_path):
        tool = PathwayAnalysisTool({
            "name": "pathway_tool",
            "description": "Pathway tool"
        })
        result = tool._run({
            "model_path": test_model_path,
            "pathway": "Test Pathway"
        })
        
        assert result.success
        assert "reaction_count" in result.data
        assert "reactions" in result.data

class TestModelUtils:
    def test_load_model(self, test_model_path):
        utils = ModelUtils()
        model = utils.load_model(test_model_path)
        assert isinstance(model, cobra.Model)
        assert len(model.reactions) > 0

    def test_verify_model(self, test_model_path):
        utils = ModelUtils()
        model = utils.load_model(test_model_path)
        verification = utils.verify_model(model)
        
        assert "is_valid" in verification
        assert "statistics" in verification
        assert "issues" in verification

    def test_find_deadend_metabolites(self, test_model_path):
        utils = ModelUtils()
        model = utils.load_model(test_model_path)
        deadends = utils.find_deadend_metabolites(model)
        
        assert "no_production" in deadends
        assert "no_consumption" in deadends
        assert "disconnected" in deadends

class TestToolRegistry:
    def test_register_tool(self):
        @ToolRegistry.register
        class CustomTool(BaseTool):
            name = "custom_tool"
            def _run(self, input_data):
                return ToolResult(success=True, message="Test")
        
        assert "custom_tool" in ToolRegistry._tools
        
    def test_get_tool(self):
        tool_class = ToolRegistry.get_tool("run_metabolic_fba")
        assert tool_class is not None
        assert issubclass(tool_class, BaseTool)
    
    def test_create_tool(self):
        tool = ToolRegistry.create_tool(
            "run_metabolic_fba",
            {"name": "fba", "description": "FBA tool"}
        )
        assert isinstance(tool, FBATool)