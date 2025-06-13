"""
Media Analysis Workflow Templates
=================================

Pre-built workflow templates that combine AI media selection with comprehensive
metabolic analysis. These templates demonstrate how to use the AI media tools
in combination with traditional analysis tools for powerful insights.

Available Templates:
1. Optimal Media Discovery Workflow
2. Media Optimization for Production Workflow
3. Auxotrophy Analysis and Media Design Workflow
4. Cross-Model Media Comparison Workflow
5. Media Troubleshooting Workflow
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WorkflowStep(BaseModel):
    """Single step in a workflow template"""

    step_id: str = Field(description="Unique identifier for this step")
    tool_name: str = Field(description="Name of the tool to execute")
    description: str = Field(description="Human-readable description of this step")
    input_template: Dict[str, Any] = Field(description="Template for tool input")
    dependencies: List[str] = Field(
        default=[], description="Step IDs this step depends on"
    )
    conditional: Optional[str] = Field(
        default=None, description="Condition for step execution"
    )


class WorkflowTemplate(BaseModel):
    """Complete workflow template"""

    template_id: str = Field(description="Unique identifier for template")
    name: str = Field(description="Human-readable template name")
    description: str = Field(description="Description of what this workflow does")
    category: str = Field(
        description="Workflow category (media, analysis, optimization)"
    )
    steps: List[WorkflowStep] = Field(description="Ordered list of workflow steps")
    expected_inputs: Dict[str, str] = Field(
        description="Required inputs with descriptions"
    )
    expected_outputs: Dict[str, str] = Field(
        description="Expected outputs with descriptions"
    )
    estimated_duration: str = Field(description="Estimated time to complete")
    difficulty: str = Field(
        description="Difficulty level: beginner, intermediate, advanced"
    )


class MediaAnalysisWorkflowTemplates:
    """Collection of pre-built media analysis workflow templates"""

    @staticmethod
    def get_optimal_media_discovery_workflow() -> WorkflowTemplate:
        """
        Workflow for discovering optimal media for a model

        Steps:
        1. Analyze model characteristics
        2. AI selects candidate media types
        3. Test growth on each media
        4. Compare media performance
        5. Generate recommendations
        """
        return WorkflowTemplate(
            template_id="optimal_media_discovery",
            name="Optimal Media Discovery",
            description="Discover the best media type for your metabolic model using AI-powered selection and comprehensive testing",
            category="media_selection",
            steps=[
                WorkflowStep(
                    step_id="model_analysis",
                    tool_name="analyze_metabolic_model",
                    description="Analyze model structure and characteristics",
                    input_template={
                        "model_path": "{model_path}",
                        "detailed_analysis": True,
                    },
                ),
                WorkflowStep(
                    step_id="ai_media_selection",
                    tool_name="select_optimal_media",
                    description="AI selects optimal media candidates based on model analysis",
                    input_template={
                        "model_path": "{model_path}",
                        "target_growth": 0.1,
                        "exclude_media": [],
                    },
                    dependencies=["model_analysis"],
                ),
                WorkflowStep(
                    step_id="media_performance_comparison",
                    tool_name="compare_media_performance",
                    description="Compare model performance across different media types",
                    input_template={
                        "model_path": "{model_path}",
                        "media_list": "auto",  # Use AI recommendations
                        "include_visualizations": True,
                    },
                    dependencies=["ai_media_selection"],
                ),
                WorkflowStep(
                    step_id="growth_rate_analysis",
                    tool_name="run_metabolic_fba",
                    description="Detailed FBA analysis on the optimal media",
                    input_template={
                        "model_path": "{model_path}",
                        "media_name": "optimal_from_selection",
                        "detailed_output": True,
                    },
                    dependencies=["media_performance_comparison"],
                ),
                WorkflowStep(
                    step_id="media_compatibility_check",
                    tool_name="analyze_media_compatibility",
                    description="Analyze compatibility between model and selected media",
                    input_template={
                        "model_path": "{model_path}",
                        "media_names": ["optimal_from_selection"],
                        "detailed_analysis": True,
                    },
                    dependencies=["ai_media_selection"],
                ),
            ],
            expected_inputs={
                "model_path": "Path to the metabolic model file (SBML or JSON format)"
            },
            expected_outputs={
                "optimal_media": "Name and composition of the optimal media",
                "growth_rate": "Maximum achievable growth rate",
                "media_comparison": "Performance comparison across different media",
                "compatibility_analysis": "Detailed compatibility assessment",
                "recommendations": "AI-generated recommendations for media use",
            },
            estimated_duration="5-10 minutes",
            difficulty="beginner",
        )

    @staticmethod
    def get_media_optimization_workflow() -> WorkflowTemplate:
        """
        Workflow for optimizing media composition for specific production targets
        """
        return WorkflowTemplate(
            template_id="media_optimization_production",
            name="Media Optimization for Production",
            description="Optimize media composition to achieve specific growth or production targets using AI-driven optimization",
            category="media_optimization",
            steps=[
                WorkflowStep(
                    step_id="baseline_analysis",
                    tool_name="run_metabolic_fba",
                    description="Establish baseline growth on standard media",
                    input_template={
                        "model_path": "{model_path}",
                        "media_name": "GMM",
                        "detailed_output": True,
                    },
                ),
                WorkflowStep(
                    step_id="production_envelope",
                    tool_name="run_production_envelope",
                    description="Analyze growth vs production trade-offs",
                    input_template={
                        "model_path": "{model_path}",
                        "target_metabolite": "{target_metabolite}",
                        "media_name": "GMM",
                    },
                    dependencies=["baseline_analysis"],
                    conditional="if target_metabolite provided",
                ),
                WorkflowStep(
                    step_id="ai_media_optimization",
                    tool_name="optimize_media_composition",
                    description="AI-driven optimization of media composition",
                    input_template={
                        "model_path": "{model_path}",
                        "target_growth_rate": "{target_growth_rate}",
                        "base_media": "GMM",
                        "max_compounds": 50,
                        "strategy": "iterative",
                    },
                    dependencies=["baseline_analysis"],
                ),
                WorkflowStep(
                    step_id="optimized_performance_test",
                    tool_name="run_metabolic_fba",
                    description="Test performance on optimized media",
                    input_template={
                        "model_path": "{model_path}",
                        "media_name": "optimized_from_previous",
                        "detailed_output": True,
                    },
                    dependencies=["ai_media_optimization"],
                ),
                WorkflowStep(
                    step_id="flux_variability_analysis",
                    tool_name="run_flux_variability_analysis",
                    description="Analyze flux ranges on optimized media",
                    input_template={
                        "model_path": "{model_path}",
                        "media_name": "optimized_from_previous",
                        "fraction_of_optimum": 0.9,
                    },
                    dependencies=["optimized_performance_test"],
                ),
                WorkflowStep(
                    step_id="optimization_comparison",
                    tool_name="compare_media_performance",
                    description="Compare baseline vs optimized media performance",
                    input_template={
                        "model_path": "{model_path}",
                        "media_list": ["GMM", "optimized_from_optimization"],
                        "include_visualizations": True,
                    },
                    dependencies=[
                        "ai_media_optimization",
                        "optimized_performance_test",
                    ],
                ),
            ],
            expected_inputs={
                "model_path": "Path to the metabolic model file",
                "target_growth_rate": "Target growth rate to optimize for (h⁻¹)",
                "target_metabolite": "Optional: specific metabolite for production analysis",
            },
            expected_outputs={
                "optimized_media": "Optimized media composition",
                "optimization_results": "Detailed optimization analysis",
                "performance_comparison": "Before/after performance comparison",
                "flux_analysis": "Flux variability analysis on optimized media",
                "production_analysis": "Production capabilities (if target metabolite specified)",
            },
            estimated_duration="10-20 minutes",
            difficulty="intermediate",
        )

    @staticmethod
    def get_auxotrophy_analysis_workflow() -> WorkflowTemplate:
        """
        Workflow for comprehensive auxotrophy analysis and media design
        """
        return WorkflowTemplate(
            template_id="auxotrophy_analysis_design",
            name="Auxotrophy Analysis and Media Design",
            description="Comprehensive analysis of model auxotrophies with AI-powered media design recommendations",
            category="auxotrophy_analysis",
            steps=[
                WorkflowStep(
                    step_id="model_gap_analysis",
                    tool_name="analyze_metabolic_model",
                    description="Analyze model for potential metabolic gaps",
                    input_template={
                        "model_path": "{model_path}",
                        "check_gaps": True,
                        "detailed_analysis": True,
                    },
                ),
                WorkflowStep(
                    step_id="ai_auxotrophy_prediction",
                    tool_name="predict_auxotrophies",
                    description="AI-powered prediction of potential auxotrophies",
                    input_template={
                        "model_path": "{model_path}",
                        "test_media": "AuxoMedia",
                        "compound_categories": [
                            "amino_acids",
                            "vitamins",
                            "nucleotides",
                        ],
                        "growth_threshold": 0.01,
                    },
                    dependencies=["model_gap_analysis"],
                ),
                WorkflowStep(
                    step_id="traditional_auxotrophy_test",
                    tool_name="identify_auxotrophies",
                    description="Traditional auxotrophy testing by nutrient removal",
                    input_template={
                        "model_path": "{model_path}",
                        "test_media": "AuxoMedia",
                        "test_individual_compounds": True,
                    },
                    dependencies=["ai_auxotrophy_prediction"],
                ),
                WorkflowStep(
                    step_id="minimal_media_determination",
                    tool_name="find_minimal_media",
                    description="Determine minimal media requirements",
                    input_template={
                        "model_path": "{model_path}",
                        "target_growth": 0.1,
                        "include_essential_only": True,
                    },
                    dependencies=["ai_auxotrophy_prediction"],
                ),
                WorkflowStep(
                    step_id="custom_media_design",
                    tool_name="manipulate_media_composition",
                    description="Design custom media based on auxotrophy findings",
                    input_template={
                        "base_media": "GMM",
                        "ai_command": "add supplements for detected auxotrophies",
                        "model_path": "{model_path}",
                        "test_growth": True,
                    },
                    dependencies=[
                        "ai_auxotrophy_prediction",
                        "minimal_media_determination",
                    ],
                ),
                WorkflowStep(
                    step_id="media_validation",
                    tool_name="compare_media_performance",
                    description="Validate designed media against standard media types",
                    input_template={
                        "model_path": "{model_path}",
                        "media_list": ["GMM", "AuxoMedia", "custom_designed"],
                        "include_visualizations": True,
                    },
                    dependencies=["custom_media_design"],
                ),
            ],
            expected_inputs={"model_path": "Path to the metabolic model file"},
            expected_outputs={
                "auxotrophy_predictions": "AI-predicted auxotrophies with confidence scores",
                "traditional_auxotrophy_results": "Results from traditional auxotrophy testing",
                "minimal_media": "Minimal media composition for growth",
                "custom_media": "Custom-designed media addressing all auxotrophies",
                "media_comparison": "Performance comparison across different media",
                "design_recommendations": "Recommendations for media formulation",
            },
            estimated_duration="15-25 minutes",
            difficulty="advanced",
        )

    @staticmethod
    def get_cross_model_comparison_workflow() -> WorkflowTemplate:
        """
        Workflow for comparing media performance across multiple models
        """
        return WorkflowTemplate(
            template_id="cross_model_media_comparison",
            name="Cross-Model Media Comparison",
            description="Compare how different metabolic models perform on the same media types for comparative analysis",
            category="comparative_analysis",
            steps=[
                WorkflowStep(
                    step_id="model1_media_selection",
                    tool_name="select_optimal_media",
                    description="Select optimal media for first model",
                    input_template={
                        "model_path": "{model1_path}",
                        "target_growth": 0.1,
                        "exclude_media": [],
                    },
                ),
                WorkflowStep(
                    step_id="model2_media_selection",
                    tool_name="select_optimal_media",
                    description="Select optimal media for second model",
                    input_template={
                        "model_path": "{model2_path}",
                        "target_growth": 0.1,
                        "exclude_media": [],
                    },
                ),
                WorkflowStep(
                    step_id="model1_media_comparison",
                    tool_name="compare_media_performance",
                    description="Test model 1 performance across different media",
                    input_template={
                        "model_path": "{model1_path}",
                        "media_list": ["GMM", "AuxoMedia", "PyruvateMinimalMedia"],
                        "include_visualizations": True,
                    },
                    dependencies=["model1_media_selection"],
                ),
                WorkflowStep(
                    step_id="model2_media_comparison",
                    tool_name="compare_media_performance",
                    description="Test model 2 performance across different media",
                    input_template={
                        "model_path": "{model2_path}",
                        "media_list": ["GMM", "AuxoMedia", "PyruvateMinimalMedia"],
                        "include_visualizations": True,
                    },
                    dependencies=["model2_media_selection"],
                ),
                WorkflowStep(
                    step_id="compatibility_analysis",
                    tool_name="analyze_media_compatibility",
                    description="Analyze media compatibility for both models",
                    input_template={
                        "model_paths": ["{model1_path}", "{model2_path}"],
                        "media_names": ["GMM", "AuxoMedia"],
                        "detailed_analysis": True,
                    },
                    dependencies=["model1_media_comparison", "model2_media_comparison"],
                ),
                WorkflowStep(
                    step_id="pathway_comparison",
                    tool_name="analyze_pathway",
                    description="Compare pathway utilization between models",
                    input_template={
                        "model_paths": ["{model1_path}", "{model2_path}"],
                        "pathway_analysis": "comparative",
                        "media_name": "optimal_shared",
                    },
                    dependencies=["compatibility_analysis"],
                ),
            ],
            expected_inputs={
                "model1_path": "Path to first metabolic model",
                "model2_path": "Path to second metabolic model",
            },
            expected_outputs={
                "model_specific_optimal_media": "Optimal media for each model",
                "cross_model_performance": "Performance matrix across models and media",
                "compatibility_analysis": "Media compatibility comparison",
                "pathway_differences": "Pathway utilization differences",
                "shared_media_recommendations": "Media that work well for both models",
            },
            estimated_duration="20-30 minutes",
            difficulty="advanced",
        )

    @staticmethod
    def get_media_troubleshooting_workflow() -> WorkflowTemplate:
        """
        Workflow for troubleshooting media-related growth issues
        """
        return WorkflowTemplate(
            template_id="media_troubleshooting",
            name="Media Troubleshooting Workflow",
            description="Systematic troubleshooting of media-related growth issues using AI-powered diagnostics",
            category="troubleshooting",
            steps=[
                WorkflowStep(
                    step_id="initial_growth_test",
                    tool_name="run_metabolic_fba",
                    description="Test growth on current media",
                    input_template={
                        "model_path": "{model_path}",
                        "media_name": "{problematic_media}",
                        "detailed_output": True,
                    },
                ),
                WorkflowStep(
                    step_id="media_compatibility_diagnosis",
                    tool_name="analyze_media_compatibility",
                    description="Diagnose media-model compatibility issues",
                    input_template={
                        "model_path": "{model_path}",
                        "media_names": ["{problematic_media}"],
                        "detailed_analysis": True,
                    },
                    dependencies=["initial_growth_test"],
                ),
                WorkflowStep(
                    step_id="missing_nutrients_check",
                    tool_name="find_missing_media",
                    description="Check for missing essential nutrients",
                    input_template={
                        "model_path": "{model_path}",
                        "current_media": "{problematic_media}",
                        "target_growth": 0.01,
                    },
                    dependencies=["media_compatibility_diagnosis"],
                ),
                WorkflowStep(
                    step_id="ai_media_fix",
                    tool_name="manipulate_media_composition",
                    description="AI-powered media modification to fix issues",
                    input_template={
                        "base_media": "{problematic_media}",
                        "ai_command": "fix growth issues and add missing nutrients",
                        "model_path": "{model_path}",
                        "test_growth": True,
                    },
                    dependencies=["missing_nutrients_check"],
                ),
                WorkflowStep(
                    step_id="alternative_media_suggestion",
                    tool_name="select_optimal_media",
                    description="Suggest alternative media if fixes don't work",
                    input_template={
                        "model_path": "{model_path}",
                        "target_growth": 0.1,
                        "exclude_media": ["{problematic_media}"],
                    },
                    dependencies=["ai_media_fix"],
                ),
                WorkflowStep(
                    step_id="validation_test",
                    tool_name="compare_media_performance",
                    description="Validate fixes by comparing original vs fixed media",
                    input_template={
                        "model_path": "{model_path}",
                        "media_list": [
                            "{problematic_media}",
                            "fixed_media",
                            "alternative_media",
                        ],
                        "include_visualizations": True,
                    },
                    dependencies=["ai_media_fix", "alternative_media_suggestion"],
                ),
            ],
            expected_inputs={
                "model_path": "Path to the metabolic model",
                "problematic_media": "Name of the media causing growth issues",
            },
            expected_outputs={
                "diagnosis": "Detailed diagnosis of media-related issues",
                "fixed_media": "Modified media composition that addresses issues",
                "alternative_media": "Alternative media recommendations",
                "performance_comparison": "Before/after performance analysis",
                "troubleshooting_report": "Complete troubleshooting analysis",
            },
            estimated_duration="10-15 minutes",
            difficulty="intermediate",
        )

    @classmethod
    def get_all_templates(cls) -> List[WorkflowTemplate]:
        """Get all available workflow templates"""
        return [
            cls.get_optimal_media_discovery_workflow(),
            cls.get_media_optimization_workflow(),
            cls.get_auxotrophy_analysis_workflow(),
            cls.get_cross_model_comparison_workflow(),
            cls.get_media_troubleshooting_workflow(),
        ]

    @classmethod
    def get_template_by_id(cls, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a specific template by ID"""
        templates = cls.get_all_templates()
        for template in templates:
            if template.template_id == template_id:
                return template
        return None

    @classmethod
    def get_templates_by_category(cls, category: str) -> List[WorkflowTemplate]:
        """Get templates by category"""
        templates = cls.get_all_templates()
        return [t for t in templates if t.category == category]

    @classmethod
    def get_beginner_templates(cls) -> List[WorkflowTemplate]:
        """Get templates suitable for beginners"""
        templates = cls.get_all_templates()
        return [t for t in templates if t.difficulty == "beginner"]


class WorkflowExecutor:
    """Executes workflow templates with actual tools"""

    def __init__(self, tools_dict: Dict[str, BaseTool]):
        """Initialize with available tools"""
        self.tools_dict = tools_dict
        self.execution_history = []

    def execute_workflow(
        self, template: WorkflowTemplate, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a workflow template with provided inputs"""

        execution_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        results = {
            "execution_id": execution_id,
            "template_id": template.template_id,
            "template_name": template.name,
            "start_time": datetime.now().isoformat(),
            "inputs": inputs,
            "step_results": {},
            "status": "running",
            "errors": [],
        }

        # Validate required inputs
        missing_inputs = []
        for required_input in template.expected_inputs.keys():
            if required_input not in inputs:
                missing_inputs.append(required_input)

        if missing_inputs:
            results["status"] = "failed"
            results["errors"].append(f"Missing required inputs: {missing_inputs}")
            return results

        # Execute steps in order
        step_outputs = {}

        for step in template.steps:
            try:
                # Check if step should be executed (dependencies and conditionals)
                if not self._should_execute_step(step, step_outputs, inputs):
                    continue

                # Prepare step input by substituting templates
                step_input = self._prepare_step_input(step, inputs, step_outputs)

                # Get tool and execute
                tool = self.tools_dict.get(step.tool_name)
                if not tool:
                    error_msg = f"Tool '{step.tool_name}' not available"
                    results["errors"].append(error_msg)
                    results["step_results"][step.step_id] = {"error": error_msg}
                    continue

                # Execute tool
                step_result = tool._run_tool(step_input)

                # Store result
                results["step_results"][step.step_id] = {
                    "step_name": step.description,
                    "tool_used": step.tool_name,
                    "success": step_result.success,
                    "message": step_result.message,
                    "data": step_result.data if step_result.success else None,
                    "error": step_result.error if not step_result.success else None,
                }

                # Store outputs for next steps
                if step_result.success and step_result.data:
                    step_outputs[step.step_id] = step_result.data

            except Exception as e:
                error_msg = f"Step '{step.step_id}' failed: {str(e)}"
                results["errors"].append(error_msg)
                results["step_results"][step.step_id] = {"error": error_msg}

        results["end_time"] = datetime.now().isoformat()
        results["status"] = (
            "completed" if not results["errors"] else "completed_with_errors"
        )

        # Store execution history
        self.execution_history.append(results)

        return results

    def _should_execute_step(
        self, step: WorkflowStep, step_outputs: Dict, inputs: Dict
    ) -> bool:
        """Check if a step should be executed based on dependencies and conditionals"""

        # Check dependencies
        for dep in step.dependencies:
            if dep not in step_outputs:
                return False

        # Check conditional (simplified - could be enhanced)
        if step.conditional:
            if "if target_metabolite provided" in step.conditional:
                return "target_metabolite" in inputs and inputs["target_metabolite"]

        return True

    def _prepare_step_input(
        self, step: WorkflowStep, inputs: Dict, step_outputs: Dict
    ) -> Dict[str, Any]:
        """Prepare input for a step by substituting template variables"""

        step_input = {}

        for key, value in step.input_template.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                # Template variable
                var_name = value[1:-1]  # Remove braces
                if var_name in inputs:
                    step_input[key] = inputs[var_name]
                elif var_name in step_outputs:
                    step_input[key] = step_outputs[var_name]
                else:
                    # Handle special cases
                    if var_name == "optimal_from_selection":
                        # Get from previous AI media selection
                        for step_id, output in step_outputs.items():
                            if "best_media" in output:
                                step_input[key] = output["best_media"]
                                break
                    elif var_name == "optimized_from_previous":
                        # Get from previous optimization
                        for step_id, output in step_outputs.items():
                            if "optimized_media" in output:
                                step_input[key] = output["optimized_media"]
                                break
            else:
                step_input[key] = value

        return step_input


# Export main classes
__all__ = [
    "WorkflowStep",
    "WorkflowTemplate",
    "MediaAnalysisWorkflowTemplates",
    "WorkflowExecutor",
]
