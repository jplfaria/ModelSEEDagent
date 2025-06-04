"""
Template Library for Advanced Workflow Automation

Provides a comprehensive collection of pre-built workflow templates
for common metabolic modeling tasks and analysis patterns.
"""

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .workflow_definition import (
    ExecutionMode,
    ResourceRequirement,
    StepType,
    WorkflowDefinition,
    WorkflowParameter,
    WorkflowStep,
    WorkflowTemplate,
)

console = Console()


class TemplateLibrary:
    """Comprehensive library of workflow templates"""

    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.categories: Dict[str, List[str]] = {}
        self._initialize_built_in_templates()

    def _initialize_built_in_templates(self) -> None:
        """Initialize built-in workflow templates"""

        # Basic Model Analysis Template
        self._create_basic_analysis_template()

        # Comprehensive Model Validation Template
        self._create_validation_template()

        # Growth Analysis Template
        self._create_growth_analysis_template()

        # Flux Balance Analysis Template
        self._create_fba_template()

        # Multi-Condition Comparison Template
        self._create_comparison_template()

        # Pathway Analysis Template
        self._create_pathway_template()

        # Model Optimization Template
        self._create_optimization_template()

        # Batch Processing Template
        self._create_batch_template()

    def _create_basic_analysis_template(self) -> None:
        """Create basic model analysis template"""
        template_data = {
            "id": "basic_analysis",
            "name": "Basic Model Analysis",
            "description": "Comprehensive basic analysis of metabolic models",
            "version": "1.0.0",
            "category": "analysis",
            "steps": [
                {
                    "id": "validate_model",
                    "name": "Model Validation",
                    "type": "model_validation",
                    "description": "Validate SBML model format and structure",
                    "parameters": {"model_path": "${model_file}"},
                    "estimated_duration": 10,
                    "resource_requirements": "low",
                },
                {
                    "id": "analyze_structure",
                    "name": "Structural Analysis",
                    "type": "tool_execution",
                    "description": "Analyze model structure and components",
                    "tool_name": "analyze_metabolic_model",
                    "parameters": {
                        "model_file": "${model_file}",
                        "include_statistics": True,
                    },
                    "dependencies": ["validate_model"],
                    "estimated_duration": 30,
                    "resource_requirements": "medium",
                },
                {
                    "id": "check_growth",
                    "name": "Growth Check",
                    "type": "tool_execution",
                    "description": "Test basic growth capabilities",
                    "tool_name": "run_metabolic_fba",
                    "parameters": {
                        "model_file": "${model_file}",
                        "objective": "biomass",
                        "media": "${media_conditions}",
                    },
                    "dependencies": ["analyze_structure"],
                    "estimated_duration": 20,
                    "resource_requirements": "medium",
                },
                {
                    "id": "generate_report",
                    "name": "Generate Report",
                    "type": "data_transformation",
                    "description": "Compile analysis results into report",
                    "parameters": {
                        "output_format": "${output_format}",
                        "include_visualizations": True,
                    },
                    "dependencies": ["check_growth"],
                    "estimated_duration": 15,
                    "resource_requirements": "low",
                },
            ],
            "parameters": [
                {
                    "name": "model_file",
                    "type": "file",
                    "description": "Path to SBML model file",
                    "required": True,
                },
                {
                    "name": "media_conditions",
                    "type": "string",
                    "description": "Growth media conditions",
                    "default_value": "minimal_glucose",
                    "choices": ["minimal_glucose", "rich_media", "custom"],
                },
                {
                    "name": "output_format",
                    "type": "string",
                    "description": "Report output format",
                    "default_value": "html",
                    "choices": ["html", "json", "pdf"],
                },
            ],
        }

        parameter_schema = [
            WorkflowParameter(
                name="model_file",
                type="file",
                description="Path to SBML model file",
                required=True,
            ),
            WorkflowParameter(
                name="media_conditions",
                type="string",
                description="Growth media conditions",
                default_value="minimal_glucose",
                choices=["minimal_glucose", "rich_media", "custom"],
            ),
            WorkflowParameter(
                name="output_format",
                type="string",
                description="Report output format",
                default_value="html",
                choices=["html", "json", "pdf"],
            ),
        ]

        template = WorkflowTemplate(
            id="basic_analysis",
            name="Basic Model Analysis",
            description="Comprehensive basic analysis of metabolic models",
            category="analysis",
            tags=["basic", "validation", "structure", "growth"],
            template_data=template_data,
            parameter_schema=parameter_schema,
        )

        self._register_template(template)

    def _create_validation_template(self) -> None:
        """Create comprehensive model validation template"""
        template_data = {
            "id": "comprehensive_validation",
            "name": "Comprehensive Model Validation",
            "description": "Thorough validation of metabolic model quality",
            "version": "1.0.0",
            "category": "validation",
            "steps": [
                {
                    "id": "format_check",
                    "name": "Format Validation",
                    "type": "model_validation",
                    "description": "Check SBML format compliance",
                    "parameters": {"model_path": "${model_file}", "strict_mode": True},
                    "estimated_duration": 10,
                    "resource_requirements": "low",
                },
                {
                    "id": "mass_balance",
                    "name": "Mass Balance Check",
                    "type": "tool_execution",
                    "description": "Verify mass balance of reactions",
                    "tool_name": "check_mass_balance",
                    "parameters": {"model_file": "${model_file}", "tolerance": 1e-6},
                    "dependencies": ["format_check"],
                    "estimated_duration": 30,
                    "resource_requirements": "medium",
                },
                {
                    "id": "charge_balance",
                    "name": "Charge Balance Check",
                    "type": "tool_execution",
                    "description": "Verify charge balance of reactions",
                    "tool_name": "check_charge_balance",
                    "parameters": {"model_file": "${model_file}", "tolerance": 1e-6},
                    "dependencies": ["format_check"],
                    "estimated_duration": 25,
                    "execution_mode": "parallel",
                },
                {
                    "id": "connectivity_check",
                    "name": "Network Connectivity",
                    "type": "tool_execution",
                    "description": "Analyze network connectivity",
                    "tool_name": "analyze_connectivity",
                    "parameters": {
                        "model_file": "${model_file}",
                        "find_dead_ends": True,
                    },
                    "dependencies": ["mass_balance", "charge_balance"],
                    "estimated_duration": 20,
                    "resource_requirements": "medium",
                },
                {
                    "id": "validation_report",
                    "name": "Validation Report",
                    "type": "data_transformation",
                    "description": "Generate comprehensive validation report",
                    "parameters": {
                        "include_recommendations": True,
                        "output_format": "${output_format}",
                    },
                    "dependencies": ["connectivity_check"],
                    "estimated_duration": 15,
                },
            ],
            "parameters": [
                {
                    "name": "model_file",
                    "type": "file",
                    "description": "Path to SBML model file",
                    "required": True,
                },
                {
                    "name": "output_format",
                    "type": "string",
                    "description": "Report output format",
                    "default_value": "html",
                    "choices": ["html", "json", "pdf"],
                },
            ],
        }

        parameter_schema = [
            WorkflowParameter(
                name="model_file",
                type="file",
                description="Path to SBML model file",
                required=True,
            ),
            WorkflowParameter(
                name="output_format",
                type="string",
                description="Report output format",
                default_value="html",
                choices=["html", "json", "pdf"],
            ),
        ]

        template = WorkflowTemplate(
            id="comprehensive_validation",
            name="Comprehensive Model Validation",
            description="Thorough validation of metabolic model quality",
            category="validation",
            tags=["validation", "quality", "mass_balance", "connectivity"],
            template_data=template_data,
            parameter_schema=parameter_schema,
        )

        self._register_template(template)

    def _create_growth_analysis_template(self) -> None:
        """Create growth analysis template"""
        template_data = {
            "id": "growth_analysis",
            "name": "Growth Analysis",
            "description": "Comprehensive growth phenotype analysis",
            "version": "1.0.0",
            "category": "analysis",
            "steps": [
                {
                    "id": "baseline_growth",
                    "name": "Baseline Growth",
                    "type": "tool_execution",
                    "description": "Calculate baseline growth rate",
                    "tool_name": "run_metabolic_fba",
                    "parameters": {
                        "model_file": "${model_file}",
                        "objective": "biomass",
                        "media": "${base_media}",
                    },
                    "estimated_duration": 20,
                    "resource_requirements": "medium",
                },
                {
                    "id": "carbon_source_screen",
                    "name": "Carbon Source Screen",
                    "type": "loop",
                    "description": "Test growth on different carbon sources",
                    "parameters": {
                        "carbon_sources": "${carbon_sources}",
                        "tool_name": "run_metabolic_fba",
                    },
                    "dependencies": ["baseline_growth"],
                    "estimated_duration": 60,
                    "resource_requirements": "high",
                },
                {
                    "id": "gene_essentiality",
                    "name": "Gene Essentiality",
                    "type": "tool_execution",
                    "description": "Identify essential genes",
                    "tool_name": "gene_essentiality_analysis",
                    "parameters": {
                        "model_file": "${model_file}",
                        "media": "${base_media}",
                        "threshold": 0.1,
                    },
                    "dependencies": ["baseline_growth"],
                    "estimated_duration": 120,
                    "resource_requirements": "high",
                    "execution_mode": "parallel",
                },
                {
                    "id": "growth_summary",
                    "name": "Growth Summary",
                    "type": "data_transformation",
                    "description": "Summarize growth analysis results",
                    "parameters": {
                        "create_visualizations": True,
                        "output_format": "${output_format}",
                    },
                    "dependencies": ["carbon_source_screen", "gene_essentiality"],
                    "estimated_duration": 25,
                },
            ],
        }

        parameter_schema = [
            WorkflowParameter(
                name="model_file",
                type="file",
                description="Path to SBML model file",
                required=True,
            ),
            WorkflowParameter(
                name="base_media",
                type="string",
                description="Base growth media",
                default_value="minimal_glucose",
                choices=["minimal_glucose", "minimal_acetate", "rich_media"],
            ),
            WorkflowParameter(
                name="carbon_sources",
                type="list",
                description="List of carbon sources to test",
                default_value=["glucose", "acetate", "glycerol", "lactate"],
            ),
            WorkflowParameter(
                name="output_format",
                type="string",
                description="Output format",
                default_value="html",
                choices=["html", "json", "pdf"],
            ),
        ]

        template = WorkflowTemplate(
            id="growth_analysis",
            name="Growth Analysis",
            description="Comprehensive growth phenotype analysis",
            category="analysis",
            tags=["growth", "phenotype", "carbon_sources", "essentiality"],
            template_data=template_data,
            parameter_schema=parameter_schema,
        )

        self._register_template(template)

    def _create_fba_template(self) -> None:
        """Create flux balance analysis template"""
        template_data = {
            "id": "flux_balance_analysis",
            "name": "Flux Balance Analysis",
            "description": "Comprehensive flux balance analysis workflow",
            "version": "1.0.0",
            "category": "flux_analysis",
        }

        parameter_schema = [
            WorkflowParameter(
                name="model_file",
                type="file",
                description="Path to SBML model file",
                required=True,
            ),
            WorkflowParameter(
                name="objective_function",
                type="string",
                description="Objective function to optimize",
                default_value="biomass",
            ),
        ]

        template = WorkflowTemplate(
            id="flux_balance_analysis",
            name="Flux Balance Analysis",
            description="Comprehensive flux balance analysis workflow",
            category="flux_analysis",
            tags=["fba", "flux", "optimization"],
            template_data=template_data,
            parameter_schema=parameter_schema,
        )

        self._register_template(template)

    def _create_comparison_template(self) -> None:
        """Create multi-condition comparison template"""
        pass  # Placeholder for brevity

    def _create_pathway_template(self) -> None:
        """Create pathway analysis template"""
        pass  # Placeholder for brevity

    def _create_optimization_template(self) -> None:
        """Create model optimization template"""
        pass  # Placeholder for brevity

    def _create_batch_template(self) -> None:
        """Create batch processing template"""
        pass  # Placeholder for brevity

    def _register_template(self, template: WorkflowTemplate) -> None:
        """Register a template in the library"""
        self.templates[template.id] = template

        # Add to category
        if template.category not in self.categories:
            self.categories[template.category] = []
        self.categories[template.category].append(template.id)

        console.print(f"✅ Registered template: [cyan]{template.name}[/cyan]")

    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a template by ID"""
        return self.templates.get(template_id)

    def list_templates(self, category: Optional[str] = None) -> List[WorkflowTemplate]:
        """List available templates"""
        if category:
            template_ids = self.categories.get(category, [])
            return [self.templates[tid] for tid in template_ids]
        else:
            return list(self.templates.values())

    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self.categories.keys())

    def search_templates(self, query: str) -> List[WorkflowTemplate]:
        """Search templates by name, description, or tags"""
        query_lower = query.lower()
        matching_templates = []

        for template in self.templates.values():
            if (
                query_lower in template.name.lower()
                or query_lower in template.description.lower()
                or any(query_lower in tag.lower() for tag in template.tags)
            ):
                matching_templates.append(template)

        return matching_templates

    def create_workflow_from_template(
        self, template_id: str, parameters: Dict[str, Any]
    ) -> WorkflowDefinition:
        """Create a workflow instance from a template"""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")

        return template.instantiate(parameters)

    def display_template_library(self) -> None:
        """Display the template library in a beautiful format"""
        console.print(
            Panel(
                "[bold blue]📚 Workflow Template Library[/bold blue]",
                border_style="blue",
            )
        )

        for category in sorted(self.categories.keys()):
            console.print(f"\n[bold yellow]📁 {category.title()}[/bold yellow]")

            category_table = Table(box=box.SIMPLE)
            category_table.add_column("Template", style="cyan")
            category_table.add_column("Description", style="white")
            category_table.add_column("Tags", style="green")

            for template_id in self.categories[category]:
                template = self.templates[template_id]
                tags_str = ", ".join(template.tags[:3])
                if len(template.tags) > 3:
                    tags_str += "..."

                category_table.add_row(
                    template.name,
                    (
                        template.description[:50] + "..."
                        if len(template.description) > 50
                        else template.description
                    ),
                    tags_str,
                )

            console.print(category_table)

    def display_template_details(self, template_id: str) -> None:
        """Display detailed information about a template"""
        template = self.get_template(template_id)
        if not template:
            console.print(f"[red]Template '{template_id}' not found[/red]")
            return

        # Template overview
        overview_table = Table(show_header=False, box=box.SIMPLE)
        overview_table.add_column("Aspect", style="bold cyan")
        overview_table.add_column("Value", style="bold white")

        overview_table.add_row("ID", template.id)
        overview_table.add_row("Name", template.name)
        overview_table.add_row("Category", template.category)
        overview_table.add_row("Description", template.description)
        overview_table.add_row("Tags", ", ".join(template.tags))

        console.print(
            Panel(
                overview_table,
                title=f"[bold blue]📋 Template: {template.name}[/bold blue]",
                border_style="blue",
            )
        )

        # Parameters
        if template.parameter_schema:
            console.print("\n[bold yellow]⚙️ Parameters:[/bold yellow]")

            params_table = Table(box=box.ROUNDED)
            params_table.add_column("Name", style="cyan")
            params_table.add_column("Type", style="green")
            params_table.add_column("Required", style="yellow")
            params_table.add_column("Default", style="blue")
            params_table.add_column("Description", style="white")

            for param in template.parameter_schema:
                required_str = "Yes" if param.required else "No"
                default_str = (
                    str(param.default_value)
                    if param.default_value is not None
                    else "None"
                )

                params_table.add_row(
                    param.name, param.type, required_str, default_str, param.description
                )

            console.print(params_table)

    def export_template(self, template_id: str, file_path: str) -> None:
        """Export a template to a file"""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")

        export_data = {
            "id": template.id,
            "name": template.name,
            "description": template.description,
            "category": template.category,
            "tags": template.tags,
            "template_data": template.template_data,
            "parameter_schema": [
                {
                    "name": param.name,
                    "type": param.type,
                    "description": param.description,
                    "default_value": param.default_value,
                    "required": param.required,
                    "validation_rule": param.validation_rule,
                    "choices": param.choices,
                }
                for param in template.parameter_schema
            ],
        }

        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        console.print(f"✅ Template exported to: [cyan]{file_path}[/cyan]")

    def import_template(self, file_path: str) -> None:
        """Import a template from a file"""
        with open(file_path, "r") as f:
            data = json.load(f)

        # Recreate parameter schema
        parameter_schema = []
        for param_data in data.get("parameter_schema", []):
            param = WorkflowParameter(
                name=param_data["name"],
                type=param_data["type"],
                description=param_data["description"],
                default_value=param_data.get("default_value"),
                required=param_data.get("required", True),
                validation_rule=param_data.get("validation_rule"),
                choices=param_data.get("choices"),
            )
            parameter_schema.append(param)

        template = WorkflowTemplate(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            category=data["category"],
            tags=data.get("tags", []),
            template_data=data["template_data"],
            parameter_schema=parameter_schema,
        )

        self._register_template(template)
        console.print(f"✅ Template imported: [cyan]{template.name}[/cyan]")
