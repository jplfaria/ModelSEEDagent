"""
Workflow Definition System for Advanced Automation

Provides comprehensive workflow definition capabilities with YAML/JSON support,
intelligent dependency management, and flexible parameter handling.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class StepType(Enum):
    """Types of workflow steps"""

    TOOL_EXECUTION = "tool_execution"
    CONDITION_CHECK = "condition_check"
    PARALLEL_EXECUTION = "parallel_execution"
    LOOP = "loop"
    SUBWORKFLOW = "subworkflow"
    SCRIPT_EXECUTION = "script_execution"
    MODEL_VALIDATION = "model_validation"
    DATA_TRANSFORMATION = "data_transformation"
    NOTIFICATION = "notification"
    CHECKPOINT = "checkpoint"


class ExecutionMode(Enum):
    """Execution modes for workflow steps"""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ASYNC = "async"
    RETRY_ON_FAILURE = "retry_on_failure"


class ResourceRequirement(Enum):
    """Resource requirements for execution"""

    LOW = "low"  # < 1GB RAM, 1 CPU
    MEDIUM = "medium"  # 1-4GB RAM, 2-4 CPU
    HIGH = "high"  # 4-16GB RAM, 4-8 CPU
    EXTREME = "extreme"  # >16GB RAM, >8 CPU


@dataclass
class WorkflowParameter:
    """Parameter definition for workflows"""

    name: str
    type: str  # "string", "integer", "float", "boolean", "list", "dict", "file"
    description: str
    default_value: Optional[Any] = None
    required: bool = True
    validation_rule: Optional[str] = None
    choices: Optional[List[Any]] = None

    def validate(self, value: Any) -> bool:
        """Validate parameter value"""
        if self.required and value is None:
            return False

        if value is None and self.default_value is not None:
            return True

        # Type validation
        if self.type == "string" and not isinstance(value, str):
            return False
        elif self.type == "integer" and not isinstance(value, int):
            return False
        elif self.type == "float" and not isinstance(value, (int, float)):
            return False
        elif self.type == "boolean" and not isinstance(value, bool):
            return False
        elif self.type == "list" and not isinstance(value, list):
            return False
        elif self.type == "dict" and not isinstance(value, dict):
            return False
        elif self.type == "file" and not Path(str(value)).exists():
            return False

        # Choice validation
        if self.choices and value not in self.choices:
            return False

        return True


@dataclass
class WorkflowStep:
    """Individual step in a workflow"""

    id: str
    name: str
    type: StepType
    description: str = ""
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    resource_requirements: ResourceRequirement = ResourceRequirement.MEDIUM
    timeout: Optional[int] = None  # seconds
    retry_count: int = 0
    retry_delay: int = 5  # seconds
    condition: Optional[str] = None  # Python expression

    # Step metadata
    estimated_duration: Optional[int] = None  # seconds
    tags: List[str] = field(default_factory=list)
    category: str = "general"

    # Output configuration
    output_variables: List[str] = field(default_factory=list)
    output_format: str = "json"
    save_output: bool = True

    def validate_dependencies(self, available_steps: List[str]) -> bool:
        """Validate that all dependencies exist"""
        return all(dep in available_steps for dep in self.dependencies)

    def can_execute(
        self, completed_steps: List[str], step_outputs: Dict[str, Any]
    ) -> bool:
        """Check if step can be executed based on dependencies and conditions"""
        # Check dependencies
        if not all(dep in completed_steps for dep in self.dependencies):
            return False

        # Check condition if specified
        if self.condition:
            try:
                # Create safe evaluation context
                context = {
                    "outputs": step_outputs,
                    "parameters": self.parameters,
                    "True": True,
                    "False": False,
                }
                return eval(self.condition, {"__builtins__": {}}, context)
            except Exception:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "execution_mode": self.execution_mode.value,
            "resource_requirements": self.resource_requirements.value,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "condition": self.condition,
            "estimated_duration": self.estimated_duration,
            "tags": self.tags,
            "category": self.category,
            "output_variables": self.output_variables,
            "output_format": self.output_format,
            "save_output": self.save_output,
        }


@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""

    id: str
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "ModelSEEDagent"
    created_at: datetime = field(default_factory=datetime.now)

    # Workflow structure
    steps: List[WorkflowStep] = field(default_factory=list)
    parameters: List[WorkflowParameter] = field(default_factory=list)

    # Execution configuration
    default_execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_parallel_steps: int = 4
    global_timeout: Optional[int] = None  # seconds

    # Workflow metadata
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    estimated_duration: Optional[int] = None

    # Resource configuration
    resource_pool: str = "default"
    priority: int = 5  # 1-10, higher is more priority

    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow"""
        if any(s.id == step.id for s in self.steps):
            raise ValueError(f"Step with ID '{step.id}' already exists")

        self.steps.append(step)

    def add_parameter(self, parameter: WorkflowParameter) -> None:
        """Add a parameter to the workflow"""
        if any(p.name == parameter.name for p in self.parameters):
            raise ValueError(f"Parameter '{parameter.name}' already exists")

        self.parameters.append(parameter)

    def validate(self) -> List[str]:
        """Validate the workflow definition"""
        errors = []

        # Check for duplicate step IDs
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Duplicate step IDs found")

        # Validate step dependencies
        for step in self.steps:
            if not step.validate_dependencies(step_ids):
                errors.append(f"Step '{step.id}' has invalid dependencies")

        # Check for circular dependencies
        if self._has_circular_dependencies():
            errors.append("Circular dependencies detected")

        # Validate parameters
        param_names = [param.name for param in self.parameters]
        if len(param_names) != len(set(param_names)):
            errors.append("Duplicate parameter names found")

        return errors

    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using DFS"""

        def has_cycle(node, visited, rec_stack):
            visited[node] = True
            rec_stack[node] = True

            # Get dependencies for current step
            current_step = next((s for s in self.steps if s.id == node), None)
            if current_step:
                for dep in current_step.dependencies:
                    if dep in step_map:
                        if not visited.get(dep, False):
                            if has_cycle(dep, visited, rec_stack):
                                return True
                        elif rec_stack.get(dep, False):
                            return True

            rec_stack[node] = False
            return False

        step_map = {step.id: step for step in self.steps}
        visited = {}
        rec_stack = {}

        for step_id in step_map:
            if not visited.get(step_id, False):
                if has_cycle(step_id, visited, rec_stack):
                    return True

        return False

    def get_execution_order(self) -> List[List[str]]:
        """Get execution order with parallel groups"""
        completed = set()
        execution_groups = []
        remaining_steps = {step.id: step for step in self.steps}

        while remaining_steps:
            # Find steps that can execute now
            ready_steps = []
            for step_id, step in remaining_steps.items():
                if all(dep in completed for dep in step.dependencies):
                    ready_steps.append(step_id)

            if not ready_steps:
                # Should not happen if validation passed
                raise ValueError(
                    "Cannot determine execution order - possible circular dependency"
                )

            execution_groups.append(ready_steps)

            # Mark steps as completed and remove from remaining
            for step_id in ready_steps:
                completed.add(step_id)
                del remaining_steps[step_id]

        return execution_groups

    def estimate_total_duration(self) -> int:
        """Estimate total workflow duration in seconds"""
        if self.estimated_duration:
            return self.estimated_duration

        execution_groups = self.get_execution_order()
        total_duration = 0

        for group in execution_groups:
            # For parallel execution, use the maximum duration in the group
            group_durations = []
            for step_id in group:
                step = next((s for s in self.steps if s.id == step_id), None)
                if step and step.estimated_duration:
                    group_durations.append(step.estimated_duration)
                else:
                    group_durations.append(60)  # Default 1 minute

            total_duration += max(group_durations) if group_durations else 60

        return total_duration

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "steps": [step.to_dict() for step in self.steps],
            "parameters": [asdict(param) for param in self.parameters],
            "default_execution_mode": self.default_execution_mode.value,
            "max_parallel_steps": self.max_parallel_steps,
            "global_timeout": self.global_timeout,
            "tags": self.tags,
            "category": self.category,
            "estimated_duration": self.estimated_duration,
            "resource_pool": self.resource_pool,
            "priority": self.priority,
        }

    def to_yaml(self) -> str:
        """Convert to YAML format"""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        """Convert to JSON format"""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowDefinition":
        """Create from dictionary"""
        workflow = cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
            author=data.get("author", "ModelSEEDagent"),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.now().isoformat())
            ),
            default_execution_mode=ExecutionMode(
                data.get("default_execution_mode", "sequential")
            ),
            max_parallel_steps=data.get("max_parallel_steps", 4),
            global_timeout=data.get("global_timeout"),
            tags=data.get("tags", []),
            category=data.get("category", "general"),
            estimated_duration=data.get("estimated_duration"),
            resource_pool=data.get("resource_pool", "default"),
            priority=data.get("priority", 5),
        )

        # Add steps
        for step_data in data.get("steps", []):
            step = WorkflowStep(
                id=step_data["id"],
                name=step_data["name"],
                type=StepType(step_data["type"]),
                description=step_data.get("description", ""),
                tool_name=step_data.get("tool_name"),
                parameters=step_data.get("parameters", {}),
                dependencies=step_data.get("dependencies", []),
                execution_mode=ExecutionMode(
                    step_data.get("execution_mode", "sequential")
                ),
                resource_requirements=ResourceRequirement(
                    step_data.get("resource_requirements", "medium")
                ),
                timeout=step_data.get("timeout"),
                retry_count=step_data.get("retry_count", 0),
                retry_delay=step_data.get("retry_delay", 5),
                condition=step_data.get("condition"),
                estimated_duration=step_data.get("estimated_duration"),
                tags=step_data.get("tags", []),
                category=step_data.get("category", "general"),
                output_variables=step_data.get("output_variables", []),
                output_format=step_data.get("output_format", "json"),
                save_output=step_data.get("save_output", True),
            )
            workflow.add_step(step)

        # Add parameters
        for param_data in data.get("parameters", []):
            param = WorkflowParameter(
                name=param_data["name"],
                type=param_data["type"],
                description=param_data["description"],
                default_value=param_data.get("default_value"),
                required=param_data.get("required", True),
                validation_rule=param_data.get("validation_rule"),
                choices=param_data.get("choices"),
            )
            workflow.add_parameter(param)

        return workflow

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "WorkflowDefinition":
        """Create from YAML content"""
        data = yaml.safe_load(yaml_content)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, json_content: str) -> "WorkflowDefinition":
        """Create from JSON content"""
        data = json.loads(json_content)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "WorkflowDefinition":
        """Load workflow from file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Workflow file not found: {file_path}")

        content = path.read_text()

        if path.suffix.lower() in [".yml", ".yaml"]:
            return cls.from_yaml(content)
        elif path.suffix.lower() == ".json":
            return cls.from_json(content)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def save_to_file(self, file_path: Union[str, Path], format: str = "yaml") -> None:
        """Save workflow to file"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "yaml":
            content = self.to_yaml()
        elif format.lower() == "json":
            content = self.to_json()
        else:
            raise ValueError(f"Unsupported format: {format}")

        path.write_text(content)
        console.print(f"✅ Workflow saved to: [cyan]{path}[/cyan]")

    def display_summary(self) -> None:
        """Display workflow summary"""
        summary_table = Table(show_header=False, box=box.SIMPLE)
        summary_table.add_column("Aspect", style="bold cyan")
        summary_table.add_column("Value", style="bold white")

        summary_table.add_row("ID", self.id)
        summary_table.add_row("Name", self.name)
        summary_table.add_row("Version", self.version)
        summary_table.add_row("Author", self.author)
        summary_table.add_row("Steps", str(len(self.steps)))
        summary_table.add_row("Parameters", str(len(self.parameters)))
        summary_table.add_row("Category", self.category)
        summary_table.add_row("Priority", str(self.priority))
        summary_table.add_row("Est. Duration", f"{self.estimate_total_duration()}s")

        console.print(
            Panel(
                summary_table,
                title=f"[bold blue]📋 Workflow: {self.name}[/bold blue]",
                border_style="blue",
            )
        )

        # Display steps
        if self.steps:
            steps_table = Table(title="🔧 Workflow Steps", box=box.ROUNDED)
            steps_table.add_column("ID", style="cyan")
            steps_table.add_column("Name", style="white")
            steps_table.add_column("Type", style="green")
            steps_table.add_column("Dependencies", style="yellow")
            steps_table.add_column("Est. Duration", style="blue")

            for step in self.steps:
                deps_str = ", ".join(step.dependencies) if step.dependencies else "None"
                duration_str = (
                    f"{step.estimated_duration}s"
                    if step.estimated_duration
                    else "Unknown"
                )

                steps_table.add_row(
                    step.id,
                    step.name,
                    step.type.value.replace("_", " ").title(),
                    deps_str,
                    duration_str,
                )

            console.print(steps_table)


@dataclass
class WorkflowTemplate:
    """Template for creating workflows"""

    id: str
    name: str
    description: str
    category: str
    tags: List[str] = field(default_factory=list)
    template_data: Dict[str, Any] = field(default_factory=dict)
    parameter_schema: List[WorkflowParameter] = field(default_factory=list)

    def instantiate(self, parameters: Dict[str, Any]) -> WorkflowDefinition:
        """Create a workflow instance from this template"""
        # Validate parameters
        for param in self.parameter_schema:
            value = parameters.get(param.name, param.default_value)
            if not param.validate(value):
                raise ValueError(f"Invalid value for parameter '{param.name}': {value}")

        # Substitute parameters in template
        instantiated_data = self._substitute_parameters(self.template_data, parameters)

        # Create workflow
        workflow = WorkflowDefinition.from_dict(instantiated_data)
        workflow.id = str(uuid.uuid4())[:8]
        workflow.created_at = datetime.now()

        return workflow

    def _substitute_parameters(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Recursively substitute parameters in template data"""
        if isinstance(data, str):
            # Simple parameter substitution
            for param_name, param_value in parameters.items():
                placeholder = f"${{{param_name}}}"
                if placeholder in data:
                    data = data.replace(placeholder, str(param_value))
            return data
        elif isinstance(data, dict):
            return {
                k: self._substitute_parameters(v, parameters) for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._substitute_parameters(item, parameters) for item in data]
        else:
            return data
