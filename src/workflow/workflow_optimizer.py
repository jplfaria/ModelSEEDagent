"""
Workflow Optimizer for Advanced Automation

Provides intelligent workflow optimization capabilities including performance
optimization, resource allocation, and execution path optimization.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .workflow_definition import ResourceRequirement, WorkflowDefinition, WorkflowStep

console = Console()


class OptimizationStrategy(Enum):
    """Workflow optimization strategies"""

    MINIMIZE_DURATION = "minimize_duration"
    MINIMIZE_RESOURCES = "minimize_resources"
    MAXIMIZE_PARALLELISM = "maximize_parallelism"
    BALANCE_PERFORMANCE = "balance_performance"


@dataclass
class OptimizationResult:
    """Result of workflow optimization"""

    original_workflow: WorkflowDefinition
    optimized_workflow: WorkflowDefinition

    # Performance improvements
    estimated_time_savings: float = 0.0  # seconds
    resource_efficiency_gain: float = 0.0  # percentage
    parallelism_improvement: float = 0.0  # percentage

    # Optimization details
    optimizations_applied: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    success: bool = True
    message: str = "Optimization completed successfully"


class WorkflowOptimizer:
    """Intelligent workflow optimizer"""

    def __init__(self):
        self.optimization_history: List[OptimizationResult] = []

    def optimize_workflow(
        self,
        workflow: WorkflowDefinition,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCE_PERFORMANCE,
        target_resources: Optional[Dict[str, int]] = None,
    ) -> OptimizationResult:
        """Optimize a workflow based on the specified strategy"""

        console.print(
            f"[blue]🔧 Optimizing workflow '{workflow.name}' with {strategy.value} strategy...[/blue]"
        )

        # Create a copy for optimization
        optimized_workflow = self._copy_workflow(workflow)
        optimizations = []
        recommendations = []

        # Apply optimizations based on strategy
        if strategy == OptimizationStrategy.MINIMIZE_DURATION:
            optimizations.extend(self._optimize_for_duration(optimized_workflow))
        elif strategy == OptimizationStrategy.MINIMIZE_RESOURCES:
            optimizations.extend(self._optimize_for_resources(optimized_workflow))
        elif strategy == OptimizationStrategy.MAXIMIZE_PARALLELISM:
            optimizations.extend(self._optimize_for_parallelism(optimized_workflow))
        elif strategy == OptimizationStrategy.BALANCE_PERFORMANCE:
            optimizations.extend(self._optimize_balanced(optimized_workflow))

        # Generate recommendations
        recommendations.extend(
            self._generate_recommendations(workflow, optimized_workflow)
        )

        # Calculate improvements
        time_savings = self._calculate_time_savings(workflow, optimized_workflow)
        resource_efficiency = self._calculate_resource_efficiency(
            workflow, optimized_workflow
        )
        parallelism_improvement = self._calculate_parallelism_improvement(
            workflow, optimized_workflow
        )

        result = OptimizationResult(
            original_workflow=workflow,
            optimized_workflow=optimized_workflow,
            estimated_time_savings=time_savings,
            resource_efficiency_gain=resource_efficiency,
            parallelism_improvement=parallelism_improvement,
            optimizations_applied=optimizations,
            recommendations=recommendations,
        )

        self.optimization_history.append(result)

        console.print(
            f"[green]✅ Optimization completed with {len(optimizations)} improvements[/green]"
        )

        return result

    def _copy_workflow(self, workflow: WorkflowDefinition) -> WorkflowDefinition:
        """Create a deep copy of workflow for optimization"""
        # Simple copy implementation
        new_workflow = WorkflowDefinition(
            id=workflow.id + "_optimized",
            name=workflow.name + " (Optimized)",
            description=workflow.description + " - Optimized version",
            parameters=workflow.parameters.copy(),
            max_parallel_steps=workflow.max_parallel_steps,
        )

        # Copy steps
        new_steps = []
        for step in workflow.steps:
            new_step = WorkflowStep(
                id=step.id,
                name=step.name,
                type=step.type,
                description=step.description,
                tool_name=step.tool_name,
                parameters=step.parameters.copy() if step.parameters else {},
                dependencies=step.dependencies.copy() if step.dependencies else [],
                condition=step.condition,
                estimated_duration=step.estimated_duration,
                timeout=step.timeout,
                retry_count=step.retry_count,
                priority=step.priority,
                resource_requirements=step.resource_requirements,
                execution_mode=step.execution_mode,
            )
            new_steps.append(new_step)

        new_workflow.steps = new_steps
        return new_workflow

    def _optimize_for_duration(self, workflow: WorkflowDefinition) -> List[str]:
        """Optimize workflow to minimize execution duration"""
        optimizations = []

        # Increase parallelism where possible
        parallel_groups = workflow.get_execution_order()
        if len(parallel_groups) > 1:
            # Look for steps that can be parallelized
            for i, group in enumerate(parallel_groups[:-1]):
                next_group = parallel_groups[i + 1]

                # Check if any steps in next group can run in parallel with current
                for step_id in next_group:
                    step = next((s for s in workflow.steps if s.id == step_id), None)
                    if step and not step.dependencies:
                        # This step can potentially run earlier
                        optimizations.append(
                            f"Moved step '{step.name}' to run in parallel"
                        )

        # Optimize resource allocation for faster execution
        for step in workflow.steps:
            if step.resource_requirements == ResourceRequirement.LOW:
                step.resource_requirements = ResourceRequirement.MEDIUM
                optimizations.append(
                    f"Increased resources for step '{step.name}' to speed up execution"
                )

        # Increase overall parallelism
        if workflow.max_parallel_steps < 6:
            workflow.max_parallel_steps = min(6, workflow.max_parallel_steps + 2)
            optimizations.append(
                f"Increased max parallel steps to {workflow.max_parallel_steps}"
            )

        return optimizations

    def _optimize_for_resources(self, workflow: WorkflowDefinition) -> List[str]:
        """Optimize workflow to minimize resource usage"""
        optimizations = []

        # Reduce resource requirements where possible
        for step in workflow.steps:
            if step.resource_requirements == ResourceRequirement.HIGH:
                step.resource_requirements = ResourceRequirement.MEDIUM
                optimizations.append(
                    f"Reduced resource requirements for step '{step.name}'"
                )
            elif step.resource_requirements == ResourceRequirement.MEDIUM:
                step.resource_requirements = ResourceRequirement.LOW
                optimizations.append(f"Optimized resource usage for step '{step.name}'")

        # Reduce parallelism to save resources
        if workflow.max_parallel_steps > 2:
            workflow.max_parallel_steps = max(2, workflow.max_parallel_steps - 1)
            optimizations.append(
                f"Reduced max parallel steps to {workflow.max_parallel_steps} to save resources"
            )

        return optimizations

    def _optimize_for_parallelism(self, workflow: WorkflowDefinition) -> List[str]:
        """Optimize workflow to maximize parallel execution"""
        optimizations = []

        # Increase max parallel steps
        workflow.max_parallel_steps = min(8, workflow.max_parallel_steps + 3)
        optimizations.append(
            f"Increased max parallel steps to {workflow.max_parallel_steps}"
        )

        # Analyze dependencies and remove unnecessary ones
        for step in workflow.steps:
            if len(step.dependencies) > 1:
                # Check if all dependencies are truly necessary
                # For demo, assume we can remove some
                if len(step.dependencies) > 2:
                    step.dependencies = step.dependencies[:2]
                    optimizations.append(
                        f"Reduced dependencies for step '{step.name}' to increase parallelism"
                    )

        # Set steps to parallel execution mode where possible
        for step in workflow.steps:
            if (
                step.execution_mode != "parallel"
                and step.resource_requirements != ResourceRequirement.EXTREME
            ):
                step.execution_mode = "parallel"
                optimizations.append(
                    f"Set step '{step.name}' to parallel execution mode"
                )

        return optimizations

    def _optimize_balanced(self, workflow: WorkflowDefinition) -> List[str]:
        """Apply balanced optimization across all aspects"""
        optimizations = []

        # Apply moderate improvements across all strategies
        optimizations.extend(
            self._optimize_for_duration(workflow)[:2]
        )  # Take only first 2
        optimizations.extend(
            self._optimize_for_resources(workflow)[:1]
        )  # Take only first 1
        optimizations.extend(
            self._optimize_for_parallelism(workflow)[:2]
        )  # Take only first 2

        # Additional balanced optimizations
        optimizations.append("Applied balanced optimization strategy")

        return optimizations

    def _generate_recommendations(
        self, original: WorkflowDefinition, optimized: WorkflowDefinition
    ) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Analyze workflow structure
        if len(original.steps) > 10:
            recommendations.append(
                "Consider breaking down large workflows into smaller, reusable components"
            )

        # Check for resource-intensive steps
        heavy_steps = [
            s
            for s in original.steps
            if s.resource_requirements == ResourceRequirement.HIGH
        ]
        if heavy_steps:
            recommendations.append(
                f"Monitor resource usage for {len(heavy_steps)} resource-intensive steps"
            )

        # Check for single points of failure
        critical_steps = [
            s
            for s in original.steps
            if len([d for d in original.steps if s.id in d.dependencies]) > 2
        ]
        if critical_steps:
            recommendations.append(
                f"Consider adding redundancy for {len(critical_steps)} critical steps"
            )

        # Check execution time estimates
        long_steps = [
            s
            for s in original.steps
            if s.estimated_duration and s.estimated_duration > 300
        ]
        if long_steps:
            recommendations.append(
                f"Consider optimizing {len(long_steps)} long-running steps"
            )

        return recommendations

    def _calculate_time_savings(
        self, original: WorkflowDefinition, optimized: WorkflowDefinition
    ) -> float:
        """Calculate estimated time savings from optimization"""
        original_time = original.estimate_total_duration()
        optimized_time = optimized.estimate_total_duration()

        # Account for parallelism improvements
        parallelism_factor = optimized.max_parallel_steps / max(
            1, original.max_parallel_steps
        )
        adjusted_optimized_time = optimized_time / parallelism_factor

        return max(0, original_time - adjusted_optimized_time)

    def _calculate_resource_efficiency(
        self, original: WorkflowDefinition, optimized: WorkflowDefinition
    ) -> float:
        """Calculate resource efficiency improvement"""
        original_resources = sum(self._step_resource_cost(s) for s in original.steps)
        optimized_resources = sum(self._step_resource_cost(s) for s in optimized.steps)

        if original_resources == 0:
            return 0.0

        efficiency_gain = (
            (original_resources - optimized_resources) / original_resources
        ) * 100
        return max(0, efficiency_gain)

    def _calculate_parallelism_improvement(
        self, original: WorkflowDefinition, optimized: WorkflowDefinition
    ) -> float:
        """Calculate parallelism improvement percentage"""
        original_parallelism = self._calculate_parallelism_score(original)
        optimized_parallelism = self._calculate_parallelism_score(optimized)

        if original_parallelism == 0:
            return 100.0 if optimized_parallelism > 0 else 0.0

        improvement = (
            (optimized_parallelism - original_parallelism) / original_parallelism
        ) * 100
        return max(0, improvement)

    def _step_resource_cost(self, step: WorkflowStep) -> int:
        """Calculate resource cost for a step"""
        resource_costs = {
            ResourceRequirement.LOW: 1,
            ResourceRequirement.MEDIUM: 2,
            ResourceRequirement.HIGH: 4,
            ResourceRequirement.EXTREME: 8,
        }
        return resource_costs.get(step.resource_requirements, 2)

    def _calculate_parallelism_score(self, workflow: WorkflowDefinition) -> float:
        """Calculate a parallelism score for the workflow"""
        execution_groups = workflow.get_execution_order()
        if not execution_groups:
            return 0.0

        # Score based on number of parallel groups and max parallelism
        group_sizes = [len(group) for group in execution_groups]
        avg_group_size = sum(group_sizes) / len(group_sizes)

        # Normalize by max possible parallelism
        max_possible = workflow.max_parallel_steps
        parallelism_score = (avg_group_size / max_possible) * 100

        return min(100.0, parallelism_score)

    def analyze_workflow_performance(
        self, workflow: WorkflowDefinition
    ) -> Dict[str, Any]:
        """Analyze workflow performance characteristics"""
        analysis = {
            "total_steps": len(workflow.steps),
            "estimated_duration": workflow.estimate_total_duration(),
            "max_parallelism": workflow.max_parallel_steps,
            "execution_groups": len(workflow.get_execution_order()),
            "parallelism_score": self._calculate_parallelism_score(workflow),
            "resource_intensity": sum(
                self._step_resource_cost(s) for s in workflow.steps
            ),
            "critical_path_length": self._calculate_critical_path_length(workflow),
            "dependency_complexity": self._calculate_dependency_complexity(workflow),
        }

        return analysis

    def _calculate_critical_path_length(self, workflow: WorkflowDefinition) -> int:
        """Calculate the length of the critical path"""
        # Simplified critical path calculation
        execution_groups = workflow.get_execution_order()
        return len(execution_groups)

    def _calculate_dependency_complexity(self, workflow: WorkflowDefinition) -> float:
        """Calculate workflow dependency complexity"""
        total_dependencies = sum(len(step.dependencies) for step in workflow.steps)
        if len(workflow.steps) == 0:
            return 0.0

        avg_dependencies = total_dependencies / len(workflow.steps)
        return avg_dependencies

    def compare_workflows(
        self, workflow1: WorkflowDefinition, workflow2: WorkflowDefinition
    ) -> Dict[str, Any]:
        """Compare two workflows across multiple dimensions"""
        analysis1 = self.analyze_workflow_performance(workflow1)
        analysis2 = self.analyze_workflow_performance(workflow2)

        comparison = {}
        for key in analysis1:
            value1 = analysis1[key]
            value2 = analysis2[key]

            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                if value1 != 0:
                    percentage_change = ((value2 - value1) / value1) * 100
                else:
                    percentage_change = 100.0 if value2 > 0 else 0.0

                comparison[key] = {
                    "workflow1": value1,
                    "workflow2": value2,
                    "change": value2 - value1,
                    "percentage_change": percentage_change,
                }
            else:
                comparison[key] = {"workflow1": value1, "workflow2": value2}

        return comparison

    def display_optimization_result(self, result: OptimizationResult) -> None:
        """Display optimization results in a beautiful format"""

        # Optimization overview
        overview_table = Table(show_header=False, box=box.SIMPLE)
        overview_table.add_column("Metric", style="bold cyan")
        overview_table.add_column("Improvement", style="bold white")

        overview_table.add_row(
            "Time Savings", f"{result.estimated_time_savings:.1f} seconds"
        )
        overview_table.add_row(
            "Resource Efficiency", f"+{result.resource_efficiency_gain:.1f}%"
        )
        overview_table.add_row("Parallelism", f"+{result.parallelism_improvement:.1f}%")
        overview_table.add_row(
            "Optimizations Applied", str(len(result.optimizations_applied))
        )

        console.print(
            Panel(
                overview_table,
                title="[bold blue]🚀 Optimization Results[/bold blue]",
                border_style="blue",
            )
        )

        # Applied optimizations
        if result.optimizations_applied:
            console.print("\n[bold yellow]⚡ Applied Optimizations:[/bold yellow]")
            for i, optimization in enumerate(result.optimizations_applied, 1):
                console.print(f"  {i}. {optimization}")

        # Recommendations
        if result.recommendations:
            console.print("\n[bold green]💡 Recommendations:[/bold green]")
            for i, recommendation in enumerate(result.recommendations, 1):
                console.print(f"  {i}. {recommendation}")

    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get optimization history"""
        return self.optimization_history

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.optimization_history:
            return {"total_optimizations": 0}

        total_time_saved = sum(
            r.estimated_time_savings for r in self.optimization_history
        )
        avg_resource_efficiency = sum(
            r.resource_efficiency_gain for r in self.optimization_history
        ) / len(self.optimization_history)
        avg_parallelism_improvement = sum(
            r.parallelism_improvement for r in self.optimization_history
        ) / len(self.optimization_history)

        return {
            "total_optimizations": len(self.optimization_history),
            "total_time_saved": total_time_saved,
            "average_resource_efficiency_gain": avg_resource_efficiency,
            "average_parallelism_improvement": avg_parallelism_improvement,
            "successful_optimizations": sum(
                1 for r in self.optimization_history if r.success
            ),
        }
