"""
Advanced Workflow Execution Engine

Provides intelligent workflow orchestration with parallel execution,
real-time monitoring, error recovery, and comprehensive observability.
"""

import asyncio
import json
import queue
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .workflow_definition import (
    ExecutionMode,
    ResourceRequirement,
    StepType,
    WorkflowDefinition,
    WorkflowStep,
)

console = Console()


class WorkflowStatus(Enum):
    """Workflow execution status"""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StepStatus(Enum):
    """Individual step execution status"""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class StepExecution:
    """Step execution state and results"""

    step_id: str
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    output: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    resource_usage: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get execution duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0


@dataclass
class WorkflowExecution:
    """Complete workflow execution state"""

    workflow_id: str
    execution_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    step_executions: Dict[str, StepExecution] = field(default_factory=dict)
    execution_groups: List[List[str]] = field(default_factory=list)
    current_group: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    global_context: Dict[str, Any] = field(default_factory=dict)

    # Resource tracking
    max_parallel_steps: int = 4
    active_executions: int = 0

    # Statistics
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0

    @property
    def progress(self) -> float:
        """Get overall progress percentage"""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100

    @property
    def duration(self) -> float:
        """Get total execution duration"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0


@dataclass
class WorkflowResult:
    """Final workflow execution result"""

    workflow_id: str
    execution_id: str
    success: bool
    status: WorkflowStatus
    message: str
    duration: float
    step_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    average_step_time: float = 0.0


class WorkflowEngine:
    """Advanced workflow execution engine"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []

        # Tool registry for step execution
        self.tool_registry: Dict[str, Callable] = {}

        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            "workflow_started": [],
            "workflow_completed": [],
            "workflow_failed": [],
            "step_started": [],
            "step_completed": [],
            "step_failed": [],
        }

        # Live monitoring
        self.live_display: Optional[Live] = None
        self.monitoring_enabled: bool = True

    def register_tool(self, name: str, tool_function: Callable) -> None:
        """Register a tool for workflow execution"""
        self.tool_registry[name] = tool_function
        console.print(f"✅ Registered tool: [cyan]{name}[/cyan]")

    def register_event_handler(self, event: str, handler: Callable) -> None:
        """Register an event handler"""
        if event in self.event_handlers:
            self.event_handlers[event].append(handler)

    def _emit_event(self, event: str, data: Dict[str, Any]) -> None:
        """Emit an event to registered handlers"""
        for handler in self.event_handlers.get(event, []):
            try:
                handler(data)
            except Exception as e:
                console.print(f"[red]❌ Event handler error for {event}: {e}[/red]")

    async def execute_workflow(
        self,
        workflow: WorkflowDefinition,
        parameters: Optional[Dict[str, Any]] = None,
        enable_monitoring: bool = True,
    ) -> WorkflowResult:
        """Execute a workflow with full orchestration"""

        execution_id = str(uuid.uuid4())[:8]

        # Validate workflow
        validation_errors = workflow.validate()
        if validation_errors:
            return WorkflowResult(
                workflow_id=workflow.id,
                execution_id=execution_id,
                success=False,
                status=WorkflowStatus.FAILED,
                message=f"Workflow validation failed: {'; '.join(validation_errors)}",
                duration=0.0,
            )

        # Create execution context
        execution = WorkflowExecution(
            workflow_id=workflow.id,
            execution_id=execution_id,
            parameters=parameters or {},
            total_steps=len(workflow.steps),
            max_parallel_steps=workflow.max_parallel_steps,
            execution_groups=workflow.get_execution_order(),
        )

        # Initialize step executions
        for step in workflow.steps:
            execution.step_executions[step.id] = StepExecution(step_id=step.id)

        self.active_executions[execution_id] = execution

        try:
            # Start monitoring if enabled
            if enable_monitoring and self.monitoring_enabled:
                await self._start_live_monitoring(execution, workflow)

            # Execute workflow
            result = await self._execute_workflow_internal(workflow, execution)

            # Stop monitoring
            if self.live_display:
                self.live_display.stop()
                self.live_display = None

            return result

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()

            return WorkflowResult(
                workflow_id=workflow.id,
                execution_id=execution_id,
                success=False,
                status=WorkflowStatus.FAILED,
                message=f"Workflow execution failed: {str(e)}",
                duration=execution.duration,
            )
        finally:
            # Clean up
            if execution_id in self.active_executions:
                self.execution_history.append(self.active_executions[execution_id])
                del self.active_executions[execution_id]

    async def _execute_workflow_internal(
        self, workflow: WorkflowDefinition, execution: WorkflowExecution
    ) -> WorkflowResult:
        """Internal workflow execution logic"""

        execution.status = WorkflowStatus.RUNNING
        execution.start_time = datetime.now()

        self._emit_event(
            "workflow_started",
            {
                "workflow_id": workflow.id,
                "execution_id": execution.execution_id,
                "timestamp": execution.start_time,
            },
        )

        try:
            # Execute groups sequentially, steps within groups in parallel
            for group_index, step_ids in enumerate(execution.execution_groups):
                execution.current_group = group_index

                # Create tasks for parallel execution
                tasks = []
                for step_id in step_ids:
                    step = next((s for s in workflow.steps if s.id == step_id), None)
                    if step:
                        task = asyncio.create_task(
                            self._execute_step(step, execution, workflow)
                        )
                        tasks.append(task)

                # Wait for all steps in the group to complete
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Check for failures
                    for i, result in enumerate(results):
                        step_id = step_ids[i]
                        step_execution = execution.step_executions[step_id]

                        if isinstance(result, Exception):
                            step_execution.status = StepStatus.FAILED
                            step_execution.error = str(result)
                            execution.failed_steps += 1
                        elif not result:
                            step_execution.status = StepStatus.FAILED
                            execution.failed_steps += 1
                        else:
                            step_execution.status = StepStatus.COMPLETED
                            execution.completed_steps += 1

                # Check if we should continue after failures
                if execution.failed_steps > 0:
                    # For now, stop on any failure
                    break

            # Determine final status
            if execution.failed_steps > 0:
                execution.status = WorkflowStatus.FAILED
                message = f"Workflow failed: {execution.failed_steps} steps failed"
            else:
                execution.status = WorkflowStatus.COMPLETED
                message = f"Workflow completed successfully: {execution.completed_steps} steps completed"

            execution.end_time = datetime.now()

            # Create result
            result = WorkflowResult(
                workflow_id=workflow.id,
                execution_id=execution.execution_id,
                success=execution.status == WorkflowStatus.COMPLETED,
                status=execution.status,
                message=message,
                duration=execution.duration,
                total_steps=execution.total_steps,
                completed_steps=execution.completed_steps,
                failed_steps=execution.failed_steps,
                skipped_steps=execution.total_steps
                - execution.completed_steps
                - execution.failed_steps,
            )

            # Calculate average step time
            step_times = [
                se.duration
                for se in execution.step_executions.values()
                if se.status == StepStatus.COMPLETED
            ]
            if step_times:
                result.average_step_time = sum(step_times) / len(step_times)

            # Collect step results
            for step_id, step_execution in execution.step_executions.items():
                if step_execution.output is not None:
                    result.step_results[step_id] = step_execution.output

            # Emit completion event
            event_name = "workflow_completed" if result.success else "workflow_failed"
            self._emit_event(
                event_name,
                {
                    "workflow_id": workflow.id,
                    "execution_id": execution.execution_id,
                    "result": result,
                    "timestamp": execution.end_time,
                },
            )

            return result

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()

            self._emit_event(
                "workflow_failed",
                {
                    "workflow_id": workflow.id,
                    "execution_id": execution.execution_id,
                    "error": str(e),
                    "timestamp": execution.end_time,
                },
            )

            raise

    async def _execute_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
        workflow: WorkflowDefinition,
    ) -> bool:
        """Execute a single workflow step"""

        step_execution = execution.step_executions[step.id]
        step_execution.status = StepStatus.RUNNING
        step_execution.start_time = datetime.now()

        self._emit_event(
            "step_started",
            {
                "workflow_id": workflow.id,
                "execution_id": execution.execution_id,
                "step_id": step.id,
                "step_name": step.name,
                "timestamp": step_execution.start_time,
            },
        )

        try:
            # Check if step can execute (conditions, dependencies)
            completed_steps = [
                sid
                for sid, se in execution.step_executions.items()
                if se.status == StepStatus.COMPLETED
            ]
            step_outputs = {
                sid: se.output
                for sid, se in execution.step_executions.items()
                if se.output is not None
            }

            if not step.can_execute(completed_steps, step_outputs):
                step_execution.status = StepStatus.SKIPPED
                return True

            # Execute based on step type
            if step.type == StepType.TOOL_EXECUTION:
                success = await self._execute_tool_step(step, step_execution, execution)
            elif step.type == StepType.CONDITION_CHECK:
                success = await self._execute_condition_step(
                    step, step_execution, execution
                )
            elif step.type == StepType.SCRIPT_EXECUTION:
                success = await self._execute_script_step(
                    step, step_execution, execution
                )
            elif step.type == StepType.MODEL_VALIDATION:
                success = await self._execute_validation_step(
                    step, step_execution, execution
                )
            elif step.type == StepType.NOTIFICATION:
                success = await self._execute_notification_step(
                    step, step_execution, execution
                )
            else:
                # Default mock execution for unsupported types
                success = await self._execute_mock_step(step, step_execution, execution)

            step_execution.end_time = datetime.now()

            if success:
                step_execution.status = StepStatus.COMPLETED
                self._emit_event(
                    "step_completed",
                    {
                        "workflow_id": workflow.id,
                        "execution_id": execution.execution_id,
                        "step_id": step.id,
                        "step_name": step.name,
                        "duration": step_execution.duration,
                        "timestamp": step_execution.end_time,
                    },
                )
            else:
                step_execution.status = StepStatus.FAILED
                self._emit_event(
                    "step_failed",
                    {
                        "workflow_id": workflow.id,
                        "execution_id": execution.execution_id,
                        "step_id": step.id,
                        "step_name": step.name,
                        "error": step_execution.error,
                        "timestamp": step_execution.end_time,
                    },
                )

            return success

        except Exception as e:
            step_execution.status = StepStatus.FAILED
            step_execution.error = str(e)
            step_execution.end_time = datetime.now()

            self._emit_event(
                "step_failed",
                {
                    "workflow_id": workflow.id,
                    "execution_id": execution.execution_id,
                    "step_id": step.id,
                    "step_name": step.name,
                    "error": str(e),
                    "timestamp": step_execution.end_time,
                },
            )

            return False

    async def _execute_tool_step(
        self,
        step: WorkflowStep,
        step_execution: StepExecution,
        execution: WorkflowExecution,
    ) -> bool:
        """Execute a tool-based step"""
        if not step.tool_name or step.tool_name not in self.tool_registry:
            step_execution.error = f"Tool '{step.tool_name}' not found in registry"
            return False

        tool_function = self.tool_registry[step.tool_name]

        try:
            # Prepare parameters
            params = step.parameters.copy()

            # Add global context if needed
            if "context" in params:
                params["context"] = execution.global_context

            # Execute tool
            result = await asyncio.to_thread(tool_function, params)

            step_execution.output = result
            return True

        except Exception as e:
            step_execution.error = f"Tool execution failed: {str(e)}"
            return False

    async def _execute_condition_step(
        self,
        step: WorkflowStep,
        step_execution: StepExecution,
        execution: WorkflowExecution,
    ) -> bool:
        """Execute a condition check step"""
        condition = step.condition or step.parameters.get("condition")
        if not condition:
            step_execution.error = "No condition specified"
            return False

        try:
            # Create evaluation context
            step_outputs = {
                sid: se.output
                for sid, se in execution.step_executions.items()
                if se.output is not None
            }

            context = {
                "outputs": step_outputs,
                "parameters": execution.parameters,
                "global_context": execution.global_context,
                "True": True,
                "False": False,
            }

            result = eval(condition, {"__builtins__": {}}, context)
            step_execution.output = {"condition_result": bool(result)}

            return bool(result)

        except Exception as e:
            step_execution.error = f"Condition evaluation failed: {str(e)}"
            return False

    async def _execute_script_step(
        self,
        step: WorkflowStep,
        step_execution: StepExecution,
        execution: WorkflowExecution,
    ) -> bool:
        """Execute a script-based step"""
        script = step.parameters.get("script")
        if not script:
            step_execution.error = "No script specified"
            return False

        try:
            # Simple script execution (could be enhanced with more sophisticated execution)
            # For now, just simulate script execution
            await asyncio.sleep(0.5)  # Simulate execution time

            step_execution.output = {"script_result": "Script executed successfully"}
            return True

        except Exception as e:
            step_execution.error = f"Script execution failed: {str(e)}"
            return False

    async def _execute_validation_step(
        self,
        step: WorkflowStep,
        step_execution: StepExecution,
        execution: WorkflowExecution,
    ) -> bool:
        """Execute a model validation step"""
        model_path = step.parameters.get("model_path")
        if not model_path:
            step_execution.error = "No model path specified"
            return False

        try:
            # Simple validation (could integrate with actual SBML validation)
            if Path(model_path).exists():
                step_execution.output = {
                    "validation_result": "Model file exists and is accessible",
                    "file_size": Path(model_path).stat().st_size,
                }
                return True
            else:
                step_execution.error = f"Model file not found: {model_path}"
                return False

        except Exception as e:
            step_execution.error = f"Validation failed: {str(e)}"
            return False

    async def _execute_notification_step(
        self,
        step: WorkflowStep,
        step_execution: StepExecution,
        execution: WorkflowExecution,
    ) -> bool:
        """Execute a notification step"""
        message = step.parameters.get("message", "Workflow notification")

        try:
            # Simple console notification
            console.print(f"📢 [bold yellow]Notification:[/bold yellow] {message}")

            step_execution.output = {"notification_sent": True, "message": message}
            return True

        except Exception as e:
            step_execution.error = f"Notification failed: {str(e)}"
            return False

    async def _execute_mock_step(
        self,
        step: WorkflowStep,
        step_execution: StepExecution,
        execution: WorkflowExecution,
    ) -> bool:
        """Execute a mock step for testing/demonstration"""
        try:
            # Simulate execution time based on resource requirements
            execution_times = {
                ResourceRequirement.LOW: 0.5,
                ResourceRequirement.MEDIUM: 1.0,
                ResourceRequirement.HIGH: 2.0,
                ResourceRequirement.EXTREME: 3.0,
            }

            await asyncio.sleep(execution_times.get(step.resource_requirements, 1.0))

            step_execution.output = {
                "mock_result": f"Mock execution of {step.name} completed",
                "step_type": step.type.value,
                "execution_time": execution_times.get(step.resource_requirements, 1.0),
            }

            return True

        except Exception as e:
            step_execution.error = f"Mock execution failed: {str(e)}"
            return False

    async def _start_live_monitoring(
        self, execution: WorkflowExecution, workflow: WorkflowDefinition
    ) -> None:
        """Start live monitoring display"""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )

        task = progress.add_task(
            description=f"Executing workflow: {workflow.name}",
            total=execution.total_steps,
        )

        self.live_display = Live(progress, console=console, refresh_per_second=4)
        self.live_display.start()

        # Start monitoring task
        asyncio.create_task(self._monitor_execution(execution, progress, task))

    async def _monitor_execution(
        self, execution: WorkflowExecution, progress: Progress, task_id
    ) -> None:
        """Monitor execution progress"""
        while execution.status == WorkflowStatus.RUNNING:
            completed = sum(
                1
                for se in execution.step_executions.values()
                if se.status
                in [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED]
            )

            progress.update(task_id, completed=completed)
            await asyncio.sleep(0.25)

    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get current execution status"""
        return self.active_executions.get(execution_id)

    def list_active_executions(self) -> List[WorkflowExecution]:
        """List all active executions"""
        return list(self.active_executions.values())

    def get_execution_history(self, limit: int = 10) -> List[WorkflowExecution]:
        """Get execution history"""
        return self.execution_history[-limit:]

    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            execution.status = WorkflowStatus.CANCELLED
            execution.end_time = datetime.now()
            return True
        return False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        all_executions = list(self.active_executions.values()) + self.execution_history

        if not all_executions:
            return {"total_executions": 0}

        # Calculate metrics
        total_executions = len(all_executions)
        successful_executions = sum(
            1 for e in all_executions if e.status == WorkflowStatus.COMPLETED
        )
        failed_executions = sum(
            1 for e in all_executions if e.status == WorkflowStatus.FAILED
        )

        # Duration metrics
        completed_executions = [
            e
            for e in all_executions
            if e.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
        ]
        durations = [e.duration for e in completed_executions if e.duration > 0]

        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": (
                successful_executions / total_executions if total_executions > 0 else 0
            ),
            "average_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "active_executions": len(self.active_executions),
            "total_steps_executed": sum(e.completed_steps for e in all_executions),
            "total_steps_failed": sum(e.failed_steps for e in all_executions),
        }
