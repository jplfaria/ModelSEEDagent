"""
Advanced Workflow Scheduler

Provides intelligent workflow orchestration with priority-based execution,
resource management, and advanced scheduling strategies.
"""

import asyncio
import heapq
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .workflow_definition import WorkflowDefinition
from .workflow_engine import WorkflowEngine, WorkflowResult, WorkflowStatus

console = Console()


class SchedulingStrategy(Enum):
    """Workflow scheduling strategies"""

    FIFO = "fifo"  # First In, First Out
    PRIORITY = "priority"  # Priority-based scheduling
    SHORTEST_JOB_FIRST = "sjf"  # Shortest estimated duration first
    ROUND_ROBIN = "round_robin"  # Round-robin scheduling
    RESOURCE_AWARE = "resource"  # Resource-aware scheduling
    DEADLINE_DRIVEN = "deadline"  # Deadline-driven scheduling


class TaskPriority(Enum):
    """Task priority levels"""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Scheduled task status"""

    QUEUED = "queued"
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ScheduledTask:
    """A scheduled workflow task"""

    id: str
    workflow: WorkflowDefinition
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.QUEUED

    # Scheduling metadata
    submit_time: datetime = field(default_factory=datetime.now)
    schedule_time: Optional[datetime] = None  # When to start
    deadline: Optional[datetime] = None  # When must be completed
    estimated_duration: Optional[int] = None  # Estimated duration in seconds

    # Execution tracking
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_id: Optional[str] = None
    result: Optional[WorkflowResult] = None
    error: Optional[str] = None

    # Resource requirements
    required_cpu: int = 1
    required_memory: int = 512  # MB
    max_retries: int = 3
    retry_count: int = 0

    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    dependent_tasks: List[str] = field(default_factory=list)

    # Callbacks
    on_complete: Optional[Callable] = None
    on_failure: Optional[Callable] = None

    @property
    def wait_time(self) -> float:
        """Time spent waiting in queue"""
        if self.start_time:
            return (self.start_time - self.submit_time).total_seconds()
        else:
            return (datetime.now() - self.submit_time).total_seconds()

    @property
    def execution_time(self) -> float:
        """Time spent executing"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0

    @property
    def is_overdue(self) -> bool:
        """Check if task is past deadline"""
        if self.deadline:
            return datetime.now() > self.deadline
        return False

    def __lt__(self, other) -> bool:
        """For priority queue ordering"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        # If same priority, use estimated duration (shortest first)
        if self.estimated_duration and other.estimated_duration:
            return self.estimated_duration < other.estimated_duration
        # If no duration estimates, use submit time (FIFO)
        return self.submit_time < other.submit_time


@dataclass
class SchedulerStats:
    """Scheduler performance statistics"""

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0

    total_wait_time: float = 0.0
    total_execution_time: float = 0.0

    average_wait_time: float = 0.0
    average_execution_time: float = 0.0

    throughput: float = 0.0  # tasks per hour

    # Resource utilization
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0


class AdvancedScheduler:
    """Advanced workflow scheduler with intelligent orchestration"""

    def __init__(
        self,
        max_concurrent_workflows: int = 4,
        strategy: SchedulingStrategy = SchedulingStrategy.PRIORITY,
    ):

        self.max_concurrent_workflows = max_concurrent_workflows
        self.strategy = strategy
        self.workflow_engine = WorkflowEngine()

        # Task management
        self.task_queue: List[ScheduledTask] = []
        self.running_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: List[ScheduledTask] = []
        self.task_history: Dict[str, ScheduledTask] = {}

        # Resource tracking
        self.available_cpu = 8
        self.available_memory = 8192  # MB
        self.used_cpu = 0
        self.used_memory = 0

        # Scheduler state
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Statistics
        self.stats = SchedulerStats()
        self.start_time = datetime.now()

        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            "task_queued": [],
            "task_started": [],
            "task_completed": [],
            "task_failed": [],
            "scheduler_started": [],
            "scheduler_stopped": [],
        }

    def schedule_workflow(
        self,
        workflow: WorkflowDefinition,
        parameters: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        schedule_time: Optional[datetime] = None,
        deadline: Optional[datetime] = None,
    ) -> str:
        """Schedule a workflow for execution"""

        task_id = str(uuid.uuid4())[:8]

        task = ScheduledTask(
            id=task_id,
            workflow=workflow,
            parameters=parameters or {},
            priority=priority,
            schedule_time=schedule_time,
            deadline=deadline,
            estimated_duration=workflow.estimate_total_duration(),
        )

        # Add to queue based on strategy
        if self.strategy == SchedulingStrategy.PRIORITY:
            heapq.heappush(self.task_queue, task)
        else:
            self.task_queue.append(task)

        self.task_history[task_id] = task
        self.stats.total_tasks += 1

        # Emit event
        self._emit_event("task_queued", {"task": task})

        console.print(
            f"✅ Scheduled workflow '{workflow.name}' (Task ID: {task_id}, Priority: {priority.name})"
        )

        return task_id

    def schedule_recurring_workflow(
        self,
        workflow: WorkflowDefinition,
        parameters: Optional[Dict[str, Any]] = None,
        interval: timedelta = timedelta(hours=1),
        max_occurrences: Optional[int] = None,
    ) -> List[str]:
        """Schedule a workflow to run at regular intervals"""

        task_ids = []
        current_time = datetime.now()
        occurrence_count = 0

        while max_occurrences is None or occurrence_count < max_occurrences:
            schedule_time = current_time + (interval * occurrence_count)

            task_id = self.schedule_workflow(
                workflow=workflow, parameters=parameters, schedule_time=schedule_time
            )

            task_ids.append(task_id)
            occurrence_count += 1

            # For demonstration, limit to reasonable number
            if occurrence_count >= 10:
                break

        console.print(
            f"✅ Scheduled {len(task_ids)} recurring instances of '{workflow.name}'"
        )
        return task_ids

    def start_scheduler(self) -> None:
        """Start the scheduler"""
        if self.is_running:
            console.print("[yellow]Scheduler is already running[/yellow]")
            return

        self.is_running = True
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True
        )
        self.scheduler_thread.start()

        self._emit_event("scheduler_started", {"timestamp": datetime.now()})
        console.print("[green]🚀 Scheduler started[/green]")

    def stop_scheduler(self) -> None:
        """Stop the scheduler"""
        if not self.is_running:
            return

        self.is_running = False
        self.stop_event.set()

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)

        self._emit_event("scheduler_stopped", {"timestamp": datetime.now()})
        console.print("[red]⏹️ Scheduler stopped[/red]")

    def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        while self.is_running and not self.stop_event.is_set():
            try:
                self._process_queue()
                self._check_running_tasks()
                self._update_statistics()

                # Sleep briefly before next iteration
                self.stop_event.wait(1.0)

            except Exception as e:
                console.print(f"[red]Scheduler error: {e}[/red]")
                self.stop_event.wait(5.0)

    def _process_queue(self) -> None:
        """Process the task queue"""
        current_time = datetime.now()

        # Check if we can start more tasks
        while (
            len(self.running_tasks) < self.max_concurrent_workflows and self.task_queue
        ):

            # Get next task based on strategy
            task = self._get_next_task()
            if not task:
                break

            # Check if task should be started now
            if task.schedule_time and task.schedule_time > current_time:
                # Put task back and wait
                if self.strategy == SchedulingStrategy.PRIORITY:
                    heapq.heappush(self.task_queue, task)
                else:
                    self.task_queue.insert(0, task)
                break

            # Check resource availability
            if not self._has_sufficient_resources(task):
                # Put task back and try later
                if self.strategy == SchedulingStrategy.PRIORITY:
                    heapq.heappush(self.task_queue, task)
                else:
                    self.task_queue.append(task)
                break

            # Check dependencies
            if not self._dependencies_satisfied(task):
                # Put task back
                if self.strategy == SchedulingStrategy.PRIORITY:
                    heapq.heappush(self.task_queue, task)
                else:
                    self.task_queue.append(task)
                break

            # Start the task
            self._start_task(task)

    def _get_next_task(self) -> Optional[ScheduledTask]:
        """Get the next task based on scheduling strategy"""
        if not self.task_queue:
            return None

        if self.strategy == SchedulingStrategy.FIFO:
            return self.task_queue.pop(0)
        elif self.strategy == SchedulingStrategy.PRIORITY:
            return heapq.heappop(self.task_queue)
        elif self.strategy == SchedulingStrategy.SHORTEST_JOB_FIRST:
            # Find task with shortest estimated duration
            if self.task_queue:
                shortest_task = min(
                    self.task_queue, key=lambda t: t.estimated_duration or 9999
                )
                self.task_queue.remove(shortest_task)
                return shortest_task
        elif self.strategy == SchedulingStrategy.DEADLINE_DRIVEN:
            # Find task with earliest deadline
            deadline_tasks = [t for t in self.task_queue if t.deadline]
            if deadline_tasks:
                earliest_task = min(deadline_tasks, key=lambda t: t.deadline)
                self.task_queue.remove(earliest_task)
                return earliest_task
            elif self.task_queue:
                return self.task_queue.pop(0)
        else:
            # Default to FIFO
            return self.task_queue.pop(0)

        return None

    def _has_sufficient_resources(self, task: ScheduledTask) -> bool:
        """Check if sufficient resources are available for task"""
        return (
            self.used_cpu + task.required_cpu <= self.available_cpu
            and self.used_memory + task.required_memory <= self.available_memory
        )

    def _dependencies_satisfied(self, task: ScheduledTask) -> bool:
        """Check if all task dependencies are satisfied"""
        for dep_id in task.dependencies:
            dep_task = self.task_history.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True

    def _start_task(self, task: ScheduledTask) -> None:
        """Start executing a task"""
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()

        # Allocate resources
        self.used_cpu += task.required_cpu
        self.used_memory += task.required_memory

        # Add to running tasks
        self.running_tasks[task.id] = task

        # Start execution in a separate thread to handle async workflow execution
        def run_task():
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Run the async task
                loop.run_until_complete(self._execute_task(task))

            except Exception as e:
                console.print(f"[red]Task execution thread error: {e}[/red]")
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.end_time = datetime.now()

                # Free resources
                self.used_cpu -= task.required_cpu
                self.used_memory -= task.required_memory

                # Remove from running tasks
                if task.id in self.running_tasks:
                    del self.running_tasks[task.id]

                # Add to completed tasks
                self.completed_tasks.append(task)

            finally:
                loop.close()

        # Start the task in a thread
        task_thread = threading.Thread(target=run_task, daemon=True)
        task_thread.start()

        self._emit_event("task_started", {"task": task})
        console.print(f"🚀 Started task: {task.workflow.name} (ID: {task.id})")

    async def _execute_task(self, task: ScheduledTask) -> None:
        """Execute a task asynchronously"""
        try:
            # Execute workflow
            result = await self.workflow_engine.execute_workflow(
                task.workflow, task.parameters, enable_monitoring=False
            )

            task.result = result
            task.execution_id = result.execution_id

            if result.success:
                task.status = TaskStatus.COMPLETED
                self.stats.completed_tasks += 1
                self._emit_event("task_completed", {"task": task, "result": result})

                # Execute success callback
                if task.on_complete:
                    try:
                        task.on_complete(task, result)
                    except Exception as e:
                        console.print(f"[yellow]Callback error: {e}[/yellow]")
            else:
                task.status = TaskStatus.FAILED
                task.error = result.message
                self.stats.failed_tasks += 1
                self._emit_event("task_failed", {"task": task, "result": result})

                # Execute failure callback
                if task.on_failure:
                    try:
                        task.on_failure(task, result)
                    except Exception as e:
                        console.print(f"[yellow]Callback error: {e}[/yellow]")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.stats.failed_tasks += 1
            self._emit_event("task_failed", {"task": task, "error": str(e)})

        finally:
            task.end_time = datetime.now()

            # Free resources
            self.used_cpu -= task.required_cpu
            self.used_memory -= task.required_memory

            # Remove from running tasks
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]

            # Add to completed tasks
            self.completed_tasks.append(task)

    def _check_running_tasks(self) -> None:
        """Check running tasks for timeouts and other issues"""
        current_time = datetime.now()

        for task in list(self.running_tasks.values()):
            # Check for deadline violations
            if task.deadline and current_time > task.deadline:
                console.print(f"[red]⚠️ Task {task.id} exceeded deadline[/red]")
                # Could implement deadline handling here

    def _update_statistics(self) -> None:
        """Update scheduler statistics"""
        if self.stats.completed_tasks > 0:
            # Calculate average times
            completed_tasks = [t for t in self.completed_tasks if t.execution_time > 0]
            if completed_tasks:
                self.stats.average_execution_time = sum(
                    t.execution_time for t in completed_tasks
                ) / len(completed_tasks)
                self.stats.average_wait_time = sum(
                    t.wait_time for t in completed_tasks
                ) / len(completed_tasks)

            # Calculate throughput (tasks per hour)
            elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            if elapsed_hours > 0:
                self.stats.throughput = self.stats.completed_tasks / elapsed_hours

        # Calculate resource utilization
        if self.available_cpu > 0:
            self.stats.cpu_utilization = (self.used_cpu / self.available_cpu) * 100
        if self.available_memory > 0:
            self.stats.memory_utilization = (
                self.used_memory / self.available_memory
            ) * 100

    def _emit_event(self, event: str, data: Dict[str, Any]) -> None:
        """Emit an event to registered handlers"""
        for handler in self.event_handlers.get(event, []):
            try:
                handler(data)
            except Exception as e:
                console.print(f"[red]❌ Event handler error for {event}: {e}[/red]")

    def register_event_handler(self, event: str, handler: Callable) -> None:
        """Register an event handler"""
        if event in self.event_handlers:
            self.event_handlers[event].append(handler)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled or running task"""
        # Check if task is in queue
        for i, task in enumerate(self.task_queue):
            if task.id == task_id:
                task.status = TaskStatus.CANCELLED
                self.task_queue.pop(i)
                self.stats.cancelled_tasks += 1
                console.print(f"✅ Cancelled queued task: {task_id}")
                return True

        # Check if task is running
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            # Note: Actual workflow cancellation would need to be implemented
            console.print(f"✅ Cancelled running task: {task_id}")
            return True

        return False

    def get_task_status(self, task_id: str) -> Optional[ScheduledTask]:
        """Get the status of a task"""
        return self.task_history.get(task_id)

    def list_queued_tasks(self) -> List[ScheduledTask]:
        """List all queued tasks"""
        return [task for task in self.task_queue]

    def list_running_tasks(self) -> List[ScheduledTask]:
        """List all running tasks"""
        return list(self.running_tasks.values())

    def list_completed_tasks(self, limit: int = 10) -> List[ScheduledTask]:
        """List completed tasks"""
        return self.completed_tasks[-limit:]

    def get_scheduler_stats(self) -> SchedulerStats:
        """Get comprehensive scheduler statistics"""
        self._update_statistics()
        return self.stats

    def display_scheduler_status(self) -> None:
        """Display comprehensive scheduler status"""
        # Status overview
        status_table = Table(show_header=False, box=box.SIMPLE)
        status_table.add_column("Metric", style="bold cyan")
        status_table.add_column("Value", style="bold white")

        status_table.add_row(
            "Status", "🟢 Running" if self.is_running else "🔴 Stopped"
        )
        status_table.add_row("Strategy", self.strategy.value.title())
        status_table.add_row("Queued Tasks", str(len(self.task_queue)))
        status_table.add_row("Running Tasks", str(len(self.running_tasks)))
        status_table.add_row("Completed Tasks", str(self.stats.completed_tasks))
        status_table.add_row("Failed Tasks", str(self.stats.failed_tasks))
        status_table.add_row(
            "CPU Usage",
            f"{self.used_cpu}/{self.available_cpu} ({self.stats.cpu_utilization:.1f}%)",
        )
        status_table.add_row(
            "Memory Usage",
            f"{self.used_memory}/{self.available_memory} MB ({self.stats.memory_utilization:.1f}%)",
        )
        status_table.add_row("Throughput", f"{self.stats.throughput:.2f} tasks/hour")

        console.print(
            Panel(
                status_table,
                title="[bold blue]📊 Scheduler Status[/bold blue]",
                border_style="blue",
            )
        )

        # Running tasks
        if self.running_tasks:
            console.print("\n[bold yellow]🏃 Running Tasks:[/bold yellow]")

            running_table = Table(box=box.ROUNDED)
            running_table.add_column("Task ID", style="cyan")
            running_table.add_column("Workflow", style="white")
            running_table.add_column("Priority", style="yellow")
            running_table.add_column("Runtime", style="green")
            running_table.add_column("CPU", style="blue")
            running_table.add_column("Memory", style="magenta")

            for task in self.running_tasks.values():
                running_table.add_row(
                    task.id,
                    task.workflow.name,
                    task.priority.name,
                    f"{task.execution_time:.1f}s",
                    str(task.required_cpu),
                    f"{task.required_memory}MB",
                )

            console.print(running_table)

        # Queued tasks (first 5)
        if self.task_queue:
            console.print("\n[bold yellow]⏳ Queued Tasks (Next 5):[/bold yellow]")

            queue_table = Table(box=box.ROUNDED)
            queue_table.add_column("Task ID", style="cyan")
            queue_table.add_column("Workflow", style="white")
            queue_table.add_column("Priority", style="yellow")
            queue_table.add_column("Wait Time", style="red")
            queue_table.add_column("Scheduled", style="blue")

            for task in self.task_queue[:5]:
                scheduled_str = (
                    "Now"
                    if not task.schedule_time
                    else task.schedule_time.strftime("%H:%M:%S")
                )

                queue_table.add_row(
                    task.id,
                    task.workflow.name,
                    task.priority.name,
                    f"{task.wait_time:.1f}s",
                    scheduled_str,
                )

            console.print(queue_table)

            if len(self.task_queue) > 5:
                console.print(
                    f"[dim]... and {len(self.task_queue) - 5} more tasks in queue[/dim]"
                )
