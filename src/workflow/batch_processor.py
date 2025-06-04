"""
Batch Processing Engine for Advanced Workflow Automation

Provides intelligent batch processing capabilities for multiple models
and workflows with resource management and comprehensive monitoring.
"""

import asyncio
import json
import multiprocessing as mp
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from .workflow_definition import WorkflowDefinition
from .workflow_engine import WorkflowEngine, WorkflowResult, WorkflowStatus

console = Console()


class BatchJobStatus(Enum):
    """Batch job execution status"""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class BatchProcessingStrategy(Enum):
    """Batch processing strategies"""

    PARALLEL = "parallel"  # Execute all jobs in parallel
    SEQUENTIAL = "sequential"  # Execute jobs one by one
    ADAPTIVE = "adaptive"  # Adapt based on resource availability
    PRIORITY_BASED = "priority"  # Execute high priority jobs first
    RESOURCE_AWARE = "resource"  # Consider resource requirements


@dataclass
class BatchJob:
    """Individual job in a batch processing operation"""

    id: str
    workflow: WorkflowDefinition
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10, higher is more priority
    status: BatchJobStatus = BatchJobStatus.PENDING

    # Execution tracking
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_id: Optional[str] = None
    result: Optional[WorkflowResult] = None
    error: Optional[str] = None
    retry_count: int = 0

    # Resource requirements
    estimated_duration: Optional[int] = None  # seconds
    max_memory: Optional[int] = None  # MB
    required_cpu: int = 1

    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    dependent_jobs: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get job duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0

    @property
    def success(self) -> bool:
        """Check if job completed successfully"""
        return (
            self.status == BatchJobStatus.COMPLETED
            and self.result
            and self.result.success
        )


@dataclass
class BatchExecution:
    """Batch execution context and tracking"""

    batch_id: str
    name: str
    jobs: List[BatchJob] = field(default_factory=list)
    strategy: BatchProcessingStrategy = BatchProcessingStrategy.PARALLEL
    max_concurrent_jobs: int = 4

    # Status tracking
    status: BatchJobStatus = BatchJobStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Progress tracking
    total_jobs: int = 0
    pending_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0

    # Resource tracking
    current_cpu_usage: int = 0
    current_memory_usage: int = 0
    max_cpu_limit: int = mp.cpu_count()
    max_memory_limit: int = 8192  # MB

    @property
    def progress(self) -> float:
        """Get overall batch progress percentage"""
        if self.total_jobs == 0:
            return 0.0
        return (
            (self.completed_jobs + self.failed_jobs + self.cancelled_jobs)
            / self.total_jobs
        ) * 100

    @property
    def duration(self) -> float:
        """Get total batch duration"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0

    @property
    def success_rate(self) -> float:
        """Get success rate percentage"""
        total_finished = self.completed_jobs + self.failed_jobs
        if total_finished == 0:
            return 0.0
        return (self.completed_jobs / total_finished) * 100


@dataclass
class BatchResult:
    """Final batch processing result"""

    batch_id: str
    batch_name: str
    success: bool
    status: BatchJobStatus
    message: str
    duration: float

    # Job statistics
    total_jobs: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0

    # Performance metrics
    average_job_time: float = 0.0
    fastest_job_time: float = 0.0
    slowest_job_time: float = 0.0

    # Resource utilization
    peak_cpu_usage: int = 0
    peak_memory_usage: int = 0

    # Results
    job_results: List[WorkflowResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchProcessor:
    """Advanced batch processing engine"""

    def __init__(self, max_concurrent_batches: int = 2):
        self.max_concurrent_batches = max_concurrent_batches
        self.workflow_engine = WorkflowEngine()

        # Batch tracking
        self.active_batches: Dict[str, BatchExecution] = {}
        self.batch_history: List[BatchExecution] = []

        # Resource monitoring
        self.global_cpu_usage = 0
        self.global_memory_usage = 0
        self.max_global_cpu = mp.cpu_count()
        self.max_global_memory = 16384  # MB

        # Live monitoring
        self.live_display: Optional[Live] = None
        self.monitoring_enabled: bool = True

    def create_batch(
        self,
        name: str,
        workflows: List[WorkflowDefinition],
        parameters_list: Optional[List[Dict[str, Any]]] = None,
        strategy: BatchProcessingStrategy = BatchProcessingStrategy.PARALLEL,
    ) -> str:
        """Create a new batch processing job"""

        batch_id = str(uuid.uuid4())[:8]

        # Create jobs
        jobs = []
        for i, workflow in enumerate(workflows):
            parameters = (
                parameters_list[i]
                if parameters_list and i < len(parameters_list)
                else {}
            )

            job = BatchJob(
                id=f"{batch_id}_job_{i+1:03d}",
                workflow=workflow,
                parameters=parameters,
                estimated_duration=workflow.estimate_total_duration(),
            )
            jobs.append(job)

        # Create batch execution
        batch = BatchExecution(
            batch_id=batch_id,
            name=name,
            jobs=jobs,
            strategy=strategy,
            total_jobs=len(jobs),
            pending_jobs=len(jobs),
        )

        self.active_batches[batch_id] = batch
        console.print(
            f"✅ Created batch '{name}' with {len(jobs)} jobs (ID: {batch_id})"
        )

        return batch_id

    def add_model_batch(
        self,
        name: str,
        model_files: List[Union[str, Path]],
        workflow_template: WorkflowDefinition,
        strategy: BatchProcessingStrategy = BatchProcessingStrategy.PARALLEL,
    ) -> str:
        """Create a batch for processing multiple models with the same workflow"""

        workflows = []
        parameters_list = []

        for model_file in model_files:
            # Create workflow instance for each model
            workflow = WorkflowDefinition(
                id=str(uuid.uuid4())[:8],
                name=f"{workflow_template.name}_{Path(model_file).stem}",
                description=f"Processing {Path(model_file).name}",
                steps=workflow_template.steps.copy(),
                parameters=workflow_template.parameters.copy(),
            )

            workflows.append(workflow)
            parameters_list.append({"model_path": str(model_file)})

        return self.create_batch(name, workflows, parameters_list, strategy)

    async def execute_batch(
        self, batch_id: str, enable_monitoring: bool = True
    ) -> BatchResult:
        """Execute a batch processing job"""

        if batch_id not in self.active_batches:
            raise ValueError(f"Batch {batch_id} not found")

        batch = self.active_batches[batch_id]
        batch.status = BatchJobStatus.RUNNING
        batch.start_time = datetime.now()

        try:
            # Start monitoring if enabled
            if enable_monitoring and self.monitoring_enabled:
                await self._start_batch_monitoring(batch)

            # Execute based on strategy
            if batch.strategy == BatchProcessingStrategy.PARALLEL:
                result = await self._execute_parallel_batch(batch)
            elif batch.strategy == BatchProcessingStrategy.SEQUENTIAL:
                result = await self._execute_sequential_batch(batch)
            elif batch.strategy == BatchProcessingStrategy.ADAPTIVE:
                result = await self._execute_adaptive_batch(batch)
            elif batch.strategy == BatchProcessingStrategy.PRIORITY_BASED:
                result = await self._execute_priority_batch(batch)
            elif batch.strategy == BatchProcessingStrategy.RESOURCE_AWARE:
                result = await self._execute_resource_aware_batch(batch)
            else:
                result = await self._execute_parallel_batch(batch)

            # Stop monitoring
            if self.live_display:
                self.live_display.stop()
                self.live_display = None

            return result

        except Exception as e:
            batch.status = BatchJobStatus.FAILED
            batch.end_time = datetime.now()

            return BatchResult(
                batch_id=batch_id,
                batch_name=batch.name,
                success=False,
                status=BatchJobStatus.FAILED,
                message=f"Batch execution failed: {str(e)}",
                duration=batch.duration,
            )
        finally:
            # Clean up
            if batch_id in self.active_batches:
                self.batch_history.append(self.active_batches[batch_id])
                del self.active_batches[batch_id]

    async def _execute_parallel_batch(self, batch: BatchExecution) -> BatchResult:
        """Execute all jobs in parallel (resource-limited)"""

        # Create semaphore to limit concurrent jobs
        semaphore = asyncio.Semaphore(batch.max_concurrent_jobs)

        async def execute_job(job: BatchJob) -> None:
            async with semaphore:
                await self._execute_single_job(job, batch)

        # Create tasks for all jobs
        tasks = [asyncio.create_task(execute_job(job)) for job in batch.jobs]

        # Wait for all jobs to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        return self._create_batch_result(batch)

    async def _execute_sequential_batch(self, batch: BatchExecution) -> BatchResult:
        """Execute jobs one by one"""

        for job in batch.jobs:
            if batch.status == BatchJobStatus.CANCELLED:
                break
            await self._execute_single_job(job, batch)

        return self._create_batch_result(batch)

    async def _execute_adaptive_batch(self, batch: BatchExecution) -> BatchResult:
        """Execute with adaptive concurrency based on system resources"""

        # Dynamically adjust concurrency based on resource usage
        max_concurrent = min(
            batch.max_concurrent_jobs,
            max(1, (self.max_global_cpu - self.global_cpu_usage) // 2),
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_job(job: BatchJob) -> None:
            async with semaphore:
                await self._execute_single_job(job, batch)

        tasks = [asyncio.create_task(execute_job(job)) for job in batch.jobs]
        await asyncio.gather(*tasks, return_exceptions=True)

        return self._create_batch_result(batch)

    async def _execute_priority_batch(self, batch: BatchExecution) -> BatchResult:
        """Execute jobs in priority order"""

        # Sort jobs by priority (highest first)
        sorted_jobs = sorted(batch.jobs, key=lambda j: j.priority, reverse=True)

        # Group jobs by priority level for parallel execution within priority
        priority_groups = {}
        for job in sorted_jobs:
            if job.priority not in priority_groups:
                priority_groups[job.priority] = []
            priority_groups[job.priority].append(job)

        # Execute priority groups sequentially, jobs within group in parallel
        for priority in sorted(priority_groups.keys(), reverse=True):
            if batch.status == BatchJobStatus.CANCELLED:
                break

            jobs_in_group = priority_groups[priority]
            semaphore = asyncio.Semaphore(
                min(len(jobs_in_group), batch.max_concurrent_jobs)
            )

            async def execute_job(job: BatchJob) -> None:
                async with semaphore:
                    await self._execute_single_job(job, batch)

            tasks = [asyncio.create_task(execute_job(job)) for job in jobs_in_group]
            await asyncio.gather(*tasks, return_exceptions=True)

        return self._create_batch_result(batch)

    async def _execute_resource_aware_batch(self, batch: BatchExecution) -> BatchResult:
        """Execute with intelligent resource management"""

        resource_queue = asyncio.Queue()

        # Add all jobs to queue
        for job in batch.jobs:
            await resource_queue.put(job)

        # Worker that considers resource requirements
        async def resource_worker():
            while not resource_queue.empty():
                try:
                    job = await asyncio.wait_for(resource_queue.get(), timeout=1.0)

                    # Check if we have enough resources
                    if (
                        self.global_cpu_usage + job.required_cpu <= self.max_global_cpu
                        and self.global_memory_usage + (job.max_memory or 512)
                        <= self.max_global_memory
                    ):

                        await self._execute_single_job(job, batch)
                    else:
                        # Put job back in queue and wait
                        await resource_queue.put(job)
                        await asyncio.sleep(1.0)

                except asyncio.TimeoutError:
                    break

        # Start workers
        num_workers = min(batch.max_concurrent_jobs, 4)
        workers = [asyncio.create_task(resource_worker()) for _ in range(num_workers)]

        # Wait for all workers to complete
        await asyncio.gather(*workers, return_exceptions=True)

        return self._create_batch_result(batch)

    async def _execute_single_job(self, job: BatchJob, batch: BatchExecution) -> None:
        """Execute a single job within a batch"""

        job.status = BatchJobStatus.RUNNING
        job.start_time = datetime.now()

        # Update batch counters
        batch.pending_jobs -= 1
        batch.running_jobs += 1

        # Update resource usage
        self.global_cpu_usage += job.required_cpu
        self.global_memory_usage += job.max_memory or 512

        try:
            # Execute workflow
            result = await self.workflow_engine.execute_workflow(
                job.workflow,
                job.parameters,
                enable_monitoring=False,  # Batch monitoring handles this
            )

            job.result = result
            job.execution_id = result.execution_id

            if result.success:
                job.status = BatchJobStatus.COMPLETED
                batch.completed_jobs += 1
            else:
                job.status = BatchJobStatus.FAILED
                job.error = result.message
                batch.failed_jobs += 1

        except Exception as e:
            job.status = BatchJobStatus.FAILED
            job.error = str(e)
            batch.failed_jobs += 1

        finally:
            job.end_time = datetime.now()
            batch.running_jobs -= 1

            # Update resource usage
            self.global_cpu_usage -= job.required_cpu
            self.global_memory_usage -= job.max_memory or 512

    def _create_batch_result(self, batch: BatchExecution) -> BatchResult:
        """Create final batch result"""

        batch.status = (
            BatchJobStatus.COMPLETED
            if batch.failed_jobs == 0
            else BatchJobStatus.FAILED
        )
        batch.end_time = datetime.now()

        # Collect job results
        job_results = [job.result for job in batch.jobs if job.result]

        # Calculate performance metrics
        job_times = [job.duration for job in batch.jobs if job.duration > 0]

        result = BatchResult(
            batch_id=batch.batch_id,
            batch_name=batch.name,
            success=batch.failed_jobs == 0,
            status=batch.status,
            message=f"Batch completed: {batch.completed_jobs} successful, {batch.failed_jobs} failed",
            duration=batch.duration,
            total_jobs=batch.total_jobs,
            successful_jobs=batch.completed_jobs,
            failed_jobs=batch.failed_jobs,
            cancelled_jobs=batch.cancelled_jobs,
            job_results=job_results,
        )

        if job_times:
            result.average_job_time = sum(job_times) / len(job_times)
            result.fastest_job_time = min(job_times)
            result.slowest_job_time = max(job_times)

        return result

    async def _start_batch_monitoring(self, batch: BatchExecution) -> None:
        """Start live batch monitoring display"""

        # Create layout
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=5),
        )

        # Header with batch info
        header_text = Text(
            f"🚀 Batch Execution: {batch.name} (ID: {batch.batch_id})",
            style="bold blue",
        )
        layout["header"].update(Panel(header_text, border_style="blue"))

        # Body with progress
        progress = Progress(
            TextColumn("[bold blue]Job Progress"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )

        main_task = progress.add_task(
            description="Overall Progress", total=batch.total_jobs
        )

        layout["body"].update(progress)

        # Footer with statistics
        layout["footer"].update(Panel("Batch Statistics", border_style="green"))

        self.live_display = Live(layout, console=console, refresh_per_second=2)
        self.live_display.start()

        # Start monitoring task
        asyncio.create_task(self._monitor_batch_execution(batch, progress, main_task))

    async def _monitor_batch_execution(
        self, batch: BatchExecution, progress: Progress, task_id
    ) -> None:
        """Monitor batch execution progress"""
        while batch.status == BatchJobStatus.RUNNING:
            completed = batch.completed_jobs + batch.failed_jobs + batch.cancelled_jobs
            progress.update(task_id, completed=completed)
            await asyncio.sleep(0.5)

    def get_batch_status(self, batch_id: str) -> Optional[BatchExecution]:
        """Get current batch execution status"""
        return self.active_batches.get(batch_id)

    def list_active_batches(self) -> List[BatchExecution]:
        """List all active batch executions"""
        return list(self.active_batches.values())

    def get_batch_history(self, limit: int = 10) -> List[BatchExecution]:
        """Get batch execution history"""
        return self.batch_history[-limit:]

    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a running batch"""
        if batch_id in self.active_batches:
            batch = self.active_batches[batch_id]
            batch.status = BatchJobStatus.CANCELLED

            # Cancel pending jobs
            for job in batch.jobs:
                if job.status == BatchJobStatus.PENDING:
                    job.status = BatchJobStatus.CANCELLED
                    batch.cancelled_jobs += 1
                    batch.pending_jobs -= 1

            return True
        return False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive batch processing metrics"""
        all_batches = list(self.active_batches.values()) + self.batch_history

        if not all_batches:
            return {"total_batches": 0}

        # Calculate metrics
        total_batches = len(all_batches)
        successful_batches = sum(
            1 for b in all_batches if b.status == BatchJobStatus.COMPLETED
        )
        failed_batches = sum(
            1 for b in all_batches if b.status == BatchJobStatus.FAILED
        )

        # Job metrics
        all_jobs = []
        for batch in all_batches:
            all_jobs.extend(batch.jobs)

        total_jobs = len(all_jobs)
        successful_jobs = sum(1 for j in all_jobs if j.success)

        # Duration metrics
        completed_batches = [
            b
            for b in all_batches
            if b.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED]
        ]
        durations = [b.duration for b in completed_batches if b.duration > 0]

        return {
            "total_batches": total_batches,
            "successful_batches": successful_batches,
            "failed_batches": failed_batches,
            "batch_success_rate": (
                successful_batches / total_batches if total_batches > 0 else 0
            ),
            "total_jobs": total_jobs,
            "successful_jobs": successful_jobs,
            "job_success_rate": successful_jobs / total_jobs if total_jobs > 0 else 0,
            "average_batch_duration": (
                sum(durations) / len(durations) if durations else 0
            ),
            "active_batches": len(self.active_batches),
            "current_cpu_usage": self.global_cpu_usage,
            "current_memory_usage": self.global_memory_usage,
        }

    def display_batch_summary(self, batch_id: str) -> None:
        """Display comprehensive batch summary"""
        batch = self.active_batches.get(batch_id) or next(
            (b for b in self.batch_history if b.batch_id == batch_id), None
        )

        if not batch:
            console.print(f"[red]Batch {batch_id} not found[/red]")
            return

        # Batch overview
        overview_table = Table(show_header=False, box=box.SIMPLE)
        overview_table.add_column("Aspect", style="bold cyan")
        overview_table.add_column("Value", style="bold white")

        overview_table.add_row("Batch ID", batch.batch_id)
        overview_table.add_row("Name", batch.name)
        overview_table.add_row("Status", batch.status.value.title())
        overview_table.add_row("Strategy", batch.strategy.value.title())
        overview_table.add_row("Duration", f"{batch.duration:.1f}s")
        overview_table.add_row("Progress", f"{batch.progress:.1f}%")
        overview_table.add_row("Success Rate", f"{batch.success_rate:.1f}%")

        console.print(
            Panel(
                overview_table,
                title=f"[bold blue]📦 Batch Overview[/bold blue]",
                border_style="blue",
            )
        )

        # Job status table
        if batch.jobs:
            job_table = Table(title="🔧 Job Status", box=box.ROUNDED)
            job_table.add_column("Job ID", style="cyan")
            job_table.add_column("Workflow", style="white")
            job_table.add_column("Status", style="green")
            job_table.add_column("Duration", style="yellow")
            job_table.add_column("Priority", style="magenta")

            for job in batch.jobs[:10]:  # Show first 10 jobs
                status_color = {
                    BatchJobStatus.COMPLETED: "green",
                    BatchJobStatus.FAILED: "red",
                    BatchJobStatus.RUNNING: "blue",
                    BatchJobStatus.PENDING: "yellow",
                }.get(job.status, "white")

                job_table.add_row(
                    job.id,
                    job.workflow.name,
                    f"[{status_color}]{job.status.value.title()}[/{status_color}]",
                    f"{job.duration:.1f}s" if job.duration > 0 else "N/A",
                    str(job.priority),
                )

            console.print(job_table)

            if len(batch.jobs) > 10:
                console.print(f"[dim]... and {len(batch.jobs) - 10} more jobs[/dim]")
