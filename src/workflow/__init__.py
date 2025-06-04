"""
Advanced Workflow Automation for ModelSEEDagent

This package provides intelligent workflow orchestration, automated batch processing,
and advanced scheduling capabilities for metabolic modeling analyses.
"""

from .batch_processor import BatchJob, BatchProcessor, BatchResult
from .monitoring import AlertManager, NotificationChannel, WorkflowMonitor
from .scheduler import AdvancedScheduler, ScheduledTask, SchedulingStrategy
from .template_library import TemplateLibrary, WorkflowTemplate
from .workflow_definition import WorkflowDefinition, WorkflowStep, WorkflowTemplate
from .workflow_engine import WorkflowEngine, WorkflowResult, WorkflowStatus
from .workflow_optimizer import OptimizationResult, WorkflowOptimizer

__all__ = [
    "WorkflowEngine",
    "WorkflowStatus",
    "WorkflowResult",
    "WorkflowDefinition",
    "WorkflowStep",
    "WorkflowTemplate",
    "BatchProcessor",
    "BatchJob",
    "BatchResult",
    "AdvancedScheduler",
    "ScheduledTask",
    "SchedulingStrategy",
    "WorkflowOptimizer",
    "OptimizationResult",
    "TemplateLibrary",
    "WorkflowMonitor",
    "AlertManager",
    "NotificationChannel",
]
