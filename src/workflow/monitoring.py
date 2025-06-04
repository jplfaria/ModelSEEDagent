"""
Advanced Workflow Monitoring and Alerting System

Provides comprehensive monitoring, alerting, and notification capabilities
for workflow execution with real-time status tracking and alert management.
"""

import asyncio
import json
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

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

console = Console()


class AlertLevel(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationType(Enum):
    """Types of notifications"""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    FILE = "file"


@dataclass
class WorkflowAlert:
    """Workflow alert definition"""

    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "id": self.id,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "metadata": self.metadata,
        }


@dataclass
class NotificationChannel:
    """Notification channel configuration"""

    id: str
    type: NotificationType
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    alert_levels: List[AlertLevel] = field(default_factory=lambda: list(AlertLevel))

    def can_handle_alert(self, alert_level: AlertLevel) -> bool:
        """Check if channel can handle the alert level"""
        return self.enabled and alert_level in self.alert_levels


class AlertManager:
    """Advanced alert management system"""

    def __init__(self):
        self.alerts: List[WorkflowAlert] = []
        self.channels: Dict[str, NotificationChannel] = {}
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = {
            level: [] for level in AlertLevel
        }

        # Alert thresholds and rules
        self.alert_rules: Dict[str, Callable] = {}
        self.alert_history: List[WorkflowAlert] = []
        self.max_history_size = 1000

        # Background processing
        self.alert_queue = queue.Queue()
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_processing = threading.Event()

        # Built-in console channel
        self._setup_default_channels()

    def _setup_default_channels(self) -> None:
        """Setup default notification channels"""
        console_channel = NotificationChannel(
            id="console",
            type=NotificationType.CONSOLE,
            name="Console Output",
            alert_levels=list(AlertLevel),
        )
        self.add_channel(console_channel)

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel"""
        self.channels[channel.id] = channel
        console.print(f"✅ Added notification channel: [cyan]{channel.name}[/cyan]")

    def remove_channel(self, channel_id: str) -> bool:
        """Remove a notification channel"""
        if channel_id in self.channels:
            del self.channels[channel_id]
            console.print(f"✅ Removed notification channel: [cyan]{channel_id}[/cyan]")
            return True
        return False

    def create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        workflow_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowAlert:
        """Create and queue an alert"""

        alert = WorkflowAlert(
            id=str(uuid.uuid4())[:8],
            level=level,
            title=title,
            message=message,
            workflow_id=workflow_id,
            execution_id=execution_id,
            metadata=metadata or {},
        )

        # Add to queue for processing
        self.alert_queue.put(alert)

        # Also add to active alerts
        self.alerts.append(alert)

        return alert

    def start_processing(self) -> None:
        """Start background alert processing"""
        if self.processing_thread and self.processing_thread.is_alive():
            return

        self.stop_processing.clear()
        self.processing_thread = threading.Thread(
            target=self._process_alerts, daemon=True
        )
        self.processing_thread.start()

        console.print("[green]🚀 Alert processing started[/green]")

    def stop_processing(self) -> None:
        """Stop background alert processing"""
        self.stop_processing.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)

        console.print("[red]⏹️ Alert processing stopped[/red]")

    def _process_alerts(self) -> None:
        """Background alert processing loop"""
        while not self.stop_processing.is_set():
            try:
                # Get alert with timeout
                alert = self.alert_queue.get(timeout=1.0)

                # Process the alert
                self._handle_alert(alert)

                # Add to history
                self.alert_history.append(alert)

                # Trim history if needed
                if len(self.alert_history) > self.max_history_size:
                    self.alert_history = self.alert_history[-self.max_history_size :]

            except queue.Empty:
                continue
            except Exception as e:
                console.print(f"[red]Alert processing error: {e}[/red]")

    def _handle_alert(self, alert: WorkflowAlert) -> None:
        """Handle an individual alert"""
        # Send to appropriate channels
        for channel in self.channels.values():
            if channel.can_handle_alert(alert.level):
                try:
                    self._send_to_channel(alert, channel)
                except Exception as e:
                    console.print(
                        f"[red]Failed to send alert to channel {channel.name}: {e}[/red]"
                    )

        # Call registered handlers
        for handler in self.alert_handlers.get(alert.level, []):
            try:
                handler(alert)
            except Exception as e:
                console.print(f"[red]Alert handler error: {e}[/red]")

    def _send_to_channel(
        self, alert: WorkflowAlert, channel: NotificationChannel
    ) -> None:
        """Send alert to a specific channel"""
        if channel.type == NotificationType.CONSOLE:
            self._send_console_alert(alert)
        elif channel.type == NotificationType.FILE:
            self._send_file_alert(alert, channel)
        elif channel.type == NotificationType.WEBHOOK:
            self._send_webhook_alert(alert, channel)
        # Add more channel types as needed

    def _send_console_alert(self, alert: WorkflowAlert) -> None:
        """Send alert to console"""
        level_colors = {
            AlertLevel.INFO: "blue",
            AlertLevel.WARNING: "yellow",
            AlertLevel.ERROR: "red",
            AlertLevel.CRITICAL: "bold red",
        }

        level_emojis = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }

        color = level_colors.get(alert.level, "white")
        emoji = level_emojis.get(alert.level, "📢")

        console.print(
            f"[{color}]{emoji} {alert.level.value.upper()}: {alert.title}[/{color}]"
        )
        console.print(f"[dim]{alert.message}[/dim]")

        if alert.workflow_id:
            console.print(f"[dim]Workflow: {alert.workflow_id}[/dim]")

    def _send_file_alert(
        self, alert: WorkflowAlert, channel: NotificationChannel
    ) -> None:
        """Send alert to file"""
        log_file = Path(channel.config.get("file_path", "alerts.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)

        log_entry = json.dumps(alert.to_dict()) + "\n"

        with open(log_file, "a") as f:
            f.write(log_entry)

    def _send_webhook_alert(
        self, alert: WorkflowAlert, channel: NotificationChannel
    ) -> None:
        """Send alert via webhook"""
        # Implementation would use requests or httpx
        # For now, just print
        webhook_url = channel.config.get("url")
        console.print(f"[dim]Would send webhook to: {webhook_url}[/dim]")

    def register_handler(self, level: AlertLevel, handler: Callable) -> None:
        """Register an alert handler"""
        self.alert_handlers[level].append(handler)

    def get_recent_alerts(self, limit: int = 10) -> List[WorkflowAlert]:
        """Get recent alerts"""
        return self.alerts[-limit:]

    def get_alerts_by_level(self, level: AlertLevel) -> List[WorkflowAlert]:
        """Get alerts by severity level"""
        return [alert for alert in self.alerts if alert.level == level]

    def clear_alerts(self) -> None:
        """Clear active alerts"""
        self.alerts.clear()
        console.print("[green]✅ Cleared all active alerts[/green]")


class WorkflowMonitor:
    """Comprehensive workflow monitoring system"""

    def __init__(self, alert_manager: Optional[AlertManager] = None):
        self.alert_manager = alert_manager or AlertManager()

        # Monitoring state
        self.monitored_workflows: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        self.monitoring_active = False

        # Thresholds
        self.execution_time_threshold = 300  # 5 minutes
        self.memory_threshold = 1024  # 1GB
        self.failure_rate_threshold = 0.2  # 20%

        # Live monitoring
        self.live_display: Optional[Live] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()

    def start_monitoring(self) -> None:
        """Start workflow monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.stop_monitoring.clear()

        # Start alert manager
        self.alert_manager.start_processing()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        console.print("[green]🚀 Workflow monitoring started[/green]")

    def stop_monitoring(self) -> None:
        """Stop workflow monitoring"""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        self.stop_monitoring.set()

        # Stop alert manager
        self.alert_manager.stop_processing()

        # Stop monitoring thread
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        # Stop live display
        if self.live_display:
            self.live_display.stop()
            self.live_display = None

        console.print("[red]⏹️ Workflow monitoring stopped[/red]")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active and not self.stop_monitoring.is_set():
            try:
                self._check_workflow_health()
                self._check_performance_metrics()
                self._update_live_display()

                # Sleep briefly before next check
                self.stop_monitoring.wait(5.0)

            except Exception as e:
                console.print(f"[red]Monitoring error: {e}[/red]")
                self.stop_monitoring.wait(10.0)

    def register_workflow(
        self, workflow_id: str, execution_id: str, workflow_name: str
    ) -> None:
        """Register a workflow for monitoring"""
        self.monitored_workflows[execution_id] = {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "start_time": datetime.now(),
            "status": "running",
            "metrics": {
                "steps_completed": 0,
                "steps_failed": 0,
                "memory_usage": 0,
                "cpu_usage": 0,
            },
        }

        # Create info alert
        self.alert_manager.create_alert(
            level=AlertLevel.INFO,
            title="Workflow Started",
            message=f"Started monitoring workflow: {workflow_name}",
            workflow_id=workflow_id,
            execution_id=execution_id,
        )

    def update_workflow_status(
        self, execution_id: str, status: str, metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update workflow status and metrics"""
        if execution_id in self.monitored_workflows:
            workflow_info = self.monitored_workflows[execution_id]
            workflow_info["status"] = status

            if metrics:
                workflow_info["metrics"].update(metrics)

            # Check for alerts
            self._check_workflow_alerts(execution_id, workflow_info)

    def _check_workflow_health(self) -> None:
        """Check health of monitored workflows"""
        current_time = datetime.now()

        for execution_id, workflow_info in self.monitored_workflows.items():
            # Check execution time
            runtime = (current_time - workflow_info["start_time"]).total_seconds()

            if (
                runtime > self.execution_time_threshold
                and workflow_info["status"] == "running"
            ):
                self.alert_manager.create_alert(
                    level=AlertLevel.WARNING,
                    title="Long Running Workflow",
                    message=f"Workflow has been running for {runtime:.1f} seconds",
                    workflow_id=workflow_info["workflow_id"],
                    execution_id=execution_id,
                    metadata={"runtime": runtime},
                )

    def _check_workflow_alerts(
        self, execution_id: str, workflow_info: Dict[str, Any]
    ) -> None:
        """Check for workflow-specific alerts"""
        metrics = workflow_info["metrics"]

        # Check failure rate
        total_steps = metrics.get("steps_completed", 0) + metrics.get("steps_failed", 0)
        if total_steps > 0:
            failure_rate = metrics.get("steps_failed", 0) / total_steps
            if failure_rate > self.failure_rate_threshold:
                self.alert_manager.create_alert(
                    level=AlertLevel.ERROR,
                    title="High Failure Rate",
                    message=f"Workflow failure rate: {failure_rate:.1%}",
                    workflow_id=workflow_info["workflow_id"],
                    execution_id=execution_id,
                    metadata={"failure_rate": failure_rate},
                )

        # Check memory usage
        memory_usage = metrics.get("memory_usage", 0)
        if memory_usage > self.memory_threshold:
            self.alert_manager.create_alert(
                level=AlertLevel.WARNING,
                title="High Memory Usage",
                message=f"Workflow memory usage: {memory_usage}MB",
                workflow_id=workflow_info["workflow_id"],
                execution_id=execution_id,
                metadata={"memory_usage": memory_usage},
            )

    def _check_performance_metrics(self) -> None:
        """Check system performance metrics"""
        # This would check system-wide metrics
        # For now, just a placeholder
        pass

    def _update_live_display(self) -> None:
        """Update live monitoring display"""
        if not self.monitored_workflows:
            return

        # Create monitoring table
        table = Table(title="🔍 Active Workflow Monitoring", box=box.ROUNDED)
        table.add_column("Execution ID", style="cyan")
        table.add_column("Workflow", style="white")
        table.add_column("Status", style="green")
        table.add_column("Runtime", style="yellow")
        table.add_column("Steps", style="blue")
        table.add_column("Memory", style="magenta")

        current_time = datetime.now()

        for execution_id, workflow_info in self.monitored_workflows.items():
            runtime = (current_time - workflow_info["start_time"]).total_seconds()
            metrics = workflow_info["metrics"]

            steps_text = f"{metrics.get('steps_completed', 0)}/{metrics.get('steps_completed', 0) + metrics.get('steps_failed', 0)}"
            memory_text = f"{metrics.get('memory_usage', 0)}MB"

            table.add_row(
                execution_id[:8],
                workflow_info["workflow_name"][:20],
                workflow_info["status"].title(),
                f"{runtime:.1f}s",
                steps_text,
                memory_text,
            )

        # Update or create live display
        if not self.live_display:
            self.live_display = Live(table, console=console, refresh_per_second=1)
            self.live_display.start()
        else:
            self.live_display.update(table)

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        total_workflows = len(self.monitored_workflows)
        running_workflows = sum(
            1 for w in self.monitored_workflows.values() if w["status"] == "running"
        )

        recent_alerts = self.alert_manager.get_recent_alerts(10)
        alert_counts = {}
        for level in AlertLevel:
            alert_counts[level.value] = len(
                self.alert_manager.get_alerts_by_level(level)
            )

        return {
            "total_workflows": total_workflows,
            "running_workflows": running_workflows,
            "recent_alerts": len(recent_alerts),
            "alert_counts": alert_counts,
            "monitoring_active": self.monitoring_active,
        }

    def display_monitoring_dashboard(self) -> None:
        """Display comprehensive monitoring dashboard"""
        summary = self.get_monitoring_summary()

        # Summary panel
        summary_table = Table(show_header=False, box=box.SIMPLE)
        summary_table.add_column("Metric", style="bold cyan")
        summary_table.add_column("Value", style="bold white")

        summary_table.add_row("Total Workflows", str(summary["total_workflows"]))
        summary_table.add_row("Running Workflows", str(summary["running_workflows"]))
        summary_table.add_row("Recent Alerts", str(summary["recent_alerts"]))
        summary_table.add_row(
            "Monitoring Status",
            "🟢 Active" if summary["monitoring_active"] else "🔴 Inactive",
        )

        console.print(
            Panel(
                summary_table,
                title="[bold blue]📊 Monitoring Dashboard[/bold blue]",
                border_style="blue",
            )
        )

        # Recent alerts
        recent_alerts = self.alert_manager.get_recent_alerts(5)
        if recent_alerts:
            console.print("\n[bold yellow]🚨 Recent Alerts:[/bold yellow]")

            alerts_table = Table(box=box.ROUNDED)
            alerts_table.add_column("Time", style="cyan")
            alerts_table.add_column("Level", style="yellow")
            alerts_table.add_column("Title", style="white")
            alerts_table.add_column("Workflow", style="green")

            for alert in recent_alerts:
                time_str = alert.timestamp.strftime("%H:%M:%S")
                workflow_str = alert.workflow_id[:8] if alert.workflow_id else "N/A"

                alerts_table.add_row(
                    time_str, alert.level.value.upper(), alert.title, workflow_str
                )

            console.print(alerts_table)
