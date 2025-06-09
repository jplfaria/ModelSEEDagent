"""
Real-Time Streaming Interface for Dynamic AI Agent

Provides live display of AI reasoning, tool execution progress, and results
as they happen, creating an immersive and transparent AI experience.
"""

import asyncio
import json
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.table import Table
from rich.text import Text

console = Console()


class StreamingEventType(Enum):
    """Types of streaming events"""

    AI_THINKING = "ai_thinking"
    TOOL_SELECTED = "tool_selected"
    TOOL_EXECUTING = "tool_executing"
    TOOL_COMPLETED = "tool_completed"
    AI_ANALYZING = "ai_analyzing"
    DECISION_MADE = "decision_made"
    WORKFLOW_COMPLETE = "workflow_complete"
    ERROR_OCCURRED = "error_occurred"


class StreamingEvent:
    """A single streaming event"""

    def __init__(
        self,
        event_type: StreamingEventType,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ):
        self.event_type = event_type
        self.message = message
        self.data = data or {}
        self.timestamp = timestamp or datetime.now()
        self.duration = 0.0


class RealTimeStreamingInterface:
    """Real-time streaming interface for AI agent interactions"""

    def __init__(self):
        self.console = Console()
        self.events: List[StreamingEvent] = []
        self.current_progress: Optional[Progress] = None
        self.current_live: Optional[Live] = None
        self.tool_tasks: Dict[str, TaskID] = {}
        self.start_time = time.time()

        # Display state
        self.is_streaming = False
        self.current_ai_thought = ""
        self.current_tool = ""
        self.executed_tools: List[Dict[str, Any]] = []
        self.ai_decisions: List[str] = []

    def start_streaming(self, query: str) -> None:
        """Start the streaming interface for a new query with robust initialization"""
        # Initialize state first (order matters!)
        self.is_streaming = True
        self.start_time = time.time()

        # Initialize collections safely
        self.events = []
        self.executed_tools = []
        self.ai_decisions = []

        # Initialize content to prevent empty panels (BEFORE creating layout)
        self.current_ai_thought = "🧠 AI agent starting analysis..."
        self.current_tool = ""

        # Validate query
        if query is None:
            query = "Unknown query"
        query = str(query).strip()
        if not query:
            query = "Analysis in progress"

        # Create the streaming layout
        layout = self._create_streaming_layout(query)

        # Start live display with robust error handling
        try:
            self.current_live = Live(layout, console=self.console, refresh_per_second=4)
            self.current_live.start()
        except Exception as e:
            # Fallback to non-live display with details
            self.console.print(
                f"[yellow]Live display unavailable ({e}), using standard output[/yellow]"
            )
            self.current_live = None

        # Initial event (after everything is set up)
        self.add_event(
            StreamingEventType.AI_THINKING, "🧠 AI agent starting analysis..."
        )

    def add_event(
        self,
        event_type: StreamingEventType,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a new streaming event"""
        if not self.is_streaming:
            return

        event = StreamingEvent(event_type, message, data)
        self.events.append(event)

        # Update display state based on event type
        self._update_display_state(event)

        # Update the live display
        if self.current_live:
            self._update_live_display(event)

    def _update_display_state(self, event: StreamingEvent) -> None:
        """Update internal display state based on event"""
        if event.event_type == StreamingEventType.AI_THINKING:
            self.current_ai_thought = event.message

        elif event.event_type == StreamingEventType.TOOL_SELECTED:
            self.current_tool = event.data.get("tool_name", "")
            self.ai_decisions.append(
                f"Selected {self.current_tool}: {event.data.get('reasoning', '')}"
            )

        elif event.event_type == StreamingEventType.TOOL_COMPLETED:
            tool_info = {
                "name": event.data.get("tool_name", ""),
                "duration": event.data.get("duration", 0.0),
                "success": event.data.get("success", True),
                "summary": event.data.get("summary", ""),
            }
            self.executed_tools.append(tool_info)

        elif event.event_type == StreamingEventType.DECISION_MADE:
            self.ai_decisions.append(event.message)

    def _create_streaming_layout(self, query: str) -> Layout:
        """Create the main streaming layout"""
        layout = Layout()

        # Split into header and main content
        layout.split_column(Layout(name="header", size=3), Layout(name="main"))

        # Header with query
        header_content = f"🧠 **Dynamic AI Analysis:** {query[:80]}{'...' if len(query) > 80 else ''}"
        layout["header"].update(Panel(header_content, style="bold blue"))

        # Main content - split into left and right
        layout["main"].split_row(Layout(name="left"), Layout(name="right"))

        # Left side: AI thinking and decisions
        layout["left"].split_column(
            Layout(name="ai_thinking", size=8), Layout(name="decisions")
        )

        # Right side: Tool execution and progress
        layout["right"].split_column(
            Layout(name="tool_progress", size=8), Layout(name="results")
        )

        # Initialize with empty content
        layout["ai_thinking"].update(self._create_ai_thinking_panel())
        layout["decisions"].update(self._create_decisions_panel())
        layout["tool_progress"].update(self._create_tool_progress_panel())
        layout["results"].update(self._create_results_panel())

        return layout

    def _create_ai_thinking_panel(self) -> Panel:
        """Create the AI thinking panel with robust content validation"""
        # Robust content validation
        thought = self.current_ai_thought
        if thought is None:
            thought = ""

        thought_str = str(thought).strip()
        if not thought_str:
            content = "[dim]🤔 Waiting for AI analysis...[/dim]"
        else:
            content = f"🧠 {thought_str}"

        # Final safety check
        if not content or not str(content).strip():
            content = "[dim]🤔 AI thinking panel ready...[/dim]"

        return Panel(
            content, title="[bold cyan]🤖 AI Reasoning[/bold cyan]", border_style="cyan"
        )

    def _create_decisions_panel(self) -> Panel:
        """Create the AI decisions panel with robust content validation"""
        # Robust content validation for decisions list
        decisions = self.ai_decisions
        if not decisions or not isinstance(decisions, list):
            content = "[dim]📋 AI decisions will appear here...[/dim]"
        else:
            # Filter and validate each decision
            valid_decisions = []
            for decision in decisions[-5:]:  # Last 5 decisions
                if decision is not None:
                    decision_str = str(decision).strip()
                    if decision_str:
                        valid_decisions.append(f"• {decision_str}")

            if valid_decisions:
                content = "\n".join(valid_decisions)
            else:
                content = "[dim]📋 Waiting for AI decisions...[/dim]"

        # Final safety check
        if not content or not str(content).strip():
            content = "[dim]📋 AI decisions panel ready...[/dim]"

        return Panel(
            content,
            title="[bold yellow]🎯 AI Decisions[/bold yellow]",
            border_style="yellow",
        )

    def _create_tool_progress_panel(self) -> Panel:
        """Create the tool progress panel with robust content validation"""
        lines = []

        # Current tool validation
        current_tool = self.current_tool
        if current_tool is not None:
            current_tool_str = str(current_tool).strip()
            if current_tool_str:
                lines.append(f"⚡ Currently executing: [bold]{current_tool_str}[/bold]")
                lines.append("")

        # Completed tools validation
        executed_tools = self.executed_tools
        if (
            executed_tools
            and isinstance(executed_tools, list)
            and len(executed_tools) > 0
        ):
            lines.append("✅ Completed tools:")
            for tool in executed_tools[-3:]:  # Last 3
                if tool and isinstance(tool, dict) and "name" in tool:
                    tool_name = tool.get("name")
                    if tool_name is not None:
                        tool_name_str = str(tool_name).strip()
                        if tool_name_str:
                            status = "✅" if tool.get("success", True) else "❌"
                            duration = tool.get("duration", 0.0)
                            try:
                                duration_float = float(duration)
                            except (ValueError, TypeError):
                                duration_float = 0.0
                            lines.append(
                                f"  {status} {tool_name_str} ({duration_float:.1f}s)"
                            )

        # Generate content
        if lines:
            content = "\n".join(lines)
        else:
            content = "[dim]🔧 Tool execution will appear here...[/dim]"

        # Final safety check
        if not content or not str(content).strip():
            content = "[dim]🔧 Tool execution panel ready...[/dim]"

        return Panel(
            content,
            title="[bold green]🔧 Tool Execution[/bold green]",
            border_style="green",
        )

    def _create_results_panel(self) -> Panel:
        """Create the results panel with robust content validation"""
        try:
            # Safe time calculation
            start_time = getattr(self, "start_time", time.time())
            elapsed = time.time() - start_time
            if elapsed < 0:
                elapsed = 0.0

            # Safe counting with validation
            tools_count = 0
            if self.executed_tools and isinstance(self.executed_tools, list):
                tools_count = len(self.executed_tools)

            decisions_count = 0
            if self.ai_decisions and isinstance(self.ai_decisions, list):
                decisions_count = len(self.ai_decisions)

            events_count = 0
            if self.events and isinstance(self.events, list):
                events_count = len(self.events)

            content = f"""⏱️ Elapsed: {elapsed:.1f}s
🔧 Tools executed: {tools_count}
🧠 AI decisions: {decisions_count}
📊 Events: {events_count}"""

        except Exception:
            # Fallback content if anything goes wrong
            content = "📊 Statistics loading..."

        # Final safety check
        if not content or not str(content).strip():
            content = "📊 Live statistics panel ready..."

        return Panel(
            content,
            title="[bold magenta]📊 Live Statistics[/bold magenta]",
            border_style="magenta",
        )

    def _update_live_display(self, event: StreamingEvent) -> None:
        """Update the live display with new event using robust error handling"""
        if not self.current_live:
            # If live display is not available, log the event instead
            self.console.print(f"[dim]{event.event_type.value}: {event.message}[/dim]")
            return

        try:
            # Get current layout with validation
            layout = self.current_live.renderable
            if not layout or not hasattr(layout, "__getitem__"):
                return

            # Create panels with robust error handling
            panels = {}
            try:
                panels["ai_thinking"] = self._create_ai_thinking_panel()
            except Exception:
                panels["ai_thinking"] = Panel(
                    "[red]AI panel error[/red]", title="AI Thinking", border_style="red"
                )

            try:
                panels["decisions"] = self._create_decisions_panel()
            except Exception:
                panels["decisions"] = Panel(
                    "[red]Decisions panel error[/red]",
                    title="Decisions",
                    border_style="red",
                )

            try:
                panels["tool_progress"] = self._create_tool_progress_panel()
            except Exception:
                panels["tool_progress"] = Panel(
                    "[red]Tools panel error[/red]", title="Tools", border_style="red"
                )

            try:
                panels["results"] = self._create_results_panel()
            except Exception:
                panels["results"] = Panel(
                    "[red]Results panel error[/red]",
                    title="Results",
                    border_style="red",
                )

            # Update layout with each panel, handling errors individually
            for panel_name, panel in panels.items():
                if panel:
                    try:
                        layout[panel_name].update(panel)
                    except (KeyError, AttributeError) as e:
                        # Log missing layout sections but don't crash
                        import logging

                        logging.debug(f"Layout section '{panel_name}' not found: {e}")
                    except Exception as e:
                        # Log other update errors but don't crash
                        import logging

                        logging.warning(f"Failed to update {panel_name}: {e}")

        except Exception as e:
            # Final safety net - log error but don't crash the streaming
            import logging

            logging.error(f"Live display critical error: {e}")
            # Try to show error in console as fallback
            try:
                self.console.print(f"[red]Display error: {event.message}[/red]")
            except:
                pass  # Even the fallback failed, but don't crash

    def show_tool_execution(self, tool_name: str, reasoning: str) -> None:
        """Show tool execution starting"""
        self.add_event(
            StreamingEventType.TOOL_SELECTED,
            f"🔧 Selected tool: {tool_name}",
            {"tool_name": tool_name, "reasoning": reasoning},
        )

        self.add_event(
            StreamingEventType.TOOL_EXECUTING,
            f"⚡ Executing {tool_name}...",
            {"tool_name": tool_name},
        )

    def show_tool_completion(
        self, tool_name: str, duration: float, success: bool, summary: str
    ) -> None:
        """Show tool execution completion"""
        status = "✅" if success else "❌"
        self.add_event(
            StreamingEventType.TOOL_COMPLETED,
            f"{status} {tool_name} completed in {duration:.1f}s",
            {
                "tool_name": tool_name,
                "duration": duration,
                "success": success,
                "summary": summary,
            },
        )

    def show_ai_analysis(self, thought: str) -> None:
        """Show AI analysis/thinking"""
        self.add_event(StreamingEventType.AI_ANALYZING, f"🧠 {thought}")

    def show_decision(self, decision: str, reasoning: str) -> None:
        """Show AI decision with reasoning"""
        self.add_event(
            StreamingEventType.DECISION_MADE,
            f"🎯 Decision: {decision}",
            {"reasoning": reasoning},
        )

    def show_workflow_complete(
        self, final_message: str, metadata: Dict[str, Any]
    ) -> None:
        """Show workflow completion"""
        self.add_event(
            StreamingEventType.WORKFLOW_COMPLETE,
            f"🎉 Analysis complete: {final_message}",
            metadata,
        )

    def show_error(self, error_message: str, details: Optional[str] = None) -> None:
        """Show error occurrence"""
        self.add_event(
            StreamingEventType.ERROR_OCCURRED,
            f"❌ Error: {error_message}",
            {"details": details},
        )

    def stop_streaming(self) -> None:
        """Stop the streaming interface"""
        if self.current_live:
            self.current_live.stop()
            self.current_live = None

        self.is_streaming = False

        # Show final summary
        self._show_final_summary()

    def _show_final_summary(self) -> None:
        """Show final summary of the streaming session"""
        elapsed = time.time() - self.start_time

        summary_table = Table(title="🎯 AI Analysis Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="bold cyan")
        summary_table.add_column("Value", style="bold white")

        summary_table.add_row("Total Duration", f"{elapsed:.1f}s")
        summary_table.add_row("Tools Executed", str(len(self.executed_tools)))
        summary_table.add_row("AI Decisions", str(len(self.ai_decisions)))
        summary_table.add_row("Total Events", str(len(self.events)))

        # Success rate
        successful_tools = sum(1 for tool in self.executed_tools if tool["success"])
        success_rate = successful_tools / max(1, len(self.executed_tools))
        summary_table.add_row("Success Rate", f"{success_rate:.1%}")

        self.console.print(summary_table)

        # Show tool execution timeline
        if self.executed_tools:
            self.console.print("\n🔧 **Tool Execution Timeline:**")
            for i, tool in enumerate(self.executed_tools, 1):
                status = "✅" if tool["success"] else "❌"
                self.console.print(
                    f"  {i}. {status} {tool['name']} ({tool['duration']:.1f}s)"
                )

        # Show key AI decisions
        if self.ai_decisions:
            self.console.print("\n🎯 **Key AI Decisions:**")
            for i, decision in enumerate(self.ai_decisions, 1):
                self.console.print(f"  {i}. {decision}")


class StreamingProgress:
    """Progress tracking for streaming operations"""

    def __init__(self, total_steps: int, description: str = "Processing"):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        )
        self.task_id = self.progress.add_task(description, total=total_steps)
        self.progress.start()

    def update(self, advance: int = 1, description: Optional[str] = None):
        """Update progress"""
        self.progress.update(self.task_id, advance=advance)
        if description:
            self.progress.update(self.task_id, description=description)

    def complete(self, description: str = "Complete!"):
        """Mark as complete"""
        self.progress.update(self.task_id, completed=100, description=description)
        self.progress.stop()


def create_streaming_interface() -> RealTimeStreamingInterface:
    """Factory function to create streaming interface"""
    return RealTimeStreamingInterface()


def demo_streaming_interface():
    """Demo the streaming interface"""
    import random

    streaming = create_streaming_interface()

    try:
        # Start streaming
        streaming.start_streaming(
            "Comprehensive analysis of E. coli metabolic capabilities"
        )

        # Simulate AI workflow
        time.sleep(1)
        streaming.show_ai_analysis(
            "Analyzing query structure and determining analysis strategy..."
        )

        time.sleep(1.5)
        streaming.show_decision(
            "Start with growth analysis", "Need baseline metabolic performance"
        )
        streaming.show_tool_execution(
            "run_metabolic_fba", "Essential for understanding growth capacity"
        )

        time.sleep(2)
        streaming.show_tool_completion(
            "run_metabolic_fba", 1.8, True, "Growth rate: 0.518 h⁻¹"
        )

        time.sleep(1)
        streaming.show_ai_analysis(
            "High growth rate detected - investigating nutritional requirements..."
        )
        streaming.show_decision(
            "Analyze nutritional dependencies",
            "High growth suggests efficient metabolism",
        )
        streaming.show_tool_execution(
            "find_minimal_media", "Determine essential nutrients"
        )

        time.sleep(1.5)
        streaming.show_tool_completion(
            "find_minimal_media", 1.2, True, "Found 20 essential nutrients"
        )

        time.sleep(1)
        streaming.show_ai_analysis(
            "Moderate nutritional complexity - checking biosynthetic capabilities..."
        )
        streaming.show_decision(
            "Investigate essential genes", "Complete metabolic characterization"
        )
        streaming.show_tool_execution(
            "analyze_essentiality", "Identify critical metabolic components"
        )

        time.sleep(2)
        streaming.show_tool_completion(
            "analyze_essentiality", 2.1, True, "Found 142 essential genes"
        )

        time.sleep(1)
        streaming.show_workflow_complete(
            "Comprehensive metabolic analysis complete",
            {
                "tools_executed": 3,
                "ai_decisions": 3,
                "key_findings": {
                    "growth_rate": "0.518 h⁻¹",
                    "essential_nutrients": 20,
                    "essential_genes": 142,
                },
            },
        )

        time.sleep(2)

    finally:
        streaming.stop_streaming()


if __name__ == "__main__":
    demo_streaming_interface()
