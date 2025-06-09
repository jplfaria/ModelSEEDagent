"""
Live Visualizer for Interactive Analysis

Provides real-time workflow visualization, interactive network graphs,
and dynamic result updates for metabolic modeling analysis.
"""

import json
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
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
from rich.text import Text

console = Console()


@dataclass
class VisualizationSpec:
    """Specification for a visualization"""

    viz_type: str
    title: str
    data: Dict[str, Any]
    config: Dict[str, Any] = field(default_factory=dict)
    interactive: bool = True
    auto_refresh: bool = False
    refresh_interval: float = 1.0


@dataclass
class WorkflowNode:
    """Node in the workflow visualization"""

    id: str
    label: str
    type: str  # "input", "tool", "output", "decision"
    status: str  # "pending", "running", "completed", "failed"
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class WorkflowEdge:
    """Edge in the workflow visualization"""

    source: str
    target: str
    label: str = ""
    type: str = "data"  # "data", "control", "dependency"
    status: str = "pending"  # "pending", "active", "completed"


class LiveVisualizer:
    """Real-time visualizer for interactive analysis"""

    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.workflow_nodes: Dict[str, WorkflowNode] = {}
        self.workflow_edges: List[WorkflowEdge] = []
        self.active_visualizations: Dict[str, str] = {}  # viz_id -> file_path

        # Progress tracking
        self.current_progress: Dict[str, Any] = {}
        self.live_display: Optional[Live] = None

    def create_workflow_visualization(
        self, workflow_data: Dict[str, Any], title: str = "Analysis Workflow"
    ) -> str:
        """Create interactive workflow visualization"""

        # Parse workflow data into nodes and edges
        self._parse_workflow_data(workflow_data)

        # Create NetworkX graph
        G = nx.DiGraph()

        # Add nodes
        for node_id, node in self.workflow_nodes.items():
            G.add_node(
                node_id,
                label=node.label,
                type=node.type,
                status=node.status,
                **node.metadata,
            )

        # Add edges
        for edge in self.workflow_edges:
            G.add_edge(
                edge.source,
                edge.target,
                label=edge.label,
                type=edge.type,
                status=edge.status,
            )

        # Create layout
        pos = self._create_workflow_layout(G)

        # Create Plotly figure
        fig = self._create_plotly_workflow(G, pos, title)

        # Save and return path
        viz_file = (
            self.output_dir
            / f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        pyo.plot(fig, filename=str(viz_file), auto_open=False)

        self.active_visualizations["workflow"] = str(viz_file)
        console.print(f"🎨 Workflow visualization created: [cyan]{viz_file}[/cyan]")

        return str(viz_file)

    def create_progress_dashboard(
        self, analysis_data: Dict[str, Any], title: str = "Analysis Dashboard"
    ) -> str:
        """Create real-time progress dashboard"""

        # Create subplot figure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Execution Timeline",
                "Tool Performance",
                "Success Rate",
                "Resource Usage",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "indicator"}, {"type": "scatter"}],
            ],
        )

        # Timeline plot
        if "execution_timeline" in analysis_data:
            timeline_data = analysis_data["execution_timeline"]
            fig.add_trace(
                go.Scatter(
                    x=timeline_data.get("timestamps", []),
                    y=timeline_data.get("tools", []),
                    mode="markers+lines",
                    name="Execution Flow",
                    marker=dict(size=10, color="blue"),
                    line=dict(width=2),
                ),
                row=1,
                col=1,
            )

        # Tool performance
        if "tool_performance" in analysis_data:
            perf_data = analysis_data["tool_performance"]
            fig.add_trace(
                go.Bar(
                    x=list(perf_data.keys()),
                    y=list(perf_data.values()),
                    name="Execution Time (s)",
                    marker_color="green",
                ),
                row=1,
                col=2,
            )

        # Success rate indicator
        success_rate = analysis_data.get("success_rate", 0.85)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=success_rate * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Success Rate (%)"},
                delta={"reference": 80},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "gray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            ),
            row=2,
            col=1,
        )

        # Resource usage
        if "resource_usage" in analysis_data:
            resource_data = analysis_data["resource_usage"]
            fig.add_trace(
                go.Scatter(
                    x=resource_data.get("time", []),
                    y=resource_data.get("memory", []),
                    mode="lines",
                    name="Memory Usage",
                    line=dict(color="red", width=2),
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title=title, height=600, showlegend=True, template="plotly_white"
        )

        # Save and return path
        viz_file = (
            self.output_dir
            / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        pyo.plot(fig, filename=str(viz_file), auto_open=False)

        self.active_visualizations["dashboard"] = str(viz_file)
        console.print(f"📊 Progress dashboard created: [cyan]{viz_file}[/cyan]")

        return str(viz_file)

    def create_network_visualization(
        self, network_data: Dict[str, Any], title: str = "Metabolic Network"
    ) -> str:
        """Create interactive metabolic network visualization"""

        # Extract nodes and edges from network data
        nodes = network_data.get("nodes", [])
        edges = network_data.get("edges", [])

        # Create NetworkX graph
        G = nx.Graph()

        # Add nodes with attributes
        for node in nodes:
            G.add_node(node["id"], **node.get("attributes", {}))

        # Add edges
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], **edge.get("attributes", {}))

        # Create layout
        if len(G.nodes()) > 100:
            pos = nx.spring_layout(G, k=1, iterations=50)
        else:
            pos = nx.spring_layout(G, k=2, iterations=100)

        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        # Create node traces
        node_x = []
        node_y = []
        node_info = []
        node_colors = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Node information for hover
            adjacencies = list(G.neighbors(node))
            node_info.append(f"{node}<br># of connections: {len(adjacencies)}")

            # Color by node type or attribute
            node_attrs = G.nodes[node]
            if "type" in node_attrs:
                type_colors = {"metabolite": "blue", "reaction": "red", "gene": "green"}
                node_colors.append(type_colors.get(node_attrs["type"], "gray"))
            else:
                node_colors.append("blue")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_info,
            marker=dict(
                showscale=True,
                colorscale="Viridis",
                color=node_colors,
                size=10,
                colorbar=dict(thickness=15, len=0.5, x=1.1, title="Node Type"),
                line=dict(width=2),
            ),
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text=title, font=dict(size=16)),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Metabolic network visualization",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                        font=dict(color="gray", size=12),
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        # Save and return path
        viz_file = (
            self.output_dir / f"network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        pyo.plot(fig, filename=str(viz_file), auto_open=False)

        self.active_visualizations["network"] = str(viz_file)
        console.print(f"🕸️ Network visualization created: [cyan]{viz_file}[/cyan]")

        return str(viz_file)

    def create_flux_heatmap(
        self, flux_data: Dict[str, Any], title: str = "Flux Distribution"
    ) -> str:
        """Create interactive flux distribution heatmap"""

        # Extract flux matrix
        fluxes = flux_data.get("flux_matrix", [])
        reactions = flux_data.get("reaction_names", [])
        conditions = flux_data.get("condition_names", [])

        if len(fluxes) == 0:
            # Create mock data for demonstration
            import numpy as np

            fluxes = np.random.rand(20, 5) * 10 - 2  # Random fluxes between -2 and 8
            reactions = [f"R{i:03d}" for i in range(20)]
            conditions = [f"Condition_{i}" for i in range(5)]

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=fluxes,
                x=conditions,
                y=reactions,
                colorscale="RdBu",
                zmid=0,
                hoverongaps=False,
                hovertemplate="Reaction: %{y}<br>Condition: %{x}<br>Flux: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Conditions",
            yaxis_title="Reactions",
            height=600,
            template="plotly_white",
        )

        # Save and return path
        viz_file = (
            self.output_dir / f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        pyo.plot(fig, filename=str(viz_file), auto_open=False)

        self.active_visualizations["heatmap"] = str(viz_file)
        console.print(f"🔥 Flux heatmap created: [cyan]{viz_file}[/cyan]")

        return str(viz_file)

    def start_live_progress(self, total_steps: int = 100) -> None:
        """Start live progress display"""
        self.current_progress = {
            "total": total_steps,
            "completed": 0,
            "current_step": "Initializing...",
            "start_time": datetime.now(),
        }

        try:
            # Create live display layout
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            )

            # Ensure step description is not empty
            step_description = self.current_progress.get(
                "current_step", "Processing..."
            )
            if not step_description or step_description.strip() == "":
                step_description = "Processing..."

            task = progress.add_task(description=step_description, total=total_steps)

            self.live_display = Live(progress, console=console, refresh_per_second=4)
            self.live_display.start()

            # Store task ID for updates
            self.current_progress["task_id"] = task
            self.current_progress["progress_obj"] = progress

        except Exception as e:
            # Fallback to console output if live display fails
            console.print(f"[yellow]Live progress display unavailable: {e}[/yellow]")
            console.print(f"[dim]Starting {step_description}...[/dim]")
            self.live_display = None

    def update_progress(self, step: str, advance: int = 1) -> None:
        """Update live progress display"""
        if not self.live_display or "progress_obj" not in self.current_progress:
            # Fallback to console output
            console.print(f"[dim]▶ {step}[/dim]")
            return

        # Validate step content
        if not step or step.strip() == "":
            step = "Processing..."

        self.current_progress["completed"] += advance
        self.current_progress["current_step"] = step

        try:
            progress = self.current_progress["progress_obj"]
            task_id = self.current_progress["task_id"]

            progress.update(task_id, description=step, advance=advance)
        except Exception:
            # Fallback if update fails
            console.print(f"[dim]▶ {step}[/dim]")

    def stop_live_progress(self) -> None:
        """Stop live progress display"""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None

        if self.current_progress:
            duration = datetime.now() - self.current_progress["start_time"]
            console.print(f"\n✅ Analysis completed in {duration.total_seconds():.1f}s")

    def _parse_workflow_data(self, workflow_data: Dict[str, Any]) -> None:
        """Parse workflow data into nodes and edges"""
        self.workflow_nodes.clear()
        self.workflow_edges.clear()

        # Parse nodes
        for node_data in workflow_data.get("nodes", []):
            node = WorkflowNode(
                id=node_data["id"],
                label=node_data.get("label", node_data["id"]),
                type=node_data.get("type", "tool"),
                status=node_data.get("status", "pending"),
                metadata=node_data.get("metadata", {}),
            )

            if "start_time" in node_data:
                node.start_time = datetime.fromisoformat(node_data["start_time"])
            if "end_time" in node_data:
                node.end_time = datetime.fromisoformat(node_data["end_time"])

            self.workflow_nodes[node.id] = node

        # Parse edges
        for edge_data in workflow_data.get("edges", []):
            edge = WorkflowEdge(
                source=edge_data["source"],
                target=edge_data["target"],
                label=edge_data.get("label", ""),
                type=edge_data.get("type", "data"),
                status=edge_data.get("status", "pending"),
            )
            self.workflow_edges.append(edge)

    def _create_workflow_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout for workflow graph"""

        # Try to create hierarchical layout
        try:
            # Identify levels based on node types
            levels = {}
            for node_id, node in self.workflow_nodes.items():
                if node.type == "input":
                    levels[node_id] = 0
                elif node.type == "tool":
                    levels[node_id] = 1
                elif node.type == "output":
                    levels[node_id] = 2
                else:
                    levels[node_id] = 1

            # Use graphviz layout if available, otherwise spring layout
            try:
                pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
            except:
                pos = nx.spring_layout(G, k=3, iterations=50)

        except Exception:
            # Fallback to spring layout
            pos = nx.spring_layout(G, k=2, iterations=50)

        return pos

    def _create_plotly_workflow(
        self, G: nx.DiGraph, pos: Dict, title: str
    ) -> go.Figure:
        """Create Plotly figure for workflow visualization"""

        # Create edge traces
        edge_x = []
        edge_y = []
        edge_info = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(edge[2].get("label", ""))

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=2, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        # Create node traces by type
        node_traces = {}
        type_colors = {
            "input": "lightblue",
            "tool": "lightgreen",
            "output": "lightcoral",
            "decision": "lightyellow",
        }

        for node_type in type_colors:
            node_traces[node_type] = {"x": [], "y": [], "text": [], "ids": []}

        for node_id in G.nodes():
            if node_id in self.workflow_nodes:
                node = self.workflow_nodes[node_id]
                x, y = pos[node_id]

                node_type = node.type
                if node_type not in node_traces:
                    node_type = "tool"  # default

                node_traces[node_type]["x"].append(x)
                node_traces[node_type]["y"].append(y)
                node_traces[node_type]["text"].append(
                    f"{node.label}<br>Status: {node.status}"
                )
                node_traces[node_type]["ids"].append(node_id)

        # Create Plotly traces
        traces = [edge_trace]

        for node_type, trace_data in node_traces.items():
            if trace_data["x"]:  # Only add if there are nodes of this type
                traces.append(
                    go.Scatter(
                        x=trace_data["x"],
                        y=trace_data["y"],
                        mode="markers+text",
                        text=[node_id.split("_")[-1] for node_id in trace_data["ids"]],
                        textposition="middle center",
                        hovertext=trace_data["text"],
                        hoverinfo="text",
                        marker=dict(
                            size=30,
                            color=type_colors[node_type],
                            line=dict(width=2, color="black"),
                        ),
                        name=node_type.title(),
                    )
                )

        # Create figure
        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title=dict(text=title, font=dict(size=16)),
                showlegend=True,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Interactive workflow visualization",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                        font=dict(color="gray", size=12),
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_white",
            ),
        )

        return fig

    def open_visualization(self, viz_key: str) -> bool:
        """Open a visualization in the browser"""
        if viz_key in self.active_visualizations:
            viz_file = self.active_visualizations[viz_key]
            try:
                webbrowser.open(f"file://{Path(viz_file).absolute()}")
                console.print(f"🌐 Opened {viz_key} visualization in browser")
                return True
            except Exception as e:
                console.print(f"❌ Error opening visualization: {e}")
                return False
        else:
            console.print(f"⚠️ Visualization '{viz_key}' not found")
            return False

    def get_visualization_summary(self) -> Dict[str, Any]:
        """Get summary of all active visualizations"""
        return {
            "total_visualizations": len(self.active_visualizations),
            "visualization_types": list(self.active_visualizations.keys()),
            "output_directory": str(self.output_dir),
            "file_paths": dict(self.active_visualizations),
        }

    def display_visualization_table(self) -> None:
        """Display table of available visualizations"""
        if not self.active_visualizations:
            console.print("[yellow]No visualizations available.[/yellow]")
            return

        table = Table(title="🎨 Available Visualizations", box=box.ROUNDED)
        table.add_column("Type", style="bold cyan")
        table.add_column("File", style="bold white")
        table.add_column("Size", style="bold green")
        table.add_column("Created", style="bold blue")

        for viz_type, file_path in self.active_visualizations.items():
            file_obj = Path(file_path)
            if file_obj.exists():
                size = file_obj.stat().st_size
                size_str = (
                    f"{size / 1024:.1f} KB"
                    if size < 1024 * 1024
                    else f"{size / (1024*1024):.1f} MB"
                )
                created = datetime.fromtimestamp(file_obj.stat().st_mtime).strftime(
                    "%H:%M:%S"
                )

                table.add_row(viz_type.title(), file_obj.name, size_str, created)

        console.print(table)
