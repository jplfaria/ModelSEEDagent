#!/usr/bin/env python3
"""
ModelSEEDagent Professional CLI

A beautiful, modern command-line interface for metabolic modeling
with intelligent workflow capabilities, real-time visualization,
and comprehensive observability.
"""

import json
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import questionary
import typer
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.enhanced_tool_integration import EnhancedToolIntegration
from agents.langgraph_metabolic import LangGraphMetabolicAgent
from llm.argo import ArgoLLM
from llm.local import LocalLLM
from llm.openai_llm import OpenAILLM
from tools.cobra.analysis import ModelAnalysisTool, PathwayAnalysisTool
from tools.cobra.fba import FBATool

# Initialize Rich console for beautiful output
console = Console()

# Create the main CLI application
app = typer.Typer(
    name="modelseed-agent",
    help="🧬 ModelSEEDagent - Intelligent Metabolic Modeling with LLM-powered Analysis",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Global state for configuration
config_state = {
    "llm_backend": None,
    "llm_config": None,
    "tools": None,
    "agent": None,
    "last_run_id": None,
    "workspace": Path.cwd(),
}

# ASCII Art Banner
BANNER = """
[bold cyan]
╔══════════════════════════════════════════════════════════════════════════════╗
║                           🧬 ModelSEEDagent                                 ║
║                  Intelligent Metabolic Modeling Platform                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
[/bold cyan]
"""


def print_banner():
    """Print the beautiful ASCII art banner"""
    console.print(BANNER)


def print_success(message: str):
    """Print success message with green styling"""
    console.print(f"✅ [bold green]{message}[/bold green]")


def print_error(message: str):
    """Print error message with red styling"""
    console.print(f"❌ [bold red]{message}[/bold red]")


def print_info(message: str):
    """Print info message with blue styling"""
    console.print(f"ℹ️  [bold blue]{message}[/bold blue]")


def print_warning(message: str):
    """Print warning message with yellow styling"""
    console.print(f"⚠️  [bold yellow]{message}[/bold yellow]")


def create_config_panel(config: Dict[str, Any]) -> Panel:
    """Create a beautiful configuration display panel"""
    config_table = Table(show_header=False, box=box.ROUNDED)
    config_table.add_column("Setting", style="bold cyan")
    config_table.add_column("Value", style="bold white")

    # Add configuration rows
    llm_backend = config.get("llm_backend", "Not configured")
    config_table.add_row(
        "🤖 LLM Backend",
        (
            f"[green]{llm_backend}[/green]"
            if llm_backend != "Not configured"
            else "[red]Not configured[/red]"
        ),
    )

    model_name = config.get("llm_config", {}).get("model_name", "Not set")
    config_table.add_row(
        "🧠 Model",
        (
            f"[green]{model_name}[/green]"
            if model_name != "Not set"
            else "[red]Not set[/red]"
        ),
    )

    tools_count = len(config.get("tools", []))
    config_table.add_row(
        "🔧 Tools",
        (
            f"[green]{tools_count} tools loaded[/green]"
            if tools_count > 0
            else "[red]No tools loaded[/red]"
        ),
    )

    agent_status = "Ready" if config.get("agent") else "Not initialized"
    config_table.add_row(
        "🚀 Agent",
        (
            f"[green]{agent_status}[/green]"
            if agent_status == "Ready"
            else "[red]{agent_status}[/red]"
        ),
    )

    workspace = config.get("workspace", Path.cwd())
    config_table.add_row("📁 Workspace", f"[cyan]{workspace}[/cyan]")

    return Panel(
        config_table,
        title="[bold blue]📋 Current Configuration[/bold blue]",
        border_style="blue",
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-v", help="Show version information"
    ),
    config: bool = typer.Option(
        False, "--config", "-c", help="Show current configuration"
    ),
):
    """
    🧬 ModelSEEDagent - Professional CLI for Intelligent Metabolic Modeling

    A powerful command-line interface featuring:
    • Intelligent workflow automation with LangGraph
    • Real-time performance monitoring and visualization
    • Advanced tool integration with parallel execution
    • Interactive analysis with beautiful output formatting
    """
    if version:
        console.print("[bold green]ModelSEEDagent CLI v1.0.0[/bold green]")
        console.print(
            "Enhanced with LangGraph workflows and intelligent tool integration"
        )
        return

    if config:
        print_banner()
        console.print(create_config_panel(config_state))
        return

    if ctx.invoked_subcommand is None:
        print_banner()
        console.print("\n[bold yellow]Welcome to ModelSEEDagent![/bold yellow] 🎉")
        console.print("\nTo get started, try one of these commands:")

        commands_table = Table(show_header=False, box=box.SIMPLE)
        commands_table.add_column("Command", style="bold cyan")
        commands_table.add_column("Description", style="white")

        commands_table.add_row(
            "modelseed-agent setup", "🔧 Configure LLM backend and tools"
        )
        commands_table.add_row(
            "modelseed-agent analyze [model]", "🧬 Analyze a metabolic model"
        )
        commands_table.add_row(
            "modelseed-agent interactive", "💬 Start interactive analysis session"
        )
        commands_table.add_row(
            "modelseed-agent status", "📊 Show system status and metrics"
        )
        commands_table.add_row(
            "modelseed-agent --help", "❓ Show detailed help information"
        )

        console.print(
            Panel(
                commands_table,
                title="[bold blue]🚀 Quick Start Commands[/bold blue]",
                border_style="blue",
            )
        )


@app.command()
def setup(
    llm_backend: str = typer.Option(
        None,
        "--backend",
        "-b",
        help="LLM backend to use [argo, openai, local]",
        prompt="Choose LLM backend (argo/openai/local)",
    ),
    interactive: bool = typer.Option(
        True, "--interactive/--non-interactive", help="Use interactive setup"
    ),
):
    """
    🔧 Configure ModelSEEDagent with LLM backend and tools

    Sets up the agent with your preferred LLM backend and initializes
    all necessary tools for metabolic modeling analysis.
    """
    print_banner()
    console.print("[bold yellow]🔧 Setting up ModelSEEDagent...[/bold yellow]\n")

    # Validate backend choice
    valid_backends = ["argo", "openai", "local"]
    if llm_backend not in valid_backends:
        print_error(
            f"Invalid backend '{llm_backend}'. Choose from: {', '.join(valid_backends)}"
        )
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:

        # Step 1: Configure LLM
        task = progress.add_task("Configuring LLM backend...", total=None)

        try:
            llm_config = {}

            if llm_backend == "argo":
                if interactive:
                    api_base = questionary.text(
                        "Argo API Base URL:", default="https://api.argilla.com/"
                    ).ask()
                    user = questionary.text("Argo Username:").ask()
                    model_name = questionary.text(
                        "Model Name:", default="llama-3.1-70b"
                    ).ask()
                else:
                    api_base = "https://api.argilla.com/"
                    user = "default_user"
                    model_name = "llama-3.1-70b"

                llm_config = {
                    "model_name": model_name,
                    "api_base": api_base,
                    "user": user,
                    "system_content": "You are an expert metabolic modeling assistant.",
                    "max_tokens": 1000,
                    "temperature": 0.1,
                }

            elif llm_backend == "openai":
                if interactive:
                    api_key = questionary.password("OpenAI API Key:").ask()
                    model_name = questionary.select(
                        "OpenAI Model:",
                        choices=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                    ).ask()
                else:
                    api_key = os.getenv("OPENAI_API_KEY", "")
                    model_name = "gpt-4"

                llm_config = {
                    "model_name": model_name,
                    "api_key": api_key,
                    "system_content": "You are an expert metabolic modeling assistant.",
                    "max_tokens": 1000,
                    "temperature": 0.1,
                }

            elif llm_backend == "local":
                if interactive:
                    model_path = questionary.path("Local model path:").ask()
                else:
                    model_path = "./models/local_model"

                llm_config = {
                    "model_name": "local_model",
                    "model_path": model_path,
                    "system_content": "You are an expert metabolic modeling assistant.",
                    "max_tokens": 1000,
                    "temperature": 0.1,
                }

            # Create LLM instance
            if llm_backend == "argo":
                llm = ArgoLLM(llm_config)
            elif llm_backend == "openai":
                llm = OpenAILLM(llm_config)
            else:
                llm = LocalLLM(llm_config)

            config_state["llm_backend"] = llm_backend
            config_state["llm_config"] = llm_config

            progress.update(task, description="✅ LLM configured successfully")

        except Exception as e:
            progress.update(task, description="❌ LLM configuration failed")
            print_error(f"Failed to configure LLM: {e}")
            return

        # Step 2: Initialize tools
        progress.update(task, description="Initializing metabolic modeling tools...")

        try:
            tools = [
                FBATool(
                    {"name": "run_metabolic_fba", "description": "Run FBA analysis"}
                ),
                ModelAnalysisTool(
                    {
                        "name": "analyze_metabolic_model",
                        "description": "Analyze model structure",
                    }
                ),
                PathwayAnalysisTool(
                    {
                        "name": "analyze_pathway",
                        "description": "Analyze metabolic pathways",
                    }
                ),
            ]

            config_state["tools"] = tools
            progress.update(task, description="✅ Tools initialized successfully")

        except Exception as e:
            progress.update(task, description="❌ Tool initialization failed")
            print_error(f"Failed to initialize tools: {e}")
            return

        # Step 3: Create agent
        progress.update(task, description="Creating enhanced LangGraph agent...")

        try:
            agent_config = {
                "name": "modelseed_langgraph_agent",
                "description": "Enhanced ModelSEED agent with LangGraph workflows",
            }

            agent = LangGraphMetabolicAgent(llm, tools, agent_config)
            config_state["agent"] = agent

            progress.update(task, description="✅ Agent created successfully")

        except Exception as e:
            progress.update(task, description="❌ Agent creation failed")
            print_error(f"Failed to create agent: {e}")
            return

    print_success("🎉 ModelSEEDagent setup completed successfully!")

    # Show configuration summary
    console.print("\n" + "=" * 70)
    console.print(create_config_panel(config_state))
    console.print("\n[bold green]Ready to analyze metabolic models! 🚀[/bold green]")
    console.print("\nTry: [bold cyan]modelseed-agent analyze [model_file][/bold cyan]")


@app.command()
def analyze(
    model_path: str = typer.Argument(
        ..., help="Path to the metabolic model file (.xml, .sbml)"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for results"
    ),
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="Custom analysis query"
    ),
    max_iterations: int = typer.Option(
        5, "--max-iter", help="Maximum workflow iterations"
    ),
    show_visualizations: bool = typer.Option(
        True, "--viz/--no-viz", help="Open visualizations in browser"
    ),
    format_output: str = typer.Option(
        "rich", "--format", help="Output format [rich, json, table]"
    ),
):
    """
    🧬 Analyze a metabolic model with intelligent workflows

    Performs comprehensive analysis using enhanced LangGraph workflows
    with intelligent tool selection, parallel execution, and real-time
    performance monitoring.
    """
    if not config_state.get("agent"):
        print_error("Agent not configured. Run 'modelseed-agent setup' first.")
        return

    # Validate model file
    model_file = Path(model_path)
    if not model_file.exists():
        print_error(f"Model file not found: {model_path}")
        return

    if not model_file.suffix.lower() in [".xml", ".sbml"]:
        print_error("Model file must be in SBML format (.xml or .sbml)")
        return

    print_banner()
    console.print(
        f"[bold yellow]🧬 Analyzing model: [cyan]{model_file.name}[/cyan][/bold yellow]\n"
    )

    # Create default query if not provided
    if not query:
        query = f"Provide a comprehensive analysis of the {model_file.name} metabolic model including structure, growth capabilities, and pathway characteristics."

    # Prepare analysis input
    analysis_input = {
        "query": query,
        "model_path": str(model_file),
        "max_iterations": max_iterations,
    }

    # Create output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"analysis_results_{timestamp}")
        output_path.mkdir(exist_ok=True)

    console.print(f"📁 Results will be saved to: [cyan]{output_path}[/cyan]\n")

    # Run analysis with beautiful progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        task = progress.add_task(
            "🔬 Running intelligent analysis...", total=max_iterations
        )

        try:
            # Execute analysis
            result = config_state["agent"].run(analysis_input)

            # Update task completion
            progress.update(task, completed=max_iterations)

            # Store last run ID for status command
            config_state["last_run_id"] = result.metadata.get("run_id")

        except Exception as e:
            progress.update(task, description="❌ Analysis failed")
            print_error(f"Analysis failed: {e}")
            return

    # Display results based on format
    if format_output == "rich":
        display_rich_results(result, output_path)
    elif format_output == "json":
        display_json_results(result, output_path)
    elif format_output == "table":
        display_table_results(result, output_path)

    # Open visualizations if requested
    if show_visualizations and result.success:
        open_visualizations(result)

    print_success(f"🎉 Analysis completed! Results saved to {output_path}")


def display_rich_results(result, output_path: Path):
    """Display analysis results with rich formatting"""
    console.print("\n" + "=" * 80)
    console.print("[bold green]📊 Analysis Results[/bold green]", justify="center")
    console.print("=" * 80)

    # Success status
    status_text = "✅ SUCCESS" if result.success else "❌ FAILED"
    status_color = "green" if result.success else "red"
    console.print(f"\n[bold {status_color}]{status_text}[/bold {status_color}]")

    # Main results
    console.print(f"\n[bold blue]📋 Analysis Summary:[/bold blue]")
    console.print(
        Panel(
            result.message[:500] + "..."
            if len(result.message) > 500
            else result.message
        )
    )

    # Metadata table
    metadata = result.metadata

    results_table = Table(title="🔍 Execution Details", box=box.ROUNDED)
    results_table.add_column("Metric", style="bold cyan")
    results_table.add_column("Value", style="bold white")

    results_table.add_row("🆔 Run ID", metadata.get("run_id", "N/A"))
    results_table.add_row("🔧 Tools Used", ", ".join(metadata.get("tools_used", [])))
    results_table.add_row("🔄 Iterations", str(metadata.get("iterations", 0)))
    results_table.add_row(
        "⏱️ Execution Time", f"{metadata.get('total_execution_time', 0):.3f}s"
    )
    results_table.add_row(
        "🎯 Workflow Complexity", metadata.get("workflow_complexity", "unknown")
    )
    results_table.add_row(
        "📊 Visualizations", str(metadata.get("visualization_count", 0))
    )

    if metadata.get("errors"):
        results_table.add_row("⚠️ Errors", str(len(metadata.get("errors", []))))

    console.print(results_table)

    # Tool execution details
    if metadata.get("performance_metrics"):
        console.print("\n[bold blue]⚡ Performance Metrics:[/bold blue]")

        perf_table = Table(box=box.ROUNDED)
        perf_table.add_column("Tool", style="bold cyan")
        perf_table.add_column("Execution Time", style="bold green")
        perf_table.add_column("Category", style="bold yellow")
        perf_table.add_column("Data Size", style="bold blue")

        for tool_name, metrics in metadata.get("performance_metrics", {}).items():
            perf_table.add_row(
                tool_name,
                f"{metrics.get('execution_time', 0):.3f}s",
                metrics.get("category", "unknown"),
                f"{metrics.get('data_size', 0)} chars",
            )

        console.print(perf_table)

    # Intent analysis
    if metadata.get("intent_analysis"):
        intent = metadata["intent_analysis"]
        console.print(f"\n[bold blue]🧠 Query Intent Analysis:[/bold blue]")

        intent_table = Table(show_header=False, box=box.SIMPLE)
        intent_table.add_column("Aspect", style="bold cyan")
        intent_table.add_column("Value", style="white")

        intent_table.add_row("Primary Intent", intent.get("primary_intent", "unknown"))
        intent_table.add_row(
            "Suggested Tools", ", ".join(intent.get("suggested_tools", []))
        )
        intent_table.add_row("Analysis Depth", intent.get("analysis_depth", "basic"))
        intent_table.add_row("Estimated Tools", str(intent.get("estimated_tools", 0)))

        console.print(intent_table)


def display_json_results(result, output_path: Path):
    """Display and save results in JSON format"""
    json_output = {
        "success": result.success,
        "message": result.message,
        "data": result.data,
        "metadata": result.metadata,
        "timestamp": datetime.now().isoformat(),
    }

    # Save to file
    json_file = output_path / "analysis_results.json"
    with open(json_file, "w") as f:
        json.dump(json_output, f, indent=2, default=str)

    console.print(f"📄 Results saved to: [cyan]{json_file}[/cyan]")
    console.print(f"[bold green]Success:[/bold green] {result.success}")
    console.print(
        f"[bold blue]Tools Used:[/bold blue] {', '.join(result.metadata.get('tools_used', []))}"
    )


def display_table_results(result, output_path: Path):
    """Display results in table format"""
    console.print("\n[bold blue]📊 Analysis Results Table[/bold blue]")

    # Create summary table
    summary_table = Table(title="Analysis Summary", box=box.DOUBLE_EDGE)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value")

    summary_table.add_row("Status", "✅ Success" if result.success else "❌ Failed")
    summary_table.add_row("Run ID", result.metadata.get("run_id", "N/A"))
    summary_table.add_row("Tools Used", str(len(result.metadata.get("tools_used", []))))
    summary_table.add_row(
        "Execution Time", f"{result.metadata.get('total_execution_time', 0):.3f}s"
    )
    summary_table.add_row(
        "Workflow Complexity", result.metadata.get("workflow_complexity", "unknown")
    )

    console.print(summary_table)


def open_visualizations(result):
    """Open generated visualizations in browser"""
    metadata = result.metadata

    visualizations = [
        metadata.get("workflow_visualization"),
        metadata.get("final_workflow_visualization"),
        metadata.get("performance_dashboard"),
    ]

    opened_count = 0
    for viz_path in visualizations:
        if viz_path and Path(viz_path).exists():
            try:
                webbrowser.open(f"file://{Path(viz_path).absolute()}")
                opened_count += 1
            except Exception as e:
                print_warning(f"Could not open visualization: {e}")

    if opened_count > 0:
        print_info(f"🌐 Opened {opened_count} visualizations in your browser")


@app.command()
def interactive():
    """Launch interactive analysis session with conversational AI"""
    console.print("[bold blue]🚀 Starting Interactive Analysis Session...[/bold blue]")

    try:
        # Import and launch interactive CLI
        from ..interactive.interactive_cli import InteractiveCLI

        cli = InteractiveCLI()
        cli.start_interactive_session()

    except ImportError as e:
        console.print(f"[red]❌ Error importing interactive components: {e}[/red]")
        console.print(
            "[yellow]💡 Try installing missing dependencies with: pip install -r requirements.txt[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]❌ Error starting interactive session: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status():
    """
    📊 Show system status, performance metrics, and recent activity

    Displays comprehensive information about the current system state,
    performance statistics, and visualization links.
    """
    print_banner()

    # System configuration status
    console.print(create_config_panel(config_state))

    # Performance metrics if available
    if config_state.get("agent") and hasattr(config_state["agent"], "tool_integration"):
        console.print("\n[bold blue]📈 Performance Overview[/bold blue]")

        try:
            tool_integration = config_state["agent"].tool_integration
            summary = tool_integration.get_execution_summary()

            if summary.get("total_executions", 0) > 0:
                perf_table = Table(box=box.ROUNDED)
                perf_table.add_column("Metric", style="bold cyan")
                perf_table.add_column("Value", style="bold white")

                perf_table.add_row(
                    "Total Executions", str(summary.get("total_executions", 0))
                )
                perf_table.add_row(
                    "Success Rate", f"{summary.get('success_rate', 0):.1%}"
                )
                perf_table.add_row(
                    "Avg Execution Time",
                    f"{summary.get('average_execution_time', 0):.3f}s",
                )
                perf_table.add_row(
                    "Tools Used", ", ".join(summary.get("tools_used", []))
                )

                console.print(perf_table)

                # Performance insights
                insights = summary.get("performance_insights", [])
                if insights:
                    console.print(
                        "\n[bold yellow]💡 Performance Insights:[/bold yellow]"
                    )
                    for insight in insights:
                        console.print(f"  • {insight}")

                # Recommendations
                recommendations = summary.get("recommendations", [])
                if recommendations:
                    console.print("\n[bold green]🎯 Recommendations:[/bold green]")
                    for rec in recommendations:
                        console.print(f"  • {rec}")
            else:
                console.print(
                    "[dim]No performance data available yet. Run some analyses to see metrics.[/dim]"
                )

        except Exception as e:
            print_warning(f"Could not load performance metrics: {e}")

    # Recent activity
    if config_state.get("last_run_id"):
        console.print(f"\n[bold blue]🕒 Recent Activity:[/bold blue]")
        console.print(f"Last Run ID: [cyan]{config_state['last_run_id']}[/cyan]")

    # Workspace info
    console.print(f"\n[bold blue]📁 Workspace:[/bold blue]")
    workspace_table = Table(show_header=False, box=box.SIMPLE)
    workspace_table.add_column("Item", style="cyan")
    workspace_table.add_column("Path", style="white")

    workspace_table.add_row("Current Directory", str(config_state["workspace"]))

    # Check for log directory
    log_dir = Path("logs")
    if log_dir.exists():
        log_count = len(list(log_dir.glob("*")))
        workspace_table.add_row("Log Files", f"{log_count} runs in logs/")

    console.print(workspace_table)


@app.command()
def logs(
    run_id: Optional[str] = typer.Argument(None, help="Specific run ID to view"),
    last: int = typer.Option(5, "--last", "-l", help="Show last N runs"),
    open_viz: bool = typer.Option(
        False, "--open-viz", help="Open visualizations for the run"
    ),
):
    """
    📜 View execution logs and run history

    Browse through previous analysis runs, view detailed logs,
    and access generated visualizations.
    """
    print_banner()
    console.print("[bold blue]📜 Execution Logs[/bold blue]\n")

    log_dir = Path("logs")
    if not log_dir.exists():
        print_warning("No logs directory found. Run some analyses first.")
        return

    # Get all run directories
    run_dirs = sorted(
        [d for d in log_dir.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if not run_dirs:
        print_warning("No run logs found.")
        return

    if run_id:
        # Show specific run
        matching_runs = [d for d in run_dirs if run_id in d.name]
        if not matching_runs:
            print_error(f"No run found with ID containing '{run_id}'")
            return

        show_run_details(matching_runs[0], open_viz)
    else:
        # Show recent runs
        console.print(
            f"[bold green]📋 Last {min(last, len(run_dirs))} runs:[/bold green]"
        )

        runs_table = Table(box=box.ROUNDED)
        runs_table.add_column("Run ID", style="bold cyan")
        runs_table.add_column("Timestamp", style="bold white")
        runs_table.add_column("Files", style="bold green")
        runs_table.add_column("Size", style="bold yellow")

        for run_dir in run_dirs[:last]:
            run_id = run_dir.name.replace("langgraph_run_", "")
            timestamp = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            file_count = len(list(run_dir.rglob("*")))
            size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
            size_str = (
                f"{size / 1024:.1f} KB"
                if size < 1024 * 1024
                else f"{size / (1024*1024):.1f} MB"
            )

            runs_table.add_row(run_id, timestamp, str(file_count), size_str)

        console.print(runs_table)
        console.print(
            f"\n[dim]Use 'modelseed-agent logs [run_id]' to view details of a specific run[/dim]"
        )


def show_run_details(run_dir: Path, open_viz: bool = False):
    """Show detailed information about a specific run"""
    console.print(f"[bold green]📋 Run Details: {run_dir.name}[/bold green]")

    # Check for execution log
    log_file = run_dir / "execution_log.json"
    if log_file.exists():
        try:
            with open(log_file) as f:
                log_data = json.load(f)

            console.print(
                f"\n[bold blue]📊 Execution Steps ({len(log_data)} steps):[/bold blue]"
            )

            for i, step in enumerate(log_data[-5:], 1):  # Show last 5 steps
                step_data = step.get("data", {})
                console.print(
                    f"  {i}. [cyan]{step.get('step', 'unknown')}[/cyan] - {step_data.get('decision', 'N/A')}"
                )

        except Exception as e:
            print_warning(f"Could not read execution log: {e}")

    # Check for visualizations
    viz_dir = run_dir / "visualizations"
    if viz_dir.exists():
        viz_files = list(viz_dir.glob("*.html"))
        console.print(f"\n[bold blue]🎨 Visualizations ({len(viz_files)}):[/bold blue]")

        for viz_file in viz_files:
            console.print(f"  • [cyan]{viz_file.name}[/cyan]")

            if open_viz:
                try:
                    webbrowser.open(f"file://{viz_file.absolute()}")
                    print_info(f"Opened {viz_file.name}")
                except Exception as e:
                    print_warning(f"Could not open {viz_file.name}: {e}")


if __name__ == "__main__":
    app()
