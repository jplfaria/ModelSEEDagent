#!/usr/bin/env python3
"""
ModelSEEDagent Professional CLI

A beautiful, modern command-line interface for metabolic modeling
with intelligent workflow capabilities, real-time visualization,
and comprehensive observability.
"""

import glob
import json
import os
import sys
import uuid
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

# Add project root to Python path for imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import warnings suppression for module reload issues
import warnings

warnings.filterwarnings(
    "ignore", message=".*found in sys.modules.*", category=RuntimeWarning
)

from src.agents import create_real_time_agent
from src.agents.langgraph_metabolic import LangGraphMetabolicAgent
from src.agents.tool_integration import EnhancedToolIntegration
from src.cli.audit_viewer import audit_app as ai_audit_app
from src.config.debug_config import configure_logging, print_debug_status
from src.interactive.streaming_interface import RealTimeStreamingInterface
from src.llm.argo import ArgoLLM
from src.llm.local_llm import LocalLLM
from src.llm.openai_llm import OpenAILLM
from src.tools.biochem.resolver import BiochemEntityResolverTool, BiochemSearchTool
from src.tools.cobra.advanced_media_ai import (
    AuxotrophyPredictionTool,
    MediaOptimizationTool,
)
from src.tools.cobra.analysis import ModelAnalysisTool, PathwayAnalysisTool
from src.tools.cobra.auxotrophy import AuxotrophyTool
from src.tools.cobra.essentiality import EssentialityAnalysisTool
from src.tools.cobra.fba import FBATool
from src.tools.cobra.flux_sampling import FluxSamplingTool
from src.tools.cobra.flux_variability import FluxVariabilityTool
from src.tools.cobra.gene_deletion import GeneDeletionTool
from src.tools.cobra.media_tools import (
    MediaComparatorTool,
    MediaCompatibilityTool,
    MediaManipulatorTool,
    MediaSelectorTool,
)
from src.tools.cobra.minimal_media import MinimalMediaTool
from src.tools.cobra.missing_media import MissingMediaTool
from src.tools.cobra.production_envelope import ProductionEnvelopeTool
from src.tools.cobra.reaction_expression import ReactionExpressionTool
from src.tools.modelseed import (
    GapFillTool,
    ModelBuildTool,
    ModelCompatibilityTool,
    RastAnnotationTool,
)

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

# CLI-specific configuration file for persistence
CLI_CONFIG_FILE = Path.home() / ".modelseed-agent-cli.json"


def load_cli_config() -> Dict[str, Any]:
    """Load CLI configuration from persistent storage"""
    if CLI_CONFIG_FILE.exists():
        try:
            with open(CLI_CONFIG_FILE, "r") as f:
                config = json.load(f)

            # Auto-recreate tools and agent if configuration is available
            if config.get("llm_config") and config.get("llm_backend"):
                try:
                    # Recreate LLM
                    llm_config = config["llm_config"]
                    llm_backend = config["llm_backend"]

                    # Use factory for connection pooling
                    from src.llm.factory import LLMFactory

                    llm = LLMFactory.create(llm_backend, llm_config)

                    # Recreate tools
                    tools = [
                        # COBRA.py tools - Basic Analysis
                        FBATool(
                            {
                                "name": "run_metabolic_fba",
                                "description": "Run FBA analysis",
                            }
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
                        # COBRA.py tools - Advanced Analysis
                        FluxVariabilityTool(
                            {
                                "name": "run_flux_variability_analysis",
                                "description": "Run FVA to determine flux ranges",
                            }
                        ),
                        GeneDeletionTool(
                            {
                                "name": "run_gene_deletion_analysis",
                                "description": "Analyze gene deletion effects",
                            }
                        ),
                        EssentialityAnalysisTool(
                            {
                                "name": "analyze_essentiality",
                                "description": "Identify essential genes and reactions",
                            }
                        ),
                        FluxSamplingTool(
                            {
                                "name": "run_flux_sampling",
                                "description": "Sample flux space for statistical analysis",
                            }
                        ),
                        ProductionEnvelopeTool(
                            {
                                "name": "run_production_envelope",
                                "description": "Analyze growth vs production trade-offs",
                            }
                        ),
                        AuxotrophyTool(
                            {
                                "name": "identify_auxotrophies",
                                "description": "Identify potential auxotrophies by testing nutrient removal",
                            }
                        ),
                        MinimalMediaTool(
                            {
                                "name": "find_minimal_media",
                                "description": "Determine minimal media components required for growth",
                            }
                        ),
                        MissingMediaTool(
                            {
                                "name": "find_missing_media",
                                "description": "Find missing media components preventing growth",
                            }
                        ),
                        # AI Media Tools - Intelligent media management
                        MediaSelectorTool(
                            {
                                "name": "select_optimal_media",
                                "description": "AI-powered selection of optimal media for models",
                            }
                        ),
                        MediaManipulatorTool(
                            {
                                "name": "manipulate_media_composition",
                                "description": "Modify media using natural language commands",
                            }
                        ),
                        MediaCompatibilityTool(
                            {
                                "name": "analyze_media_compatibility",
                                "description": "Analyze media-model compatibility with AI suggestions",
                            }
                        ),
                        MediaComparatorTool(
                            {
                                "name": "compare_media_performance",
                                "description": "Compare model performance across different media",
                            }
                        ),
                        # Advanced AI Media Tools
                        MediaOptimizationTool(
                            {
                                "name": "optimize_media_composition",
                                "description": "AI-driven media optimization for specific growth targets",
                            }
                        ),
                        AuxotrophyPredictionTool(
                            {
                                "name": "predict_auxotrophies",
                                "description": "AI-powered auxotrophy prediction from model gaps",
                            }
                        ),
                        ReactionExpressionTool(
                            {
                                "name": "analyze_reaction_expression",
                                "description": "Analyze reaction expression levels from omics data",
                            }
                        ),
                        # ModelSEED tools
                        RastAnnotationTool(
                            {
                                "name": "annotate_genome_rast",
                                "description": "Annotate genome using RAST",
                            }
                        ),
                        ModelBuildTool(
                            {
                                "name": "build_metabolic_model",
                                "description": "Build metabolic model from genome",
                            }
                        ),
                        GapFillTool(
                            {
                                "name": "gapfill_model",
                                "description": "Gapfill metabolic model",
                            }
                        ),
                        ModelCompatibilityTool(
                            {
                                "name": "test_modelseed_cobra_compatibility",
                                "description": "Test ModelSEED-COBRApy compatibility",
                            }
                        ),
                        # Biochemistry resolution tools
                        BiochemEntityResolverTool(
                            {
                                "name": "resolve_biochem_entity",
                                "description": "Resolve biochemistry entity IDs to human-readable names",
                            }
                        ),
                        BiochemSearchTool(
                            {
                                "name": "search_biochem",
                                "description": "Search biochemistry database for compounds and reactions",
                            }
                        ),
                    ]

                    # Store configuration for lazy agent creation
                    # Don't create agent immediately to avoid initialization spam
                    config["tools"] = tools
                    config["agent"] = None  # Will be created on demand
                    config["agent_factory"] = lambda: get_or_create_cached_agent(
                        llm, tools
                    )

                except Exception as e:
                    # If recreation fails, just log warning and continue with basic config
                    print(f"Warning: Could not recreate agent from saved config: {e}")

            return config
        except Exception:
            pass

    # Return default config if file doesn't exist or is corrupted
    return {
        "llm_backend": None,
        "llm_config": None,
        "tools": None,
        "agent": None,
        "last_run_id": None,
        "workspace": str(Path.cwd()),
    }


def save_cli_config(config: Dict[str, Any]) -> None:
    """Save CLI configuration to persistent storage"""
    try:
        # Don't save runtime objects like agents
        saveable_config = {
            "llm_backend": config.get("llm_backend"),
            "llm_config": config.get("llm_config"),
            "tools": None,  # Tools are recreated from config
            "agent": None,  # Agents are runtime objects
            "last_run_id": config.get("last_run_id"),
            "workspace": config.get("workspace", str(Path.cwd())),
        }

        with open(CLI_CONFIG_FILE, "w") as f:
            json.dump(saveable_config, f, indent=2)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save configuration: {e}[/yellow]")


# Global state for configuration (now persistent)
config_state = load_cli_config()

# Initialize debug configuration early
configure_logging()

# Agent caching to prevent excessive recreations
_agent_cache = {}


def get_or_create_cached_agent(llm, tools):
    """Get cached agent or create new one if needed"""
    # Create cache key based on LLM type and tool count
    llm_key = f"{type(llm).__name__}_{getattr(llm, 'model_name', 'unknown')}"
    cache_key = f"{llm_key}_{len(tools)}"

    if cache_key not in _agent_cache:
        from src.agents.langgraph_metabolic import LangGraphMetabolicAgent

        agent_config = {
            "name": "modelseed_langgraph_agent",
            "description": "Enhanced ModelSEED agent with LangGraph workflows",
        }
        _agent_cache[cache_key] = LangGraphMetabolicAgent(llm, tools, agent_config)

    return _agent_cache[cache_key]


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

    model_name = config.get("llm_config") or {}
    model_name = model_name.get("model_name", "Not set")
    config_table.add_row(
        "🧠 Model",
        (
            f"[green]{model_name}[/green]"
            if model_name != "Not set"
            else "[red]Not set[/red]"
        ),
    )

    tools_count = len(config.get("tools") or [])
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
            else f"[red]{agent_status}[/red]"
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
            "modelseed-agent debug", "🔍 Show debug configuration and control logging"
        )
        commands_table.add_row(
            "modelseed-agent audit list", "🔍 Review tool execution history"
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
    ),
    model_name: str = typer.Option(
        None, "--model", "-m", help="Model name to use (overrides defaults)"
    ),
    interactive: bool = typer.Option(
        True, "--interactive/--non-interactive", help="Use interactive setup"
    ),
):
    """
    🔧 Configure ModelSEEDagent with LLM backend and tools

    Sets up the agent with your preferred LLM backend and initializes
    all necessary tools for metabolic modeling analysis.

    Environment variables for defaults:
    - DEFAULT_LLM_BACKEND: argo, openai, or local
    - DEFAULT_MODEL_NAME: model name for the backend
    - ARGO_USER: username for Argo Gateway
    """
    print_banner()
    console.print("[bold yellow]🔧 Setting up ModelSEEDagent...[/bold yellow]\n")

    # Get defaults from environment or use fallbacks
    default_backend = os.getenv("DEFAULT_LLM_BACKEND", "argo")
    default_model = os.getenv(
        "DEFAULT_MODEL_NAME", "gpto1"
    )  # Default to gpto1 reasoning model

    # Use provided backend or fall back to default/prompt
    if not llm_backend:
        if interactive:
            # Create simple string choices for questionary
            choice_map = {
                "🧬 Argo Gateway (Recommended)": "argo",
                "🤖 OpenAI API": "openai",
                "💻 Local LLM": "local",
            }

            # Map backend values to display text for default
            backend_display_map = {
                "argo": "🧬 Argo Gateway (Recommended)",
                "openai": "🤖 OpenAI API",
                "local": "💻 Local LLM",
            }

            selected_display = questionary.select(
                "Choose LLM backend:",
                choices=list(choice_map.keys()),
                default=backend_display_map.get(
                    default_backend, "🧬 Argo Gateway (Recommended)"
                ),
            ).ask()

            # Map display text back to backend value
            llm_backend = choice_map.get(selected_display, "argo")
        else:
            llm_backend = default_backend

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
                # Argo Gateway configuration with ACTUAL available models
                argo_models = {
                    # Dev Environment Models - prefer gpto1 as default
                    "gpto1": "GPT-o1 (Reasoning, Recommended)",
                    "gpt4o": "GPT-4o (Latest)",
                    "gpt4olatest": "GPT-4o Latest",
                    "gpto1mini": "GPT-o1 Mini",
                    "gpto1preview": "GPT-o1 Preview",
                    "gpto3mini": "GPT-o3 Mini",
                    # Prod Environment Models
                    "gpt35": "GPT-3.5",
                    "gpt35large": "GPT-3.5 Large",
                    "gpt4": "GPT-4",
                    "gpt4large": "GPT-4 Large",
                    "gpt4turbo": "GPT-4 Turbo",
                }

                if interactive:
                    user = questionary.text(
                        "Argo Username:", default=os.getenv("ARGO_USER", os.getlogin())
                    ).ask()

                    # Use provided model or prompt for selection
                    if not model_name:
                        # Create simple string choices and mapping (same fix as backend selection)
                        model_display_map = {v: k for k, v in argo_models.items()}

                        # Set default - prefer gpto1 if available, otherwise use provided default
                        if default_model in argo_models:
                            default_display = argo_models[default_model]
                        else:
                            default_display = (
                                "GPT-o1 (Reasoning, Recommended)"  # Default to gpto1
                            )

                        selected_display = questionary.select(
                            "Choose Argo model:",
                            choices=list(argo_models.values()),
                            default=default_display,
                        ).ask()

                        # Map display text back to model name
                        model_name = model_display_map.get(selected_display, "gpto1")

                    # Show helpful info about o-series models
                    if model_name.startswith("gpto"):
                        console.print(
                            f"[yellow]ℹ️  Note: {model_name} is a reasoning model with special behavior:[/yellow]"
                        )
                        console.print(
                            "   • No temperature parameter (reasoning models use fixed temperature)"
                        )
                        console.print(
                            "   • Uses max_completion_tokens instead of max_tokens"
                        )
                        console.print(
                            "   • May work better without token limits for complex queries"
                        )
                else:
                    user = os.getenv("ARGO_USER", "default_user")
                    model_name = (
                        model_name or "gpto1"
                    )  # Default to gpto1 in non-interactive mode

                # Build configuration with proper handling for o-series models
                llm_config = {
                    "model_name": model_name,
                    "user": user,
                    "system_content": "You are an expert metabolic modeling assistant.",
                }

                # Handle token limits based on model type
                if model_name.startswith("gpto") or model_name.startswith("o"):
                    # For o-series models, be more conservative with max_completion_tokens
                    # and allow option to disable it completely
                    if interactive:
                        use_token_limit = questionary.confirm(
                            f"Set token limit for {model_name}? (Some queries work better without limits)",
                            default=False,
                        ).ask()
                        if use_token_limit:
                            llm_config["max_tokens"] = questionary.text(
                                "Max completion tokens:", default="1000"
                            ).ask()
                    # For non-interactive, don't set token limit for o-series by default
                    # Note: o-series models don't support temperature parameter
                else:
                    # Standard models get normal configuration
                    llm_config["max_tokens"] = 1000
                    llm_config["temperature"] = 0.1

                # Note: API key is optional for users on ANL network
                # llm_config does not include api_key field unless explicitly set

            elif llm_backend == "openai":
                if interactive:
                    api_key = questionary.password("OpenAI API Key:").ask()
                    model_choice = questionary.select(
                        "OpenAI Model:",
                        choices=["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
                        default="gpt-4o",
                    ).ask()
                else:
                    api_key = os.getenv("OPENAI_API_KEY", "")
                    model_choice = model_name or "gpt-4o"

                llm_config = {
                    "model_name": model_choice,
                    "api_key": api_key,
                    "system_content": "You are an expert metabolic modeling assistant.",
                    "max_tokens": 1000,
                    "temperature": 0.1,
                }

            elif llm_backend == "local":
                # Map model names to actual paths
                model_name = model_name or "llama-3.2-3b"  # Default to smaller model

                # Define available local models with their paths
                local_model_paths = {
                    "llama-3.1-8b": os.getenv(
                        "LLAMA_8B_PATH",
                        "/Users/jplfaria/.llama/checkpoints/Llama3.1-8B",
                    ),
                    "llama-3.2-3b": os.getenv(
                        "LLAMA_3B_PATH",
                        "/Users/jplfaria/.llama/checkpoints/Llama3.2-3B",
                    ),
                }

                # Check if the provided model is a known name or a direct path
                if model_name in local_model_paths:
                    model_path = local_model_paths[model_name]
                elif model_name and Path(model_name).exists():
                    # User provided a direct path
                    model_path = model_name
                    # Extract a friendly name from the path
                    model_name = Path(model_name).name
                else:
                    # Default to 3B model if unknown model requested
                    if model_name not in local_model_paths:
                        print_warning(
                            f"Unknown local model '{model_name}', using llama-3.2-3b"
                        )
                        model_name = "llama-3.2-3b"
                    model_path = local_model_paths[model_name]

                llm_config = {
                    "model_name": model_name,
                    "model_path": model_path,
                    "system_content": "You are an expert metabolic modeling assistant.",
                    "max_tokens": 1000,
                    "temperature": 0.1,
                    "device": "mps",  # Default to MPS for Mac
                }

            # Create LLM instance
            # Use factory for connection pooling
            from src.llm.factory import LLMFactory

            llm = LLMFactory.create(llm_backend, llm_config)

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
                # COBRA.py tools - Basic Analysis
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
                # COBRA.py tools - Advanced Analysis
                FluxVariabilityTool(
                    {
                        "name": "run_flux_variability_analysis",
                        "description": "Run FVA to determine flux ranges",
                    }
                ),
                GeneDeletionTool(
                    {
                        "name": "run_gene_deletion_analysis",
                        "description": "Analyze gene deletion effects",
                    }
                ),
                EssentialityAnalysisTool(
                    {
                        "name": "analyze_essentiality",
                        "description": "Identify essential genes and reactions",
                    }
                ),
                FluxSamplingTool(
                    {
                        "name": "run_flux_sampling",
                        "description": "Sample flux space for statistical analysis",
                    }
                ),
                ProductionEnvelopeTool(
                    {
                        "name": "run_production_envelope",
                        "description": "Analyze growth vs production trade-offs",
                    }
                ),
                AuxotrophyTool(
                    {
                        "name": "identify_auxotrophies",
                        "description": "Identify potential auxotrophies by testing nutrient removal",
                    }
                ),
                MinimalMediaTool(
                    {
                        "name": "find_minimal_media",
                        "description": "Determine minimal media components required for growth",
                    }
                ),
                MissingMediaTool(
                    {
                        "name": "find_missing_media",
                        "description": "Find missing media components preventing growth",
                    }
                ),
                # AI Media Tools - Intelligent media management
                MediaSelectorTool(
                    {
                        "name": "select_optimal_media",
                        "description": "AI-powered selection of optimal media for models",
                    }
                ),
                MediaManipulatorTool(
                    {
                        "name": "manipulate_media_composition",
                        "description": "Modify media using natural language commands",
                    }
                ),
                MediaCompatibilityTool(
                    {
                        "name": "analyze_media_compatibility",
                        "description": "Analyze media-model compatibility with AI suggestions",
                    }
                ),
                MediaComparatorTool(
                    {
                        "name": "compare_media_performance",
                        "description": "Compare model performance across different media",
                    }
                ),
                # Advanced AI Media Tools - Optimization and prediction
                MediaOptimizationTool(
                    {
                        "name": "optimize_media_composition",
                        "description": "AI-driven media optimization for specific growth targets",
                    }
                ),
                AuxotrophyPredictionTool(
                    {
                        "name": "predict_auxotrophies",
                        "description": "AI-powered auxotrophy prediction from model gaps",
                    }
                ),
                ReactionExpressionTool(
                    {
                        "name": "analyze_reaction_expression",
                        "description": "Analyze reaction expression levels from omics data",
                    }
                ),
                # ModelSEED tools
                RastAnnotationTool(
                    {
                        "name": "annotate_genome_rast",
                        "description": "Annotate genome using RAST",
                    }
                ),
                ModelBuildTool(
                    {
                        "name": "build_metabolic_model",
                        "description": "Build metabolic model from genome",
                    }
                ),
                GapFillTool(
                    {"name": "gapfill_model", "description": "Gapfill metabolic model"}
                ),
                ModelCompatibilityTool(
                    {
                        "name": "test_modelseed_cobra_compatibility",
                        "description": "Test ModelSEED-COBRApy compatibility",
                    }
                ),
                # Biochemistry resolution tools
                BiochemEntityResolverTool(
                    {
                        "name": "resolve_biochem_entity",
                        "description": "Resolve biochemistry entity IDs to human-readable names",
                    }
                ),
                BiochemSearchTool(
                    {
                        "name": "search_biochem",
                        "description": "Search biochemistry database for compounds and reactions",
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

    # Save configuration
    save_cli_config(config_state)


def run_streaming_analysis(
    query: str,
    model_file: Path,
    analysis_input: Dict[str, Any],
    log_llm_inputs: bool = False,
):
    """Run analysis with real-time streaming interface"""
    console.print("\n[bold cyan]🚀 Starting Real-Time AI Analysis[/bold cyan]")
    console.print("[dim]Watch the AI make decisions in real-time...[/dim]\n")

    try:
        # Get LLM and tools from config
        llm_config = config_state.get("llm_config")
        llm_backend = config_state.get("llm_backend")

        if not llm_config or not llm_backend:
            print_error("LLM not configured. Run 'modelseed-agent setup' first.")
            return None

        # Create LLM using factory for connection pooling
        try:
            from src.llm.factory import LLMFactory

            llm = LLMFactory.create(llm_backend, llm_config)
        except Exception as e:
            print_error(f"Failed to create LLM: {e}")
            return None

        # Get tools
        tools = config_state.get("tools", [])
        if not tools:
            print_error("No tools configured. Run 'modelseed-agent setup' first.")
            return None

        # Create dynamic agent with optional LLM input logging
        agent_config = {
            "max_iterations": 6,
            "log_llm_inputs": log_llm_inputs,  # Pass the CLI flag to agent
        }
        dynamic_agent = create_real_time_agent(llm, tools, agent_config)

        # Create streaming interface
        streaming = RealTimeStreamingInterface()

        # Start streaming analysis
        streaming.start_streaming(query)

        # Show initial AI thinking
        streaming.show_ai_analysis(
            "Planning comprehensive metabolic analysis approach..."
        )

        # Add model file to the query context
        enhanced_query = f"{query}\n\nModel file: {model_file}\n\nPlease load and analyze this metabolic model comprehensively."

        # Show model loading
        streaming.show_ai_analysis(f"Loading metabolic model: {model_file.name}")

        # Run the dynamic agent
        result = dynamic_agent.run({"query": enhanced_query})

        # Show completion
        if result.success:
            streaming.show_workflow_complete(result.message, result.metadata)
        else:
            streaming.show_error(result.error)

        # Stop streaming after brief pause
        import time

        time.sleep(1.5)
        streaming.stop_streaming()

        return result

    except Exception as e:
        print_error(f"Streaming analysis failed: {e}")
        return None


def run_regular_analysis(analysis_input: Dict[str, Any], max_iterations: int):
    """Run analysis with regular progress display"""
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

            return result

        except Exception as e:
            progress.update(task, description="❌ Analysis failed")
            print_error(f"Analysis failed: {e}")
            return None


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
    stream: bool = typer.Option(
        False,
        "--stream",
        help="Use real-time streaming AI agent with live reasoning display",
    ),
    log_llm_inputs: bool = typer.Option(
        False,
        "--log-llm-inputs",
        help="🔍 Log complete LLM inputs (prompts + tool data) for analysis",
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

    # Choose between streaming and regular analysis
    if stream:
        result = run_streaming_analysis(
            query, model_file, analysis_input, log_llm_inputs
        )
    else:
        result = run_regular_analysis(analysis_input, max_iterations)

    if not result:
        return

    # Store last run ID for status command
    config_state["last_run_id"] = result.metadata.get("run_id")

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
            "[yellow]💡 Try installing missing dependencies with: pip install .[all][/yellow]"
        )
    except Exception as e:
        console.print(f"[red]❌ Error starting interactive session: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def debug():
    """
    🔍 Show debug configuration and logging control

    Display current debug settings and environment variables that control
    different levels of logging verbosity for different components.

    Environment Variables:
    - MODELSEED_DEBUG_LEVEL: overall debug level (quiet, normal, verbose, trace)
    - MODELSEED_DEBUG_COBRAKBASE: enable cobrakbase debug messages (true/false)
    - MODELSEED_DEBUG_LANGGRAPH: enable LangGraph initialization debug (true/false)
    - MODELSEED_DEBUG_HTTP: enable HTTP/SSL debug messages (true/false)
    - MODELSEED_DEBUG_TOOLS: enable tool execution debug (true/false)
    - MODELSEED_DEBUG_LLM: enable LLM interaction debug (true/false)
    - MODELSEED_LOG_LLM_INPUTS: enable complete LLM input logging (true/false)
    - MODELSEED_CAPTURE_CONSOLE_DEBUG: capture console debug output (true/false)
    - MODELSEED_CAPTURE_AI_REASONING_FLOW: capture AI reasoning steps (true/false)
    - MODELSEED_CAPTURE_FORMATTED_RESULTS: capture final formatted results (true/false)

    Intelligence Framework Variables:
    - MODELSEED_DEBUG_INTELLIGENCE: enable Intelligence Framework debug (true/false)
    - MODELSEED_DEBUG_PROMPTS: enable prompt registry debug (true/false)
    - MODELSEED_DEBUG_CONTEXT_ENHANCEMENT: enable context enhancer debug (true/false)
    - MODELSEED_DEBUG_QUALITY_ASSESSMENT: enable quality validator debug (true/false)
    - MODELSEED_DEBUG_ARTIFACT_INTELLIGENCE: enable artifact intelligence debug (true/false)
    - MODELSEED_DEBUG_SELF_REFLECTION: enable self-reflection engine debug (true/false)
    - MODELSEED_TRACE_REASONING_WORKFLOW: trace complete reasoning workflow (true/false)
    """
    print_banner()
    console.print("[bold blue]🔍 Debug Configuration Status[/bold blue]\n")

    # Print debug status using the dedicated function
    print_debug_status()

    console.print("\n[bold green]💡 Tips for Debug Control:[/bold green]")
    console.print("   • Set MODELSEED_DEBUG_LEVEL=quiet to minimize all debug output")
    console.print("   • Set MODELSEED_DEBUG_LEVEL=trace to enable all debug messages")
    console.print("   • Use component-specific flags to control individual debug areas")
    console.print("   • Set MODELSEED_LOG_LLM_INPUTS=true for detailed LLM analysis")

    console.print("\n[bold cyan]💡 Console Capture Control:[/bold cyan]")
    console.print(
        "   • Set MODELSEED_CAPTURE_CONSOLE_DEBUG=true to capture console debug output"
    )
    console.print(
        "   • Set MODELSEED_CAPTURE_AI_REASONING_FLOW=true to capture AI reasoning steps"
    )
    console.print(
        "   • Set MODELSEED_CAPTURE_FORMATTED_RESULTS=true to capture final results"
    )

    console.print("\n[bold magenta]🧠 Intelligence Framework Debug:[/bold magenta]")
    console.print(
        "   • Set MODELSEED_DEBUG_INTELLIGENCE=true to enable Intelligence Framework debug"
    )
    console.print(
        "   • Set MODELSEED_DEBUG_PROMPTS=true to debug prompt registry issues"
    )
    console.print(
        "   • Set MODELSEED_DEBUG_QUALITY_ASSESSMENT=true to debug quality validation"
    )
    console.print(
        "   • Set MODELSEED_TRACE_REASONING_WORKFLOW=true to trace complete reasoning workflow"
    )
    console.print(
        "   • Use component-specific flags to isolate Intelligence Framework issues"
    )

    console.print("\n[bold yellow]📝 Example Usage:[/bold yellow]")
    console.print("   # General debugging")
    console.print("   export MODELSEED_DEBUG_LEVEL=verbose")
    console.print("   export MODELSEED_DEBUG_COBRAKBASE=true")
    console.print("   export MODELSEED_DEBUG_LANGGRAPH=false")

    console.print("\n   # Intelligence Framework debugging")
    console.print("   export MODELSEED_DEBUG_INTELLIGENCE=true")
    console.print("   export MODELSEED_DEBUG_PROMPTS=true")
    console.print("   export MODELSEED_TRACE_REASONING_WORKFLOW=true")

    console.print("\n   # Console capture")
    console.print("   export MODELSEED_CAPTURE_AI_REASONING_FLOW=true")
    console.print("   modelseed-agent interactive")


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

    # Intelligence Framework status
    try:
        from src.config.debug_config import (
            get_intelligence_debug_config,
            is_intelligence_debug_enabled,
        )

        console.print("\n[bold magenta]🧠 Intelligence Framework Status[/bold magenta]")

        intelligence_table = Table(box=box.ROUNDED)
        intelligence_table.add_column("Component", style="bold cyan")
        intelligence_table.add_column("Status", style="bold white")
        intelligence_table.add_column("Debug Level", style="yellow")

        # Check component availability
        components = [
            ("Enhanced Prompt Provider", "src.reasoning.enhanced_prompt_provider"),
            ("Context Enhancer", "src.reasoning.context_enhancer"),
            ("Quality Assessment", "src.reasoning.quality_validator"),
            ("Artifact Intelligence", "src.reasoning.artifact_intelligence"),
            ("Self-Reflection Engine", "src.reasoning.self_reflection_engine"),
            (
                "Intelligent Reasoning System",
                "src.reasoning.intelligent_reasoning_system",
            ),
        ]

        debug_config = get_intelligence_debug_config()

        for component_name, module_name in components:
            try:
                __import__(module_name)
                status = "[green]✅ Available[/green]"
            except ImportError:
                status = "[red]❌ Missing[/red]"

            # Determine debug level for this component
            component_key = (
                component_name.lower()
                .replace(" ", "_")
                .replace("enhanced_", "")
                .replace("_provider", "")
                .replace("_engine", "")
                .replace("_system", "")
            )
            debug_key = f"{component_key}_debug"

            if debug_key in debug_config and debug_config[debug_key]:
                debug_level = "[green]DEBUG[/green]"
            elif debug_config.get("intelligence_debug", False):
                debug_level = "[yellow]INFO[/yellow]"
            else:
                debug_level = "[dim]NORMAL[/dim]"

            intelligence_table.add_row(component_name, status, debug_level)

        console.print(intelligence_table)

        # Intelligence Framework configuration summary
        if is_intelligence_debug_enabled():
            console.print(
                "\n[green]Intelligence Framework debugging is ENABLED[/green]"
            )
        else:
            console.print("\n[dim]Intelligence Framework debugging is disabled[/dim]")
            console.print("   Use: export MODELSEED_DEBUG_INTELLIGENCE=true to enable")

    except Exception as e:
        console.print(
            f"\n[red]Could not check Intelligence Framework status: {e}[/red]"
        )

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


@app.command()
def switch(
    backend: str = typer.Argument(
        ..., help="Backend to switch to [argo, openai, local]"
    ),
    model: str = typer.Option(
        None, "--model", "-m", help="Model to use with the backend"
    ),
):
    """
    🔄 Quick switch between LLM backends

    Examples:
      modelseed-agent switch argo              # Switch to Argo with default gpt4o
      modelseed-agent switch argo --model gpto1  # Switch to Argo with gpt-o1
      modelseed-agent switch openai           # Switch to OpenAI with default
      modelseed-agent switch local            # Switch to local LLM
    """
    valid_backends = ["argo", "openai", "local"]
    if backend not in valid_backends:
        print_error(
            f"Invalid backend '{backend}'. Choose from: {', '.join(valid_backends)}"
        )
        return

    console.print(f"[bold cyan]🔄 Switching to {backend} backend...[/bold cyan]")

    # Quick configuration based on backend
    if backend == "argo":
        default_model = model or os.getenv("DEFAULT_MODEL_NAME", "gpt4o")
        user = os.getenv("ARGO_USER", os.getlogin())

        llm_config = {
            "model_name": default_model,
            "user": user,
            "system_content": "You are an expert metabolic modeling assistant.",
        }

        # Handle token configuration for o-series models
        if not (default_model.startswith("gpto") or default_model.startswith("o")):
            llm_config["max_tokens"] = 1000
            llm_config["temperature"] = 0.1
        # Note: o-series models don't get temperature parameter

    elif backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print_error("OPENAI_API_KEY environment variable not set")
            print_info("Set it with: export OPENAI_API_KEY='your_key_here'")
            return

        llm_config = {
            "model_name": model or "gpt-4o",
            "api_key": api_key,
            "system_content": "You are an expert metabolic modeling assistant.",
            "max_tokens": 1000,
            "temperature": 0.1,
        }

    else:  # local
        # Map model names to actual paths
        model_name = model or "llama-3.2-3b"  # Default to smaller model

        # Define available local models with their paths
        local_model_paths = {
            "llama-3.1-8b": os.getenv(
                "LLAMA_8B_PATH", "/Users/jplfaria/.llama/checkpoints/Llama3.1-8B"
            ),
            "llama-3.2-3b": os.getenv(
                "LLAMA_3B_PATH", "/Users/jplfaria/.llama/checkpoints/Llama3.2-3B"
            ),
        }

        # Check if the provided model is a known name or a direct path
        if model_name in local_model_paths:
            model_path = local_model_paths[model_name]
        elif model_name and Path(model_name).exists():
            # User provided a direct path
            model_path = model_name
            # Extract a friendly name from the path
            model_name = Path(model_name).name
        else:
            # Default to 3B model if unknown model requested
            if model_name not in local_model_paths:
                print_warning(f"Unknown local model '{model_name}', using llama-3.2-3b")
                model_name = "llama-3.2-3b"
            model_path = local_model_paths[model_name]

        llm_config = {
            "model_name": model_name,
            "model_path": model_path,
            "system_content": "You are an expert metabolic modeling assistant.",
            "max_tokens": 1000,
            "temperature": 0.1,
            "device": "mps",  # Default to MPS for Mac
        }

    try:
        # Create LLM instance using factory for connection pooling
        from src.llm.factory import LLMFactory

        llm = LLMFactory.create(backend, llm_config)

        # Initialize tools (reuse existing if available)
        tools = config_state.get("tools") or [
            FBATool({"name": "run_metabolic_fba", "description": "Run FBA analysis"}),
            ModelAnalysisTool(
                {
                    "name": "analyze_metabolic_model",
                    "description": "Analyze model structure",
                }
            ),
            PathwayAnalysisTool(
                {"name": "analyze_pathway", "description": "Analyze metabolic pathways"}
            ),
            # Add AI Media Tools to fallback list
            MediaSelectorTool(
                {
                    "name": "select_optimal_media",
                    "description": "AI-powered selection of optimal media for models",
                }
            ),
            MediaManipulatorTool(
                {
                    "name": "manipulate_media_composition",
                    "description": "Modify media using natural language commands",
                }
            ),
            # Add advanced AI media tools to fallback
            MediaOptimizationTool(
                {
                    "name": "optimize_media_composition",
                    "description": "AI-driven media optimization for specific growth targets",
                }
            ),
        ]

        # Create agent
        agent_config = {
            "name": "modelseed_langgraph_agent",
            "description": "Enhanced ModelSEED agent with LangGraph workflows",
        }
        agent = LangGraphMetabolicAgent(llm, tools, agent_config)

        # Update configuration
        config_state["llm_backend"] = backend
        config_state["llm_config"] = llm_config
        config_state["tools"] = tools
        config_state["agent"] = agent

        # Save configuration
        save_cli_config(config_state)

        print_success(f"✅ Switched to {backend} backend!")

        # Show model info
        model_name = llm_config["model_name"]
        if model_name.startswith("gpto") or model_name.startswith("o"):
            console.print(f"[yellow]Using reasoning model: {model_name}[/yellow]")
            console.print("• Optimized for complex reasoning tasks")
            console.print("• No temperature control (uses fixed reasoning temperature)")
            console.print("• Uses prompt array format instead of messages")
            if "max_tokens" not in llm_config:
                console.print("• No token limit set (recommended for complex queries)")
        else:
            console.print(f"[green]Using model: {model_name}[/green]")

        console.print(
            f"\n[dim]Run 'modelseed-agent status' to see full configuration[/dim]"
        )

    except Exception as e:
        print_error(f"Failed to switch to {backend}: {e}")
        return


# Create audit subcommand group
audit_app = typer.Typer(
    name="audit",
    help="🔍 Tool Execution Audit System - Review and analyze tool execution history for hallucination detection",
    rich_markup_mode="rich",
)

app.add_typer(audit_app, name="audit")
app.add_typer(ai_audit_app, name="ai-audit")


@audit_app.command("list")
def audit_list(
    limit: int = typer.Option(
        10, "--limit", "-l", help="Number of recent audits to show"
    ),
    session_id: Optional[str] = typer.Option(
        None, "--session", "-s", help="Filter by session ID"
    ),
    tool_name: Optional[str] = typer.Option(
        None, "--tool", "-t", help="Filter by tool name"
    ),
):
    """
    📋 List recent tool executions

    Shows recent tool audit records with execution details and success status.
    """
    console.print("[bold blue]🔍 Tool Execution Audit History[/bold blue]\n")

    # Find audit files
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print_warning("No logs directory found. No audit records available.")
        return

    # Collect audit files
    audit_files = []
    search_pattern = "*/tool_audits/*.json"

    if session_id:
        search_pattern = f"{session_id}/tool_audits/*.json"

    for audit_file in logs_dir.glob(search_pattern):
        try:
            with open(audit_file, "r") as f:
                audit_data = json.load(f)

            # Filter by tool name if specified
            if tool_name and audit_data.get("tool_name") != tool_name:
                continue

            audit_files.append((audit_file, audit_data))
        except Exception:
            # Skip corrupted files
            continue

    # Sort by timestamp (newest first)
    audit_files.sort(key=lambda x: x[1].get("timestamp", ""), reverse=True)

    if not audit_files:
        print_warning("No audit records found matching the criteria.")
        return

    # Create results table
    audit_table = Table(box=box.ROUNDED)
    audit_table.add_column("Audit ID", style="bold cyan", width=12)
    audit_table.add_column("Tool", style="bold yellow", width=20)
    audit_table.add_column("Timestamp", style="bold white", width=19)
    audit_table.add_column("Duration", style="bold green", width=10)
    audit_table.add_column("Status", style="bold", width=10)
    audit_table.add_column("Session", style="dim cyan", width=12)

    # Show requested number of records
    for audit_file, audit_data in audit_files[:limit]:
        audit_id = audit_data.get("audit_id", "unknown")[:8]  # Short ID
        tool_name = audit_data.get("tool_name", "unknown")
        timestamp = audit_data.get("timestamp", "")

        # Format timestamp
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = timestamp[:19]
        else:
            formatted_time = "unknown"

        # Execution details
        execution = audit_data.get("execution", {})
        duration = execution.get("duration_seconds", 0)
        success = execution.get("success", False)

        duration_str = f"{duration:.2f}s" if duration else "N/A"
        status_str = "[green]✅ Success[/green]" if success else "[red]❌ Failed[/red]"

        session_id = audit_data.get("session_id", "unknown")
        session_id = session_id[:8] if session_id else "default"  # Short session ID

        audit_table.add_row(
            audit_id, tool_name, formatted_time, duration_str, status_str, session_id
        )

    console.print(audit_table)

    # Summary info
    total_shown = min(limit, len(audit_files))
    total_available = len(audit_files)

    console.print(
        f"\n[dim]Showing {total_shown} of {total_available} audit records[/dim]"
    )

    if session_id:
        console.print(f"[dim]Filtered by session: {session_id}[/dim]")
    if tool_name:
        console.print(f"[dim]Filtered by tool: {tool_name}[/dim]")

    console.print(
        f"\n[dim]Use 'modelseed-agent audit show <audit_id>' to view detailed execution data[/dim]"
    )


@audit_app.command("show")
def audit_show(
    audit_id: str = typer.Argument(..., help="Audit ID to display (full or partial)"),
    show_console: bool = typer.Option(
        True, "--console/--no-console", help="Show console output"
    ),
    show_files: bool = typer.Option(
        True, "--files/--no-files", help="Show created files"
    ),
):
    """
    🔍 Show detailed audit information for a specific tool execution

    Displays comprehensive execution details including inputs, outputs, console logs,
    and file artifacts for hallucination detection analysis.
    """
    console.print(f"[bold blue]🔍 Tool Execution Audit: {audit_id}[/bold blue]\n")

    # Find matching audit file
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print_error("No logs directory found.")
        return

    # Search for audit files matching the ID
    matching_files = []
    for audit_file in logs_dir.glob("*/tool_audits/*.json"):
        try:
            with open(audit_file, "r") as f:
                audit_data = json.load(f)

            full_audit_id = audit_data.get("audit_id", "")
            if audit_id in full_audit_id:
                matching_files.append((audit_file, audit_data))
        except:
            continue

    if not matching_files:
        print_error(f"No audit record found matching ID: {audit_id}")
        return

    if len(matching_files) > 1:
        print_warning(f"Multiple matches found for '{audit_id}'. Showing first match.")

    audit_file, audit_data = matching_files[0]

    # Display audit header
    header_table = Table(show_header=False, box=box.SIMPLE)
    header_table.add_column("Field", style="bold cyan")
    header_table.add_column("Value", style="bold white")

    header_table.add_row("Audit ID", audit_data.get("audit_id", "N/A"))
    header_table.add_row("Session ID", audit_data.get("session_id", "N/A"))
    header_table.add_row("Tool Name", audit_data.get("tool_name", "N/A"))
    header_table.add_row("Timestamp", audit_data.get("timestamp", "N/A"))

    console.print(
        Panel(
            header_table,
            title="[bold blue]📋 Audit Header[/bold blue]",
            border_style="blue",
        )
    )

    # Input data
    input_data = audit_data.get("input", {})
    if input_data:
        console.print(f"\n[bold green]📥 Input Data:[/bold green]")
        input_json = json.dumps(input_data, indent=2)
        if len(input_json) > 500:
            input_json = input_json[:500] + "\n... (truncated)"
        console.print(Panel(input_json, border_style="green"))

    # Output data
    output_data = audit_data.get("output", {})
    if output_data:
        console.print(f"\n[bold yellow]📤 Output Data:[/bold yellow]")

        # Structured output
        structured = output_data.get("structured", {})
        if structured:
            console.print(f"\n[bold]🔧 Structured Results:[/bold]")
            structured_json = json.dumps(structured, indent=2)
            if len(structured_json) > 1000:
                structured_json = structured_json[:1000] + "\n... (truncated)"
            console.print(Panel(structured_json, border_style="yellow"))

        # Console output
        console_output = output_data.get("console", {})
        if console_output and show_console:
            console.print(f"\n[bold]💻 Console Output:[/bold]")
            # Handle both string and dict formats for console output
            if isinstance(console_output, dict):
                stdout = console_output.get("stdout", "")
                stderr = console_output.get("stderr", "")
                console_text = (
                    f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                    if stdout or stderr
                    else "No console output"
                )
            else:
                console_text = str(console_output)

            if len(console_text) > 1000:
                console_text = console_text[:1000] + "\n... (truncated)"
            console.print(Panel(console_text, border_style="cyan"))

        # Created files
        files = output_data.get("files", [])
        if files and show_files:
            console.print(f"\n[bold]📄 Created Files:[/bold]")
            files_table = Table(box=box.ROUNDED)
            files_table.add_column("File", style="cyan")
            files_table.add_column("Exists", style="bold")
            files_table.add_column("Size", style="green")

            for file_path in files:
                file_obj = Path(file_path)
                exists = file_obj.exists()
                if exists:
                    size = file_obj.stat().st_size
                    size_str = f"{size} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                    status = "[green]✅ Yes[/green]"
                else:
                    size_str = "N/A"
                    status = "[red]❌ No[/red]"

                files_table.add_row(str(file_path), status, size_str)

            console.print(files_table)

    # Execution details
    execution = audit_data.get("execution", {})
    if execution:
        console.print(f"\n[bold blue]⚡ Execution Details:[/bold blue]")

        exec_table = Table(show_header=False, box=box.SIMPLE)
        exec_table.add_column("Metric", style="bold cyan")
        exec_table.add_column("Value", style="bold white")

        duration = execution.get("duration_seconds", 0)
        success = execution.get("success", False)
        error = execution.get("error")

        exec_table.add_row("Duration", f"{duration:.3f} seconds")
        exec_table.add_row("Success", "✅ Yes" if success else "❌ No")

        if error:
            exec_table.add_row(
                "Error",
                str(error)[:100] + "..." if len(str(error)) > 100 else str(error),
            )

        console.print(exec_table)

    console.print(f"\n[dim]Audit file: {audit_file}[/dim]")


@audit_app.command("session")
def audit_session(
    session_id: str = typer.Argument(..., help="Session ID to analyze"),
    summary: bool = typer.Option(
        False, "--summary", "-s", help="Show summary statistics only"
    ),
):
    """
    📊 Show all tool executions for a specific session

    Displays comprehensive session-level audit information for workflow analysis
    and hallucination pattern detection.
    """
    console.print(f"[bold blue]📊 Session Audit: {session_id}[/bold blue]\n")

    # Find session directory
    logs_dir = Path("logs")
    session_dir = logs_dir / session_id

    if not session_dir.exists():
        print_error(f"Session directory not found: {session_id}")
        return

    audit_dir = session_dir / "tool_audits"
    if not audit_dir.exists():
        print_warning(f"No tool audits found for session: {session_id}")
        return

    # Load all audit files for this session
    audit_records = []
    for audit_file in audit_dir.glob("*.json"):
        try:
            with open(audit_file, "r") as f:
                audit_data = json.load(f)
            audit_records.append(audit_data)
        except Exception as e:
            print_warning(f"Could not load audit file {audit_file}: {e}")
            continue

    if not audit_records:
        print_warning(f"No valid audit records found for session: {session_id}")
        return

    # Sort by timestamp
    audit_records.sort(key=lambda x: x.get("timestamp", ""))

    # Session summary statistics
    total_tools = len(audit_records)
    successful_tools = sum(
        1 for r in audit_records if r.get("execution", {}).get("success", False)
    )
    total_duration = sum(
        r.get("execution", {}).get("duration_seconds", 0) for r in audit_records
    )
    unique_tools = len(set(r.get("tool_name") for r in audit_records))

    # Display session summary
    summary_table = Table(box=box.ROUNDED)
    summary_table.add_column("Metric", style="bold cyan")
    summary_table.add_column("Value", style="bold white")

    summary_table.add_row("Session ID", session_id)
    summary_table.add_row("Total Executions", str(total_tools))
    summary_table.add_row(
        "Successful", f"{successful_tools} ({successful_tools/total_tools*100:.1f}%)"
    )
    summary_table.add_row("Failed", str(total_tools - successful_tools))
    summary_table.add_row("Total Duration", f"{total_duration:.2f} seconds")
    summary_table.add_row("Unique Tools", str(unique_tools))

    if audit_records:
        first_time = audit_records[0].get("timestamp", "")
        last_time = audit_records[-1].get("timestamp", "")
        if first_time and last_time:
            try:
                start_dt = datetime.fromisoformat(first_time.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(last_time.replace("Z", "+00:00"))
                session_duration = (end_dt - start_dt).total_seconds()
                summary_table.add_row(
                    "Session Duration", f"{session_duration:.0f} seconds"
                )
            except:
                pass

    console.print(
        Panel(
            summary_table,
            title="[bold blue]📊 Session Summary[/bold blue]",
            border_style="blue",
        )
    )

    if summary:
        return

    # Detailed tool execution timeline
    console.print(f"\n[bold green]📋 Tool Execution Timeline:[/bold green]")

    timeline_table = Table(box=box.ROUNDED)
    timeline_table.add_column("#", style="bold blue", width=3)
    timeline_table.add_column("Tool", style="bold yellow", width=25)
    timeline_table.add_column("Time", style="bold white", width=8)
    timeline_table.add_column("Duration", style="bold green", width=10)
    timeline_table.add_column("Status", style="bold", width=10)
    timeline_table.add_column("Files", style="cyan", width=8)

    for i, record in enumerate(audit_records, 1):
        tool_name = record.get("tool_name", "unknown")
        timestamp = record.get("timestamp", "")

        # Format time (just time part)
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                time_str = dt.strftime("%H:%M:%S")
            except:
                time_str = timestamp[11:19] if len(timestamp) > 19 else "unknown"
        else:
            time_str = "unknown"

        execution = record.get("execution", {})
        duration = execution.get("duration_seconds", 0)
        success = execution.get("success", False)

        duration_str = f"{duration:.2f}s"
        status_str = "[green]✅[/green]" if success else "[red]❌[/red]"

        # Count created files
        files_count = len(record.get("output", {}).get("files", []))
        files_str = str(files_count) if files_count > 0 else "-"

        timeline_table.add_row(
            str(i), tool_name, time_str, duration_str, status_str, files_str
        )

    console.print(timeline_table)

    # Tool usage statistics
    tool_stats = {}
    for record in audit_records:
        tool_name = record.get("tool_name", "unknown")
        if tool_name not in tool_stats:
            tool_stats[tool_name] = {"count": 0, "successes": 0, "total_duration": 0}

        tool_stats[tool_name]["count"] += 1
        if record.get("execution", {}).get("success", False):
            tool_stats[tool_name]["successes"] += 1
        tool_stats[tool_name]["total_duration"] += record.get("execution", {}).get(
            "duration_seconds", 0
        )

    console.print(f"\n[bold yellow]🔧 Tool Usage Statistics:[/bold yellow]")

    stats_table = Table(box=box.ROUNDED)
    stats_table.add_column("Tool", style="bold cyan")
    stats_table.add_column("Count", style="bold white")
    stats_table.add_column("Success Rate", style="bold green")
    stats_table.add_column("Avg Duration", style="bold yellow")

    for tool_name, stats in sorted(
        tool_stats.items(), key=lambda x: x[1]["count"], reverse=True
    ):
        count = stats["count"]
        successes = stats["successes"]
        success_rate = f"{successes/count*100:.1f}%" if count > 0 else "0%"
        avg_duration = f"{stats['total_duration']/count:.2f}s" if count > 0 else "0s"

        stats_table.add_row(tool_name, str(count), success_rate, avg_duration)

    console.print(stats_table)

    console.print(
        f"\n[dim]Use 'modelseed-agent audit show <audit_id>' to view specific execution details[/dim]"
    )


@audit_app.command("verify")
def audit_verify(
    audit_id: str = typer.Argument(..., help="Audit ID to verify for hallucinations"),
    check_files: bool = typer.Option(
        True, "--check-files/--no-check-files", help="Verify file outputs exist"
    ),
    check_claims: bool = typer.Option(
        True, "--check-claims/--no-check-claims", help="Compare claims vs data"
    ),
):
    """
    🔍 Verify tool execution for potential hallucinations

    Performs automated checks to detect discrepancies between tool claims
    and actual execution results, helping identify AI hallucinations.
    """
    console.print(f"[bold blue]🔍 Hallucination Verification: {audit_id}[/bold blue]\n")

    # Find and load audit record
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print_error("No logs directory found.")
        return

    # Find matching audit file
    audit_data = None
    for candidate_file in logs_dir.glob("*/tool_audits/*.json"):
        try:
            with open(candidate_file, "r") as f:
                data = json.load(f)

            if audit_id in data.get("audit_id", ""):
                audit_data = data
                break
        except:
            continue

    if not audit_data:
        print_error(f"No audit record found matching ID: {audit_id}")
        return

    console.print(
        f"[bold green]📋 Verifying: {audit_data.get('tool_name', 'unknown')}[/bold green]\n"
    )

    # Verification results
    issues = []
    verifications = []

    # 1. Basic execution verification
    execution = audit_data.get("execution", {})
    success = execution.get("success", False)
    error = execution.get("error")

    if success:
        verifications.append("✅ Tool execution completed successfully")
    else:
        issues.append(f"❌ Tool execution failed: {error}")

    # 2. File verification
    if check_files:
        files = audit_data.get("output", {}).get("files", [])
        if files:
            existing_files = 0
            for file_path in files:
                if Path(file_path).exists():
                    existing_files += 1
                else:
                    issues.append(f"❌ Claimed file does not exist: {file_path}")

            if existing_files == len(files):
                verifications.append(f"✅ All {len(files)} claimed files exist")
            else:
                issues.append(
                    f"⚠️ Only {existing_files}/{len(files)} claimed files exist"
                )
        else:
            verifications.append("ℹ️ No file outputs claimed")

    # 3. Console vs structured output consistency
    output_data = audit_data.get("output", {})
    console_output = output_data.get("console", {})
    structured = output_data.get("structured", {})

    if console_output and structured:
        # Check for error messages in console but success in structured output
        if isinstance(console_output, dict):
            console_text = (
                console_output.get("stdout", "")
                + " "
                + console_output.get("stderr", "")
            )
        else:
            console_text = str(console_output)
        console_lower = console_text.lower()
        if any(
            error_word in console_lower
            for error_word in ["error", "failed", "exception", "traceback"]
        ):
            if structured.get("success", True):  # Assuming success if not specified
                issues.append(
                    "⚠️ Console shows errors but structured output indicates success"
                )
            else:
                verifications.append(
                    "✅ Console errors match structured failure status"
                )
        else:
            verifications.append("✅ No obvious error patterns in console output")

    # 4. Data consistency checks
    if structured:
        # Check for None/null values in critical fields
        def check_none_values(obj, path=""):
            none_fields = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if value is None:
                        none_fields.append(current_path)
                    elif isinstance(value, (dict, list)):
                        none_fields.extend(check_none_values(value, current_path))
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]"
                    if item is None:
                        none_fields.append(current_path)
                    elif isinstance(item, (dict, list)):
                        none_fields.extend(check_none_values(item, current_path))
            return none_fields

        none_fields = check_none_values(structured)
        if none_fields:
            issues.append(
                f"⚠️ Found null/None values in: {', '.join(none_fields[:5])}{'...' if len(none_fields) > 5 else ''}"
            )
        else:
            verifications.append("✅ No null/None values in structured output")

    # 5. Execution time consistency
    duration = execution.get("duration_seconds", 0)
    if duration > 0:
        if duration < 0.001:
            issues.append("⚠️ Execution time suspiciously fast (< 1ms)")
        elif duration > 300:  # 5 minutes
            issues.append("⚠️ Execution time suspiciously long (> 5 minutes)")
        else:
            verifications.append(f"✅ Execution time reasonable ({duration:.3f}s)")

    # Display verification results
    if verifications:
        console.print("[bold green]✅ Verification Passed:[/bold green]")
        for verification in verifications:
            console.print(f"  {verification}")

    if issues:
        console.print(f"\n[bold red]⚠️ Issues Found ({len(issues)}):[/bold red]")
        for issue in issues:
            console.print(f"  {issue}")

    if not issues:
        console.print(
            f"\n[bold green]🎉 No hallucination indicators detected![/bold green]"
        )
        console.print(
            "[dim]This execution appears to be consistent and trustworthy.[/dim]"
        )
    else:
        console.print(
            f"\n[bold yellow]🚨 Potential hallucination indicators found.[/bold yellow]"
        )
        console.print("[dim]Manual review recommended for this execution.[/dim]")

    # Summary
    console.print(f"\n[bold blue]📊 Verification Summary:[/bold blue]")
    summary_table = Table(show_header=False, box=box.SIMPLE)
    summary_table.add_column("Check", style="cyan")
    summary_table.add_column("Result", style="bold")

    summary_table.add_row("Total Checks", str(len(verifications) + len(issues)))
    summary_table.add_row("Passed", f"[green]{len(verifications)}[/green]")
    summary_table.add_row(
        "Issues", f"[red]{len(issues)}[/red]" if issues else "[green]0[/green]"
    )
    summary_table.add_row(
        "Trust Level",
        (
            "[green]High[/green]"
            if not issues
            else "[yellow]Medium[/yellow]" if len(issues) < 3 else "[red]Low[/red]"
        ),
    )

    console.print(summary_table)


if __name__ == "__main__":
    app()
