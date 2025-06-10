"""
Interactive CLI for ModelSEEDagent

Provides a seamless interactive analysis experience with conversational AI,
real-time visualization, and intelligent session management.
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import questionary
import typer
from questionary import Style
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

# Configure logging
logger = logging.getLogger(__name__)

# Debug flag from environment variable
DEBUG_MODE = os.getenv("MODELSEED_DEBUG", "false").lower() == "true"

if DEBUG_MODE:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

from ..cli.argo_health import display_argo_health
from .conversation_engine import ConversationResponse, DynamicAIConversationEngine
from .live_visualizer import LiveVisualizer
from .query_processor import QueryAnalysis, QueryProcessor
from .session_manager import (
    AnalysisSession,
    Interaction,
    InteractionType,
    SessionManager,
)

# Initialize Rich console
console = Console()

# Custom questionary style
custom_style = Style(
    [
        ("question", "bold"),
        ("answer", "fg:#ffffff bold"),
        ("pointer", "fg:#00aa00 bold"),
        ("highlighted", "fg:#00aa00 bold"),
        ("selected", "fg:#00aa00"),
        ("separator", "fg:#666666"),
        ("instruction", ""),
        ("text", ""),
        ("disabled", "fg:#858585 italic"),
    ]
)


class InteractiveCLI:
    """Main interactive CLI for ModelSEEDagent"""

    def __init__(self):
        self.session_manager = SessionManager()
        self.query_processor = QueryProcessor()
        self.visualizer = LiveVisualizer()

        self.current_session: Optional[AnalysisSession] = None
        self.conversation_engine: Optional[DynamicAIConversationEngine] = None

        # CLI state
        self.running = False
        self.exit_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        self.exit_requested = True
        console.print("\n[yellow]Gracefully shutting down...[/yellow]")
        if self.current_session:
            self.session_manager.save_session(self.current_session)
        sys.exit(0)

    def start_interactive_session(self) -> None:
        """Start the main interactive session"""
        self.running = True

        # Log debug mode once at session start
        if DEBUG_MODE:
            logger.info("🔍 DEBUG MODE ENABLED")

        # Print banner
        self._print_banner()

        # Argo Gateway health check and configuration display
        try:
            display_argo_health()
        except Exception as e:
            console.print(f"⚠️ [yellow]Argo health check failed: {e}[/yellow]")

        # Session selection or creation
        session = self._select_or_create_session()
        if not session:
            console.print("[red]No session selected. Exiting.[/red]")
            return

        self.current_session = session

        # Initialize conversation engine
        self.conversation_engine = DynamicAIConversationEngine(session)

        # Start conversation
        response = self.conversation_engine.start_conversation()
        self._display_response(response)

        # Main interaction loop
        try:
            self._interaction_loop()
        except KeyboardInterrupt:
            self._handle_exit()

    def _print_banner(self) -> None:
        """Print the interactive CLI banner"""
        banner_text = """
[bold cyan]
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🧬 ModelSEEDagent Interactive Analysis                   ║
║                       Intelligent Metabolic Modeling                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
[/bold cyan]

[bold yellow]✨ Welcome to the Interactive Analysis Interface! ✨[/bold yellow]

This interface provides:
• [green]💬 Natural language conversation[/green] for metabolic modeling
• [blue]🎨 Real-time visualization[/blue] of analysis workflows
• [magenta]📊 Live progress tracking[/magenta] and performance monitoring
• [cyan]💾 Persistent sessions[/cyan] with full history

[dim]Type 'help' for assistance, 'exit' to quit, or start asking questions![/dim]
        """
        console.print(Panel(banner_text, border_style="blue"))

    def _select_or_create_session(self) -> Optional[AnalysisSession]:
        """Select existing session or create new one"""
        sessions = self.session_manager.list_sessions()

        if sessions:
            console.print(
                f"\n[bold green]Found {len(sessions)} existing sessions:[/bold green]"
            )
            self.session_manager.display_sessions_table(sessions[:5])  # Show last 5

            choice = questionary.select(
                "What would you like to do?",
                choices=[
                    "Create new session",
                    "Continue existing session",
                    "View all sessions",
                    "Exit",
                ],
                style=custom_style,
            ).ask()

            if choice == "Continue existing session":
                session_choices = [f"{s.id}: {s.name}" for s in sessions[:10]]
                selected = questionary.select(
                    "Select session:", choices=session_choices, style=custom_style
                ).ask()

                if selected:
                    session_id = selected.split(":")[0]
                    return self.session_manager.load_session(session_id)

            elif choice == "View all sessions":
                self.session_manager.display_sessions_table()
                return self._select_or_create_session()  # Recurse

            elif choice == "Exit":
                return None

        # Create new session
        name = questionary.text(
            "Session name:",
            default=f"Analysis_{datetime.now().strftime('%Y%m%d_%H%M')}",
            style=custom_style,
        ).ask()

        if not name:
            return None

        description = questionary.text(
            "Session description (optional):",
            default="Interactive metabolic modeling analysis",
            style=custom_style,
        ).ask()

        return self.session_manager.create_session(name, description or "")

    def _interaction_loop(self) -> None:
        """Main interaction loop"""
        while self.running and not self.exit_requested:
            try:
                # Get user input
                user_input = self._get_user_input()

                if not user_input:
                    continue

                # Handle special commands
                if self._handle_special_commands(user_input):
                    continue

                # Process with conversation engine (avoid Rich status that interferes with agent output)
                console.print(
                    "🧠 [cyan]AI analyzing your query and executing tools...[/cyan]"
                )

                # Debug mode indicator
                if DEBUG_MODE:
                    console.print("[yellow]🔍 DEBUG MODE: Processing query[/yellow]")
                    logger.debug(f"🔍 Processing user input: '{user_input[:50]}...'")
                    logger.debug(
                        f"🔍 Conversation engine type: {type(self.conversation_engine).__name__}"
                    )
                    logger.debug(f"🔍 Session: {self.current_session.id}")

                # Add timeout protection for the entire interaction
                import time

                start_time = time.time()
                timeout_seconds = 300  # 5 minutes total timeout

                try:
                    # Use asyncio to add timeout to the synchronous call
                    response = self._process_with_timeout(user_input, timeout_seconds)

                    processing_time = time.time() - start_time
                    if DEBUG_MODE:
                        logger.debug(
                            f"🔍 Processing completed in {processing_time:.2f}s"
                        )
                        logger.debug(f"🔍 Response type: {response.response_type}")
                        logger.debug(
                            f"🔍 Response success: {response.metadata.get('ai_agent_result', 'unknown')}"
                        )

                except TimeoutError:
                    console.print(
                        f"[red]⏰ Request timed out after {timeout_seconds}s[/red]"
                    )
                    console.print(
                        "[yellow]This may indicate a complex analysis or system issue.[/yellow]"
                    )
                    continue
                except Exception as e:
                    if DEBUG_MODE:
                        logger.error(f"🔍 Processing failed with error: {e}")
                        import traceback

                        logger.debug(f"🔍 Full traceback: {traceback.format_exc()}")
                    raise

                # Display response
                self._display_response(response)

                # Handle post-response actions
                self._handle_response_actions(response)

                # Auto-save session
                self.session_manager.save_session(self.current_session)

            except KeyboardInterrupt:
                self._handle_exit()
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")
                console.print(
                    "[dim]Please try again or type 'help' for assistance.[/dim]"
                )

    def _get_user_input(self) -> str:
        """Get user input with rich prompt"""
        try:
            # Show session info in prompt
            session_name = (
                self.current_session.name[:15] + "..."
                if len(self.current_session.name) > 15
                else self.current_session.name
            )
            # Use plain text for questionary (no Rich markup)
            prompt_text = f"{session_name} ❯ "

            user_input = questionary.text(
                "", qmark=prompt_text, style=custom_style
            ).ask()

            return user_input.strip() if user_input else ""

        except (KeyboardInterrupt, EOFError):
            return "exit"

    def _handle_special_commands(self, user_input: str) -> bool:
        """Handle special CLI commands"""
        command = user_input.lower().strip()

        if command in ["exit", "quit", "q"]:
            self._handle_exit()
            return True

        elif command in ["help", "h", "?"]:
            self._show_help()
            return True

        elif command in ["status", "info"]:
            self._show_status()
            return True

        elif command in ["sessions", "ls"]:
            self.session_manager.display_sessions_table()
            return True

        elif command.startswith("switch "):
            session_id = command.split(" ", 1)[1]
            if self.session_manager.set_current_session(session_id):
                self.current_session = self.session_manager.current_session
                self.conversation_engine.session = self.current_session
            return True

        elif command in ["visualizations", "viz"]:
            self.visualizer.display_visualization_table()
            return True

        elif command.startswith("open "):
            viz_key = command.split(" ", 1)[1]
            self.visualizer.open_visualization(viz_key)
            return True

        elif command in ["clear", "cls"]:
            console.clear()
            return True

        elif command in ["analytics", "stats"]:
            self.session_manager.display_analytics()
            return True

        # AI Media Tools Commands
        elif command in ["media", "select-media"]:
            self._show_media_selector()
            return True

        elif command.startswith("media-select "):
            model_arg = (
                command.split(" ", 1)[1] if len(command.split(" ")) > 1 else None
            )
            self._handle_media_selection(model_arg)
            return True

        elif command.startswith("media-modify "):
            modification = (
                command.split(" ", 1)[1] if len(command.split(" ")) > 1 else None
            )
            self._handle_media_modification(modification)
            return True

        elif command in ["media-compare"]:
            self._handle_media_comparison()
            return True

        return False

    def _process_with_timeout(
        self, user_input: str, timeout_seconds: int
    ) -> ConversationResponse:
        """Process user input with timeout protection"""
        import signal
        import threading

        # Use signal-based timeout if in main thread, otherwise use simple timeout
        if threading.current_thread() is threading.main_thread():

            def timeout_handler(signum, frame):
                raise TimeoutError(f"Processing timed out after {timeout_seconds}s")

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)

            try:
                response = self.conversation_engine.process_user_input(user_input)
                signal.alarm(0)  # Cancel timeout
                return response
            except TimeoutError:
                signal.alarm(0)  # Cancel timeout
                raise
            finally:
                signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
        else:
            # Fallback for non-main threads
            return self.conversation_engine.process_user_input(user_input)

    def _show_help(self) -> None:
        """Display help information"""
        help_table = Table(title="🔧 Interactive Commands", box=box.ROUNDED)
        help_table.add_column("Command", style="bold cyan")
        help_table.add_column("Description", style="white")

        commands = [
            ("help, h, ?", "Show this help message"),
            ("status, info", "Show current session status"),
            ("sessions, ls", "List all available sessions"),
            ("switch <id>", "Switch to a different session"),
            ("visualizations, viz", "Show available visualizations"),
            ("open <viz>", "Open visualization in browser"),
            ("analytics, stats", "Show session analytics"),
            ("clear, cls", "Clear the terminal"),
            # AI Media Tools Commands
            ("media", "🧬 Show interactive media selection interface"),
            ("media-select <model>", "🧬 AI-powered media selection for model"),
            (
                "media-modify <cmd>",
                "🧬 Modify media ('make anaerobic', 'add vitamins')",
            ),
            ("media-compare", "🧬 Compare media performance across models"),
            ("exit, quit, q", "Exit the interactive session"),
        ]

        # Add debug mode indicator
        if DEBUG_MODE:
            commands.append(("DEBUG MODE", "🔍 Detailed logging enabled"))
        else:
            commands.append(
                ("Debug Mode", "Set MODELSEED_DEBUG=true for detailed logs")
            )

        for cmd, desc in commands:
            help_table.add_row(cmd, desc)

        console.print(help_table)

        # Analysis help
        analysis_help = """
[bold yellow]🧬 Analysis Capabilities:[/bold yellow]

You can ask natural language questions about metabolic modeling:

[green]📊 Model Analysis:[/green]
• "Analyze the structure of my E. coli model"
• "What is the growth rate on glucose?"
• "Show me the model statistics"

[blue]🛣️ Pathway Analysis:[/blue]
• "Analyze glycolysis pathway fluxes"
• "Compare central carbon metabolism"
• "Show pathway connectivity"

[red]⚡ Flux Analysis:[/red]
• "Run FBA optimization"
• "Calculate flux variability"
• "Find essential genes"

[magenta]🎨 Visualization:[/magenta]
• "Create a network visualization"
• "Plot flux distributions"
• "Generate pathway diagrams"

[dim]Just ask your questions in natural language and I'll understand![/dim]
        """
        console.print(Panel(analysis_help, border_style="green"))

    def _show_status(self) -> None:
        """Show current session status"""
        if not self.current_session or not self.conversation_engine:
            console.print("[red]No active session.[/red]")
            return

        # Session information
        session_summary = self.current_session.get_session_summary()
        conv_summary = self.conversation_engine.get_conversation_summary()

        # Create status table
        status_table = Table(show_header=False, box=box.SIMPLE)
        status_table.add_column("Aspect", style="bold cyan")
        status_table.add_column("Value", style="bold white")

        # Session info
        status_table.add_row("Session ID", session_summary["session_info"]["id"])
        status_table.add_row("Session Name", session_summary["session_info"]["name"])
        status_table.add_row("Status", session_summary["session_info"]["status"])
        status_table.add_row("Duration", session_summary["session_info"]["duration"])

        # Statistics
        status_table.add_row("", "")  # Separator
        status_table.add_row(
            "Total Interactions",
            str(session_summary["statistics"]["total_interactions"]),
        )
        status_table.add_row(
            "Success Rate", session_summary["statistics"]["success_rate"]
        )
        status_table.add_row(
            "Avg Execution Time", session_summary["statistics"]["avg_execution_time"]
        )

        # Conversation state
        status_table.add_row("", "")  # Separator
        status_table.add_row(
            "Conversation State",
            conv_summary["current_state"].replace("_", " ").title(),
        )
        if conv_summary["current_context"]["last_query_type"]:
            status_table.add_row(
                "Last Query Type",
                conv_summary["current_context"]["last_query_type"]
                .replace("_", " ")
                .title(),
            )

        console.print(
            Panel(
                status_table,
                title="[bold blue]📊 Session Status[/bold blue]",
                border_style="blue",
            )
        )

        # Recent activity
        if session_summary["recent_activity"]:
            console.print(f"\n[bold yellow]🕒 Recent Activity:[/bold yellow]")
            for i, activity in enumerate(session_summary["recent_activity"], 1):
                console.print(f"  {i}. {activity}")

    def _display_response(self, response: ConversationResponse) -> None:
        """Display conversation response with beautiful formatting"""
        # Response content
        console.print(
            Panel(
                response.content,
                title=f"[bold green]🤖 Assistant ({response.response_type.value.title()})[/bold green]",
                border_style="green",
            )
        )

        # Suggested actions
        if response.suggested_actions:
            actions_text = "\n".join(
                [
                    f"  {i}. {action}"
                    for i, action in enumerate(response.suggested_actions, 1)
                ]
            )
            console.print(
                Panel(
                    actions_text,
                    title="[bold yellow]💡 Suggested Actions[/bold yellow]",
                    border_style="yellow",
                )
            )

        # Processing time
        if response.processing_time > 0:
            console.print(f"[dim]⏱️ Processed in {response.processing_time:.3f}s[/dim]")

    def _handle_response_actions(self, response: ConversationResponse) -> None:
        """Handle post-response actions"""
        # Create visualizations if analysis was performed
        if response.response_type.name == "ANALYSIS_RESULT" and response.metadata.get(
            "analysis"
        ):
            analysis_data = response.metadata["analysis"]

            # Ask if user wants visualizations
            if Confirm.ask(
                "Would you like me to create visualizations for this analysis?"
            ):
                self._create_analysis_visualizations(analysis_data)

        # Handle clarification responses
        if response.clarification_needed and response.suggested_actions:
            console.print(
                "\n[dim]You can respond to any of the questions above, or continue with a new query.[/dim]"
            )

    def _create_analysis_visualizations(self, analysis_data: Dict[str, Any]) -> None:
        """Create visualizations for analysis results"""
        query_type = analysis_data.get("query_type", "general")

        # Start live progress
        self.visualizer.start_live_progress(4)

        try:
            # Create workflow visualization
            self.visualizer.update_progress("Creating workflow visualization...")
            workflow_data = self._create_mock_workflow_data(analysis_data)
            self.visualizer.create_workflow_visualization(workflow_data)

            # Create progress dashboard
            self.visualizer.update_progress("Creating progress dashboard...")
            dashboard_data = self._create_mock_dashboard_data(analysis_data)
            self.visualizer.create_progress_dashboard(dashboard_data)

            # Create specific visualizations based on query type
            if query_type in ["flux_analysis", "pathway_analysis"]:
                self.visualizer.update_progress("Creating flux heatmap...")
                flux_data = self._create_mock_flux_data()
                self.visualizer.create_flux_heatmap(flux_data)

            if query_type in ["structural_analysis", "network_analysis"]:
                self.visualizer.update_progress("Creating network visualization...")
                network_data = self._create_mock_network_data()
                self.visualizer.create_network_visualization(network_data)

            self.visualizer.update_progress("Visualizations complete!", 1)

        except Exception as e:
            console.print(f"[red]❌ Error creating visualizations: {e}[/red]")
        finally:
            self.visualizer.stop_live_progress()

        # Display available visualizations
        self.visualizer.display_visualization_table()

        # Ask if user wants to open them
        if Confirm.ask("Would you like to open the visualizations in your browser?"):
            for viz_key in self.visualizer.active_visualizations:
                self.visualizer.open_visualization(viz_key)

    def _create_mock_workflow_data(
        self, analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create mock workflow data for visualization"""
        return {
            "nodes": [
                {
                    "id": "input_model",
                    "label": "Model Input",
                    "type": "input",
                    "status": "completed",
                },
                {
                    "id": "preprocess",
                    "label": "Preprocessing",
                    "type": "tool",
                    "status": "completed",
                },
                {
                    "id": "analyze",
                    "label": "Analysis",
                    "type": "tool",
                    "status": "completed",
                },
                {
                    "id": "optimize",
                    "label": "Optimization",
                    "type": "tool",
                    "status": "completed",
                },
                {
                    "id": "results",
                    "label": "Results",
                    "type": "output",
                    "status": "completed",
                },
            ],
            "edges": [
                {"source": "input_model", "target": "preprocess", "label": "SBML"},
                {
                    "source": "preprocess",
                    "target": "analyze",
                    "label": "Validated Model",
                },
                {"source": "analyze", "target": "optimize", "label": "Analysis Data"},
                {
                    "source": "optimize",
                    "target": "results",
                    "label": "Optimized Solution",
                },
            ],
        }

    def _create_mock_dashboard_data(
        self, analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create mock dashboard data"""
        return {
            "execution_timeline": {
                "timestamps": ["0s", "2s", "5s", "8s", "10s"],
                "tools": ["Input", "Preprocess", "Analyze", "Optimize", "Complete"],
            },
            "tool_performance": {
                "Preprocessing": 1.2,
                "Analysis": 3.5,
                "Optimization": 2.8,
                "Visualization": 1.0,
            },
            "success_rate": 0.92,
            "resource_usage": {
                "time": [0, 2, 4, 6, 8, 10],
                "memory": [50, 65, 80, 75, 60, 45],
            },
        }

    def _create_mock_flux_data(self) -> Dict[str, Any]:
        """Create mock flux data for heatmap"""
        import numpy as np

        return {
            "flux_matrix": np.random.rand(15, 4) * 10 - 2,
            "reaction_names": [f"R{i:03d}" for i in range(15)],
            "condition_names": ["Glucose", "Acetate", "Glycerol", "Lactate"],
        }

    def _create_mock_network_data(self) -> Dict[str, Any]:
        """Create mock network data"""
        nodes = [
            {"id": f"M{i}", "attributes": {"type": "metabolite"}} for i in range(20)
        ]
        nodes.extend(
            [{"id": f"R{i}", "attributes": {"type": "reaction"}} for i in range(15)]
        )

        edges = [
            {
                "source": f"M{i}",
                "target": f"R{i%15}",
                "attributes": {"type": "substrate"},
            }
            for i in range(20)
        ]

        return {"nodes": nodes, "edges": edges}

    def _show_media_selector(self) -> None:
        """Show interactive media selection interface"""
        console.print(
            "\n[bold blue]🧬 AI Media Tools - Interactive Selection[/bold blue]"
        )

        # Create media options table
        media_table = Table(title="Available AI Media Commands", box=box.ROUNDED)
        media_table.add_column("Command", style="bold cyan")
        media_table.add_column("Description", style="white")
        media_table.add_column("Example", style="dim")

        media_commands = [
            (
                "media-select",
                "AI selects optimal media for your model",
                "media-select e_coli_core.xml",
            ),
            (
                "media-modify",
                "Modify media with natural language",
                "media-modify make anaerobic",
            ),
            (
                "media-compare",
                "Compare performance across media types",
                "media-compare",
            ),
        ]

        for cmd, desc, example in media_commands:
            media_table.add_row(cmd, desc, example)

        console.print(media_table)

        # Show available models
        console.print("\n[bold]📁 Available Models:[/bold]")
        models = ["e_coli_core.xml", "EcoliMG1655.xml", "BuchnMG37.xml"]
        for model in models:
            console.print(f"  • {model}")

        console.print(
            "\n[dim]💡 Tip: Type any of the commands above or ask naturally like:[/dim]"
        )
        console.print("[dim]    'select the best media for my E. coli model'[/dim]")
        console.print("[dim]    'make my media anaerobic for fermentation'[/dim]")

    def _handle_media_selection(self, model_arg: Optional[str]) -> None:
        """Handle AI-powered media selection"""
        if not model_arg:
            console.print(
                "[yellow]⚠️  Please specify a model: media-select <model>[/yellow]"
            )
            return

        console.print(f"[cyan]🧠 AI selecting optimal media for {model_arg}...[/cyan]")

        # Create a natural language query for the conversation engine
        query = f"Please use the select_optimal_media tool to find the best media for model {model_arg}. Analyze the model characteristics and recommend appropriate media types."

        try:
            self.conversation_engine.process_user_input(query)
            console.print("\n[green]✅ Media selection completed![/green]")
        except Exception as e:
            console.print(f"[red]❌ Media selection failed: {e}[/red]")

    def _handle_media_modification(self, modification: Optional[str]) -> None:
        """Handle AI-powered media modification"""
        if not modification:
            console.print(
                "[yellow]⚠️  Please specify modification: media-modify <command>[/yellow]"
            )
            console.print(
                "[dim]Examples: 'make anaerobic', 'add vitamins', 'remove amino acids'[/dim]"
            )
            return

        console.print(f"[cyan]🧠 AI modifying media: {modification}...[/cyan]")

        # Create a natural language query for the conversation engine
        query = f"Please use the manipulate_media_composition tool to modify media with this command: '{modification}'. Use GMM as the base media and test the results."

        try:
            self.conversation_engine.process_user_input(query)
            console.print("\n[green]✅ Media modification completed![/green]")
        except Exception as e:
            console.print(f"[red]❌ Media modification failed: {e}[/red]")

    def _handle_media_comparison(self) -> None:
        """Handle AI-powered media comparison"""
        console.print("[cyan]🧠 AI comparing media performance across models...[/cyan]")

        # Create a natural language query for the conversation engine
        query = "Please use the compare_media_performance tool to compare how different models perform across various media types. Test with both E. coli and B. aphidicola models if available."

        try:
            self.conversation_engine.process_user_input(query)
            console.print("\n[green]✅ Media comparison completed![/green]")
        except Exception as e:
            console.print(f"[red]❌ Media comparison failed: {e}[/red]")

    def _handle_exit(self) -> None:
        """Handle graceful exit"""
        if self.current_session:
            # Save session
            self.session_manager.save_session(self.current_session)

            # Show session summary
            console.print("\n[bold blue]📊 Session Summary:[/bold blue]")
            summary = self.current_session.get_session_summary()

            console.print(
                f"• Total interactions: {summary['statistics']['total_interactions']}"
            )
            console.print(f"• Success rate: {summary['statistics']['success_rate']}")
            console.print(
                f"• Total time: {summary['statistics']['total_execution_time']}"
            )
            console.print(f"• Session saved: [green]{self.current_session.id}[/green]")

        console.print(
            "\n[bold green]👋 Thank you for using ModelSEEDagent![/bold green]"
        )
        console.print(
            "[dim]Your session has been saved and can be resumed anytime.[/dim]"
        )

        self.running = False


# CLI entry point
def main():
    """Main entry point for interactive CLI"""
    try:
        cli = InteractiveCLI()
        cli.start_interactive_session()
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
