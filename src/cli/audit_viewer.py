#!/usr/bin/env python3
"""
Interactive AI Audit Verification Interface - Phase 7.2

Advanced CLI interface for exploring and verifying AI agent workflows,
providing detailed insights into AI reasoning chains, decision-making
processes, and tool execution patterns.

Features:
- Interactive workflow browsing
- AI reasoning step analysis
- Tool execution verification
- Hallucination detection results
- Confidence scoring visualization
- Decision pattern analysis
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from ..tools.ai_audit import (
    AIDecisionVerifier,
    AIWorkflowAudit,
    create_ai_decision_verifier,
)

console = Console()

# Create the audit viewer CLI application
audit_app = typer.Typer(
    name="audit",
    help="🔍 AI Agent Audit System - Verify and explore AI reasoning workflows",
    add_completion=False,
    rich_markup_mode="rich",
)


def load_workflow_audits(logs_dir: Path) -> List[Dict[str, Any]]:
    """Load all AI workflow audit files from logs directory"""
    workflow_audits = []

    if not logs_dir.exists():
        return workflow_audits

    # Search for AI audit files
    for session_dir in logs_dir.iterdir():
        if session_dir.is_dir():
            ai_audit_dir = session_dir / "ai_audits"
            if ai_audit_dir.exists():
                for audit_file in ai_audit_dir.glob("ai_workflow_*.json"):
                    try:
                        with open(audit_file, "r") as f:
                            audit_data = json.load(f)
                            audit_data["_file_path"] = str(audit_file)
                            audit_data["_session_name"] = session_dir.name
                            workflow_audits.append(audit_data)
                    except Exception as e:
                        console.print(f"[red]Error loading {audit_file}: {e}[/red]")

    # Sort by timestamp
    workflow_audits.sort(key=lambda x: x.get("timestamp_start", ""), reverse=True)
    return workflow_audits


@audit_app.command()
def list(
    logs_dir: str = typer.Option("logs", "--logs", "-l", help="Logs directory path"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of workflows to show"),
    session_filter: Optional[str] = typer.Option(
        None, "--session", "-s", help="Filter by session ID"
    ),
):
    """
    📋 List recent AI workflow audits

    Shows a summary of recent AI agent workflows with basic statistics
    and verification status.
    """
    logs_path = Path(logs_dir)
    workflows = load_workflow_audits(logs_path)

    if session_filter:
        workflows = [
            w for w in workflows if session_filter in w.get("_session_name", "")
        ]

    if not workflows:
        console.print("[yellow]No AI workflow audits found.[/yellow]")
        return

    # Create workflows table
    table = Table(title="🔍 AI Workflow Audits", box=box.ROUNDED)
    table.add_column("Workflow ID", style="bold cyan")
    table.add_column("Session", style="dim")
    table.add_column("Query", style="white", max_width=40)
    table.add_column("Steps", style="bold green")
    table.add_column("Tools", style="blue")
    table.add_column("Duration", style="magenta")
    table.add_column("Status", style="bold")

    for workflow in workflows[:limit]:
        workflow_id = workflow.get("workflow_id", "Unknown")[:8]
        session_name = workflow.get("_session_name", "Unknown")[:12]
        query = workflow.get("user_query", "No query")[:35] + "..."
        steps = str(workflow.get("total_reasoning_steps", 0))
        tools = str(len(workflow.get("tools_executed", [])))

        duration = workflow.get("total_duration_seconds", 0)
        duration_str = f"{duration:.1f}s" if duration else "N/A"

        success = workflow.get("success", False)
        status = "✅ Success" if success else "❌ Failed"
        status_style = "green" if success else "red"

        table.add_row(
            workflow_id,
            session_name,
            query,
            steps,
            tools,
            duration_str,
            f"[{status_style}]{status}[/{status_style}]",
        )

    console.print(table)

    if len(workflows) > limit:
        console.print(
            f"\n[dim]Showing {limit} of {len(workflows)} workflows. Use --limit to show more.[/dim]"
        )


@audit_app.command()
def show(
    workflow_id: str = typer.Argument(..., help="Workflow ID to examine"),
    logs_dir: str = typer.Option("logs", "--logs", "-l", help="Logs directory path"),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed reasoning steps"
    ),
):
    """
    📊 Show detailed information about a specific AI workflow

    Displays comprehensive information about an AI agent workflow including
    reasoning steps, tool selections, and decision-making patterns.
    """
    logs_path = Path(logs_dir)
    workflows = load_workflow_audits(logs_path)

    # Find matching workflow
    workflow = None
    for w in workflows:
        if w.get("workflow_id", "").startswith(workflow_id):
            workflow = w
            break

    if not workflow:
        console.print(f"[red]Workflow {workflow_id} not found.[/red]")
        return

    # Display workflow overview
    display_workflow_overview(workflow)

    # Display reasoning chain
    if detailed:
        display_reasoning_chain(workflow)

    # Display tool execution summary
    display_tool_execution_summary(workflow)


@audit_app.command()
def verify(
    workflow_id: str = typer.Argument(..., help="Workflow ID to verify"),
    logs_dir: str = typer.Option("logs", "--logs", "-l", help="Logs directory path"),
):
    """
    🔍 Run AI reasoning verification on a specific workflow

    Performs comprehensive analysis of AI decision-making accuracy,
    reasoning coherence, and potential hallucination detection.
    """
    logs_path = Path(logs_dir)
    workflows = load_workflow_audits(logs_path)

    # Find matching workflow
    workflow_data = None
    for w in workflows:
        if w.get("workflow_id", "").startswith(workflow_id):
            workflow_data = w
            break

    if not workflow_data:
        console.print(f"[red]Workflow {workflow_id} not found.[/red]")
        return

    # Convert to AIWorkflowAudit object
    try:
        workflow_audit = AIWorkflowAudit(**workflow_data)
    except Exception as e:
        console.print(f"[red]Error parsing workflow data: {e}[/red]")
        return

    # Run verification
    with console.status("🔍 Running AI reasoning verification...", spinner="dots"):
        verifier = create_ai_decision_verifier(logs_path)
        verification_result = verifier.verify_reasoning_chain(workflow_audit)

    # Display verification results
    display_verification_results(verification_result)


@audit_app.command()
def interactive(
    logs_dir: str = typer.Option("logs", "--logs", "-l", help="Logs directory path"),
):
    """
    🎮 Launch interactive audit exploration mode

    Interactive interface for browsing AI workflows, analyzing reasoning
    patterns, and performing verification across multiple workflows.
    """
    logs_path = Path(logs_dir)
    workflows = load_workflow_audits(logs_path)

    if not workflows:
        console.print("[yellow]No AI workflow audits found.[/yellow]")
        return

    console.print(
        Panel(
            "🔍 **Interactive AI Audit Explorer**\n\n"
            "Explore AI agent workflows, analyze reasoning patterns,\n"
            "and verify decision-making accuracy.\n\n"
            "Commands: list, show <id>, verify <id>, stats, search, quit",
            title="[bold blue]AI Audit Explorer[/bold blue]",
            border_style="blue",
        )
    )

    while True:
        try:
            command = Prompt.ask(
                "\n[bold cyan]audit-explorer[/bold cyan]", default="list"
            )

            if command.lower() in ["quit", "exit", "q"]:
                break
            elif command.lower() == "list":
                display_workflow_list_interactive(workflows)
            elif command.lower().startswith("show "):
                workflow_id = command.split(" ", 1)[1]
                show_workflow_interactive(workflows, workflow_id)
            elif command.lower().startswith("verify "):
                workflow_id = command.split(" ", 1)[1]
                verify_workflow_interactive(workflows, workflow_id, logs_path)
            elif command.lower() == "stats":
                display_audit_statistics(workflows)
            elif command.lower().startswith("search "):
                search_term = command.split(" ", 1)[1]
                search_workflows(workflows, search_term)
            else:
                console.print(
                    "[yellow]Unknown command. Try: list, show <id>, verify <id>, stats, search <term>, quit[/yellow]"
                )

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("\n[bold green]👋 Thanks for using AI Audit Explorer![/bold green]")


def display_workflow_overview(workflow: Dict[str, Any]):
    """Display workflow overview information"""

    # Basic information
    info_table = Table(show_header=False, box=box.SIMPLE)
    info_table.add_column("Property", style="bold cyan")
    info_table.add_column("Value", style="white")

    info_table.add_row("Workflow ID", workflow.get("workflow_id", "Unknown"))
    info_table.add_row("Session", workflow.get("_session_name", "Unknown"))
    info_table.add_row("User Query", workflow.get("user_query", "No query"))
    info_table.add_row("Start Time", workflow.get("timestamp_start", "Unknown"))
    info_table.add_row("End Time", workflow.get("timestamp_end", "Unknown"))
    info_table.add_row(
        "Duration", f"{workflow.get('total_duration_seconds', 0):.2f} seconds"
    )
    info_table.add_row("Success", "✅ Yes" if workflow.get("success") else "❌ No")
    info_table.add_row("AI Confidence", f"{workflow.get('ai_confidence_final', 0):.2f}")

    console.print(
        Panel(
            info_table,
            title="[bold blue]📊 Workflow Overview[/bold blue]",
            border_style="blue",
        )
    )

    # Execution summary
    steps = workflow.get("total_reasoning_steps", 0)
    tools = workflow.get("tools_executed", [])

    summary_text = f"""🧠 **AI Reasoning Steps:** {steps}
🔧 **Tools Executed:** {len(tools)} ({', '.join(tools) if tools else 'None'})
📈 **Final Result:** {workflow.get('final_result', 'No result')[:100]}..."""

    console.print(
        Panel(
            summary_text,
            title="[bold green]📋 Execution Summary[/bold green]",
            border_style="green",
        )
    )


def display_reasoning_chain(workflow: Dict[str, Any]):
    """Display detailed reasoning chain"""
    reasoning_steps = workflow.get("reasoning_steps", [])

    if not reasoning_steps:
        console.print("[yellow]No reasoning steps found in this workflow.[/yellow]")
        return

    console.print(
        f"\n[bold yellow]🧠 AI Reasoning Chain ({len(reasoning_steps)} steps):[/bold yellow]"
    )

    for i, step in enumerate(reasoning_steps, 1):
        # Step header
        console.print(
            f"\n[bold cyan]Step {i}: {step.get('selected_tool', 'Decision')}[/bold cyan]"
        )

        # AI thought
        ai_thought = step.get("ai_thought", "No reasoning recorded")
        console.print(f"[dim]💭 AI Thought:[/dim] {ai_thought[:200]}...")

        # Selection rationale
        rationale = step.get("selection_rationale", "No rationale provided")
        console.print(f"[dim]🎯 Rationale:[/dim] {rationale}")

        # Confidence
        confidence = step.get("confidence_score", 0)
        confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        console.print(f"[dim]📊 Confidence:[/dim] {confidence:.2f} [{confidence_bar}]")


def display_tool_execution_summary(workflow: Dict[str, Any]):
    """Display tool execution summary"""
    tools_executed = workflow.get("tools_executed", [])
    tool_execution_order = workflow.get("tool_execution_order", [])

    if not tools_executed:
        console.print("[yellow]No tools were executed in this workflow.[/yellow]")
        return

    # Tool execution table
    exec_table = Table(title="🔧 Tool Execution Timeline", box=box.ROUNDED)
    exec_table.add_column("Order", style="bold cyan")
    exec_table.add_column("Tool Name", style="bold white")
    exec_table.add_column("Timestamp", style="dim")
    exec_table.add_column("Status", style="bold green")

    for i, (tool_name, timestamp) in enumerate(tool_execution_order, 1):
        exec_table.add_row(
            str(i),
            tool_name,
            timestamp.split("T")[1][:8] if "T" in timestamp else timestamp,
            "✅ Success",
        )

    console.print(exec_table)


def display_verification_results(verification: Dict[str, Any]):
    """Display comprehensive verification results"""

    # Overall assessment
    overall = verification.get("overall_assessment", {})
    score = overall.get("overall_score", 0)
    grade = overall.get("overall_grade", "N/A")
    reliability = overall.get("reliability_assessment", "Unknown")

    overall_panel = f"""🎯 **Overall Score:** {score:.3f} ({grade})
🔍 **Reliability:** {reliability}

📊 **Component Scores:**
• Reasoning Coherence: {verification.get('reasoning_verification', {}).get('average_coherence_score', 0):.3f}
• Tool Selection Logic: {verification.get('tool_selection_verification', {}).get('average_selection_score', 0):.3f}
• Outcome Accuracy: {verification.get('outcome_verification', {}).get('accuracy_score', 0):.3f}"""

    console.print(
        Panel(
            overall_panel,
            title="[bold blue]🎯 Verification Summary[/bold blue]",
            border_style="blue",
        )
    )

    # Detailed reasoning verification
    reasoning_verification = verification.get("reasoning_verification", {})
    if reasoning_verification.get("coherence_issues"):
        console.print("\n[bold red]🚨 Reasoning Issues Found:[/bold red]")
        for issue in reasoning_verification["coherence_issues"]:
            console.print(
                f"  • Step {issue['step_number']}: {', '.join(issue['issues'])}"
            )

    # Tool selection verification
    tool_verification = verification.get("tool_selection_verification", {})
    if tool_verification.get("selection_issues"):
        console.print("\n[bold yellow]⚠️ Tool Selection Issues:[/bold yellow]")
        for issue in tool_verification["selection_issues"]:
            console.print(f"  • {issue['selected_tool']}: {', '.join(issue['issues'])}")

    # Improvement suggestions
    suggestions = overall.get("improvement_suggestions", [])
    if suggestions:
        console.print("\n[bold green]💡 Improvement Suggestions:[/bold green]")
        for suggestion in suggestions:
            console.print(f"  • {suggestion}")


def display_workflow_list_interactive(workflows: List[Dict[str, Any]]):
    """Display workflow list in interactive mode"""
    table = Table(title="🔍 Available AI Workflows", box=box.ROUNDED)
    table.add_column("#", style="dim")
    table.add_column("ID", style="bold cyan")
    table.add_column("Query", style="white", max_width=30)
    table.add_column("Steps", style="green")
    table.add_column("Status", style="bold")

    for i, workflow in enumerate(workflows[:20], 1):
        workflow_id = workflow.get("workflow_id", "Unknown")[:8]
        query = workflow.get("user_query", "No query")[:25] + "..."
        steps = str(workflow.get("total_reasoning_steps", 0))
        success = workflow.get("success", False)
        status = "✅" if success else "❌"

        table.add_row(str(i), workflow_id, query, steps, status)

    console.print(table)


def show_workflow_interactive(workflows: List[Dict[str, Any]], workflow_id: str):
    """Show workflow details in interactive mode"""
    workflow = None
    for w in workflows:
        if w.get("workflow_id", "").startswith(workflow_id):
            workflow = w
            break

    if not workflow:
        console.print(f"[red]Workflow {workflow_id} not found.[/red]")
        return

    display_workflow_overview(workflow)

    if Confirm.ask("Show detailed reasoning steps?"):
        display_reasoning_chain(workflow)


def verify_workflow_interactive(
    workflows: List[Dict[str, Any]], workflow_id: str, logs_path: Path
):
    """Verify workflow in interactive mode"""
    workflow_data = None
    for w in workflows:
        if w.get("workflow_id", "").startswith(workflow_id):
            workflow_data = w
            break

    if not workflow_data:
        console.print(f"[red]Workflow {workflow_id} not found.[/red]")
        return

    try:
        workflow_audit = AIWorkflowAudit(**workflow_data)
        with console.status("🔍 Verifying AI reasoning...", spinner="dots"):
            verifier = create_ai_decision_verifier(logs_path)
            verification_result = verifier.verify_reasoning_chain(workflow_audit)

        display_verification_results(verification_result)

    except Exception as e:
        console.print(f"[red]Verification failed: {e}[/red]")


def display_audit_statistics(workflows: List[Dict[str, Any]]):
    """Display overall audit statistics"""
    if not workflows:
        console.print("[yellow]No workflows to analyze.[/yellow]")
        return

    # Calculate statistics
    total_workflows = len(workflows)
    successful_workflows = sum(1 for w in workflows if w.get("success", False))
    success_rate = successful_workflows / total_workflows

    total_reasoning_steps = sum(w.get("total_reasoning_steps", 0) for w in workflows)
    avg_reasoning_steps = total_reasoning_steps / total_workflows

    total_tools_executed = sum(len(w.get("tools_executed", [])) for w in workflows)
    avg_tools_per_workflow = total_tools_executed / total_workflows

    avg_duration = (
        sum(w.get("total_duration_seconds", 0) for w in workflows) / total_workflows
    )

    stats_text = f"""📊 **Audit Statistics**

🔢 **Workflows:** {total_workflows}
✅ **Success Rate:** {success_rate:.1%} ({successful_workflows}/{total_workflows})
🧠 **Avg Reasoning Steps:** {avg_reasoning_steps:.1f}
🔧 **Avg Tools per Workflow:** {avg_tools_per_workflow:.1f}
⏱️ **Avg Duration:** {avg_duration:.1f} seconds

🏆 **Most Common Tools:**"""

    # Tool usage frequency
    tool_usage = {}
    for workflow in workflows:
        for tool in workflow.get("tools_executed", []):
            tool_usage[tool] = tool_usage.get(tool, 0) + 1

    sorted_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)
    for tool, count in sorted_tools[:5]:
        stats_text += f"\n  • {tool}: {count} times"

    console.print(
        Panel(
            stats_text,
            title="[bold green]📈 AI Audit Statistics[/bold green]",
            border_style="green",
        )
    )


def search_workflows(workflows: List[Dict[str, Any]], search_term: str):
    """Search workflows by query content"""
    matching_workflows = []

    for workflow in workflows:
        query = workflow.get("user_query", "").lower()
        tools = " ".join(workflow.get("tools_executed", [])).lower()

        if search_term.lower() in query or search_term.lower() in tools:
            matching_workflows.append(workflow)

    if not matching_workflows:
        console.print(f"[yellow]No workflows found matching '{search_term}'.[/yellow]")
        return

    console.print(
        f"[bold green]Found {len(matching_workflows)} workflows matching '{search_term}':[/bold green]"
    )
    display_workflow_list_interactive(matching_workflows)


if __name__ == "__main__":
    audit_app()
