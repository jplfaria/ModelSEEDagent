#!/usr/bin/env python3
"""
Phase 8 User Experience Interface

Intuitive interfaces for accessing advanced agentic capabilities including:
- Interactive reasoning chain builder
- Hypothesis testing wizard
- Collaborative decision assistant
- Pattern learning dashboard

Provides both CLI and programmatic interfaces for sophisticated AI reasoning.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Rich formatting for beautiful CLI output
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.tree import Tree

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

    # Fallback to basic console
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

    class Panel:
        def __init__(self, content, title=None):
            self.content = content
            self.title = title


console = Console()


class ReasoningChainBuilder:
    """Interactive builder for multi-step reasoning chains"""

    def __init__(self, config):
        self.config = config
        self.available_tools = [
            "run_metabolic_fba",
            "find_minimal_media",
            "analyze_essentiality",
            "flux_variability_analysis",
            "gene_deletion_analysis",
            "identify_auxotrophies",
            "flux_sampling",
        ]

    async def interactive_chain_builder(self) -> Dict[str, Any]:
        """Interactive CLI for building reasoning chains"""

        if RICH_AVAILABLE:
            console.print(
                Panel(
                    "[bold blue]🔗 Multi-Step Reasoning Chain Builder[/bold blue]\n\n"
                    "Build sophisticated analysis workflows with AI-guided reasoning",
                    title="Phase 8.1 - Reasoning Chains",
                )
            )
        else:
            print("🔗 Multi-Step Reasoning Chain Builder")
            print("Build sophisticated analysis workflows with AI-guided reasoning")

        # Get user query
        query = (
            Prompt.ask("\n[bold]What analysis would you like to perform?[/bold]")
            if RICH_AVAILABLE
            else input("\nWhat analysis would you like to perform? ")
        )

        # Analysis goal
        goal = (
            Prompt.ask("[bold]What is your analysis goal?[/bold]")
            if RICH_AVAILABLE
            else input("What is your analysis goal? ")
        )

        # Build chain interactively
        chain_steps = []
        step_number = 1

        while True:
            if RICH_AVAILABLE:
                console.print(f"\n[bold cyan]Step {step_number}:[/bold cyan]")
            else:
                print(f"\nStep {step_number}:")

            # Show available tools
            if RICH_AVAILABLE:
                table = Table(title="Available Tools")
                table.add_column("Tool", style="cyan")
                table.add_column("Description", style="white")

                tool_descriptions = {
                    "run_metabolic_fba": "Flux Balance Analysis for growth prediction",
                    "find_minimal_media": "Identify minimal nutritional requirements",
                    "analyze_essentiality": "Find essential genes and reactions",
                    "flux_variability_analysis": "Analyze flux ranges and variability",
                    "gene_deletion_analysis": "Single/double gene knockout analysis",
                    "identify_auxotrophies": "Find biosynthetic pathway gaps",
                    "flux_sampling": "Statistical flux space exploration",
                }

                for tool in self.available_tools:
                    table.add_row(
                        tool, tool_descriptions.get(tool, "Advanced analysis tool")
                    )

                console.print(table)
            else:
                print("Available tools:")
                for i, tool in enumerate(self.available_tools, 1):
                    print(f"  {i}. {tool}")

            # Get tool selection
            if RICH_AVAILABLE:
                selected_tool = Prompt.ask(
                    "[bold]Select tool[/bold]", choices=self.available_tools + ["done"]
                )
            else:
                print("Enter tool name or 'done' to finish:")
                selected_tool = input().strip()

            if selected_tool == "done":
                break

            if selected_tool not in self.available_tools:
                (
                    console.print("[red]Invalid tool selection[/red]")
                    if RICH_AVAILABLE
                    else print("Invalid tool selection")
                )
                continue

            # Get reasoning for this step
            reasoning = (
                Prompt.ask("[bold]Why use this tool at this step?[/bold]")
                if RICH_AVAILABLE
                else input("Why use this tool at this step? ")
            )

            # Get confidence
            confidence_str = (
                Prompt.ask("[bold]Confidence (0.0-1.0)[/bold]", default="0.8")
                if RICH_AVAILABLE
                else input("Confidence (0.0-1.0, default 0.8): ") or "0.8"
            )
            try:
                confidence = float(confidence_str)
            except ValueError:
                confidence = 0.8

            step = {
                "step_number": step_number,
                "tool_selected": selected_tool,
                "reasoning": reasoning,
                "confidence": confidence,
                "step_id": f"step_{step_number:03d}",
                "timestamp": datetime.now().isoformat(),
            }

            chain_steps.append(step)
            step_number += 1

        # Build complete chain
        reasoning_chain = {
            "chain_id": f"user_chain_{int(datetime.now().timestamp())}",
            "session_id": "interactive_session",
            "user_query": query,
            "analysis_goal": goal,
            "timestamp_start": datetime.now().isoformat(),
            "planned_steps": chain_steps,
            "status": "planned",
        }

        # Show summary
        if RICH_AVAILABLE:
            console.print(
                Panel(
                    f"[bold green]✅ Reasoning Chain Created[/bold green]\n\n"
                    f"Query: {query}\n"
                    f"Goal: {goal}\n"
                    f"Steps: {len(chain_steps)}\n"
                    f"Chain ID: {reasoning_chain['chain_id']}",
                    title="Chain Summary",
                )
            )
        else:
            print(f"\n✅ Reasoning Chain Created")
            print(f"Query: {query}")
            print(f"Goal: {goal}")
            print(f"Steps: {len(chain_steps)}")

        return reasoning_chain

    def quick_chain_templates(self) -> Dict[str, Dict[str, Any]]:
        """Predefined chain templates for common analyses"""
        return {
            "comprehensive": {
                "name": "Comprehensive Model Analysis",
                "description": "Complete characterization of metabolic model",
                "steps": [
                    "run_metabolic_fba",
                    "find_minimal_media",
                    "analyze_essentiality",
                ],
                "reasoning": "Baseline growth → nutritional requirements → robustness analysis",
            },
            "growth_optimization": {
                "name": "Growth Optimization Analysis",
                "description": "Optimize model for maximum growth",
                "steps": [
                    "run_metabolic_fba",
                    "flux_variability_analysis",
                    "gene_deletion_analysis",
                ],
                "reasoning": "Current growth → flux alternatives → knockout opportunities",
            },
            "nutrition_analysis": {
                "name": "Nutritional Requirements Analysis",
                "description": "Detailed nutritional dependency analysis",
                "steps": [
                    "find_minimal_media",
                    "identify_auxotrophies",
                    "flux_sampling",
                ],
                "reasoning": "Essential nutrients → biosynthetic gaps → metabolic flexibility",
            },
            "robustness_testing": {
                "name": "Model Robustness Testing",
                "description": "Test model stability and essential components",
                "steps": [
                    "analyze_essentiality",
                    "gene_deletion_analysis",
                    "flux_variability_analysis",
                ],
                "reasoning": "Essential components → knockout effects → flux robustness",
            },
        }


class HypothesisTestingWizard:
    """Interactive wizard for hypothesis-driven analysis"""

    def __init__(self, config):
        self.config = config

    async def interactive_hypothesis_wizard(self) -> Dict[str, Any]:
        """Interactive hypothesis generation and testing"""

        if RICH_AVAILABLE:
            console.print(
                Panel(
                    "[bold blue]🔬 Hypothesis Testing Wizard[/bold blue]\n\n"
                    "Generate and test scientific hypotheses about metabolic behavior",
                    title="Phase 8.2 - Hypothesis-Driven Analysis",
                )
            )
        else:
            print("🔬 Hypothesis Testing Wizard")

        # Get observation
        observation = (
            Prompt.ask("\n[bold]What observation would you like to investigate?[/bold]")
            if RICH_AVAILABLE
            else input("\nWhat observation would you like to investigate? ")
        )

        # Generate hypothesis suggestions
        hypothesis_suggestions = [
            {
                "type": "nutritional_gap",
                "statement": "Model has specific nutritional limitations",
                "tests": ["find_minimal_media", "identify_auxotrophies"],
            },
            {
                "type": "gene_essentiality",
                "statement": "Essential genes constrain metabolic flexibility",
                "tests": ["analyze_essentiality", "gene_deletion_analysis"],
            },
            {
                "type": "pathway_activity",
                "statement": "Alternative metabolic pathways are available",
                "tests": ["flux_variability_analysis", "flux_sampling"],
            },
            {
                "type": "metabolic_efficiency",
                "statement": "Metabolic efficiency can be improved",
                "tests": ["run_metabolic_fba", "flux_variability_analysis"],
            },
        ]

        # Show suggestions
        if RICH_AVAILABLE:
            console.print(
                "\n[bold]Suggested hypotheses based on your observation:[/bold]"
            )
            table = Table()
            table.add_column("Type", style="cyan")
            table.add_column("Hypothesis", style="white")
            table.add_column("Suggested Tests", style="yellow")

            for i, hyp in enumerate(hypothesis_suggestions, 1):
                table.add_row(hyp["type"], hyp["statement"], ", ".join(hyp["tests"]))

            console.print(table)
        else:
            print("\nSuggested hypotheses:")
            for i, hyp in enumerate(hypothesis_suggestions, 1):
                print(f"  {i}. {hyp['type']}: {hyp['statement']}")
                print(f"     Tests: {', '.join(hyp['tests'])}")

        # Get user selection or custom hypothesis
        custom = (
            Confirm.ask("\n[bold]Would you like to create a custom hypothesis?[/bold]")
            if RICH_AVAILABLE
            else input("\nWould you like to create a custom hypothesis? (y/n): ")
            .lower()
            .startswith("y")
        )

        if custom:
            hypothesis_statement = (
                Prompt.ask("[bold]Enter your hypothesis statement[/bold]")
                if RICH_AVAILABLE
                else input("Enter your hypothesis statement: ")
            )
            hypothesis_type = (
                Prompt.ask("[bold]Hypothesis type[/bold]", default="custom")
                if RICH_AVAILABLE
                else input("Hypothesis type (default: custom): ") or "custom"
            )
            test_tools = (
                Prompt.ask(
                    "[bold]Which tools should test this hypothesis? (comma-separated)[/bold]"
                )
                if RICH_AVAILABLE
                else input(
                    "Which tools should test this hypothesis? (comma-separated): "
                )
            )
            test_tools_list = [tool.strip() for tool in test_tools.split(",")]
        else:
            selection = (
                Prompt.ask(
                    "[bold]Select hypothesis number[/bold]",
                    choices=["1", "2", "3", "4"],
                )
                if RICH_AVAILABLE
                else input("Select hypothesis number (1-4): ")
            )
            try:
                selected_hyp = hypothesis_suggestions[int(selection) - 1]
                hypothesis_statement = selected_hyp["statement"]
                hypothesis_type = selected_hyp["type"]
                test_tools_list = selected_hyp["tests"]
            except (ValueError, IndexError):
                hypothesis_statement = hypothesis_suggestions[0]["statement"]
                hypothesis_type = hypothesis_suggestions[0]["type"]
                test_tools_list = hypothesis_suggestions[0]["tests"]

        # Get confidence
        confidence_str = (
            Prompt.ask("[bold]Confidence in hypothesis (0.0-1.0)[/bold]", default="0.7")
            if RICH_AVAILABLE
            else input("Confidence in hypothesis (0.0-1.0, default 0.7): ") or "0.7"
        )
        try:
            confidence = float(confidence_str)
        except ValueError:
            confidence = 0.7

        # Create hypothesis
        hypothesis = {
            "hypothesis_id": f"user_hyp_{int(datetime.now().timestamp())}",
            "hypothesis_type": hypothesis_type,
            "statement": hypothesis_statement,
            "rationale": f"Generated from observation: {observation}",
            "predictions": [f"Testing will reveal {hypothesis_type} characteristics"],
            "testable_with_tools": test_tools_list,
            "confidence_score": confidence,
            "timestamp": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "status": "generated",
            "source_observation": observation,
        }

        # Show summary
        if RICH_AVAILABLE:
            console.print(
                Panel(
                    f"[bold green]🧪 Hypothesis Generated[/bold green]\n\n"
                    f"Statement: {hypothesis_statement}\n"
                    f"Type: {hypothesis_type}\n"
                    f"Confidence: {confidence:.1%}\n"
                    f"Test Tools: {', '.join(test_tools_list)}",
                    title="Hypothesis Summary",
                )
            )
        else:
            print(f"\n🧪 Hypothesis Generated")
            print(f"Statement: {hypothesis_statement}")
            print(f"Type: {hypothesis_type}")
            print(f"Confidence: {confidence:.1%}")

        return hypothesis


class CollaborativeDecisionAssistant:
    """Assistant for collaborative AI-human decision making"""

    def __init__(self, config):
        self.config = config

    async def interactive_collaboration(
        self, context: str, options: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Interactive collaborative decision interface"""

        if RICH_AVAILABLE:
            console.print(
                Panel(
                    "[bold blue]🤝 Collaborative Decision Assistant[/bold blue]\n\n"
                    "Work with AI to make optimal analysis decisions",
                    title="Phase 8.3 - Collaborative Reasoning",
                )
            )
        else:
            print("🤝 Collaborative Decision Assistant")

        # Show context
        if RICH_AVAILABLE:
            console.print(f"\n[bold]Context:[/bold]\n{context}")
        else:
            print(f"\nContext: {context}")

        # Show AI analysis and options
        if RICH_AVAILABLE:
            console.print("\n[bold]AI Analysis & Recommendations:[/bold]")
            table = Table()
            table.add_column("Option", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("AI Assessment", style="yellow")

            for i, option in enumerate(options, 1):
                table.add_row(
                    option.get("name", f"Option {i}"),
                    option.get("description", "Analysis option"),
                    option.get("ai_assessment", "AI assessment pending"),
                )

            console.print(table)
        else:
            print("\nAvailable options:")
            for i, option in enumerate(options, 1):
                print(f"  {i}. {option.get('name', f'Option {i}')}")
                print(
                    f"     Description: {option.get('description', 'Analysis option')}"
                )
                print(f"     AI Assessment: {option.get('ai_assessment', 'Pending')}")

        # Get user input
        if RICH_AVAILABLE:
            choice_options = [str(i) for i in range(1, len(options) + 1)]
            choice = Prompt.ask(
                f"[bold]Select your preferred option (1-{len(options)})[/bold]",
                choices=choice_options,
            )
        else:
            choice = input(f"Select your preferred option (1-{len(options)}): ")

        try:
            selected_option = options[int(choice) - 1]
        except (ValueError, IndexError):
            selected_option = options[0]

        # Get user rationale
        user_rationale = (
            Prompt.ask("[bold]Why did you choose this option?[/bold]")
            if RICH_AVAILABLE
            else input("Why did you choose this option? ")
        )

        # Additional context
        additional_context = (
            Prompt.ask(
                "[bold]Any additional context or constraints?[/bold]", default="None"
            )
            if RICH_AVAILABLE
            else input("Any additional context or constraints? (default: None): ")
            or "None"
        )

        # Create collaborative decision
        decision = {
            "decision_id": f"collab_dec_{int(datetime.now().timestamp())}",
            "context": context,
            "options_considered": options,
            "selected_option": selected_option,
            "user_choice": int(choice),
            "user_rationale": user_rationale,
            "additional_context": additional_context,
            "decision_type": "collaborative",
            "confidence_score": 0.85,  # High confidence for human-guided decisions
            "timestamp": datetime.now().isoformat(),
        }

        # Show decision summary
        if RICH_AVAILABLE:
            console.print(
                Panel(
                    f"[bold green]✅ Collaborative Decision Made[/bold green]\n\n"
                    f"Selected: {selected_option.get('name', 'Selected option')}\n"
                    f"Rationale: {user_rationale}\n"
                    f"Confidence: 85%",
                    title="Decision Summary",
                )
            )
        else:
            print(f"\n✅ Collaborative Decision Made")
            print(f"Selected: {selected_option.get('name', 'Selected option')}")
            print(f"Rationale: {user_rationale}")

        return decision


class PatternLearningDashboard:
    """Dashboard for viewing and managing learned patterns"""

    def __init__(self, config):
        self.config = config
        self.mock_patterns = self._generate_mock_patterns()

    def _generate_mock_patterns(self) -> List[Dict[str, Any]]:
        """Generate mock patterns for demonstration"""
        return [
            {
                "pattern_id": "pat_001",
                "type": "tool_sequence",
                "description": "High growth models → nutrition analysis",
                "success_rate": 0.87,
                "confidence": 0.91,
                "usage_count": 23,
                "organisms": ["E. coli", "S. cerevisiae"],
            },
            {
                "pattern_id": "pat_002",
                "type": "insight_correlation",
                "description": "Complex nutrition → essential gene clusters",
                "success_rate": 0.79,
                "confidence": 0.84,
                "usage_count": 15,
                "organisms": ["E. coli", "P. putida"],
            },
            {
                "pattern_id": "pat_003",
                "type": "optimization_strategy",
                "description": "Balanced growth-robustness optimization",
                "success_rate": 0.92,
                "confidence": 0.89,
                "usage_count": 31,
                "organisms": ["Multiple"],
            },
        ]

    def show_pattern_dashboard(self) -> None:
        """Display pattern learning dashboard"""

        if RICH_AVAILABLE:
            console.print(
                Panel(
                    "[bold blue]📚 Pattern Learning Dashboard[/bold blue]\n\n"
                    "View learned patterns and analysis insights",
                    title="Phase 8.4 - Cross-Model Learning",
                )
            )

            # Patterns table
            table = Table(title="Learned Analysis Patterns")
            table.add_column("Pattern ID", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Description", style="white")
            table.add_column("Success Rate", style="green")
            table.add_column("Usage", style="yellow")

            for pattern in self.mock_patterns:
                table.add_row(
                    pattern["pattern_id"],
                    pattern["type"],
                    pattern["description"],
                    f"{pattern['success_rate']:.1%}",
                    str(pattern["usage_count"]),
                )

            console.print(table)

            # Summary stats
            total_patterns = len(self.mock_patterns)
            avg_success = (
                sum(p["success_rate"] for p in self.mock_patterns) / total_patterns
            )
            total_usage = sum(p["usage_count"] for p in self.mock_patterns)

            console.print(
                Panel(
                    f"[bold]Learning Statistics[/bold]\n\n"
                    f"Total Patterns: {total_patterns}\n"
                    f"Average Success Rate: {avg_success:.1%}\n"
                    f"Total Pattern Applications: {total_usage}",
                    title="Summary",
                )
            )
        else:
            print("📚 Pattern Learning Dashboard")
            print("\nLearned Analysis Patterns:")
            for pattern in self.mock_patterns:
                print(f"  {pattern['pattern_id']}: {pattern['description']}")
                print(
                    f"    Success: {pattern['success_rate']:.1%}, Used: {pattern['usage_count']} times"
                )

    def get_pattern_recommendations(self, query: str) -> List[Dict[str, Any]]:
        """Get pattern-based recommendations for a query"""
        recommendations = []

        query_lower = query.lower()

        for pattern in self.mock_patterns:
            relevance_score = 0.0

            # Simple keyword matching
            if "growth" in query_lower and "growth" in pattern["description"].lower():
                relevance_score += 0.4
            if (
                "nutrition" in query_lower
                and "nutrition" in pattern["description"].lower()
            ):
                relevance_score += 0.4
            if (
                "optimization" in query_lower
                and "optimization" in pattern["description"].lower()
            ):
                relevance_score += 0.4
            if "comprehensive" in query_lower:
                relevance_score += 0.2

            if relevance_score > 0:
                recommendation = {
                    **pattern,
                    "relevance_score": relevance_score,
                    "recommendation": f"Based on {pattern['usage_count']} successful uses",
                }
                recommendations.append(recommendation)

        # Sort by relevance and success rate
        recommendations.sort(
            key=lambda x: (x["relevance_score"], x["success_rate"]), reverse=True
        )

        return recommendations[:3]  # Top 3 recommendations


class Phase8Interface:
    """Main interface coordinator for Phase 8 features"""

    def __init__(self, config):
        self.config = config
        self.chain_builder = ReasoningChainBuilder(config)
        self.hypothesis_wizard = HypothesisTestingWizard(config)
        self.decision_assistant = CollaborativeDecisionAssistant(config)
        self.pattern_dashboard = PatternLearningDashboard(config)

    async def interactive_menu(self) -> None:
        """Main interactive menu for Phase 8 features"""

        while True:
            if RICH_AVAILABLE:
                console.print(
                    Panel(
                        "[bold blue]🚀 Phase 8 Advanced Agentic Capabilities[/bold blue]\n\n"
                        "Choose an advanced AI reasoning feature:",
                        title="ModelSEEDagent - Advanced AI",
                    )
                )

                options_table = Table()
                options_table.add_column("Option", style="cyan")
                options_table.add_column("Feature", style="white")
                options_table.add_column("Description", style="yellow")

                options_table.add_row(
                    "1", "Reasoning Chains", "Build multi-step analysis workflows"
                )
                options_table.add_row(
                    "2", "Hypothesis Testing", "Generate and test scientific hypotheses"
                )
                options_table.add_row(
                    "3", "Collaborative Decisions", "Work with AI on complex decisions"
                )
                options_table.add_row(
                    "4", "Pattern Dashboard", "View learned analysis patterns"
                )
                options_table.add_row(
                    "5", "Quick Templates", "Use predefined analysis templates"
                )
                options_table.add_row("0", "Exit", "Return to main menu")

                console.print(options_table)

                choice = Prompt.ask(
                    "\n[bold]Select option[/bold]",
                    choices=["0", "1", "2", "3", "4", "5"],
                )
            else:
                print("\n🚀 Phase 8 Advanced Agentic Capabilities")
                print("1. Reasoning Chains - Build multi-step analysis workflows")
                print("2. Hypothesis Testing - Generate and test scientific hypotheses")
                print("3. Collaborative Decisions - Work with AI on complex decisions")
                print("4. Pattern Dashboard - View learned analysis patterns")
                print("5. Quick Templates - Use predefined analysis templates")
                print("0. Exit")
                choice = input("Select option: ")

            if choice == "0":
                break
            elif choice == "1":
                result = await self.chain_builder.interactive_chain_builder()
                self._save_result("reasoning_chain", result)
            elif choice == "2":
                result = await self.hypothesis_wizard.interactive_hypothesis_wizard()
                self._save_result("hypothesis", result)
            elif choice == "3":
                # Mock collaborative scenario
                context = "Analysis shows multiple optimization strategies available"
                options = [
                    {
                        "name": "growth_focus",
                        "description": "Maximize growth rate",
                        "ai_assessment": "High yield but may be unstable",
                    },
                    {
                        "name": "robustness_focus",
                        "description": "Maximize stability",
                        "ai_assessment": "Lower yield but more reliable",
                    },
                    {
                        "name": "balanced",
                        "description": "Balance both objectives",
                        "ai_assessment": "Moderate performance, good reliability",
                    },
                ]
                result = await self.decision_assistant.interactive_collaboration(
                    context, options
                )
                self._save_result("collaborative_decision", result)
            elif choice == "4":
                self.pattern_dashboard.show_pattern_dashboard()
                (
                    input("\nPress Enter to continue...")
                    if not RICH_AVAILABLE
                    else Prompt.ask("[dim]Press Enter to continue...[/dim]", default="")
                )
            elif choice == "5":
                self._show_quick_templates()

    def _show_quick_templates(self) -> None:
        """Show quick template options"""
        templates = self.chain_builder.quick_chain_templates()

        if RICH_AVAILABLE:
            console.print(
                Panel(
                    "[bold]🚀 Quick Analysis Templates[/bold]",
                    title="Ready-to-Use Workflows",
                )
            )

            for key, template in templates.items():
                console.print(f"[bold cyan]{template['name']}[/bold cyan]")
                console.print(f"  {template['description']}")
                console.print(f"  Tools: {' → '.join(template['steps'])}")
                console.print(f"  Reasoning: {template['reasoning']}\n")
        else:
            print("\n🚀 Quick Analysis Templates")
            for key, template in templates.items():
                print(f"\n{template['name']}")
                print(f"  Description: {template['description']}")
                print(f"  Tools: {' → '.join(template['steps'])}")

    def _save_result(self, result_type: str, result: Dict[str, Any]) -> None:
        """Save interface results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase8_{result_type}_{timestamp}.json"

        # Create logs directory if it doesn't exist
        logs_dir = Path("logs/phase8_interface")
        logs_dir.mkdir(parents=True, exist_ok=True)

        filepath = logs_dir / filename

        with open(filepath, "w") as f:
            json.dump(result, f, indent=2, default=str)

        if RICH_AVAILABLE:
            console.print(f"[dim]💾 Saved to {filepath}[/dim]")
        else:
            print(f"💾 Saved to {filepath}")


# Quick access functions
async def quick_reasoning_chain():
    """Quick access to reasoning chain builder"""
    from src.config.settings import Config

    config = Config(llm={}, tools={})  # Mock config
    interface = Phase8Interface(config)
    return await interface.chain_builder.interactive_chain_builder()


async def quick_hypothesis_test():
    """Quick access to hypothesis testing wizard"""
    from src.config.settings import Config

    config = Config(llm={}, tools={})  # Mock config
    interface = Phase8Interface(config)
    return await interface.hypothesis_wizard.interactive_hypothesis_wizard()


def quick_pattern_dashboard():
    """Quick access to pattern dashboard"""
    from src.config.settings import Config

    config = Config(llm={}, tools={})  # Mock config
    interface = Phase8Interface(config)
    interface.pattern_dashboard.show_pattern_dashboard()


# CLI entry point
async def main():
    """CLI entry point for Phase 8 interface"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "chains":
            await quick_reasoning_chain()
        elif command == "hypothesis":
            await quick_hypothesis_test()
        elif command == "patterns":
            quick_pattern_dashboard()
        else:
            print("Available commands: chains, hypothesis, patterns")
    else:
        # Mock config for CLI
        class MockConfig:
            pass

        interface = Phase8Interface(MockConfig())
        await interface.interactive_menu()


if __name__ == "__main__":
    asyncio.run(main())
