"""
Enhanced Tool Integration for LangGraph Metabolic Agent

This module provides:
* Enhanced tool nodes adapted for LangGraph workflows
* Conditional workflow logic based on tool results
* Workflow visualization and monitoring
* Comprehensive observability and metrics
* Smart tool selection and parameter optimization
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..tools.base import BaseTool, ToolResult
from ..tools.cobra.analysis import ModelAnalysisTool, PathwayAnalysisTool
from ..tools.cobra.fba import FBATool
from ..tools.cobra.utils import ModelUtils

logger = logging.getLogger(__name__)

# -------------------------------
# Enhanced Tool System Architecture
# -------------------------------


class ToolPriority(Enum):
    """Tool execution priority levels"""

    CRITICAL = 1  # Must execute (e.g., model loading)
    HIGH = 2  # Should execute (e.g., basic analysis)
    MEDIUM = 3  # Can execute (e.g., detailed analysis)
    LOW = 4  # Optional (e.g., visualization)


class ToolCategory(Enum):
    """Tool categories for workflow organization"""

    STRUCTURAL = "structural"  # Model structure analysis
    FUNCTIONAL = "functional"  # FBA, growth analysis
    OPTIMIZATION = "optimization"  # Media optimization, design
    VALIDATION = "validation"  # Model validation, testing
    VISUALIZATION = "visualization"  # Plotting, reporting


@dataclass
class ToolExecutionPlan:
    """Plan for tool execution with dependencies and conditions"""

    tool_name: str
    priority: ToolPriority
    category: ToolCategory
    depends_on: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    max_runtime: float = 30.0  # seconds
    retry_count: int = 0
    parallel_group: Optional[str] = None


@dataclass
class ToolExecutionResult:
    """Enhanced result with performance metrics"""

    tool_name: str
    success: bool
    result: ToolResult
    execution_time: float
    memory_usage: Optional[float] = None
    retry_count: int = 0
    error_type: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class WorkflowState(Enum):
    """Current state of the workflow"""

    PLANNING = "planning"
    EXECUTING = "executing"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    COMPLETING = "completing"
    ERROR = "error"


# -------------------------------
# Enhanced Tool Integration Manager
# -------------------------------


class EnhancedToolIntegration:
    """
    Enhanced tool integration manager for LangGraph workflows

    Features:
    - Intelligent tool selection based on query analysis
    - Conditional workflow execution
    - Performance monitoring and optimization
    - Workflow visualization
    - Comprehensive observability
    """

    def __init__(self, tools: List[BaseTool], run_dir: Path):
        self.tools = {tool.tool_name: tool for tool in tools}
        self.run_dir = run_dir
        self.execution_history: List[ToolExecutionResult] = []
        self.workflow_metrics = {}
        self.visualization_dir = run_dir / "visualizations"
        self.visualization_dir.mkdir(exist_ok=True)

        # Tool categorization
        self.tool_categories = self._categorize_tools()

        # Performance baselines
        self.performance_baselines = self._load_performance_baselines()

        logger.info(f"Enhanced Tool Integration initialized with {len(tools)} tools")

    def _categorize_tools(self) -> Dict[str, ToolCategory]:
        """Categorize tools for intelligent workflow planning"""
        categories = {
            "analyze_metabolic_model": ToolCategory.STRUCTURAL,
            "run_metabolic_fba": ToolCategory.FUNCTIONAL,
            "analyze_pathway": ToolCategory.STRUCTURAL,
            "find_minimal_media": ToolCategory.OPTIMIZATION,
            "check_missing_media": ToolCategory.VALIDATION,
            "analyze_reaction_expression": ToolCategory.FUNCTIONAL,
            "check_auxotrophy": ToolCategory.VALIDATION,
        }
        return categories

    def _load_performance_baselines(self) -> Dict[str, Dict[str, float]]:
        """Load performance baselines for tool execution"""
        # Default baselines - would be learned over time
        return {
            "analyze_metabolic_model": {"avg_time": 2.0, "max_time": 10.0},
            "run_metabolic_fba": {"avg_time": 3.0, "max_time": 15.0},
            "analyze_pathway": {"avg_time": 1.5, "max_time": 8.0},
        }

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine optimal tool selection and workflow
        """
        query_lower = query.lower()

        intent_analysis = {
            "primary_intent": None,
            "secondary_intents": [],
            "suggested_tools": [],
            "workflow_complexity": "simple",
            "estimated_tools": 1,
            "analysis_depth": "basic",
        }

        # Intent detection patterns
        intent_patterns = {
            "structural_analysis": [
                "structure",
                "network",
                "connectivity",
                "topology",
                "reactions",
                "metabolites",
                "genes",
            ],
            "growth_analysis": ["growth", "fba", "flux", "biomass", "objective"],
            "pathway_analysis": [
                "pathway",
                "subsystem",
                "central carbon",
                "glycolysis",
            ],
            "optimization": ["optimize", "minimal", "media", "improve", "design"],
            "validation": ["validate", "check", "missing", "auxotrophy", "gaps"],
            "comprehensive": [
                "comprehensive",
                "complete",
                "full",
                "detailed",
                "thorough",
            ],
        }

        # Analyze intent
        intent_scores = {}
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score

        if not intent_scores:
            # Default to basic analysis
            intent_analysis["primary_intent"] = "structural_analysis"
            intent_analysis["suggested_tools"] = ["analyze_metabolic_model"]
        else:
            # Sort by score
            sorted_intents = sorted(
                intent_scores.items(), key=lambda x: x[1], reverse=True
            )
            intent_analysis["primary_intent"] = sorted_intents[0][0]
            intent_analysis["secondary_intents"] = [
                intent for intent, _ in sorted_intents[1:]
            ]

        # Map intents to tools
        intent_tool_mapping = {
            "structural_analysis": ["analyze_metabolic_model"],
            "growth_analysis": ["run_metabolic_fba"],
            "pathway_analysis": ["analyze_pathway"],
            "optimization": ["find_minimal_media"],
            "validation": ["check_missing_media", "check_auxotrophy"],
            "comprehensive": [
                "analyze_metabolic_model",
                "run_metabolic_fba",
                "analyze_pathway",
            ],
        }

        # Build suggested tools list
        suggested_tools = set()
        for intent in [intent_analysis["primary_intent"]] + intent_analysis[
            "secondary_intents"
        ]:
            if intent in intent_tool_mapping:
                suggested_tools.update(intent_tool_mapping[intent])

        intent_analysis["suggested_tools"] = list(suggested_tools)
        intent_analysis["estimated_tools"] = len(suggested_tools)

        # Determine complexity
        if len(suggested_tools) >= 3:
            intent_analysis["workflow_complexity"] = "complex"
            intent_analysis["analysis_depth"] = "comprehensive"
        elif len(suggested_tools) == 2:
            intent_analysis["workflow_complexity"] = "moderate"
            intent_analysis["analysis_depth"] = "detailed"

        logger.info(
            f"Query intent analysis: {intent_analysis['primary_intent']} with {len(suggested_tools)} tools"
        )
        return intent_analysis

    def create_execution_plan(
        self, intent_analysis: Dict[str, Any], model_path: str = None
    ) -> List[ToolExecutionPlan]:
        """
        Create optimized execution plan based on intent analysis
        """
        suggested_tools = intent_analysis["suggested_tools"]
        complexity = intent_analysis["workflow_complexity"]

        execution_plan = []

        # Always start with structural analysis if model is involved
        if model_path and "analyze_metabolic_model" not in suggested_tools:
            suggested_tools.insert(0, "analyze_metabolic_model")

        # Create execution plans with dependencies and conditions
        for i, tool_name in enumerate(suggested_tools):
            if tool_name not in self.tools:
                logger.warning(f"Tool {tool_name} not available, skipping")
                continue

            category = self.tool_categories.get(tool_name, ToolCategory.STRUCTURAL)

            # Determine priority
            if tool_name == "analyze_metabolic_model":
                priority = ToolPriority.CRITICAL
            elif category == ToolCategory.FUNCTIONAL:
                priority = ToolPriority.HIGH
            else:
                priority = ToolPriority.MEDIUM

            # Set dependencies
            depends_on = []
            if i > 0 and tool_name != "analyze_metabolic_model":
                depends_on = [suggested_tools[0]]  # Depend on first tool

            # Set parallel groups for independent tools
            parallel_group = None
            if complexity == "complex" and category in [
                ToolCategory.FUNCTIONAL,
                ToolCategory.VALIDATION,
            ]:
                parallel_group = f"parallel_group_{category.value}"

            # Create execution plan
            plan = ToolExecutionPlan(
                tool_name=tool_name,
                priority=priority,
                category=category,
                depends_on=depends_on,
                parallel_group=parallel_group,
                parameters={"model_path": model_path} if model_path else {},
            )

            execution_plan.append(plan)

        # Sort by priority
        execution_plan.sort(key=lambda x: x.priority.value)

        logger.info(f"Created execution plan with {len(execution_plan)} tools")
        return execution_plan

    def execute_tool_with_monitoring(
        self, plan: ToolExecutionPlan
    ) -> ToolExecutionResult:
        """
        Execute a tool with comprehensive monitoring and error handling
        """
        tool = self.tools[plan.tool_name]
        start_time = time.time()

        try:
            logger.info(f"Executing tool: {plan.tool_name}")

            # Prepare input based on tool requirements
            if plan.parameters:
                input_data = plan.parameters
            else:
                # For analysis tools, provide model path
                if plan.tool_name in [
                    "run_metabolic_fba",
                    "find_minimal_media",
                    "analyze_essentiality",
                    "run_flux_variability_analysis",
                    "identify_auxotrophies",
                    "run_flux_sampling",
                    "run_gene_deletion_analysis",
                    "run_production_envelope",
                    "analyze_metabolic_model",
                    "analyze_pathway",
                    "check_missing_media",
                    "analyze_reaction_expression",
                ]:
                    from pathlib import Path

                    default_model_path = str(
                        Path(__file__).parent.parent.parent
                        / "data"
                        / "examples"
                        / "e_coli_core.xml"
                    )
                    input_data = {"model_path": default_model_path}
                else:
                    input_data = "default analysis"

            # Execute tool
            result = tool._run(input_data)
            execution_time = time.time() - start_time

            # Create execution result
            exec_result = ToolExecutionResult(
                tool_name=plan.tool_name,
                success=result.success,
                result=result,
                execution_time=execution_time,
                performance_metrics={
                    "execution_time": execution_time,
                    "data_size": len(str(result.data)) if result.data else 0,
                    "category": plan.category.value,
                },
            )

            # Check performance against baselines
            self._check_performance(exec_result)

            self.execution_history.append(exec_result)

            logger.info(f"Tool {plan.tool_name} completed in {execution_time:.2f}s")
            return exec_result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool {plan.tool_name} failed: {e}")

            # Create error result
            error_result = ToolResult(
                success=False, message=f"Tool execution failed: {str(e)}", error=str(e)
            )

            exec_result = ToolExecutionResult(
                tool_name=plan.tool_name,
                success=False,
                result=error_result,
                execution_time=execution_time,
                error_type=type(e).__name__,
            )

            self.execution_history.append(exec_result)
            return exec_result

    def _check_performance(self, exec_result: ToolExecutionResult):
        """Check performance against baselines and log warnings"""
        tool_name = exec_result.tool_name
        execution_time = exec_result.execution_time

        if tool_name in self.performance_baselines:
            baseline = self.performance_baselines[tool_name]
            avg_time = baseline.get("avg_time", 5.0)
            max_time = baseline.get("max_time", 20.0)

            if execution_time > max_time:
                logger.warning(
                    f"Tool {tool_name} exceeded max time: {execution_time:.2f}s > {max_time}s"
                )
            elif execution_time > avg_time * 2:
                logger.info(
                    f"Tool {tool_name} slower than expected: {execution_time:.2f}s > {avg_time * 2}s"
                )

    def analyze_workflow_dependencies(
        self, execution_plan: List[ToolExecutionPlan]
    ) -> Dict[str, Any]:
        """
        Analyze workflow dependencies and suggest optimizations
        """
        analysis = {
            "total_tools": len(execution_plan),
            "critical_path": [],
            "parallel_opportunities": [],
            "bottlenecks": [],
            "estimated_runtime": 0.0,
        }

        # Build dependency graph
        dependency_graph = {}
        for plan in execution_plan:
            dependency_graph[plan.tool_name] = plan.depends_on

        # Find critical path
        critical_path = self._find_critical_path(dependency_graph, execution_plan)
        analysis["critical_path"] = critical_path

        # Find parallel opportunities
        parallel_groups = {}
        for plan in execution_plan:
            if plan.parallel_group:
                if plan.parallel_group not in parallel_groups:
                    parallel_groups[plan.parallel_group] = []
                parallel_groups[plan.parallel_group].append(plan.tool_name)

        analysis["parallel_opportunities"] = parallel_groups

        # Estimate total runtime
        sequential_time = sum(
            self.performance_baselines.get(plan.tool_name, {}).get("avg_time", 3.0)
            for plan in execution_plan
        )

        # Account for parallelization
        parallel_savings = 0
        for group_tools in parallel_groups.values():
            if len(group_tools) > 1:
                group_times = [
                    self.performance_baselines.get(tool, {}).get("avg_time", 3.0)
                    for tool in group_tools
                ]
                parallel_savings += sum(group_times) - max(group_times)

        analysis["estimated_runtime"] = max(sequential_time - parallel_savings, 5.0)

        return analysis

    def _find_critical_path(
        self,
        dependency_graph: Dict[str, List[str]],
        execution_plan: List[ToolExecutionPlan],
    ) -> List[str]:
        """Find the critical path through the workflow"""
        # Simple implementation - would use proper topological sort for complex graphs
        critical_path = []
        processed = set()

        def add_to_path(tool_name: str):
            if tool_name in processed:
                return

            # Add dependencies first
            for dep in dependency_graph.get(tool_name, []):
                add_to_path(dep)

            critical_path.append(tool_name)
            processed.add(tool_name)

        # Process in priority order
        for plan in sorted(execution_plan, key=lambda x: x.priority.value):
            add_to_path(plan.tool_name)

        return critical_path

    def create_workflow_visualization(
        self,
        execution_plan: List[ToolExecutionPlan],
        execution_results: List[ToolExecutionResult] = None,
    ) -> str:
        """
        Create interactive workflow visualization
        """
        # Create network graph
        G = nx.DiGraph()

        # Add nodes
        for plan in execution_plan:
            status = "pending"
            execution_time = None

            if execution_results:
                for result in execution_results:
                    if result.tool_name == plan.tool_name:
                        status = "success" if result.success else "failed"
                        execution_time = result.execution_time
                        break

            G.add_node(
                plan.tool_name,
                category=plan.category.value,
                priority=plan.priority.value,
                status=status,
                execution_time=execution_time,
            )

        # Add edges for dependencies
        for plan in execution_plan:
            for dep in plan.depends_on:
                G.add_edge(dep, plan.tool_name)

        # Create plotly visualization
        pos = nx.spring_layout(G, k=3, iterations=50)

        # Prepare node traces
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode="markers+text",
            text=[node for node in G.nodes()],
            textposition="middle center",
            marker=dict(
                size=50,
                color=[self._get_node_color(G.nodes[node]) for node in G.nodes()],
                line=dict(width=2, color="black"),
            ),
            hovertemplate="<b>%{text}</b><br>"
            + "Category: %{customdata[0]}<br>"
            + "Status: %{customdata[1]}<br>"
            + "Time: %{customdata[2]:.2f}s<extra></extra>",
            customdata=[
                [
                    G.nodes[node].get("category", "unknown"),
                    G.nodes[node].get("status", "pending"),
                    G.nodes[node].get("execution_time", 0) or 0,
                ]
                for node in G.nodes()
            ],
        )

        # Prepare edge traces
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
            line=dict(width=2, color="gray"),
            hoverinfo="none",
            mode="lines",
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Workflow Execution Graph",
                title_font_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Tool Workflow Visualization",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(color="gray", size=12),
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        # Save visualization
        viz_path = self.visualization_dir / "workflow_graph.html"
        fig.write_html(str(viz_path))

        logger.info(f"Workflow visualization saved to {viz_path}")
        return str(viz_path)

    def _get_node_color(self, node_data: Dict[str, Any]) -> str:
        """Get color for workflow node based on status and category"""
        status = node_data.get("status", "pending")
        category = node_data.get("category", "unknown")

        if status == "success":
            return "lightgreen"
        elif status == "failed":
            return "lightcoral"
        elif status == "pending":
            # Color by category
            category_colors = {
                "structural": "lightblue",
                "functional": "lightgoldenrodyellow",
                "optimization": "lightpink",
                "validation": "lightcyan",
                "visualization": "lavender",
            }
            return category_colors.get(category, "lightgray")
        else:
            return "lightgray"

    def create_performance_dashboard(self) -> str:
        """
        Create performance monitoring dashboard
        """
        if not self.execution_history:
            return ""

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Execution Times",
                "Success Rate",
                "Tool Categories",
                "Performance Trends",
            ),
            specs=[
                [{"secondary_y": False}, {"type": "pie"}],
                [{"type": "bar"}, {"secondary_y": False}],
            ],
        )

        # Execution times
        tools = [r.tool_name for r in self.execution_history]
        times = [r.execution_time for r in self.execution_history]

        fig.add_trace(
            go.Scatter(x=tools, y=times, mode="markers+lines", name="Execution Time"),
            row=1,
            col=1,
        )

        # Success rate
        success_count = sum(1 for r in self.execution_history if r.success)
        fail_count = len(self.execution_history) - success_count

        fig.add_trace(
            go.Pie(labels=["Success", "Failed"], values=[success_count, fail_count]),
            row=1,
            col=2,
        )

        # Tool categories
        categories = {}
        for result in self.execution_history:
            category = result.performance_metrics.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1

        fig.add_trace(
            go.Bar(
                x=list(categories.keys()),
                y=list(categories.values()),
                name="Category Usage",
            ),
            row=2,
            col=1,
        )

        # Performance trends (moving average)
        if len(self.execution_history) > 1:
            window = min(5, len(times))
            moving_avg = []
            for i in range(len(times)):
                start_idx = max(0, i - window + 1)
                moving_avg.append(sum(times[start_idx : i + 1]) / (i - start_idx + 1))

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(moving_avg))),
                    y=moving_avg,
                    mode="lines",
                    name="Moving Average",
                ),
                row=2,
                col=2,
            )

        fig.update_layout(height=800, title_text="Tool Execution Performance Dashboard")

        # Save dashboard
        dashboard_path = self.visualization_dir / "performance_dashboard.html"
        fig.write_html(str(dashboard_path))

        logger.info(f"Performance dashboard saved to {dashboard_path}")
        return str(dashboard_path)

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive execution summary with insights
        """
        if not self.execution_history:
            return {"total_executions": 0}

        total_time = sum(r.execution_time for r in self.execution_history)
        success_count = sum(1 for r in self.execution_history if r.success)

        summary = {
            "total_executions": len(self.execution_history),
            "total_execution_time": total_time,
            "average_execution_time": total_time / len(self.execution_history),
            "success_rate": success_count / len(self.execution_history),
            "tools_used": list(set(r.tool_name for r in self.execution_history)),
            "performance_insights": self._generate_performance_insights(),
            "recommendations": self._generate_recommendations(),
        }

        return summary

    def _generate_performance_insights(self) -> List[str]:
        """Generate insights based on execution history"""
        insights = []

        if not self.execution_history:
            return insights

        # Analyze execution times
        times = [r.execution_time for r in self.execution_history]
        avg_time = sum(times) / len(times)

        if avg_time > 10:
            insights.append("Average execution time is high - consider optimization")

        # Analyze success rates
        success_rate = sum(1 for r in self.execution_history if r.success) / len(
            self.execution_history
        )
        if success_rate < 0.9:
            insights.append("Success rate below 90% - investigate failure patterns")

        # Tool-specific insights
        tool_performance = {}
        for result in self.execution_history:
            tool = result.tool_name
            if tool not in tool_performance:
                tool_performance[tool] = {"times": [], "successes": 0, "total": 0}

            tool_performance[tool]["times"].append(result.execution_time)
            tool_performance[tool]["total"] += 1
            if result.success:
                tool_performance[tool]["successes"] += 1

        # Find slowest tool
        slowest_tool = max(
            tool_performance.items(),
            key=lambda x: sum(x[1]["times"]) / len(x[1]["times"]),
        )
        insights.append(
            f"Slowest tool: {slowest_tool[0]} (avg: {sum(slowest_tool[1]['times'])/len(slowest_tool[1]['times']):.2f}s)"
        )

        return insights

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        if not self.execution_history:
            return recommendations

        # Analyze patterns
        tool_usage = {}
        for result in self.execution_history:
            tool = result.tool_name
            tool_usage[tool] = tool_usage.get(tool, 0) + 1

        # Most used tools
        if tool_usage:
            most_used = max(tool_usage.items(), key=lambda x: x[1])
            recommendations.append(
                f"Consider caching results for {most_used[0]} (used {most_used[1]} times)"
            )

        # Performance recommendations
        avg_time = sum(r.execution_time for r in self.execution_history) / len(
            self.execution_history
        )
        if avg_time > 5:
            recommendations.append(
                "Consider implementing parallel execution for independent tools"
            )

        # Failure analysis
        failed_tools = [r.tool_name for r in self.execution_history if not r.success]
        if failed_tools:
            most_failed = max(set(failed_tools), key=failed_tools.count)
            recommendations.append(
                f"Investigate and improve reliability of {most_failed}"
            )

        return recommendations
