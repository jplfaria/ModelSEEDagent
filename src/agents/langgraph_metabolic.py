"""
LangGraph-based Metabolic Agent

Advanced implementation using LangGraph StateGraph with:
* Graph-based workflow execution instead of linear ReAct
* Parallel tool execution capabilities
* State persistence and checkpointing
* Advanced error recovery and graceful degradation
* Integration with advanced Argo LLM client
* Comprehensive observability and debugging
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

# Additional imports for enhanced features
import numpy as np
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config.debug_config import get_debug_config
from ..config.prompts import load_prompt_template, load_prompts
from ..llm.base import BaseLLM
from ..reasoning.artifact_intelligence import ArtifactIntelligenceEngine
from ..reasoning.composite_metrics import CompositeMetricsCalculator
from ..reasoning.context_enhancer import BiochemContextEnhancer
from ..reasoning.enhanced_prompt_provider import EnhancedPromptProvider
from ..reasoning.integrated_quality_system import QualityAwarePromptProvider

# Intelligence Enhancement Framework imports
from ..reasoning.intelligent_reasoning_system import (
    IntelligentAnalysisRequest,
    IntelligentReasoningSystem,
)
from ..reasoning.self_reflection_engine import SelfReflectionEngine
from ..tools.base import BaseTool, ToolResult
from .base import AgentConfig, AgentResult, BaseAgent
from .tool_integration import (
    EnhancedToolIntegration,
    ToolCategory,
    ToolExecutionPlan,
    ToolExecutionResult,
    ToolPriority,
    WorkflowState,
)

logger = logging.getLogger(__name__)

# -------------------------------
# State Management
# -------------------------------


class AgentState(TypedDict):
    """State schema for the LangGraph agent"""

    # Input/Output
    query: str
    messages: List[BaseMessage]
    final_answer: Optional[str]

    # Tool execution
    tools_called: List[str]
    tool_results: Dict[str, Any]
    parallel_results: Dict[str, ToolResult]
    tools_to_call: Optional[List[str]]

    # Enhanced tool integration
    execution_plan: Optional[List[ToolExecutionPlan]]
    intent_analysis: Optional[Dict[str, Any]]
    workflow_analysis: Optional[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, Dict[str, Any]]]
    workflow_visualization: Optional[str]

    # Execution control
    iteration: int
    max_iterations: int
    next_action: Optional[str]

    # Error handling
    errors: List[str]
    retry_count: int
    recovery_strategy: Optional[str]
    continue_analysis: Optional[bool]

    # Memory and context
    previous_results: List[str]
    context_summary: str

    # Metadata
    run_id: str
    timestamp: str
    workflow_state: Literal[
        "thinking",
        "planning",
        "executing",
        "parallel_exec",
        "analyzing",
        "completed",
        "error",
    ]


# -------------------------------
# Enhanced Memory and Storage
# -------------------------------


class EnhancedVectorStore:
    """Enhanced vector store with better retrieval capabilities"""

    def __init__(self):
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        self.embeddings = None

    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> None:
        """Add document with metadata"""
        self.documents.append(text)
        self.metadata.append(metadata or {})

        if len(self.documents) > 1:
            self.embeddings = self.vectorizer.fit_transform(self.documents)

        logger.debug(f"VectorStore: Added document. Total: {len(self.documents)}")

    def query(
        self, query_text: str, top_n: int = 3, min_similarity: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Query with similarity threshold and metadata"""
        if not self.documents or self.embeddings is None:
            return []

        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.embeddings)[0]

        # Filter by minimum similarity and get top_n
        relevant_indices = [
            (i, sim) for i, sim in enumerate(similarities) if sim >= min_similarity
        ]
        relevant_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = relevant_indices[:top_n]

        results = []
        for idx, similarity in top_indices:
            results.append(
                {
                    "text": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "similarity": similarity,
                }
            )

        return results


class SimulationResultsStore:
    """Enhanced simulation results storage with better organization"""

    def __init__(self, run_dir: Path):
        self.results = {}
        self.run_dir = run_dir
        self.results_dir = run_dir / "simulation_results"
        self.results_dir.mkdir(exist_ok=True)

    def save_results(
        self, tool_name: str, model_id: str, solution, metadata: Dict[str, Any] = None
    ) -> str:
        """Save simulation results with enhanced metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_id = f"{tool_name}_{model_id}_{timestamp}"

        # Enhanced result data structure
        result_data = {
            "id": result_id,
            "tool_name": tool_name,
            "model_id": model_id,
            "timestamp": timestamp,
            "objective_value": (
                float(solution.objective_value)
                if hasattr(solution, "objective_value")
                else None
            ),
            "status": (
                str(solution.status) if hasattr(solution, "status") else "unknown"
            ),
            "fluxes": solution.fluxes.to_dict() if hasattr(solution, "fluxes") else {},
            "metadata": metadata or {},
            "file_paths": {},
        }

        # Save to memory
        self.results[result_id] = result_data

        # Export files
        result_data["file_paths"] = self._export_result_files(result_data)

        logger.info(f"SimulationResultsStore: Saved {tool_name} results as {result_id}")
        return result_id

    def _export_result_files(self, result_data: Dict[str, Any]) -> Dict[str, str]:
        """Export result files in multiple formats"""
        result_id = result_data["id"]
        files = {}

        # JSON export
        json_path = self.results_dir / f"{result_id}.json"
        with open(json_path, "w") as f:
            json.dump(result_data, f, indent=2)
        files["json"] = str(json_path)

        # CSV export for fluxes
        if result_data.get("fluxes"):
            try:
                import pandas as pd

                flux_df = pd.DataFrame.from_dict(
                    result_data["fluxes"], orient="index", columns=["flux"]
                )
                csv_path = self.results_dir / f"{result_id}_fluxes.csv"
                flux_df.to_csv(csv_path)
                files["fluxes_csv"] = str(csv_path)
            except Exception as e:
                logger.warning(f"Could not export fluxes CSV: {e}")

        return files


# -------------------------------
# LangGraph Metabolic Agent
# -------------------------------


class LangGraphMetabolicAgent(BaseAgent):
    """
    Advanced LangGraph-based metabolic agent with modern workflow capabilities
    """

    def __init__(
        self, llm: BaseLLM, tools: List[BaseTool], config: Dict[str, Any] | AgentConfig
    ):
        # Setup logging and directories BEFORE parent init
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_base = Path(__file__).parent.parent.parent / "logs"
        self.run_dir = self.log_base / f"langgraph_run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage systems
        self.vector_store = EnhancedVectorStore()
        self.simulation_store = SimulationResultsStore(self.run_dir)

        # Enhanced tool integration
        self.tool_integration = EnhancedToolIntegration(tools, self.run_dir)

        # Tool management
        self._tools_dict = {t.tool_name: t for t in tools}
        self._available_tools = tools

        # Initialize parent class
        super().__init__(llm, tools, config)

        # LangGraph specific initialization
        self.memory = MemorySaver()
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)

        # Enhanced tracking
        self.execution_log = []
        self.performance_metrics = {}

        # Initialize Intelligence Enhancement Framework (Phase 1-5)
        try:
            logger.info("🧠 Initializing Intelligence Enhancement Framework...")

            # Phase 1: Enhanced Prompt Provider
            self.enhanced_prompt_provider = EnhancedPromptProvider()

            # Phase 2: Context Enhancer
            self.context_enhancer = BiochemContextEnhancer()

            # Phase 3: Quality System
            self.quality_provider = QualityAwarePromptProvider()
            self.metrics_calculator = CompositeMetricsCalculator()

            # Phase 4: Artifact Intelligence
            self.artifact_intelligence = ArtifactIntelligenceEngine()
            self.self_reflection_engine = SelfReflectionEngine()

            # Phase 5: Intelligent Reasoning System (Main Coordinator)
            storage_path = str(self.run_dir / "intelligence_framework")
            self.intelligent_reasoning_system = IntelligentReasoningSystem(
                storage_path=storage_path
            )

            logger.info(
                "✅ Intelligence Enhancement Framework initialized successfully"
            )
            self.intelligence_enabled = True

        except Exception as e:
            logger.warning(
                f"⚠️ Intelligence Enhancement Framework initialization failed: {e}"
            )
            self.intelligence_enabled = False

        # Only log initialization if LangGraph debug is enabled
        debug_config = get_debug_config()
        if debug_config.langgraph_debug:
            logger.info(f"LangGraphMetabolicAgent initialized with {len(tools)} tools")
            logger.info(f"Run directory: {self.run_dir}")
        else:
            # Use very low log level for quiet operation
            logger.log(5, f"LangGraph agent initialized with {len(tools)} tools")

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with parallel execution capabilities"""

        # Define the workflow graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("planner", self._planning_node)
        workflow.add_node("single_tool", self._single_tool_node)
        workflow.add_node("parallel_tools", self._parallel_tools_node)
        workflow.add_node("analyzer", self._analysis_node)
        workflow.add_node("error_handler", self._error_handler_node)
        workflow.add_node("finalizer", self._finalizer_node)

        # Define workflow edges
        workflow.add_edge(START, "planner")

        # Conditional routing from planner
        workflow.add_conditional_edges(
            "planner",
            self._should_continue,
            {
                "single_tool": "single_tool",
                "parallel_tools": "parallel_tools",
                "analyze": "analyzer",
                "finalize": "finalizer",
                "error": "error_handler",
            },
        )

        # Single tool execution
        workflow.add_edge("single_tool", "analyzer")

        # Parallel tool execution
        workflow.add_edge("parallel_tools", "analyzer")

        # Analysis routing
        workflow.add_conditional_edges(
            "analyzer",
            self._analyze_results,
            {"continue": "planner", "finalize": "finalizer", "error": "error_handler"},
        )

        # Error handling
        workflow.add_conditional_edges(
            "error_handler",
            self._handle_error,
            {"retry": "planner", "finalize": "finalizer"},
        )

        # Finalizer ends the workflow
        workflow.add_edge("finalizer", END)

        return workflow

    def _planning_node(self, state: AgentState) -> AgentState:
        """Enhanced planning node with intelligent tool selection"""
        logger.debug(f"Enhanced planning node - iteration {state['iteration']}")

        # Update workflow state
        state["workflow_state"] = "planning"
        state["timestamp"] = datetime.now().isoformat()

        # Check iteration limits
        if state["iteration"] >= state["max_iterations"]:
            state["next_action"] = "finalize"
            state["final_answer"] = (
                f"Reached maximum iterations ({state['max_iterations']}). "
                + self._summarize_results(state)
            )
            return state

        try:
            # Enhanced planning with intent analysis
            if not state.get("execution_plan"):
                # First iteration - analyze intent and create execution plan
                intent_analysis = self.tool_integration.analyze_query_intent(
                    state["query"]
                )

                # Extract model path if available
                model_path = self._extract_model_path(state["query"])

                # Create optimized execution plan
                execution_plan = self.tool_integration.create_execution_plan(
                    intent_analysis, model_path
                )

                # Analyze workflow dependencies
                workflow_analysis = self.tool_integration.analyze_workflow_dependencies(
                    execution_plan
                )

                # Store in state
                state["execution_plan"] = execution_plan
                state["intent_analysis"] = intent_analysis
                state["workflow_analysis"] = workflow_analysis

                logger.info(
                    f"Created execution plan: {len(execution_plan)} tools, "
                    f"estimated runtime: {workflow_analysis['estimated_runtime']:.1f}s"
                )

                # Create workflow visualization
                viz_path = self.tool_integration.create_workflow_visualization(
                    execution_plan
                )
                state["workflow_visualization"] = viz_path

            # Determine next action based on execution plan
            execution_plan = state["execution_plan"]
            completed_tools = set(state.get("tools_called", []))

            # Find next tools to execute
            next_tools = []
            parallel_groups = {}

            for plan in execution_plan:
                if plan.tool_name in completed_tools:
                    continue

                # Check dependencies
                deps_satisfied = all(dep in completed_tools for dep in plan.depends_on)
                if not deps_satisfied:
                    continue

                # Group parallel tools
                if plan.parallel_group:
                    if plan.parallel_group not in parallel_groups:
                        parallel_groups[plan.parallel_group] = []
                    parallel_groups[plan.parallel_group].append(plan.tool_name)
                else:
                    next_tools.append(plan.tool_name)

            # Decide execution strategy
            if parallel_groups and len(list(parallel_groups.values())[0]) > 1:
                # Execute parallel group
                state["next_action"] = "parallel_tools"
                state["tools_to_call"] = list(parallel_groups.values())[0]
                logger.info(f"Planned parallel execution: {state['tools_to_call']}")
            elif next_tools:
                # Execute single tool
                state["next_action"] = "single_tool"
                state["tools_to_call"] = [next_tools[0]]
                logger.info(f"Planned single tool execution: {next_tools[0]}")
            elif len(completed_tools) < len(execution_plan):
                # Still have tools to execute - find any tool without unsatisfied dependencies
                remaining_tools = [
                    p for p in execution_plan if p.tool_name not in completed_tools
                ]

                # Try to find a tool we can execute (ignore dependencies for comprehensive analysis)
                query_lower = state["query"].lower()
                is_comprehensive = any(
                    word in query_lower
                    for word in [
                        "comprehensive",
                        "complete",
                        "full",
                        "detailed",
                        "thorough",
                    ]
                )

                if is_comprehensive and remaining_tools:
                    # For comprehensive analysis, execute remaining tools regardless of dependencies
                    state["next_action"] = "single_tool"
                    state["tools_to_call"] = [remaining_tools[0].tool_name]
                    logger.info(
                        f"Comprehensive analysis: executing remaining tool {remaining_tools[0].tool_name}"
                    )
                else:
                    # Original dependency logic
                    state["next_action"] = "analyze"
                    logger.info("Waiting for dependencies, analyzing current results")
            else:
                # All tools completed
                state["next_action"] = "finalize"
                logger.info("All planned tools completed, finalizing")

            # Log planning decision with enhanced context
            self._log_execution(
                "enhanced_planning",
                {
                    "iteration": state["iteration"],
                    "decision": state["next_action"],
                    "tools_to_call": state.get("tools_to_call", []),
                    "completed_tools": list(completed_tools),
                    "total_planned_tools": len(execution_plan),
                    "intent_analysis": state.get("intent_analysis", {}),
                    "workflow_complexity": state.get("intent_analysis", {}).get(
                        "workflow_complexity", "unknown"
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Enhanced planning error: {e}")
            state["errors"].append(f"Enhanced planning error: {str(e)}")
            state["next_action"] = "error"

        return state

    def _single_tool_node(self, state: AgentState) -> AgentState:
        """Execute a single tool with enhanced monitoring"""
        logger.debug("Executing single tool with enhanced monitoring")

        state["workflow_state"] = "executing"
        tool_name = state.get("tools_to_call", [None])[0]

        if not tool_name or tool_name not in self._tools_dict:
            state["errors"].append(f"Invalid tool: {tool_name}")
            state["next_action"] = "error"
            return state

        try:
            # Find execution plan for this tool
            execution_plan = state.get("execution_plan", [])
            plan = next((p for p in execution_plan if p.tool_name == tool_name), None)

            if plan:
                # Execute with enhanced monitoring
                exec_result = self.tool_integration.execute_tool_with_monitoring(plan)

                # Store enhanced results
                state["tool_results"][tool_name] = (
                    exec_result.result.model_dump()
                    if hasattr(exec_result.result, "model_dump")
                    else exec_result.result.__dict__
                )
                state["tools_called"].append(tool_name)

                # Store performance metrics
                if "performance_metrics" not in state:
                    state["performance_metrics"] = {}
                if state["performance_metrics"] is None:
                    state["performance_metrics"] = {}
                state["performance_metrics"][
                    tool_name
                ] = exec_result.performance_metrics

                # Add to vector store for memory
                self.vector_store.add_document(
                    f"Enhanced Tool: {tool_name}, Result: {exec_result.result.message}",
                    {
                        "tool": tool_name,
                        "success": exec_result.success,
                        "iteration": state["iteration"],
                        "execution_time": exec_result.execution_time,
                        "category": plan.category.value,
                    },
                )

                logger.info(
                    f"Enhanced tool {tool_name} executed - success: {exec_result.success}, time: {exec_result.execution_time:.2f}s"
                )
            else:
                # Fallback to basic execution
                tool = self._tools_dict[tool_name]
                tool_input = self._prepare_tool_input(state, tool_name)
                # Handle special cases for tools that expect string instead of dict
                if (
                    tool_name == "analyze_metabolic_model"
                    and isinstance(tool_input, dict)
                    and "model_path" in tool_input
                ):
                    result = tool._run(tool_input["model_path"])
                else:
                    result = tool._run(tool_input)

                state["tool_results"][tool_name] = (
                    result.model_dump()
                    if hasattr(result, "model_dump")
                    else result.__dict__
                )
                state["tools_called"].append(tool_name)

                logger.info(f"Basic tool {tool_name} executed successfully")

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            state["errors"].append(f"Tool {tool_name} error: {str(e)}")
            state["next_action"] = "error"

        return state

    def _parallel_tools_node(self, state: AgentState) -> AgentState:
        """Execute multiple tools in parallel with enhanced monitoring"""
        logger.debug("Executing parallel tools with enhanced monitoring")

        state["workflow_state"] = "parallel_exec"
        tools_to_call = state.get("tools_to_call", [])

        logger.info(f"Enhanced parallel tools node - tools to call: {tools_to_call}")

        if not tools_to_call:
            logger.error("No tools specified for parallel execution")
            state["errors"].append("No tools specified for parallel execution")
            state["next_action"] = "error"
            return state

        parallel_results = {}
        execution_plan = state.get("execution_plan", [])

        for tool_name in tools_to_call:
            if tool_name not in self._tools_dict:
                error_result = ToolResult(
                    success=False,
                    message=f"Tool {tool_name} not found",
                    error=f"Unknown tool: {tool_name}",
                )
                parallel_results[tool_name] = error_result
                continue

            try:
                # Find execution plan for this tool
                plan = next(
                    (p for p in execution_plan if p.tool_name == tool_name), None
                )

                if plan:
                    # Execute with enhanced monitoring
                    exec_result = self.tool_integration.execute_tool_with_monitoring(
                        plan
                    )
                    parallel_results[tool_name] = exec_result.result

                    # Store performance metrics
                    if "performance_metrics" not in state:
                        state["performance_metrics"] = {}
                    if state["performance_metrics"] is None:
                        state["performance_metrics"] = {}
                    state["performance_metrics"][
                        tool_name
                    ] = exec_result.performance_metrics

                    logger.info(
                        f"Enhanced parallel tool {tool_name} completed - time: {exec_result.execution_time:.2f}s"
                    )
                else:
                    # Fallback to basic execution
                    tool = self._tools_dict[tool_name]
                    tool_input = self._prepare_tool_input(state, tool_name)
                    result = tool._run(tool_input)
                    parallel_results[tool_name] = result

                    logger.info(f"Basic parallel tool {tool_name} completed")

                # Store in main results
                state["tool_results"][tool_name] = (
                    parallel_results[tool_name].model_dump()
                    if hasattr(parallel_results[tool_name], "model_dump")
                    else parallel_results[tool_name].__dict__
                )
                state["tools_called"].append(tool_name)

                # Add to memory with enhanced metadata
                plan_metadata = (
                    {"category": plan.category.value}
                    if plan
                    else {"category": "unknown"}
                )
                self.vector_store.add_document(
                    f"Enhanced Parallel Tool: {tool_name}, Result: {parallel_results[tool_name].message}",
                    {
                        "tool": tool_name,
                        "success": parallel_results[tool_name].success,
                        "parallel": True,
                        "iteration": state["iteration"],
                        **plan_metadata,
                    },
                )

            except Exception as e:
                logger.error(f"Enhanced parallel tool {tool_name} error: {e}")
                parallel_results[tool_name] = ToolResult(
                    success=False, message=f"Error executing {tool_name}", error=str(e)
                )
                state["errors"].append(f"Enhanced parallel tool {tool_name}: {str(e)}")

        state["parallel_results"] = {
            k: v.model_dump() if hasattr(v, "model_dump") else v.__dict__
            for k, v in parallel_results.items()
        }

        logger.info(
            f"Enhanced parallel execution completed: {len(parallel_results)} tools"
        )
        return state

    def _analysis_node(self, state: AgentState) -> AgentState:
        """Analyze results and determine next steps"""
        logger.debug("Analyzing results")

        state["workflow_state"] = "analyzing"

        # Create analysis prompt
        analysis_prompt = self._create_analysis_prompt(state)

        try:
            response = self.llm._generate_response(analysis_prompt)

            # Store analysis
            analysis_summary = response.text
            state["context_summary"] = analysis_summary
            state["previous_results"].append(analysis_summary)

            # Add to messages
            state["messages"].append(AIMessage(content=analysis_summary))

            # Determine if we should continue or finalize
            should_continue = self._should_continue_analysis(analysis_summary, state)
            state["continue_analysis"] = should_continue

            logger.info(f"Analysis completed. Continue: {should_continue}")

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            state["errors"].append(f"Analysis error: {str(e)}")
            state["next_action"] = "error"

        return state

    def _error_handler_node(self, state: AgentState) -> AgentState:
        """Handle errors with recovery strategies"""
        logger.debug("Handling errors")

        state["workflow_state"] = "error"
        state["retry_count"] = state.get("retry_count", 0) + 1

        # Implement recovery strategies
        if state["retry_count"] <= 2:  # Allow up to 2 retries
            # Strategy 1: Simplify the query
            if state["retry_count"] == 1:
                state["recovery_strategy"] = "simplify"
                state["query"] = self._simplify_query(state["query"])
                state["next_action"] = "retry"
                logger.info("Recovery strategy: Simplifying query")

            # Strategy 2: Use different tools
            elif state["retry_count"] == 2:
                state["recovery_strategy"] = "alternative_tools"
                state["next_action"] = "retry"
                logger.info("Recovery strategy: Using alternative tools")

        else:
            # Max retries reached - finalize with error summary
            state["recovery_strategy"] = "finalize_with_errors"
            error_summary = self._create_error_summary(state)
            state["final_answer"] = f"Analysis completed with errors: {error_summary}"
            state["next_action"] = "finalize"
            logger.warning("Max retries reached, finalizing with errors")

        return state

    def _finalizer_node(self, state: AgentState) -> AgentState:
        """Enhanced finalizer with comprehensive reporting and visualization"""
        logger.debug("Enhanced finalizing workflow")

        state["workflow_state"] = "completed"

        if not state.get("final_answer"):
            state["final_answer"] = self._create_final_answer(state)

        # Create comprehensive summary
        summary = self._create_execution_summary(state)

        # Generate enhanced reports
        try:
            # Create performance dashboard
            dashboard_path = self.tool_integration.create_performance_dashboard()
            if dashboard_path:
                state["performance_dashboard"] = dashboard_path

            # Create final workflow visualization with results
            execution_results = []
            if hasattr(self.tool_integration, "execution_history"):
                execution_results = self.tool_integration.execution_history

            execution_plan = state.get("execution_plan", [])
            if execution_plan:
                final_viz_path = self.tool_integration.create_workflow_visualization(
                    execution_plan, execution_results
                )
                state["final_workflow_visualization"] = final_viz_path

            # Get tool integration summary
            tool_summary = self.tool_integration.get_execution_summary()
            state["tool_execution_summary"] = tool_summary

            logger.info("Enhanced reporting completed successfully")

        except Exception as e:
            logger.warning(f"Enhanced reporting failed: {e}")
            # Continue with basic finalization

        # Log enhanced final execution details
        self._log_execution(
            "enhanced_finalization",
            {
                "total_iterations": state["iteration"],
                "tools_used": list(set(state["tools_called"])),
                "total_errors": len(state["errors"]),
                "final_answer_length": len(state["final_answer"]),
                "execution_summary": summary,
                "performance_metrics": state.get("performance_metrics", {}),
                "workflow_complexity": state.get("intent_analysis", {}).get(
                    "workflow_complexity", "unknown"
                ),
                "total_execution_time": sum(
                    metrics.get("execution_time", 0)
                    for metrics in (state.get("performance_metrics") or {}).values()
                ),
                "visualizations_created": [
                    state.get("workflow_visualization"),
                    state.get("final_workflow_visualization"),
                    state.get("performance_dashboard"),
                ],
            },
        )

        logger.info(
            f"Enhanced workflow completed after {state['iteration']} iterations"
        )
        return state

    # -------------------------------
    # Conditional Routing Functions
    # -------------------------------

    def _should_continue(self, state: AgentState) -> str:
        """Determine next action from planner"""
        next_action = state.get("next_action", "finalize")

        if next_action in [
            "single_tool",
            "parallel_tools",
            "analyze",
            "finalize",
            "error",
        ]:
            return next_action

        # Default fallback
        return "finalize"

    def _analyze_results(self, state: AgentState) -> str:
        """Determine next action after analysis with optimized logic"""
        if state.get("errors"):
            return "error"

        # For comprehensive queries, skip intermediate analysis if more tools are planned
        query_lower = state["query"].lower()
        is_comprehensive = any(
            word in query_lower
            for word in ["comprehensive", "complete", "full", "detailed", "thorough"]
        )

        # Get planned tools
        planned_tools = []
        if "intent_analysis" in state and "suggested_tools" in state["intent_analysis"]:
            planned_tools = state["intent_analysis"]["suggested_tools"]

        tools_called = len(state["tools_called"])
        planned_count = len(planned_tools)

        # For comprehensive analysis, skip intermediate LLM analysis to improve performance
        if is_comprehensive and tools_called < planned_count:
            state["iteration"] += 1
            return "continue"

        if (
            state.get("continue_analysis", False)
            and state["iteration"] < state["max_iterations"]
        ):
            state["iteration"] += 1
            return "continue"

        return "finalize"

    def _handle_error(self, state: AgentState) -> str:
        """Determine action after error handling"""
        return state.get("next_action", "finalize")

    # -------------------------------
    # Helper Methods
    # -------------------------------

    def _create_planning_prompt(self, state: AgentState) -> str:
        """Create prompt for planning phase"""
        context = ""
        if state["previous_results"]:
            context = f"Previous analysis: {state['context_summary']}\n\n"

        available_tools = ", ".join(self._tools_dict.keys())

        prompt = f"""You are an expert metabolic modeling agent. Analyze the query and determine the best approach.

Query: {state["query"]}

{context}Available tools: {available_tools}

Current iteration: {state["iteration"]}/{state["max_iterations"]}
Tools already called: {", ".join(state["tools_called"]) if state["tools_called"] else "None"}

Based on the query and context, determine your next action:

1. SINGLE_TOOL: If you need to call one specific tool
   Format: ACTION: SINGLE_TOOL
   TOOL: [tool_name]
   REASON: [why this tool]

2. PARALLEL_TOOLS: If you can call multiple tools simultaneously
   Format: ACTION: PARALLEL_TOOLS
   TOOLS: [tool1, tool2, tool3]
   REASON: [why these tools together]

3. ANALYZE: If you have enough information and need to analyze results
   Format: ACTION: ANALYZE
   REASON: [what to analyze]

4. FINALIZE: If you can provide a complete answer
   Format: ACTION: FINALIZE
   ANSWER: [your final answer]

Choose the most efficient approach:"""

        return prompt

    def _parse_planning_response(self, response: str) -> tuple[str, List[str]]:
        """Parse LLM planning response"""
        response_upper = response.strip().upper()

        if "ACTION: SINGLE_TOOL" in response_upper:
            tool_match = re.search(r"TOOL:\s*(\w+)", response_upper)
            if tool_match:
                return "single_tool", [tool_match.group(1).lower()]

        elif "ACTION: PARALLEL_TOOLS" in response_upper:
            # Try multiple patterns for tool list parsing
            tools_match = re.search(r"TOOLS:\s*\[(.*?)\]", response_upper)
            if tools_match:
                tools_str = tools_match.group(1)
                # Split by comma and clean up
                tools = [t.strip().lower() for t in tools_str.split(",")]
                # Remove any quotes or extra whitespace
                tools = [t.strip("\"'") for t in tools if t.strip()]
                if tools:
                    return "parallel_tools", tools

            # Alternative pattern: tools separated by commas without brackets
            tools_match = re.search(r"TOOLS:\s*([a-z_,\s]+)", response_upper.lower())
            if tools_match:
                tools_str = tools_match.group(1)
                tools = [t.strip() for t in tools_str.split(",")]
                tools = [t for t in tools if t and t in self._tools_dict]
                if tools:
                    return "parallel_tools", tools

        elif "ACTION: ANALYZE" in response_upper:
            return "analyze", []

        elif "ACTION: FINALIZE" in response_upper:
            return "finalize", []

        # Debug logging
        logger.debug(f"Could not parse planning response: {response[:100]}")

        # Default fallback
        return "analyze", []

    def _create_analysis_prompt(self, state: AgentState) -> str:
        """Create enhanced analysis prompt with rich tool data"""
        results_summary = ""

        for tool_name, result in state["tool_results"].items():
            status = "Success" if result.get("success", True) else "Failed"
            message = result.get("message", "No message")

            # Include rich data summary for better analysis
            data_summary = ""
            data = result.get("data")
            if data and isinstance(data, dict):
                # Provide key insights from the data
                data_summary = self._summarize_tool_data(tool_name, data)

            results_summary += f"\n\n{tool_name} ({status}):\n- Message: {message}"
            if data_summary:
                results_summary += f"\n- Key Findings: {data_summary}"

        return f"""You are analyzing metabolic modeling results. Based on the tool execution results, provide insights and determine next steps.

Original query: {state["query"]}

Detailed tool results:{results_summary}

Please provide:
1. Key metabolic insights discovered so far
2. Whether you have sufficient information for a comprehensive answer
3. What additional analysis might enhance the results
4. A brief summary of the most important findings

Respond with actionable metabolic insights, not just procedural information."""

    def _should_continue_analysis(self, analysis: str, state: AgentState) -> bool:
        """Optimized analysis continuation logic - reduce LLM overhead"""

        # Get planned tools from intent analysis
        planned_tools = []
        if "intent_analysis" in state and "suggested_tools" in state["intent_analysis"]:
            planned_tools = state["intent_analysis"]["suggested_tools"]

        tools_called = len(state["tools_called"])
        planned_count = len(planned_tools)

        # For comprehensive queries, execute all planned tools with minimal LLM analysis
        query_lower = state["query"].lower()
        is_comprehensive = any(
            word in query_lower
            for word in ["comprehensive", "complete", "full", "detailed", "thorough"]
        )

        if is_comprehensive:
            # Simple rule: continue until all planned tools are executed
            return tools_called < planned_count

        # For non-comprehensive queries, use lightweight analysis
        if tools_called >= planned_count:
            return False

        # Check for obvious completion signals without complex analysis
        analysis_lower = analysis.lower()
        if (
            any(
                phrase in analysis_lower
                for phrase in [
                    "sufficient",
                    "complete",
                    "ready to answer",
                    "have enough",
                ]
            )
            and tools_called >= 2
        ):
            return False

        # Default: continue if we haven't reached our target tool count
        return tools_called < min(planned_count, 3)  # Cap at 3 tools for efficiency

    def _prepare_tool_input(self, state: AgentState, tool_name: str) -> Dict[str, Any]:
        """Prepare appropriate input for each tool"""
        query = state["query"]

        # Most tools need a model path
        if tool_name in [
            "run_metabolic_fba",
            "find_minimal_media",
            "analyze_essentiality",
            "run_flux_variability_analysis",
            "identify_auxotrophies",
            "run_flux_sampling",
            "run_gene_deletion_analysis",
            "run_production_envelope",
            "analyze_metabolic_model",
            "check_missing_media",
            "analyze_reaction_expression",
        ]:
            # Use default E. coli core model path
            default_model_path = str(
                Path(__file__).parent.parent.parent
                / "data"
                / "examples"
                / "e_coli_core.xml"
            )
            return {"model_path": default_model_path}

        # analyze_pathway needs both model_path and pathway
        elif tool_name == "analyze_pathway":
            default_model_path = str(
                Path(__file__).parent.parent.parent
                / "data"
                / "examples"
                / "e_coli_core.xml"
            )
            return {"model_path": default_model_path, "pathway": "glycolysis"}

        # Biochemistry tools need query
        elif tool_name in ["search_biochem"]:
            return {"query": "ATP"}  # Default biochemistry query

        elif tool_name in ["resolve_biochem_entity"]:
            return {"entity_id": "cpd00027"}  # ATP entity ID

        # Default to simple input
        return {"input": query}

    def _simplify_query(self, query: str) -> str:
        """Simplify query for error recovery"""
        # Remove complex parts and focus on core request
        simplified = re.sub(
            r"\b(detailed|comprehensive|complex|advanced)\b", "basic", query.lower()
        )
        return simplified

    def _create_error_summary(self, state: AgentState) -> str:
        """Create summary of errors encountered"""
        errors = state.get("errors", [])
        if not errors:
            return "No specific errors recorded"

        return f"Encountered {len(errors)} errors: " + "; ".join(errors[:3])

    def _create_final_answer(self, state: AgentState) -> str:
        """Create comprehensive final answer with rich metabolic insights"""
        if not state["tool_results"]:
            return "I was unable to execute any tools to analyze your query. Please check your query and try again."

        # Create comprehensive summary with rich data
        summary = f"# Comprehensive Metabolic Analysis Results\n\n"
        summary += f"**Query:** {state['query']}\n\n"

        total_tools = len(state["tool_results"])
        successful_tools = sum(
            1
            for result in state["tool_results"].values()
            if result.get("success", True)
        )
        summary += f"**Execution Summary:** {successful_tools}/{total_tools} tools completed successfully\n\n"

        # Process each tool result with rich data display
        for tool_name, result in state["tool_results"].items():
            summary += f"## {tool_name.replace('_', ' ').title()}\n\n"

            # Show basic status
            status = "✅ Success" if result.get("success", True) else "❌ Failed"
            summary += f"**Status:** {status}\n"
            summary += (
                f"**Message:** {result.get('message', 'No message available')}\n\n"
            )

            # Extract and display rich data if available
            data = result.get("data")
            if data and isinstance(data, dict):
                summary += self._format_tool_data(tool_name, data)

            summary += "\n---\n\n"

        # Add performance summary
        if "performance_metrics" in state and state["performance_metrics"]:
            summary += "## Performance Summary\n\n"
            total_time = sum(
                metrics.get("execution_time", 0)
                for metrics in state["performance_metrics"].values()
            )
            summary += f"**Total Tool Execution Time:** {total_time:.2f}s\n"

            for tool, metrics in state["performance_metrics"].items():
                exec_time = metrics.get("execution_time", 0)
                data_size = metrics.get("data_size", 0)
                summary += f"- {tool}: {exec_time:.2f}s ({data_size:,} bytes data)\n"
            summary += "\n"

        return summary

    def _format_tool_data(self, tool_name: str, data: Dict[str, Any]) -> str:
        """Format rich tool data for display"""
        formatted = ""

        if tool_name == "analyze_metabolic_model":
            if "model_statistics" in data:
                stats = data["model_statistics"]
                formatted += f"**Model Overview:**\n"
                formatted += f"- Reactions: {stats.get('num_reactions', 'N/A')}\n"
                formatted += f"- Metabolites: {stats.get('num_metabolites', 'N/A')}\n"
                formatted += f"- Genes: {stats.get('num_genes', 'N/A')}\n"
                formatted += f"- Subsystems: {stats.get('num_subsystems', 'N/A')}\n\n"

            if "network_properties" in data:
                network = data["network_properties"]
                if "connectivity_summary" in network:
                    conn = network["connectivity_summary"]
                    formatted += f"**Network Connectivity:**\n"
                    formatted += f"- Average connections per metabolite: {conn.get('avg_connections_per_metabolite', 0):.1f}\n"
                    formatted += f"- Most connected metabolite: {conn.get('max_connections', 0)} connections\n\n"

                if (
                    "highly_connected_metabolites" in network
                    and network["highly_connected_metabolites"]
                ):
                    formatted += f"**Hub Metabolites (top 3):**\n"
                    for i, met in enumerate(
                        network["highly_connected_metabolites"][:3]
                    ):
                        formatted += f"{i+1}. {met.get('name', met.get('id', 'Unknown'))}: {met.get('num_reactions', 0)} reactions\n"
                    formatted += "\n"

        elif tool_name == "run_metabolic_fba":
            if "objective_value" in data:
                formatted += f"**Growth Analysis:**\n"

                # Extract actual biomass growth rate from fluxes, not objective value
                growth_rate = 0.0

                # Check in both 'fluxes' and 'significant_fluxes' for biomass reaction
                flux_data = None
                if "fluxes" in data:
                    flux_data = data["fluxes"]
                elif "significant_fluxes" in data:
                    flux_data = data["significant_fluxes"]

                if flux_data:
                    # Find biomass reaction (usually contains "BIOMASS" in the name)
                    biomass_reactions = [
                        k for k in flux_data.keys() if "BIOMASS" in k.upper()
                    ]
                    if biomass_reactions:
                        growth_rate = flux_data[biomass_reactions[0]]

                formatted += f"- Predicted growth rate: {growth_rate:.4f} h⁻¹\n"
                formatted += (
                    f"- FBA objective value: {data.get('objective_value', 0):.4f}\n"
                )
                formatted += (
                    f"- Optimization status: {data.get('status', 'Unknown')}\n\n"
                )

            if "active_reactions" in data:
                active = data["active_reactions"]
                formatted += f"**Metabolic Activity:**\n"
                formatted += (
                    f"- Active reactions: {len(active)} reactions carrying flux\n"
                )
                if active:
                    formatted += f"- Key active reactions: {', '.join(list(active.keys())[:5])}\n"
                formatted += "\n"

        elif tool_name == "analyze_pathway":
            if "summary" in data:
                summary = data["summary"]
                formatted += f"**Pathway Statistics:**\n"
                formatted += (
                    f"- Reactions in pathway: {summary.get('reaction_count', 0)}\n"
                )
                formatted += f"- Associated genes: {summary.get('gene_coverage', 0)}\n"
                formatted += (
                    f"- Metabolites involved: {summary.get('metabolite_count', 0)}\n"
                )
                formatted += f"- Reversible reactions: {summary.get('reversible_reactions', 0)}\n\n"

        # Add data for other tools as needed
        elif "summary" in data:
            # Generic summary formatting
            formatted += f"**Analysis Summary:**\n"
            for key, value in data["summary"].items():
                formatted += f"- {key.replace('_', ' ').title()}: {value}\n"
            formatted += "\n"

        return formatted

    def _summarize_tool_data(self, tool_name: str, data: Dict[str, Any]) -> str:
        """Create brief summary of tool data for analysis prompt"""
        if tool_name == "analyze_metabolic_model" and "model_statistics" in data:
            stats = data["model_statistics"]
            return f"{stats.get('num_reactions', 0)} reactions, {stats.get('num_metabolites', 0)} metabolites, {stats.get('num_genes', 0)} genes"

        elif tool_name == "run_metabolic_fba" and "objective_value" in data:
            # Extract actual biomass growth rate from fluxes
            growth_rate = 0.0

            # Check in both 'fluxes' and 'significant_fluxes' for biomass reaction
            flux_data = None
            if "fluxes" in data:
                flux_data = data["fluxes"]
            elif "significant_fluxes" in data:
                flux_data = data["significant_fluxes"]

            if flux_data:
                biomass_reactions = [
                    k for k in flux_data.keys() if "BIOMASS" in k.upper()
                ]
                if biomass_reactions:
                    growth_rate = flux_data[biomass_reactions[0]]

            status = data.get("status", "unknown")
            active = len(data.get("active_reactions", {}))
            return f"Growth rate: {growth_rate:.4f}, Status: {status}, Active reactions: {active}"

        elif tool_name == "analyze_pathway" and "summary" in data:
            summary = data["summary"]
            return f"{summary.get('reaction_count', 0)} reactions, {summary.get('gene_coverage', 0)} genes"

        return "Analysis completed with detailed results"

    def _create_execution_summary(self, state: AgentState) -> Dict[str, Any]:
        """Create comprehensive execution summary"""
        return {
            "run_id": state["run_id"],
            "query": state["query"],
            "total_iterations": state["iteration"],
            "tools_executed": list(set(state["tools_called"])),
            "parallel_executions": len(state.get("parallel_results", {})),
            "errors_encountered": len(state["errors"]),
            "recovery_attempts": state.get("retry_count", 0),
            "final_status": state["workflow_state"],
            "execution_time": state["timestamp"],
        }

    def _summarize_results(self, state: AgentState) -> str:
        """Create a brief summary of results"""
        if not state["tool_results"]:
            return "No analysis results available."

        return f"Executed {len(state['tool_results'])} tools with {len(state['errors'])} errors."

    def _log_execution(self, step: str, data: Dict[str, Any]) -> None:
        """Log execution details"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "data": data,
        }
        self.execution_log.append(log_entry)

        # Also save to file
        log_file = self.run_dir / "execution_log.json"
        with open(log_file, "w") as f:
            json.dump(self.execution_log, f, indent=2)

    def _extract_model_path(self, query: str) -> Optional[str]:
        """Enhanced model path extraction with dynamic model registry"""
        import re
        from pathlib import Path

        # Model registry for production scalability
        model_registry = {
            # E. coli models
            "ecoli_core": "data/examples/e_coli_core.xml",
            "e_coli_core": "data/examples/e_coli_core.xml",
            "core": "data/examples/e_coli_core.xml",
            "iml1515": "data/models/iML1515.xml",
            "ecoli_full": "data/models/iML1515.xml",
            "full": "data/models/iML1515.xml",
            # Pattern-based matching for user models
            "default": "data/examples/e_coli_core.xml",
        }

        query_lower = query.lower()

        # 1. Check for explicit model names in query
        for model_key, model_path in model_registry.items():
            if model_key in query_lower:
                full_path = Path(__file__).parent.parent.parent / model_path
                if full_path.exists():
                    logger.info(
                        f"Model selected from query: {model_key} -> {model_path}"
                    )
                    return str(full_path)

        # 2. Look for direct file patterns
        file_patterns = [
            r"([\w-]+\.xml)",
            r"([\w-]+\.sbml)",
            r"(data/models/[\w-]+\.xml)",
            r"models/([\w-]+)",
        ]

        for pattern in file_patterns:
            match = re.search(pattern, query_lower)
            if match:
                file_ref = match.group(1)
                # Try to resolve to actual file
                potential_paths = [
                    Path(__file__).parent.parent.parent
                    / "data"
                    / "examples"
                    / file_ref,
                    Path(__file__).parent.parent.parent / "data" / "models" / file_ref,
                    Path(file_ref),  # Direct path
                ]
                for path in potential_paths:
                    if path.exists():
                        logger.info(f"Model file found: {file_ref} -> {path}")
                        return str(path)

        # 3. Numeric model IDs (future-proofing for production)
        numeric_patterns = [
            r"model[_\s]+(\d+)",
            r"genome[_\s]+(\d+)",
            r"organism[_\s]+(\d+)",
        ]

        for pattern in numeric_patterns:
            match = re.search(pattern, query_lower)
            if match:
                model_id = match.group(1)
                # Future: query database/API for model_id mapping
                # For now, return default with informative logging
                logger.info(
                    f"Numeric model ID detected: {model_id}, using default model (future: query model database)"
                )
                break

        # 4. Default fallback
        default_path = Path(__file__).parent.parent.parent / model_registry["default"]
        if default_path.exists():
            logger.info(f"Using default model: {default_path}")
            return str(default_path)

        return None

    # -------------------------------
    # Required Abstract Methods (for BaseAgent compatibility)
    # -------------------------------

    def _create_prompt(self) -> Optional[str]:
        """Create prompt template - not used in LangGraph but required by BaseAgent"""
        return "LangGraph agent uses dynamic prompts in nodes"

    def _create_agent(self) -> None:
        """Create agent executor - not used in LangGraph but required by BaseAgent"""
        # LangGraph agent doesn't use traditional AgentExecutor
        # The workflow graph serves as the execution engine
        return None

    # -------------------------------
    # Public Interface
    # -------------------------------

    def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """Run the LangGraph workflow with Intelligence Enhancement Framework"""
        query = input_data.get("query", input_data.get("input", ""))

        # Apply Intelligence Enhancement Framework if available
        if self.intelligence_enabled:
            try:
                logger.info(
                    "🧠 Using Intelligence Enhancement Framework for LangGraph workflow"
                )

                # Create intelligent analysis request (following MetabolicAgent pattern)
                intelligence_request = IntelligentAnalysisRequest(
                    request_id=f"langgraph_{self.run_id}_{hash(query) % 10000}",
                    query=query,
                    context={
                        "agent_type": "langgraph",
                        "max_iterations": input_data.get("max_iterations", 5),
                        **input_data,
                    },
                )

                # Execute through intelligent reasoning system (correct API)
                import asyncio

                workflow_result = asyncio.run(
                    self.intelligent_reasoning_system.execute_comprehensive_workflow(
                        intelligence_request
                    )
                )

                # Use Intelligence Framework result if successful
                if workflow_result.success:
                    logger.info(
                        "✅ Intelligence Framework analysis completed, using enhanced workflow"
                    )

                    # Return Intelligence Framework result directly
                    return AgentResult(
                        success=workflow_result.success,
                        message=workflow_result.final_response,
                        data={
                            "workflow_result": workflow_result.model_dump(),
                            "intelligence_enabled": True,
                            "quality_scores": workflow_result.quality_scores,
                            "artifacts_generated": len(
                                workflow_result.artifacts_generated
                            ),
                            "execution_time": workflow_result.total_execution_time,
                        },
                        metadata={
                            "agent_type": "langgraph",
                            "intelligence_framework": "enabled",
                            "overall_confidence": workflow_result.overall_confidence,
                        },
                    )

                debug_config = get_debug_config()
                if debug_config.intelligence_debug:
                    logger.info(
                        f"🧠 Intelligence Framework enhanced query: {query[:100]}..."
                    )

            except Exception as e:
                logger.warning(
                    f"⚠️ Intelligence Framework processing failed, using standard LangGraph workflow: {e}"
                )

        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "final_answer": None,
            "tools_called": [],
            "tool_results": {},
            "parallel_results": {},
            "tools_to_call": None,
            "execution_plan": None,
            "intent_analysis": None,
            "workflow_analysis": None,
            "performance_metrics": None,
            "workflow_visualization": None,
            "iteration": 0,
            "max_iterations": input_data.get("max_iterations", 5),
            "next_action": None,
            "errors": [],
            "retry_count": 0,
            "recovery_strategy": None,
            "continue_analysis": None,
            "previous_results": [],
            "context_summary": "",
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "workflow_state": "thinking",
        }

        try:
            # Execute the workflow
            config = {"configurable": {"thread_id": self.run_id}}
            final_state = self.app.invoke(initial_state, config)

            # Standard LangGraph fallback path (Intelligence Framework handles its own processing)

            # Convert to AgentResult
            return AgentResult(
                success=True,
                message=final_state.get("final_answer", "Analysis completed"),
                data=final_state.get("tool_results", {}),
                intermediate_steps=self.execution_log,
                metadata={
                    "run_id": self.run_id,
                    "workflow_state": final_state.get("workflow_state"),
                    "tools_used": final_state.get("tools_called", []),
                    "iterations": final_state.get("iteration", 0),
                    "errors": final_state.get("errors", []),
                    "execution_summary": self._create_execution_summary(final_state),
                    "intelligence_enabled": False,  # Standard fallback path
                    # Enhanced tool integration metadata
                    "intent_analysis": final_state.get("intent_analysis", {}),
                    "workflow_analysis": final_state.get("workflow_analysis", {}),
                    "performance_metrics": final_state.get("performance_metrics", {}),
                    "tool_execution_summary": final_state.get(
                        "tool_execution_summary", {}
                    ),
                    # Visualization paths
                    "workflow_visualization": final_state.get("workflow_visualization"),
                    "final_workflow_visualization": final_state.get(
                        "final_workflow_visualization"
                    ),
                    "performance_dashboard": final_state.get("performance_dashboard"),
                    # Enhanced execution metrics
                    "total_execution_time": sum(
                        metrics.get("execution_time", 0)
                        for metrics in (
                            final_state.get("performance_metrics") or {}
                        ).values()
                    ),
                    "workflow_complexity": final_state.get("intent_analysis", {}).get(
                        "workflow_complexity", "unknown"
                    ),
                    "parallel_executions": len(final_state.get("parallel_results", {})),
                    "visualization_count": len(
                        [
                            path
                            for path in [
                                final_state.get("workflow_visualization"),
                                final_state.get("final_workflow_visualization"),
                                final_state.get("performance_dashboard"),
                            ]
                            if path
                        ]
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return AgentResult(
                success=False,
                message=f"Workflow execution failed: {str(e)}",
                data={},
                intermediate_steps=self.execution_log,
                error=str(e),
                metadata={"run_id": self.run_id, "workflow_state": "error"},
            )

    def analyze_model(
        self, query: str, model_path: Optional[str] = None
    ) -> AgentResult:
        """Analyze a metabolic model - compatibility method"""
        input_data = {"query": query}
        if model_path:
            input_data["model_path"] = model_path

        return self.run(input_data)

    async def arun(self, input_data: Dict[str, Any]) -> AgentResult:
        """Async version - delegates to sync for now"""
        # LangGraph supports async, but implementing sync first for compatibility
        return self.run(input_data)
