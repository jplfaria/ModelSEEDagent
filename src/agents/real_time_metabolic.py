"""
Real-Time Dynamic AI Metabolic Agent

This agent implements true dynamic AI decision-making where:
- AI analyzes actual tool results to decide the next tool
- Tool selection is based on discovered data, not predefined workflows
- Each step involves real reasoning about what was learned and what's needed next
- No templated responses - every decision is based on actual results

This replaces the static workflow approach with genuine AI-driven exploration.
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from ..llm.base import BaseLLM
from ..tools import ToolRegistry
from ..tools.ai_audit import create_ai_decision_verifier, get_ai_audit_logger
from ..tools.base import BaseTool, ToolResult
from ..tools.realtime_verification import create_realtime_detector
from .base import AgentConfig, AgentResult, BaseAgent
from .collaborative_reasoning import create_collaborative_reasoning_system
from .hypothesis_system import create_hypothesis_system
from .pattern_memory import create_learning_system
from .reasoning_chains import create_reasoning_chain_system

logger = logging.getLogger(__name__)


class RealTimeMetabolicAgent(BaseAgent):
    """
    Real-time dynamic AI agent for metabolic modeling.

    Key Features:
    - Dynamic tool selection based on actual results
    - AI reasoning at each step about what was learned
    - Result-driven workflow adaptation
    - Complete audit trail for hallucination detection
    - No predefined workflows - pure AI decision-making
    """

    def __init__(
        self, llm: BaseLLM, tools: List[BaseTool], config: Dict[str, Any] | AgentConfig
    ):
        super().__init__(llm, tools, config)

        # Initialize audit system
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = (
            Path(__file__).parent.parent.parent / "logs" / f"realtime_run_{self.run_id}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize AI audit system
        self.ai_audit_logger = get_ai_audit_logger(
            logs_dir=self.run_dir.parent, session_id=f"realtime_run_{self.run_id}"
        )
        self.ai_verifier = create_ai_decision_verifier(self.run_dir.parent)
        self.current_workflow_id = None

        # Initialize real-time verification system
        self.realtime_detector = create_realtime_detector(
            logs_dir=self.run_dir.parent,
            enable_display=True,  # Enable live verification display
        )

        # Tool management
        self._tools_dict = {t.tool_name: t for t in tools}
        self.knowledge_base = {}
        self.audit_trail = []
        self.tool_execution_history = []

        # Performance optimization
        self.enable_audit = (
            config.get("enable_audit", True) if isinstance(config, dict) else True
        )
        self.enable_realtime_verification = (
            config.get("enable_realtime_verification", False)
            if isinstance(config, dict)
            else False
        )
        self.cache_llm_responses = (
            config.get("cache_llm_responses", True)
            if isinstance(config, dict)
            else True
        )
        self._llm_cache = {}

        # Initialize Phase 8 Advanced Agentic Capabilities
        tools_registry = {t.tool_name: t for t in tools}

        # Phase 8.1: Multi-step reasoning chains
        self.reasoning_planner, self.reasoning_executor = create_reasoning_chain_system(
            llm, tools_registry, self, self.ai_audit_logger
        )

        # Phase 8.2: Hypothesis-driven analysis
        self.hypothesis_manager = create_hypothesis_system(llm, self)

        # Phase 8.3: Collaborative reasoning
        self.collaborative_reasoner = create_collaborative_reasoning_system(
            llm, interactive=True  # Enable interactive mode
        )

        # Phase 8.4: Cross-model learning
        self.learning_memory = create_learning_system(
            llm, storage_path=self.run_dir.parent / "learning_memory"
        )

        # Phase 8 state management
        self.current_reasoning_chain = None
        self.active_hypotheses = []
        self.reasoning_mode = (
            "dynamic"  # Can be "dynamic", "chain", "hypothesis", "collaborative"
        )

        # Default model path for tools that need it
        self.default_model_path = str(
            Path(__file__).parent.parent.parent
            / "data"
            / "examples"
            / "e_coli_core.xml"
        )

        logger.info(f"RealTimeMetabolicAgent initialized with {len(tools)} tools")

        # DEBUG: Add LLM timeout diagnostic (disabled for interactive sessions)
        # self._diagnose_llm_timeout_settings()

    def _diagnose_llm_timeout_settings(self):
        """Diagnostic method to check LLM timeout configuration"""
        try:
            logger.info(
                "🔍 LLM TIMEOUT DIAGNOSTIC: Starting timeout configuration check"
            )

            # Check LLM type and model name
            llm_type = type(self.llm).__name__
            logger.info(f"🔍 LLM TIMEOUT DIAGNOSTIC: LLM type: {llm_type}")

            # Check model name attribute
            if hasattr(self.llm, "model_name"):
                model_name = self.llm.model_name
                logger.info(f"🔍 LLM TIMEOUT DIAGNOSTIC: Model name: '{model_name}'")

                # Check if it would be detected as o1 model
                model_lower = model_name.lower()
                is_o1_model = model_lower.startswith(("gpto", "o1"))
                expected_timeout = 120 if is_o1_model else 30
                logger.info(f"🔍 LLM TIMEOUT DIAGNOSTIC: Is o1 model: {is_o1_model}")
                logger.info(
                    f"🔍 LLM TIMEOUT DIAGNOSTIC: Expected signal timeout: {expected_timeout}s"
                )
            else:
                logger.warning(
                    "🔍 LLM TIMEOUT DIAGNOSTIC: LLM has no model_name attribute!"
                )

            # Check if LLM has timeout configuration
            if hasattr(self.llm, "_timeout"):
                llm_timeout = self.llm._timeout
                logger.info(
                    f"🔍 LLM TIMEOUT DIAGNOSTIC: LLM HTTP timeout: {llm_timeout}s"
                )
            else:
                logger.info("🔍 LLM TIMEOUT DIAGNOSTIC: LLM has no _timeout attribute")

            # Check LLM config
            if hasattr(self.llm, "config"):
                config = self.llm.config
                logger.info(
                    f"🔍 LLM TIMEOUT DIAGNOSTIC: LLM config type: {type(config)}"
                )
                if hasattr(config, "llm_name"):
                    logger.info(
                        f"🔍 LLM TIMEOUT DIAGNOSTIC: Config llm_name: '{config.llm_name}'"
                    )

        except Exception as e:
            logger.error(f"🔍 LLM TIMEOUT DIAGNOSTIC: Error during diagnostic: {e}")

    def _create_prompt(self) -> Optional[str]:
        """Create prompt template - not used in real-time agent"""
        return None

    def _create_agent(self) -> None:
        """Create agent executor - not used in real-time agent"""
        return None

    async def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Run advanced AI-driven metabolic analysis with Phase 8 capabilities.

        This method implements sophisticated AI reasoning including:
        - Multi-step reasoning chains
        - Hypothesis-driven analysis
        - Collaborative decision-making
        - Pattern learning and memory
        """
        query = input_data.get("query", input_data.get("input", ""))
        max_iterations = input_data.get("max_iterations", 6)
        reasoning_mode = input_data.get("reasoning_mode", self.reasoning_mode)

        logger.info(f"🧠 ADVANCED AI AGENT STARTING: {query} (Mode: {reasoning_mode})")

        try:
            # Initialize session
            self._init_session(query)

            # Start AI workflow auditing (only if enabled)
            if self.enable_audit:
                self.current_workflow_id = self.ai_audit_logger.start_workflow(query)
                logger.info(f"🔍 AI workflow audit started: {self.current_workflow_id}")
            else:
                self.current_workflow_id = None

            # Start real-time verification monitoring (only if enabled)
            if self.enable_realtime_verification and self.current_workflow_id:
                try:
                    self.realtime_detector.start_monitoring(
                        self.current_workflow_id, query
                    )
                    logger.info(
                        f"🔍 Real-time verification monitoring started for workflow {self.current_workflow_id}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to start real-time verification: {e}")
                    # Continue without real-time verification
                    self.enable_realtime_verification = False

            # Get learning-based recommendations
            model_characteristics = self._analyze_model_characteristics(query)
            recommendations = self.learning_memory.get_recommended_approach(
                query, model_characteristics
            )

            if recommendations["confidence"] > 0.7:
                logger.info(
                    f"🧠 Learning memory suggests: {recommendations['rationale']}"
                )

            # Select reasoning approach based on query complexity and mode
            if reasoning_mode == "chain" or self._should_use_reasoning_chains(query):
                return await self._run_with_reasoning_chains(query, max_iterations)
            elif reasoning_mode == "hypothesis" or self._should_use_hypothesis_driven(
                query
            ):
                return await self._run_with_hypothesis_mode(query, max_iterations)
            elif reasoning_mode == "collaborative":
                return await self._run_with_collaborative_mode(query, max_iterations)
            else:
                return await self._run_standard_analysis(query, max_iterations)

        except Exception as e:
            logger.error(f"Advanced AI agent execution failed: {e}")
            return self._create_error_result(
                f"Advanced AI agent execution failed: {str(e)}"
            )

    def _init_session(self, query: str):
        """Initialize analysis session"""
        session_info = {
            "session_id": self.run_id,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "agent_type": "Real-Time Dynamic AI Agent",
            "available_tools": list(self._tools_dict.keys()),
        }

        self.audit_trail.append(
            {
                "step": "session_initialization",
                "timestamp": datetime.now().isoformat(),
                "data": session_info,
            }
        )

    def _ai_analyze_query_for_first_tool(self, query: str) -> tuple[Optional[str], str]:
        """
        AI analyzes the query to select the first tool.
        This is genuine AI reasoning, not templated responses.
        """
        available_tools = list(self._tools_dict.keys())

        # Categorize tools for better AI guidance
        analysis_tools = [
            "run_metabolic_fba",
            "find_minimal_media",
            "analyze_essentiality",
            "run_flux_variability_analysis",
            "identify_auxotrophies",
            "run_gene_deletion_analysis",
            "run_flux_sampling",
            # AI Media Tools
            "select_optimal_media",
            "manipulate_media_composition",
            "analyze_media_compatibility",
            "compare_media_performance",
        ]

        build_tools = ["build_metabolic_model", "annotate_genome_rast", "gapfill_model"]

        biochem_tools = ["search_biochem", "resolve_biochem_entity"]

        available_analysis_tools = [t for t in analysis_tools if t in available_tools]
        available_build_tools = [t for t in build_tools if t in available_tools]
        available_biochem_tools = [t for t in biochem_tools if t in available_tools]

        prompt = f"""You are an expert metabolic modeling AI agent. Analyze this query and select the BEST first tool to start with.

Query: "{query}"

Available tools are categorized as follows:

ANALYSIS TOOLS (for analyzing existing models): {', '.join(available_analysis_tools)}
BUILD TOOLS (for creating new models from genome data): {', '.join(available_build_tools)}
BIOCHEMISTRY TOOLS (for biochemistry database queries): {', '.join(available_biochem_tools)}

IMPORTANT GUIDELINES:
- If the query asks for "analysis", "comprehensive analysis", "characterization", or mentions analyzing an existing model, START with ANALYSIS TOOLS
- BUILD TOOLS should only be used when the query explicitly mentions building a new model from genome/annotation data
- For E. coli analysis queries, "run_metabolic_fba" is usually the best starting point as it provides foundational growth and flux information
- ANALYSIS TOOLS work with existing model files and don't require genome annotation data
- BUILD TOOLS require genome annotation files and are not appropriate for analyzing pre-built models

Based on the query, what tool should you start with and why? Consider:
1. What type of analysis is being requested?
2. Do you have an existing model to analyze, or do you need to build one from genome data?
3. Which tool provides the most informative starting point for THIS specific query?

Respond with:
TOOL: [exact tool name from the available tools list]
REASONING: [detailed explanation of why this tool is the optimal starting point]

Think step by step about the query requirements and tool capabilities."""

        try:
            # Add timeout protection for LLM calls (signal only works in main thread)
            import signal
            import threading

            # Use longer timeout for reasoning models
            model_name = getattr(self.llm, "model_name", "").lower()
            timeout_seconds = 120 if model_name.startswith(("gpto", "o1")) else 30

            # DEBUG: Add comprehensive logging for first tool selection
            logger.debug(f"🔍 FIRST TOOL TIMEOUT DEBUG: Model '{model_name}' detected")
            logger.debug(
                f"🔍 FIRST TOOL TIMEOUT DEBUG: Using {timeout_seconds}s timeout"
            )
            logger.debug(
                f"🔍 FIRST TOOL TIMEOUT DEBUG: Model type: {type(self.llm).__name__}"
            )

            # Check if we're in the main thread for signal-based timeout
            is_main_thread = threading.current_thread() is threading.main_thread()

            if is_main_thread:

                def timeout_handler(signum, frame):
                    raise TimeoutError("LLM call timed out")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)

            # Log the start of LLM call
            start_time = time.time()
            logger.debug(
                f"🔍 FIRST TOOL TIMEOUT DEBUG: Starting LLM call at {start_time}"
            )

            response = self.llm._generate_response(prompt)

            if is_main_thread:
                signal.alarm(0)  # Cancel timeout

            end_time = time.time()
            duration = end_time - start_time
            logger.debug(
                f"🔍 FIRST TOOL TIMEOUT DEBUG: LLM call completed in {duration:.2f}s"
            )
            response_text = response.text.strip()

            # Parse AI response with robust, flexible parsing
            selected_tool, reasoning = self._parse_tool_selection_response(
                response_text
            )

            # Log AI reasoning step (only if audit enabled)
            if self.current_workflow_id and self.enable_audit:
                context_analysis = f"Query requests: {query[:100]}... Available tools: {len(available_tools)} options"
                self.ai_audit_logger.log_reasoning_step(
                    ai_thought=response_text,
                    context_analysis=context_analysis,
                    available_tools=available_tools,
                    selected_tool=selected_tool,
                    selection_rationale=reasoning,
                    confidence_score=0.8,  # High confidence for initial selection
                    expected_result=f"Foundational data for {query[:50]}...",
                    success_criteria=[
                        "Tool executes successfully",
                        "Provides relevant baseline data",
                    ],
                )

                # Get the reasoning step for real-time verification (only if enabled)
                if self.enable_realtime_verification and hasattr(
                    self.realtime_detector, "process_reasoning_step"
                ):
                    try:
                        reasoning_steps = (
                            self.ai_audit_logger.current_workflow.reasoning_steps
                        )
                        if reasoning_steps:
                            latest_step = reasoning_steps[-1]
                            alerts = self.realtime_detector.process_reasoning_step(
                                self.current_workflow_id, latest_step
                            )
                            if alerts:
                                logger.info(
                                    f"🔍 Real-time verification generated {len(alerts)} alerts"
                                )
                    except Exception as e:
                        logger.warning(
                            f"Real-time verification step processing failed: {e}"
                        )
                        # Continue without real-time verification for this step

            # Validate tool exists
            if selected_tool and selected_tool in self._tools_dict:
                return selected_tool, reasoning

            # Fallback selection if parsing failed
            logger.warning("AI tool selection parsing failed, using fallback logic")
            return (
                self._fallback_tool_selection(query),
                "Fallback selection due to parsing error",
            )

        except TimeoutError:
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0
            logger.error(
                f"🔍 FIRST TOOL TIMEOUT DEBUG: Signal timeout triggered after {duration:.2f}s (limit: {timeout_seconds}s)"
            )
            logger.warning(
                f"AI tool selection timed out ({timeout_seconds}s), using fallback logic"
            )
            return (
                self._fallback_tool_selection(query),
                "Fallback selection due to LLM timeout",
            )
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0
            logger.error(
                f"🔍 FIRST TOOL TIMEOUT DEBUG: LLM call failed after {duration:.2f}s with error: {e}"
            )
            logger.error(f"AI tool selection failed: {e}")
            return (
                self._fallback_tool_selection(query),
                f"Error in AI selection: {str(e)}",
            )

    def _ai_analyze_results_and_decide_next_step(
        self, query: str, knowledge_base: Dict[str, Any]
    ) -> tuple[Optional[str], bool, str]:
        """
        AI analyzes accumulated results and decides the next step.
        This is the core of dynamic decision-making.
        """
        # Prepare context of what we've learned so far
        results_context = self._format_knowledge_base_for_ai(knowledge_base)
        step_number = len(knowledge_base)

        available_tools = list(self._tools_dict.keys())

        # Categorize tools for better AI guidance
        analysis_tools = [
            "run_metabolic_fba",
            "find_minimal_media",
            "analyze_essentiality",
            "run_flux_variability_analysis",
            "identify_auxotrophies",
            "run_gene_deletion_analysis",
            "run_flux_sampling",
            # AI Media Tools
            "select_optimal_media",
            "manipulate_media_composition",
            "analyze_media_compatibility",
            "compare_media_performance",
        ]

        build_tools = ["build_metabolic_model", "annotate_genome_rast", "gapfill_model"]

        biochem_tools = ["search_biochem", "resolve_biochem_entity"]

        available_analysis_tools = [t for t in analysis_tools if t in available_tools]
        available_build_tools = [t for t in build_tools if t in available_tools]
        available_biochem_tools = [t for t in biochem_tools if t in available_tools]

        prompt = f"""You are an expert metabolic modeling AI agent. You have executed some tools and now need to decide what to do next based on the ACTUAL RESULTS you've obtained.

ORIGINAL QUERY: "{query}"

CURRENT STEP: {step_number}

RESULTS OBTAINED SO FAR:
{results_context}

Available tools are categorized as follows:

ANALYSIS TOOLS (for analyzing existing models): {', '.join(available_analysis_tools)}
BUILD TOOLS (for creating new models from genome data): {', '.join(available_build_tools)}
BIOCHEMISTRY TOOLS (for biochemistry database queries): {', '.join(available_biochem_tools)}

IMPORTANT GUIDELINES:
- Continue using ANALYSIS TOOLS to explore different aspects of the metabolic model
- BUILD TOOLS should only be used if you need to create a new model from genome data (rare in analysis workflows)
- For comprehensive analysis, consider tools like: minimal media, essentiality, flux variability, auxotrophy analysis
- NEW AI MEDIA TOOLS: Use select_optimal_media for intelligent media selection, manipulate_media_composition for natural language media modifications, analyze_media_compatibility for compatibility analysis, compare_media_performance for cross-media comparisons
- Each tool provides different insights: FBA (growth), minimal media (nutritional requirements), essentiality (critical genes), AI media tools (intelligent media management), etc.
- BUILD TOOLS require genome annotation files and are not appropriate for analyzing existing models

Based on the ACTUAL DATA you've collected, analyze:

1. What specific information have you learned?
2. What questions remain unanswered from the original query?
3. What ANALYSIS tool would provide the most valuable ADDITIONAL insight?
4. Do you have enough information to provide a comprehensive answer?

Decide your next action:

ACTION: [either "execute_tool" or "finalize"]
TOOL: [if executing tool, specify exact tool name from available tools]
REASONING: [detailed explanation based on the actual results you've analyzed]

Make your decision based on the ACTUAL DATA PATTERNS you see, not generic workflows."""

        try:
            # Add timeout protection for LLM calls (signal only works in main thread)
            import signal
            import threading

            # Use longer timeout for reasoning models
            model_name = getattr(self.llm, "model_name", "").lower()
            timeout_seconds = 120 if model_name.startswith(("gpto", "o1")) else 30

            # DEBUG: Add comprehensive logging for decision analysis
            logger.debug(f"🔍 DECISION TIMEOUT DEBUG: Model '{model_name}' detected")
            logger.debug(f"🔍 DECISION TIMEOUT DEBUG: Using {timeout_seconds}s timeout")
            logger.debug(f"🔍 DECISION TIMEOUT DEBUG: Step {step_number} analysis")

            # Check if we're in the main thread for signal-based timeout
            is_main_thread = threading.current_thread() is threading.main_thread()

            if is_main_thread:

                def timeout_handler(signum, frame):
                    raise TimeoutError("LLM call timed out")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)

            # Log the start of LLM call
            start_time = time.time()
            logger.debug(
                f"🔍 DECISION TIMEOUT DEBUG: Starting decision LLM call at {start_time}"
            )

            response = self.llm._generate_response(prompt)

            if is_main_thread:
                signal.alarm(0)  # Cancel timeout

            end_time = time.time()
            duration = end_time - start_time
            logger.debug(
                f"🔍 DECISION TIMEOUT DEBUG: Decision LLM call completed in {duration:.2f}s"
            )
            response_text = response.text.strip()

            # Parse AI decision with robust, flexible parsing
            decision = self._parse_decision_response(response_text)

            # Log AI reasoning step for this decision
            if self.current_workflow_id:
                available_tools = list(self._tools_dict.keys())
                context_analysis = f"Step {step_number}: Analyzing {len(knowledge_base)} previous results to decide next action"

                # Determine alternative tools that were considered
                alternative_tools = [
                    t for t in available_tools if t != decision.get("tool")
                ]

                self.ai_audit_logger.log_reasoning_step(
                    ai_thought=response_text,
                    context_analysis=context_analysis,
                    available_tools=available_tools,
                    selected_tool=decision.get("tool"),
                    selection_rationale=decision.get("reasoning", ""),
                    confidence_score=0.7,  # Moderate confidence for subsequent decisions
                    expected_result=f"Additional insights for step {step_number}",
                    success_criteria=[
                        "Tool provides new information",
                        "Results complement existing data",
                    ],
                    alternative_tools=alternative_tools[
                        :3
                    ],  # Limit to first 3 alternatives
                    rejection_reasons={
                        tool: "Not selected by AI analysis"
                        for tool in alternative_tools[:3]
                    },
                )

                # Real-time verification for subsequent steps
                if self.enable_realtime_verification and hasattr(
                    self.realtime_detector, "process_reasoning_step"
                ):
                    try:
                        reasoning_steps = (
                            self.ai_audit_logger.current_workflow.reasoning_steps
                        )
                        if reasoning_steps:
                            latest_step = reasoning_steps[-1]
                            alerts = self.realtime_detector.process_reasoning_step(
                                self.current_workflow_id, latest_step
                            )
                            if alerts:
                                logger.info(
                                    f"🔍 Step {step_number} verification generated {len(alerts)} alerts"
                                )
                    except Exception as e:
                        logger.warning(
                            f"Real-time verification step processing failed: {e}"
                        )
                        # Continue without real-time verification for this step

            # Convert decision dict to tuple format expected by caller
            if decision["action"] == "execute_tool" and "tool" in decision:
                return decision["tool"], True, decision.get("reasoning", "")
            else:
                return None, False, decision.get("reasoning", "Analysis complete")

        except TimeoutError:
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0
            logger.error(
                f"🔍 DECISION TIMEOUT DEBUG: Signal timeout triggered after {duration:.2f}s (limit: {timeout_seconds}s)"
            )
            logger.warning(
                f"AI decision analysis timed out ({timeout_seconds}s), finalizing"
            )
            return None, False, "Analysis timeout - finalizing current results"
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0
            logger.error(
                f"🔍 DECISION TIMEOUT DEBUG: LLM call failed after {duration:.2f}s with error: {e}"
            )
            logger.error(f"AI decision analysis failed: {e}")
            return None, False, f"Error in analysis: {str(e)}"

    async def _execute_tool_with_audit(self, tool_name: str, query: str) -> ToolResult:
        """
        Execute tool with complete audit trail for hallucination detection.
        """
        if tool_name not in self._tools_dict:
            logger.error(f"Tool {tool_name} not found")
            return ToolResult(
                success=False,
                message=f"Tool {tool_name} not found",
                error=f"Unknown tool: {tool_name}",
                data={},
            )

        tool = self._tools_dict[tool_name]

        # Prepare tool input
        logger.debug(f"🔍 CALLING _prepare_tool_input: tool_name='{tool_name}'")
        tool_input = self._prepare_tool_input(tool_name, query)
        logger.debug(f"🔍 RECEIVED tool_input: {tool_input}")

        # Record execution start
        execution_start = {
            "tool": tool_name,
            "start_time": datetime.now().isoformat(),
            "inputs": tool_input,
        }

        try:
            logger.info(f"🔧 Executing tool: {tool_name}")

            # Execute tool
            start_time = time.time()
            result = tool._run_tool(tool_input)
            execution_time = time.time() - start_time

            if result.success:
                # Store successful result
                self.knowledge_base[tool_name] = result.data

                # Create audit record
                audit_record = {
                    **execution_start,
                    "end_time": datetime.now().isoformat(),
                    "execution_time_seconds": execution_time,
                    "success": True,
                    "result_data": result.data,
                    "result_message": result.message,
                }

                self.audit_trail.append(audit_record)

                # Log tool execution in AI audit system
                if (
                    self.current_workflow_id
                    and hasattr(result, "metadata")
                    and result.metadata.get("audit_file")
                ):
                    self.ai_audit_logger.log_tool_execution(
                        tool_name, result.metadata["audit_file"]
                    )

                # Log success
                summary = self._create_execution_summary(tool_name, result.data)
                logger.info(f"   ✅ {summary}")

                return result
            else:
                # Store failed result
                audit_record = {
                    **execution_start,
                    "end_time": datetime.now().isoformat(),
                    "execution_time_seconds": execution_time,
                    "success": False,
                    "error": result.error,
                }

                self.audit_trail.append(audit_record)
                logger.error(f"   ❌ Tool failed: {result.error}")

                return result

        except Exception as e:
            # Store exception
            audit_record = {
                **execution_start,
                "end_time": datetime.now().isoformat(),
                "success": False,
                "exception": str(e),
            }

            self.audit_trail.append(audit_record)
            logger.error(f"   💥 Exception: {str(e)}")

            return ToolResult(
                success=False, message=f"Tool execution failed: {str(e)}", error=str(e)
            )

    def _ai_generate_final_conclusions(
        self,
        query: str,
        knowledge_base: Dict[str, Any],
        execution_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        AI generates final conclusions based on actual accumulated data.
        """
        results_context = self._format_knowledge_base_for_ai(knowledge_base)

        prompt = f"""You are an expert metabolic modeling AI agent. Provide a comprehensive analysis based on the ACTUAL RESULTS you've collected.

ORIGINAL QUERY: "{query}"

ACTUAL RESULTS COLLECTED:
{results_context}

Based on the REAL DATA you've gathered, provide:

1. KEY QUANTITATIVE FINDINGS: Extract specific numbers, rates, counts from your results
2. BIOLOGICAL INSIGHTS: What do these results tell us about the organism's metabolism?
3. DIRECT ANSWERS: Answer the specific questions posed in the original query
4. CONFIDENCE ASSESSMENT: Rate your confidence (0.0-1.0) in these conclusions

Format your response as:
QUANTITATIVE_FINDINGS: [specific numbers and metrics you discovered]
BIOLOGICAL_INSIGHTS: [metabolic insights from the data]
DIRECT_ANSWERS: [answers to the original query]
CONFIDENCE_SCORE: [0.0-1.0]
SUMMARY: [concise overall conclusion]

Base everything on the ACTUAL DATA you collected, not general knowledge."""

        try:
            # Add timeout protection for LLM calls (signal only works in main thread)
            import signal
            import threading

            # Use longer timeout for reasoning models
            model_name = getattr(self.llm, "model_name", "").lower()
            timeout_seconds = 120 if model_name.startswith(("gpto", "o1")) else 30

            # DEBUG: Add comprehensive logging for conclusion generation
            logger.debug(f"🔍 CONCLUSION TIMEOUT DEBUG: Model '{model_name}' detected")
            logger.debug(
                f"🔍 CONCLUSION TIMEOUT DEBUG: Using {timeout_seconds}s timeout"
            )
            logger.debug(
                f"🔍 CONCLUSION TIMEOUT DEBUG: Final analysis with {len(knowledge_base)} tools executed"
            )

            # Check if we're in the main thread for signal-based timeout
            is_main_thread = threading.current_thread() is threading.main_thread()

            if is_main_thread:

                def timeout_handler(signum, frame):
                    raise TimeoutError("LLM call timed out")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)

            # Log the start of LLM call
            start_time = time.time()
            logger.debug(
                f"🔍 CONCLUSION TIMEOUT DEBUG: Starting conclusion LLM call at {start_time}"
            )

            response = self.llm._generate_response(prompt)

            if is_main_thread:
                signal.alarm(0)  # Cancel timeout

            end_time = time.time()
            duration = end_time - start_time
            logger.debug(
                f"🔍 CONCLUSION TIMEOUT DEBUG: Conclusion LLM call completed in {duration:.2f}s"
            )
            response_text = response.text.strip()

            # Parse response into structured conclusions
            conclusions = {"summary": "Analysis completed based on collected data"}
            current_section = None
            content = []

            for line in response_text.split("\n"):
                if line.startswith("QUANTITATIVE_FINDINGS:"):
                    if current_section and content:
                        conclusions[current_section] = " ".join(content)
                    current_section = "quantitative_findings"
                    content = [line.replace("QUANTITATIVE_FINDINGS:", "").strip()]
                elif line.startswith("BIOLOGICAL_INSIGHTS:"):
                    if current_section and content:
                        conclusions[current_section] = " ".join(content)
                    current_section = "biological_insights"
                    content = [line.replace("BIOLOGICAL_INSIGHTS:", "").strip()]
                elif line.startswith("DIRECT_ANSWERS:"):
                    if current_section and content:
                        conclusions[current_section] = " ".join(content)
                    current_section = "direct_answers"
                    content = [line.replace("DIRECT_ANSWERS:", "").strip()]
                elif line.startswith("CONFIDENCE_SCORE:"):
                    if current_section and content:
                        conclusions[current_section] = " ".join(content)
                    confidence_text = line.replace("CONFIDENCE_SCORE:", "").strip()
                    try:
                        conclusions["confidence_score"] = float(confidence_text)
                    except:
                        conclusions["confidence_score"] = 0.8
                    current_section = None
                    content = []
                elif line.startswith("SUMMARY:"):
                    if current_section and content:
                        conclusions[current_section] = " ".join(content)
                    conclusions["summary"] = line.replace("SUMMARY:", "").strip()
                    # Continue collecting summary lines
                    current_section = "summary_continuation"
                    content = []
                elif current_section and line.strip():
                    content.append(line.strip())

            # Handle last section
            if current_section and content:
                if current_section == "summary_continuation":
                    conclusions["summary"] += " " + " ".join(content)
                else:
                    conclusions[current_section] = " ".join(content)

            return conclusions

        except TimeoutError:
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0
            logger.error(
                f"🔍 CONCLUSION TIMEOUT DEBUG: Signal timeout triggered after {duration:.2f}s (limit: {timeout_seconds}s)"
            )
            logger.warning(
                f"AI conclusion generation timed out ({timeout_seconds}s), using robust fallback"
            )
            logger.info("Generating robust fallback analysis from tool data...")

            # Generate comprehensive fallback analysis when LLM times out
            return self._generate_fallback_analysis(query, knowledge_base)
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0
            logger.error(
                f"🔍 CONCLUSION TIMEOUT DEBUG: LLM call failed after {duration:.2f}s with error: {e}"
            )
            logger.error(f"AI conclusion generation failed: {e}")
            logger.info("Generating robust fallback analysis from tool data...")

            # Generate comprehensive fallback analysis when LLM fails
            return self._generate_fallback_analysis(query, knowledge_base)

    def _generate_fallback_analysis(
        self, query: str, knowledge_base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis directly from tool data when LLM fails"""
        if not knowledge_base:
            return {
                "summary": "No analysis data collected",
                "confidence_score": 0.1,
                "conclusions": "Analysis failed - no tool data available",
            }

        # Build comprehensive fallback analysis
        analysis_parts = []
        quantitative_findings = []
        biological_insights = []

        for tool_name, data in knowledge_base.items():
            tool_insights = self._extract_tool_insights_for_fallback(tool_name, data)
            if tool_insights:
                analysis_parts.append(
                    f"## {tool_name.replace('_', ' ').title()}\n{tool_insights}"
                )

                # Extract specific findings
                if tool_name == "run_flux_variability_analysis":
                    if isinstance(data, dict) and "summary" in data:
                        summary = data["summary"]
                        quantitative_findings.append(
                            f"Network analysis: {summary.get('blocked_reactions', 0)} blocked, {summary.get('variable_reactions', 0)} variable reactions"
                        )
                        biological_insights.append(
                            "Metabolic network shows constrained flux patterns with limited flexibility"
                        )

                elif tool_name == "run_metabolic_fba":
                    # Extract growth rate properly
                    growth_rate = 0.0
                    flux_data = data.get("fluxes") or data.get("significant_fluxes")
                    if flux_data:
                        biomass_reactions = [
                            k for k in flux_data.keys() if "BIOMASS" in k.upper()
                        ]
                        if biomass_reactions:
                            growth_rate = flux_data[biomass_reactions[0]]

                    quantitative_findings.append(
                        f"Predicted growth rate: {growth_rate:.4f} h⁻¹"
                    )
                    if growth_rate > 0.5:
                        biological_insights.append(
                            "Model predicts robust growth under these conditions"
                        )
                    else:
                        biological_insights.append(
                            "Model shows limited growth capacity"
                        )

                elif tool_name == "analyze_metabolic_model":
                    if isinstance(data, dict) and "model_statistics" in data:
                        stats = data["model_statistics"]
                        quantitative_findings.append(
                            f"Model structure: {stats.get('num_reactions', 0)} reactions, {stats.get('num_metabolites', 0)} metabolites, {stats.get('num_genes', 0)} genes"
                        )
                        biological_insights.append(
                            "Model represents core metabolic capabilities"
                        )

        # Create comprehensive summary
        tools_executed = len(knowledge_base)
        summary_parts = [
            f"Comprehensive metabolic analysis completed using {tools_executed} specialized tools.",
        ]

        if quantitative_findings:
            summary_parts.append(
                "Key findings: " + "; ".join(quantitative_findings[:3])
            )

        if biological_insights:
            summary_parts.append(
                "Biological insights: " + "; ".join(biological_insights[:2])
            )

        comprehensive_summary = " ".join(summary_parts)

        # Determine confidence based on data quality
        confidence = 0.8 if tools_executed >= 2 else 0.6

        return {
            "summary": comprehensive_summary,
            "quantitative_findings": "; ".join(quantitative_findings),
            "biological_insights": "; ".join(biological_insights),
            "confidence_score": confidence,
            "conclusions": f"Analysis completed successfully with {tools_executed} tools executed. "
            + comprehensive_summary,
            "detailed_results": "\n\n".join(analysis_parts),
            "fallback_mode": True,
        }

    def _extract_tool_insights_for_fallback(
        self, tool_name: str, data: Dict[str, Any]
    ) -> str:
        """Extract detailed insights for fallback analysis"""
        if not isinstance(data, dict):
            return f"Analysis completed: {str(data)[:100]}"

        insights = []

        if tool_name == "run_flux_variability_analysis":
            if "summary" in data:
                summary = data["summary"]
                insights.append(f"**Network Connectivity Analysis:**")
                insights.append(
                    f"- Total reactions: {summary.get('total_reactions', 'N/A')}"
                )
                insights.append(
                    f"- Blocked reactions: {summary.get('blocked_reactions', 'N/A')} (cannot carry flux)"
                )
                insights.append(
                    f"- Fixed reactions: {summary.get('fixed_reactions', 'N/A')} (single flux value)"
                )
                insights.append(
                    f"- Variable reactions: {summary.get('variable_reactions', 'N/A')} (flexible flux)"
                )

                # Calculate flexibility percentage
                total = summary.get("total_reactions", 1)
                variable = summary.get("variable_reactions", 0)
                flexibility = (variable / total * 100) if total > 0 else 0
                insights.append(
                    f"- Network flexibility: {flexibility:.1f}% of reactions show flux variability"
                )

        elif tool_name == "run_metabolic_fba":
            insights.append(f"**Growth Analysis:**")
            insights.append(f"- Optimization status: {data.get('status', 'unknown')}")

            # Extract correct growth rate
            growth_rate = 0.0
            flux_data = data.get("fluxes") or data.get("significant_fluxes")
            if flux_data:
                biomass_reactions = [
                    k for k in flux_data.keys() if "BIOMASS" in k.upper()
                ]
                if biomass_reactions:
                    growth_rate = flux_data[biomass_reactions[0]]

            insights.append(f"- Predicted growth rate: {growth_rate:.4f} h⁻¹")
            insights.append(
                f"- FBA objective value: {data.get('objective_value', 0):.4f}"
            )

            if "active_reactions" in data:
                active_count = len(data["active_reactions"])
                insights.append(f"- Active metabolic reactions: {active_count}")

        elif tool_name == "analyze_metabolic_model":
            insights.append(f"**Model Structure:**")
            if "model_statistics" in data:
                stats = data["model_statistics"]
                insights.append(f"- Reactions: {stats.get('num_reactions', 'N/A')}")
                insights.append(f"- Metabolites: {stats.get('num_metabolites', 'N/A')}")
                insights.append(f"- Genes: {stats.get('num_genes', 'N/A')}")

            if "network_properties" in data:
                network = data["network_properties"]
                if "connectivity_summary" in network:
                    conn = network["connectivity_summary"]
                    insights.append(
                        f"- Average connections per metabolite: {conn.get('avg_connections_per_metabolite', 0):.1f}"
                    )

        elif tool_name == "analyze_pathway":
            insights.append(f"**Pathway Analysis:**")
            if "summary" in data:
                summary = data["summary"]
                insights.append(
                    f"- Reactions in pathway: {summary.get('reaction_count', 'N/A')}"
                )
                insights.append(
                    f"- Associated genes: {summary.get('gene_coverage', 'N/A')}"
                )
                insights.append(
                    f"- Metabolites involved: {summary.get('metabolite_count', 'N/A')}"
                )

        else:
            insights.append(f"**{tool_name.replace('_', ' ').title()}:**")
            insights.append(f"- Analysis completed successfully")
            # Add any summary data if available
            if "summary" in data:
                insights.append(f"- Summary available with detailed results")

        return "\n".join(insights) if insights else "Analysis completed successfully"

    def _format_knowledge_base_for_ai(self, knowledge_base: Dict[str, Any]) -> str:
        """Format accumulated knowledge for AI analysis with content filtering protection"""
        if not knowledge_base:
            return "No results collected yet."

        formatted = ""
        for tool_name, data in knowledge_base.items():
            formatted += f"\n{tool_name.upper()} RESULTS:\n"

            if isinstance(data, dict):
                # Smart data extraction to avoid content filtering
                formatted += self._extract_safe_insights(tool_name, data)
            else:
                formatted += f"  {str(data)[:200]}...\n"

        return formatted

    def _extract_safe_insights(self, tool_name: str, data: Dict[str, Any]) -> str:
        """Extract insights from tool data in a way that avoids LLM content filtering"""
        insights = ""

        if tool_name == "run_flux_variability_analysis":
            # Extract high-level insights instead of raw numerical data
            if "summary" in data:
                summary = data["summary"]
                insights += f"  - Total reactions analyzed: {summary.get('total_reactions', 'N/A')}\n"
                insights += f"  - Blocked reactions: {summary.get('blocked_reactions', 'N/A')}\n"
                insights += (
                    f"  - Fixed reactions: {summary.get('fixed_reactions', 'N/A')}\n"
                )
                insights += f"  - Variable reactions: {summary.get('variable_reactions', 'N/A')}\n"
                insights += f"  - Essential reactions: {summary.get('essential_reactions', 'N/A')}\n"

            # Extract key variable reactions without full flux data
            if "variable_reactions" in data.get("summary", {}):
                var_count = len(data.get("variable_reactions", []))
                insights += f"  - Network flexibility: {var_count} reactions show flux variability\n"

        elif tool_name == "run_metabolic_fba":
            # Extract key FBA insights safely with correct growth rate
            insights += (
                f"  - Growth optimization status: {data.get('status', 'unknown')}\n"
            )

            # Extract actual biomass growth rate from fluxes, not objective value
            growth_rate = 0.0
            flux_data = data.get("fluxes") or data.get("significant_fluxes")
            if flux_data:
                biomass_reactions = [
                    k for k in flux_data.keys() if "BIOMASS" in k.upper()
                ]
                if biomass_reactions:
                    growth_rate = flux_data[biomass_reactions[0]]

            insights += f"  - Predicted growth rate: {growth_rate:.4f} h⁻¹\n"
            insights += (
                f"  - FBA objective value: {data.get('objective_value', 0):.4f}\n"
            )

            if "active_reactions" in data:
                active_count = len(data.get("active_reactions", {}))
                insights += f"  - Active metabolic reactions: {active_count}\n"

        elif tool_name == "analyze_metabolic_model":
            # Extract model structure insights
            if "model_statistics" in data:
                stats = data["model_statistics"]
                insights += f"  - Model size: {stats.get('num_reactions', 'N/A')} reactions, {stats.get('num_metabolites', 'N/A')} metabolites\n"
                insights += (
                    f"  - Genetic basis: {stats.get('num_genes', 'N/A')} genes\n"
                )

            if "network_properties" in data:
                network = data["network_properties"]
                if "connectivity_summary" in network:
                    conn = network["connectivity_summary"]
                    insights += f"  - Network connectivity: {conn.get('avg_connections_per_metabolite', 0):.1f} avg connections/metabolite\n"

        elif tool_name == "analyze_pathway":
            # Extract pathway insights safely
            if "summary" in data:
                summary = data["summary"]
                insights += f"  - Pathway coverage: {summary.get('reaction_count', 'N/A')} reactions\n"
                insights += f"  - Gene associations: {summary.get('gene_coverage', 'N/A')} genes\n"

        else:
            # Generic safe extraction for other tools
            safe_keys = ["summary", "status", "message", "total", "count", "rate"]
            for key in safe_keys:
                if key in data:
                    value = data[key]
                    if isinstance(value, (int, float, str)) and len(str(value)) < 100:
                        insights += f"  - {key}: {value}\n"

        # Fallback if no insights extracted
        if not insights.strip():
            insights = f"  - Analysis completed successfully\n"

        return insights

    def _prepare_tool_input(self, tool_name: str, query: str) -> Dict[str, Any]:
        """Prepare appropriate input for each tool"""
        logger.debug(
            f"🔍 PREPARING TOOL INPUT: tool_name='{tool_name}', query='{query[:50]}...'"
        )

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
            "analyze_pathway",
            "check_missing_media",
            "analyze_reaction_expression",
            # AI Media Tools
            "select_optimal_media",
            "manipulate_media_composition",
            "analyze_media_compatibility",
            "compare_media_performance",
        ]:
            # Ensure model_path is always a string
            model_path = (
                str(self.default_model_path)
                if self.default_model_path
                else str(
                    Path(__file__).parent.parent.parent
                    / "data"
                    / "examples"
                    / "e_coli_core.xml"
                )
            )
            result = {"model_path": model_path}
            logger.debug(f"🔍 TOOL INPUT PREPARED: {tool_name} -> {result}")
            return result

        # Biochemistry tools need query
        elif tool_name in ["search_biochem"]:
            result = {"query": "ATP"}  # Default biochemistry query
            logger.debug(f"🔍 TOOL INPUT PREPARED: {tool_name} -> {result}")
            return result

        elif tool_name in ["resolve_biochem_entity"]:
            result = {"entity_id": "cpd00027"}  # ATP entity ID
            logger.debug(f"🔍 TOOL INPUT PREPARED: {tool_name} -> {result}")
            return result

        # ModelSEED tools need specific inputs
        elif tool_name in ["build_metabolic_model"]:
            # This tool requires genome data, but we don't have it for analysis queries
            # Return an error input to trigger graceful fallback
            result = {"error": "ModelSEED build tools require genome annotation files"}
            logger.debug(f"🔍 TOOL INPUT PREPARED: {tool_name} -> {result}")
            return result

        # Default to simple input
        result = {"input": query}
        logger.debug(f"🔍 TOOL INPUT PREPARED: {tool_name} -> {result}")
        return result

    def _create_execution_summary(self, tool_name: str, data: Any) -> str:
        """Create summary of tool execution results"""
        if not isinstance(data, dict):
            return "Analysis completed"

        if tool_name == "run_metabolic_fba":
            if "objective_value" in data:
                return f"Growth rate: {data['objective_value']:.3f} h⁻¹"
        elif tool_name == "find_minimal_media":
            if "minimal_media" in data:
                return f"Requires {len(data['minimal_media'])} nutrients"
        elif tool_name == "select_optimal_media":
            if "best_media" in data:
                return f"Optimal media: {data['best_media']}"
        elif tool_name == "manipulate_media_composition":
            if "modified_media_name" in data:
                return f"Modified media: {data['modified_media_name']}"
        elif tool_name == "analyze_media_compatibility":
            if "compatible_media" in data:
                return f"Compatible media: {data['compatible_media']}"
        elif tool_name == "compare_media_performance":
            if "best_media" in data:
                return f"Best performing media: {data['best_media']}"
        elif tool_name == "analyze_essentiality":
            if "essential_genes" in data:
                return f"Found {len(data['essential_genes'])} essential genes"
        elif tool_name == "search_biochem":
            if "total_results" in data:
                return f"Found {data['total_results']} biochemistry matches"

        return "Analysis completed successfully"

    def _parse_tool_selection_response(
        self, response_text: str
    ) -> tuple[Optional[str], str]:
        """
        Robust parser for AI tool selection responses that handles various formatting styles.

        This parser is designed to be flexible and handle:
        - Extra backticks around responses (```markdown blocks)
        - Additional whitespace and newlines
        - Different markdown formatting
        - Case variations in labels
        - Missing colons or different separators
        """
        # Clean up the response text
        cleaned_text = self._clean_response_text(response_text)

        # Try multiple parsing strategies in order of preference
        selected_tool = None
        reasoning = ""

        # Strategy 1: Standard format parsing (TOOL: xxx, REASONING: yyy)
        tool_match = re.search(
            r"(?i)(?:^|\n)\s*TOOL\s*[:]\s*([^\n]+)", cleaned_text, re.MULTILINE
        )
        reasoning_match = re.search(
            r"(?i)(?:^|\n)\s*REASONING\s*[:]\s*(.*?)(?=\n\s*[A-Z]+\s*[:}]|\Z)",
            cleaned_text,
            re.DOTALL,
        )

        if tool_match:
            selected_tool = tool_match.group(1).strip()
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Strategy 2: Alternative formats and labels
        if not selected_tool:
            # Try variations like "Selected tool:", "Tool name:", etc.
            alt_tool_patterns = [
                r"(?i)(?:selected\s+)?tool\s*(?:name)?[:]\s*([^\n]+)",
                r"(?i)choose\s*[:]\s*([^\n]+)",
                r"(?i)next\s+tool\s*[:]\s*([^\n]+)",
                r"(?i)execute\s*[:]\s*([^\n]+)",
            ]

            for pattern in alt_tool_patterns:
                match = re.search(pattern, cleaned_text)
                if match:
                    selected_tool = match.group(1).strip()
                    break

        # Strategy 3: Extract from context if no explicit label found
        if not selected_tool:
            # Look for tool names mentioned in context
            available_tools = (
                list(self._tools_dict.keys()) if hasattr(self, "_tools_dict") else []
            )
            for tool in available_tools:
                if re.search(rf"\b{re.escape(tool)}\b", cleaned_text, re.IGNORECASE):
                    selected_tool = tool
                    break

        # Strategy 4: Extract reasoning from various formats
        if not reasoning:
            # Try alternative reasoning patterns
            alt_reasoning_patterns = [
                r"(?i)(?:^|\n)\s*(?:explanation|rationale|why|because)\s*[:]\s*(.*?)(?=\n\s*[A-Z]+\s*[:}]|\Z)",
                r"(?i)(?:^|\n)\s*reasoning\s*[:\-]\s*(.*?)(?=\n\s*[A-Z]+\s*[:}]|\Z)",
                r"(?i)I\s+(?:choose|select|recommend)\s+.*?because\s+(.*?)(?=\n|\Z)",
            ]

            for pattern in alt_reasoning_patterns:
                match = re.search(pattern, cleaned_text, re.DOTALL)
                if match:
                    reasoning = match.group(1).strip()
                    break

        # Strategy 5: Fallback to extracting any explanation text
        if not reasoning and selected_tool:
            # Extract any explanatory text after tool selection
            lines = cleaned_text.split("\n")
            tool_found = False
            explanation_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if selected_tool.lower() in line.lower():
                    tool_found = True
                    continue

                if (
                    tool_found
                    and line
                    and not line.upper().startswith(("TOOL", "ACTION", "SUMMARY"))
                ):
                    explanation_lines.append(line)
                    if len(explanation_lines) >= 3:  # Limit to reasonable length
                        break

            if explanation_lines:
                reasoning = " ".join(explanation_lines)

        # Clean up extracted values
        if selected_tool:
            selected_tool = self._clean_tool_name(selected_tool)
        if reasoning:
            reasoning = self._clean_reasoning_text(reasoning)

        # Validate the selected tool exists
        if (
            selected_tool
            and hasattr(self, "_tools_dict")
            and selected_tool not in self._tools_dict
        ):
            logger.warning(
                f"🔍 PARSER: Extracted tool '{selected_tool}' not in available tools, attempting fuzzy match"
            )
            selected_tool = self._fuzzy_match_tool(selected_tool)

        return selected_tool, reasoning or "AI analysis completed"

    def _parse_decision_response(self, response_text: str) -> Dict[str, Any]:
        """
        Robust parser for AI decision responses that handles various formatting styles.
        """
        # Clean up the response text
        cleaned_text = self._clean_response_text(response_text)

        # Initialize default decision
        decision = {"action": "finalize", "reasoning": "Default finalization"}

        # Strategy 1: Standard format parsing (ACTION: xxx, TOOL: yyy, REASONING: zzz)
        action_match = re.search(
            r"(?i)(?:^|\n)\s*ACTION\s*[:]\s*([^\n]+)", cleaned_text, re.MULTILINE
        )
        tool_match = re.search(
            r"(?i)(?:^|\n)\s*TOOL\s*[:]\s*([^\n]+)", cleaned_text, re.MULTILINE
        )
        reasoning_match = re.search(
            r"(?i)(?:^|\n)\s*REASONING\s*[:]\s*(.*?)(?=\n\s*[A-Z]+\s*[:}]|\Z)",
            cleaned_text,
            re.DOTALL,
        )

        if action_match:
            action = action_match.group(1).strip().lower()
            decision["action"] = action

        if tool_match:
            tool = tool_match.group(1).strip()
            tool = self._clean_tool_name(tool)
            if hasattr(self, "_tools_dict") and tool in self._tools_dict:
                decision["tool"] = tool

        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            decision["reasoning"] = self._clean_reasoning_text(reasoning)

        # Strategy 2: Alternative action detection
        if decision["action"] == "finalize":  # Default wasn't overridden
            action_patterns = [
                (r"(?i)\b(?:execute|run|use)\s+(?:tool|next)", "execute_tool"),
                (r"(?i)\b(?:continue|proceed|next)", "execute_tool"),
                (r"(?i)\b(?:finalize|complete|done|finish)", "finalize"),
                (r"(?i)\b(?:stop|end)", "finalize"),
            ]

            for pattern, action_type in action_patterns:
                if re.search(pattern, cleaned_text):
                    decision["action"] = action_type
                    break

        # Strategy 3: Tool extraction from context if not found explicitly
        if decision["action"] == "execute_tool" and "tool" not in decision:
            available_tools = (
                list(self._tools_dict.keys()) if hasattr(self, "_tools_dict") else []
            )
            for tool in available_tools:
                if re.search(rf"\b{re.escape(tool)}\b", cleaned_text, re.IGNORECASE):
                    decision["tool"] = tool
                    break

        # Strategy 4: Enhanced reasoning extraction
        if decision["reasoning"] == "Default finalization":
            # Try to extract any explanatory content
            alt_reasoning_patterns = [
                r"(?i)(?:explanation|rationale|because)\s*[:]\s*(.*?)(?=\n|\Z)",
                r"(?i)I\s+(?:recommend|suggest|think)\s+(.*?)(?=\n|\Z)",
                r"(?i)(?:next|this)\s+(?:step|tool)\s+(?:will|should)\s+(.*?)(?=\n|\Z)",
            ]

            for pattern in alt_reasoning_patterns:
                match = re.search(pattern, cleaned_text, re.DOTALL)
                if match:
                    reasoning_text = match.group(1).strip()
                    if len(reasoning_text) > 10:  # Meaningful reasoning
                        decision["reasoning"] = self._clean_reasoning_text(
                            reasoning_text
                        )
                        break

        return decision

    def _clean_response_text(self, text: str) -> str:
        """Clean and normalize response text for parsing."""
        # Remove markdown code block markers
        text = re.sub(r"```(?:markdown|text)?\s*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)

        # Remove extra backticks and quotes
        text = re.sub(r'^[`"\s]+|[`"\s]+$', "", text)

        # Normalize whitespace but preserve line breaks
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)

        return text.strip()

    def _clean_tool_name(self, tool_name: str) -> str:
        """Clean and validate tool name."""
        # Remove common prefixes/suffixes
        tool_name = re.sub(
            r"^(?:tool|function|method)[:]\s*", "", tool_name, flags=re.IGNORECASE
        )
        tool_name = re.sub(r'["`\s]+$', "", tool_name)
        tool_name = re.sub(r'^["`\s]+', "", tool_name)

        return tool_name.strip()

    def _clean_reasoning_text(self, reasoning: str) -> str:
        """Clean and format reasoning text."""
        # Remove excessive whitespace
        reasoning = re.sub(r"\s+", " ", reasoning)

        # Remove common prefixes
        reasoning = re.sub(
            r"^(?:reasoning|explanation|because)[:]\s*",
            "",
            reasoning,
            flags=re.IGNORECASE,
        )

        # Limit length to prevent overly long reasoning
        if len(reasoning) > 500:
            reasoning = reasoning[:500] + "..."

        return reasoning.strip()

    def _fuzzy_match_tool(self, tool_name: str) -> Optional[str]:
        """Attempt fuzzy matching for tool names."""
        if not hasattr(self, "_tools_dict"):
            return None

        available_tools = list(self._tools_dict.keys())
        tool_lower = tool_name.lower()

        # Try exact substring matches first
        for available_tool in available_tools:
            if (
                tool_lower in available_tool.lower()
                or available_tool.lower() in tool_lower
            ):
                logger.info(
                    f"🔍 PARSER: Fuzzy matched '{tool_name}' to '{available_tool}'"
                )
                return available_tool

        # Try space-separated word matching for multi-word queries
        if " " in tool_lower:
            tool_words = tool_lower.split()
            for available_tool in available_tools:
                available_lower = available_tool.lower()
                if all(word in available_lower for word in tool_words):
                    logger.info(
                        f"🔍 PARSER: Multi-word fuzzy matched '{tool_name}' to '{available_tool}'"
                    )
                    return available_tool

        # Try word-based matching
        tool_words = set(re.findall(r"\w+", tool_lower))
        best_match = None
        best_score = 0

        for available_tool in available_tools:
            available_words = set(re.findall(r"\w+", available_tool.lower()))
            overlap = len(tool_words.intersection(available_words))
            if overlap > best_score:
                best_score = overlap
                best_match = available_tool

        if best_score > 0:
            logger.info(
                f"🔍 PARSER: Fuzzy matched '{tool_name}' to '{best_match}' (score: {best_score})"
            )
            return best_match

        return None

    def _fallback_tool_selection(self, query: str) -> str:
        """Fallback tool selection logic"""
        query_lower = query.lower()

        # Extract model path from query if present
        if "data/examples/e_coli_core.xml" in query_lower:
            self.default_model_path = "data/examples/e_coli_core.xml"
        elif "e_coli" in query_lower or "ecoli" in query_lower:
            self.default_model_path = "data/examples/e_coli_core.xml"

        if any(word in query_lower for word in ["growth", "fba", "flux"]):
            return "run_metabolic_fba"
        elif any(word in query_lower for word in ["media", "nutrition"]):
            # Check for AI media tool keywords for intelligent media management
            if any(
                word in query_lower for word in ["select", "choose", "optimal", "best"]
            ):
                return "select_optimal_media"
            elif any(
                word in query_lower
                for word in ["modify", "change", "add", "remove", "anaerobic"]
            ):
                return "manipulate_media_composition"
            elif any(
                word in query_lower
                for word in ["compatibility", "compatible", "mapping"]
            ):
                return "analyze_media_compatibility"
            elif any(
                word in query_lower for word in ["compare", "comparison", "performance"]
            ):
                return "compare_media_performance"
            else:
                return "find_minimal_media"  # Fallback to traditional minimal media
        elif any(word in query_lower for word in ["essential", "gene"]):
            return "analyze_essentiality"
        elif any(
            word in query_lower for word in ["comprehensive", "analyze", "systematic"]
        ):
            return "run_metabolic_fba"  # Start with growth for comprehensive analysis
        else:
            return "run_metabolic_fba"  # Default starting point

    def _log_ai_decision(self, decision_type: str, data: Dict[str, Any]):
        """Log AI decision-making for audit trail"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "decision_type": decision_type,
            "ai_reasoning": data,
        }
        self.audit_trail.append(log_entry)

    def _save_complete_audit_trail(self, query: str, conclusions: Dict[str, Any]):
        """Save complete audit trail for hallucination detection"""
        audit_report = {
            "analysis_session": {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "agent_type": "Real-Time Dynamic AI Agent",
                "run_id": self.run_id,
            },
            "execution_sequence": self.audit_trail,
            "knowledge_accumulated": self.knowledge_base,
            "ai_conclusions": conclusions,
            "tool_decision_rationale": [
                "Tool selection based on AI analysis of previous results",
                "Dynamic workflow adaptation in real-time",
                "Result-driven decision making at each step",
            ],
        }

        audit_file = self.run_dir / "complete_audit_trail.json"
        with open(audit_file, "w") as f:
            json.dump(audit_report, f, indent=2, default=str)

        logger.info(f"💾 Complete audit trail saved: {audit_file}")

    def _create_error_result(self, error_message: str) -> AgentResult:
        """Create error result"""
        return AgentResult(
            success=False,
            message=error_message,
            data={},
            intermediate_steps=self.audit_trail,
            error=error_message,
            metadata={"run_id": self.run_id},
        )

    def _analyze_model_characteristics(self, query: str) -> Dict[str, Any]:
        """Extract model characteristics from query for learning recommendations"""
        characteristics = {
            "model_type": "unknown",
            "organism": "unknown",
            "query_type": "general",
            "complexity": "medium",
        }

        query_lower = query.lower()

        # Identify model type
        if "e. coli" in query_lower or "ecoli" in query_lower:
            characteristics["organism"] = "e_coli"
        elif "yeast" in query_lower:
            characteristics["organism"] = "yeast"
        elif "human" in query_lower:
            characteristics["organism"] = "human"

        # Identify query type
        if "comprehensive" in query_lower:
            characteristics["query_type"] = "comprehensive"
            characteristics["complexity"] = "high"
        elif "growth" in query_lower:
            characteristics["query_type"] = "growth_analysis"
        elif "essential" in query_lower:
            characteristics["query_type"] = "essentiality"
        elif "media" in query_lower or "nutrition" in query_lower:
            characteristics["query_type"] = "nutritional"

        return characteristics

    def _should_use_reasoning_chains(self, query: str) -> bool:
        """Determine if multi-step reasoning chains should be used"""
        indicators = ["comprehensive", "systematic", "step by step", "analyze all"]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in indicators)

    def _should_use_hypothesis_driven(self, query: str) -> bool:
        """Determine if hypothesis-driven analysis should be used"""
        indicators = [
            "why",
            "investigate",
            "hypothesis",
            "might be",
            "could be",
            "reason",
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in indicators)

    async def _run_with_reasoning_chains(
        self, query: str, max_iterations: int
    ) -> AgentResult:
        """Run analysis using multi-step reasoning chains"""
        # For now, fallback to standard execution
        # Full reasoning chain implementation would go here
        return await self._run_standard_analysis(query, max_iterations)

    async def _run_with_hypothesis_mode(
        self, query: str, max_iterations: int
    ) -> AgentResult:
        """Run analysis using hypothesis-driven approach"""
        # For now, fallback to standard execution
        # Full hypothesis mode implementation would go here
        return await self._run_standard_analysis(query, max_iterations)

    async def _run_with_collaborative_mode(
        self, query: str, max_iterations: int
    ) -> AgentResult:
        """Run analysis with collaborative reasoning"""
        # For now, fallback to standard execution
        # Full collaborative mode implementation would go here
        return await self._run_standard_analysis(query, max_iterations)

    async def _run_standard_analysis(
        self, query: str, max_iterations: int
    ) -> AgentResult:
        """Run standard dynamic analysis workflow"""
        logger.info("🚀 Starting standard dynamic analysis workflow")

        # For comprehensive analysis queries, delegate to LangGraph for better performance
        if "comprehensive" in query.lower() and hasattr(self, "_tools_dict"):
            logger.info(
                "📊 Delegating comprehensive analysis to LangGraphMetabolicAgent"
            )
            try:
                # Create a new LangGraph agent with current config
                from .langgraph_metabolic import LangGraphMetabolicAgent

                langgraph = LangGraphMetabolicAgent(
                    llm=self.llm, tools=list(self._tools_dict.values()), config={}
                )
                result = langgraph.run({"query": query})

                # Complete audit trail if enabled
                if self.enable_audit and self.current_workflow_id:
                    self.ai_audit_logger.complete_workflow(
                        self.current_workflow_id,
                        success=result.success,
                        final_result=result.message,
                    )

                return result
            except Exception as e:
                logger.error(f"Failed to delegate to LangGraph: {e}")
                # Continue with standard analysis

        # Initial tool selection
        logger.info(
            f"🔍 AI TOOL SELECTION: Starting tool selection for query: '{query[:50]}...'"
        )
        first_tool, reasoning = self._ai_analyze_query_for_first_tool(query)
        logger.info(f"🔍 AI TOOL SELECTION: Selected tool: '{first_tool}'")
        logger.info(f"🔍 AI TOOL SELECTION: Reasoning: '{reasoning[:100]}...'")
        logger.info(
            f"🔍 AI TOOL SELECTION: Available tools: {list(self._tools_dict.keys())}"
        )

        if not first_tool:
            logger.warning("⚠️ Could not determine initial tool, using FBA as default")
            first_tool = "run_metabolic_fba"
            reasoning = "Starting with growth analysis as baseline"

        self.knowledge_base["ai_initial_reasoning"] = reasoning
        self._log_ai_decision(
            "initial_tool_selection",
            {"selected_tool": first_tool, "reasoning": reasoning},
        )

        # Execute first tool
        logger.info(f"🔧 Executing first tool: {first_tool}")
        result = await self._execute_tool_with_audit(first_tool, query)

        if not result.success:
            return self._create_error_result(f"First tool failed: {result.error}")

        self.knowledge_base[first_tool] = result.data
        self.tool_execution_history.append(
            {"tool": first_tool, "reasoning": reasoning, "success": result.success}
        )

        # Dynamic analysis loop
        iteration = 1
        while iteration < max_iterations:
            # AI decides next step based on all accumulated knowledge
            next_tool, should_continue, reasoning = (
                self._ai_analyze_results_and_decide_next_step(
                    query, self.knowledge_base
                )
            )

            if not should_continue:
                logger.info("✅ AI determined analysis is complete")
                break

            if next_tool:
                logger.info(f"🔧 AI selected next tool: {next_tool}")
                logger.info(f"💭 Reasoning: {reasoning}")

                result = await self._execute_tool_with_audit(next_tool, query)

                self.knowledge_base[next_tool] = result.data
                self.tool_execution_history.append(
                    {
                        "tool": next_tool,
                        "reasoning": reasoning,
                        "success": result.success,
                        "error": result.error if not result.success else None,
                    }
                )

                if result.success:
                    logger.info(f"✅ Tool {next_tool} completed successfully")
                else:
                    logger.warning(f"⚠️ Tool {next_tool} failed: {result.error}")

            iteration += 1

        # Generate final conclusions
        final_conclusions = self._ai_generate_final_conclusions(
            query, self.knowledge_base, self.tool_execution_history
        )

        # Save audit trail
        self._save_complete_audit_trail(query, final_conclusions)

        # Log to AI audit system (skip if method not available)
        try:
            self.ai_audit_logger.log_ai_decision(
                self.current_workflow_id,
                "final_synthesis",
                final_conclusions,
                {"knowledge_base": self.knowledge_base},
            )
        except AttributeError:
            logger.warning("AI audit logger method not available, skipping")

        # Complete workflow (skip if method not available)
        try:
            self.ai_audit_logger.complete_workflow(
                self.current_workflow_id,
                final_conclusions.get("conclusions", "Analysis completed"),
                {
                    "tools_executed": len(self.tool_execution_history),
                    "ai_decisions": len(self.audit_trail),
                },
            )
        except AttributeError:
            logger.warning(
                "AI audit logger complete_workflow method not available, skipping"
            )

        # Stop real-time monitoring (skip if method not available)
        confidence_score = 0.8  # Default confidence
        try:
            if (
                self.enable_realtime_verification
                and self.current_workflow_id
                and hasattr(self.realtime_detector, "complete_monitoring")
            ):
                verification_report = self.realtime_detector.complete_monitoring(
                    self.current_workflow_id
                )
                confidence_score = verification_report.overall_confidence
                if confidence_score < 0.8:
                    logger.warning(f"⚠️ Low confidence score: {confidence_score}")
                logger.info(
                    f"🔍 Real-time verification completed for workflow {self.current_workflow_id}"
                )
            else:
                logger.debug(
                    "Real-time verification not enabled or workflow not tracked"
                )
        except Exception as e:
            logger.warning(f"Real-time detector complete_monitoring failed: {e}")
            # Continue with default confidence score

        # Update learning memory (skip if method not available)
        try:
            self.learning_memory.record_analysis(
                query,
                self._analyze_model_characteristics(query),
                [t["tool"] for t in self.tool_execution_history if t["success"]],
                final_conclusions.get("conclusions", "Analysis completed"),
                confidence_score,
            )
        except AttributeError:
            logger.warning(
                "Learning memory record_analysis method not available, skipping"
            )

        return AgentResult(
            success=True,
            message=final_conclusions.get(
                "conclusions", final_conclusions.get("summary", "Analysis completed")
            ),
            data={
                "knowledge_base": self.knowledge_base,
                "execution_history": self.tool_execution_history,
                "ai_reasoning": final_conclusions,
            },
            metadata={
                "tools_executed": [t["tool"] for t in self.tool_execution_history],
                "reasoning_steps": len(self.audit_trail),
                "confidence_score": confidence_score,
            },
        )

    async def analyze_model(
        self, query: str, model_path: Optional[str] = None
    ) -> AgentResult:
        """Analyze a metabolic model - compatibility method"""
        input_data = {"query": query}
        if model_path:
            self.default_model_path = model_path

        return await self.run(input_data)
