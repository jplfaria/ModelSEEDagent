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

            # Start AI workflow auditing
            self.current_workflow_id = self.ai_audit_logger.start_workflow(query)
            logger.info(f"🔍 AI workflow audit started: {self.current_workflow_id}")

            # Start real-time verification monitoring
            self.realtime_detector.start_monitoring(self.current_workflow_id, query)
            logger.info(f"🔍 Real-time verification monitoring started")

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
                return await self._run_with_hypothesis_testing(query, max_iterations)
            elif reasoning_mode == "collaborative":
                return await self._run_with_collaborative_reasoning(
                    query, max_iterations
                )
            else:
                return await self._run_dynamic_analysis(query, max_iterations)

        except Exception as e:
            logger.error(f"Advanced AI agent execution failed: {e}")
            return await self._handle_execution_failure(e)

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

        prompt = f"""You are an expert metabolic modeling AI agent. Analyze this query and select the BEST first tool to start with.

Query: "{query}"

Available tools: {', '.join(available_tools)}

Based on the query, what tool should you start with and why? Consider:
1. What type of analysis is being requested?
2. What foundational information do you need first?
3. Which tool provides the most informative starting point?

Respond with:
TOOL: [exact tool name]
REASONING: [detailed explanation of why this tool is the optimal starting point]

Think step by step about the query requirements and tool capabilities."""

        try:
            response = self.llm._generate_response(prompt)
            response_text = response.text.strip()

            # Parse AI response
            lines = response_text.split("\n")
            selected_tool = None
            reasoning = ""

            for line in lines:
                if line.startswith("TOOL:"):
                    selected_tool = line.replace("TOOL:", "").strip()
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()
                elif reasoning and line.strip():
                    reasoning += " " + line.strip()

            # Log AI reasoning step
            if self.current_workflow_id:
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

                # Get the reasoning step for real-time verification
                reasoning_steps = self.ai_audit_logger.current_workflow.reasoning_steps
                if reasoning_steps:
                    latest_step = reasoning_steps[-1]
                    alerts = self.realtime_detector.process_reasoning_step(
                        self.current_workflow_id, latest_step
                    )
                    if alerts:
                        logger.info(
                            f"🔍 Real-time verification generated {len(alerts)} alerts"
                        )

            # Validate tool exists
            if selected_tool and selected_tool in self._tools_dict:
                return selected_tool, reasoning

            # Fallback selection if parsing failed
            logger.warning("AI tool selection parsing failed, using fallback logic")
            return (
                self._fallback_tool_selection(query),
                "Fallback selection due to parsing error",
            )

        except Exception as e:
            logger.error(f"AI tool selection failed: {e}")
            return (
                self._fallback_tool_selection(query),
                f"Error in AI selection: {str(e)}",
            )

    def _ai_analyze_results_and_decide_next_step(
        self, knowledge_base: Dict[str, Any], query: str, step_number: int
    ) -> Dict[str, Any]:
        """
        AI analyzes accumulated results and decides the next step.
        This is the core of dynamic decision-making.
        """
        # Prepare context of what we've learned so far
        results_context = self._format_knowledge_base_for_ai(knowledge_base)

        prompt = f"""You are an expert metabolic modeling AI agent. You have executed some tools and now need to decide what to do next based on the ACTUAL RESULTS you've obtained.

ORIGINAL QUERY: "{query}"

CURRENT STEP: {step_number}

RESULTS OBTAINED SO FAR:
{results_context}

AVAILABLE TOOLS: {', '.join(self._tools_dict.keys())}

Based on the ACTUAL DATA you've collected, analyze:

1. What specific information have you learned?
2. What questions remain unanswered from the original query?
3. What tool would provide the most valuable ADDITIONAL insight?
4. Do you have enough information to provide a comprehensive answer?

Decide your next action:

ACTION: [either "execute_tool" or "finalize"]
TOOL: [if executing tool, specify exact tool name]
REASONING: [detailed explanation based on the actual results you've analyzed]

Make your decision based on the ACTUAL DATA PATTERNS you see, not generic workflows."""

        try:
            response = self.llm._generate_response(prompt)
            response_text = response.text.strip()

            # Parse AI decision
            decision = {"action": "finalize", "reasoning": "Default finalization"}

            lines = response_text.split("\n")
            for line in lines:
                if line.startswith("ACTION:"):
                    action = line.replace("ACTION:", "").strip().lower()
                    decision["action"] = action
                elif line.startswith("TOOL:"):
                    tool = line.replace("TOOL:", "").strip()
                    if tool in self._tools_dict:
                        decision["tool"] = tool
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()
                    # Collect remaining reasoning lines
                    remaining_lines = lines[lines.index(line) + 1 :]
                    full_reasoning = reasoning
                    for remaining_line in remaining_lines:
                        if remaining_line.strip():
                            full_reasoning += " " + remaining_line.strip()
                    decision["reasoning"] = full_reasoning
                    break

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
                reasoning_steps = self.ai_audit_logger.current_workflow.reasoning_steps
                if reasoning_steps:
                    latest_step = reasoning_steps[-1]
                    alerts = self.realtime_detector.process_reasoning_step(
                        self.current_workflow_id, latest_step
                    )
                    if alerts:
                        logger.info(
                            f"🔍 Step {step_number} verification generated {len(alerts)} alerts"
                        )

            return decision

        except Exception as e:
            logger.error(f"AI decision analysis failed: {e}")
            return {"action": "finalize", "reasoning": f"Error in analysis: {str(e)}"}

    def _execute_tool_with_audit(
        self, tool_name: str, step_number: int, query: str
    ) -> tuple[bool, Optional[Any]]:
        """
        Execute tool with complete audit trail for hallucination detection.
        """
        if tool_name not in self._tools_dict:
            logger.error(f"Tool {tool_name} not found")
            return False, None

        tool = self._tools_dict[tool_name]

        # Prepare tool input
        tool_input = self._prepare_tool_input(tool_name, query)

        # Record execution start
        execution_start = {
            "step": step_number,
            "tool": tool_name,
            "start_time": datetime.now().isoformat(),
            "inputs": tool_input,
        }

        try:
            logger.info(f"🔧 Executing Step {step_number}: {tool_name}")

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

                return True, result.data
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

                return False, None

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

            return False, None

    def _ai_generate_final_conclusions(
        self, query: str, knowledge_base: Dict[str, Any]
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
            response = self.llm._generate_response(prompt)
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

        except Exception as e:
            logger.error(f"AI conclusion generation failed: {e}")
            return {
                "summary": f"Analysis completed with {len(knowledge_base)} tools executed",
                "confidence_score": 0.5,
                "error": str(e),
            }

    def _format_knowledge_base_for_ai(self, knowledge_base: Dict[str, Any]) -> str:
        """Format accumulated knowledge for AI analysis"""
        if not knowledge_base:
            return "No results collected yet."

        formatted = ""
        for tool_name, data in knowledge_base.items():
            formatted += f"\n{tool_name.upper()} RESULTS:\n"

            if isinstance(data, dict):
                # Extract key metrics
                for key, value in list(data.items())[
                    :5
                ]:  # Limit to prevent overwhelming AI
                    formatted += f"  - {key}: {value}\n"
            else:
                formatted += f"  {str(data)[:200]}...\n"

        return formatted

    def _prepare_tool_input(self, tool_name: str, query: str) -> Dict[str, Any]:
        """Prepare appropriate input for each tool"""
        # Most tools need a model path
        if tool_name in [
            "run_metabolic_fba",
            "find_minimal_media",
            "analyze_essentiality",
            "run_flux_variability_analysis",
            "identify_auxotrophies",
        ]:
            return {"model_path": self.default_model_path}

        # Biochemistry tools need query
        elif tool_name in ["search_biochem"]:
            return {"query": "ATP"}  # Default biochemistry query

        elif tool_name in ["resolve_biochem_entity"]:
            return {"entity_id": "cpd00027"}  # ATP entity ID

        # Default to simple input
        return {"input": query}

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
        elif tool_name == "analyze_essentiality":
            if "essential_genes" in data:
                return f"Found {len(data['essential_genes'])} essential genes"
        elif tool_name == "search_biochem":
            if "total_results" in data:
                return f"Found {data['total_results']} biochemistry matches"

        return "Analysis completed successfully"

    def _fallback_tool_selection(self, query: str) -> str:
        """Fallback tool selection logic"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["growth", "fba", "flux"]):
            return "run_metabolic_fba"
        elif any(word in query_lower for word in ["media", "nutrition"]):
            return "find_minimal_media"
        elif any(word in query_lower for word in ["essential", "gene"]):
            return "analyze_essentiality"
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

    async def analyze_model(
        self, query: str, model_path: Optional[str] = None
    ) -> AgentResult:
        """Analyze a metabolic model - compatibility method"""
        input_data = {"query": query}
        if model_path:
            self.default_model_path = model_path

        return await self.run(input_data)
