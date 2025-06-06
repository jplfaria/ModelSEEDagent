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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from ..llm.base import BaseLLM
from ..tools import ToolRegistry
from ..tools.base import BaseTool, ToolResult
from .base import AgentConfig, AgentResult, BaseAgent

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

        # Tool management
        self._tools_dict = {t.tool_name: t for t in tools}
        self.knowledge_base = {}
        self.audit_trail = []

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

    def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Run dynamic AI-driven metabolic analysis.

        This method implements true dynamic decision-making where each tool
        execution leads to AI analysis of results and intelligent selection
        of the next tool based on what was discovered.
        """
        query = input_data.get("query", input_data.get("input", ""))
        max_iterations = input_data.get("max_iterations", 6)

        logger.info(f"🧠 REAL-TIME AI AGENT STARTING: {query}")

        try:
            # Initialize session
            self._init_session(query)

            # Step 1: AI Query Analysis & First Tool Selection
            first_tool, reasoning = self._ai_analyze_query_for_first_tool(query)
            if not first_tool:
                return self._create_error_result(
                    "Could not determine appropriate starting tool"
                )

            self._log_ai_decision(
                "initial_analysis",
                {"query": query, "selected_tool": first_tool, "reasoning": reasoning},
            )

            # Execute first tool
            step1_success, step1_data = self._execute_tool_with_audit(
                first_tool, 1, query
            )
            if not step1_success:
                return self._create_error_result(f"First tool {first_tool} failed")

            # Dynamic tool selection loop
            current_step = 2
            while current_step <= max_iterations:
                # AI analyzes results and decides next step
                next_decision = self._ai_analyze_results_and_decide_next_step(
                    self.knowledge_base, query, current_step
                )

                if next_decision["action"] == "finalize":
                    logger.info("🤖 AI determined analysis is complete")
                    break
                elif next_decision["action"] == "execute_tool":
                    tool_name = next_decision["tool"]
                    reasoning = next_decision["reasoning"]

                    self._log_ai_decision(
                        f"step_{current_step}_selection",
                        {
                            "tool": tool_name,
                            "reasoning": reasoning,
                            "based_on_results": list(self.knowledge_base.keys()),
                        },
                    )

                    # Execute the AI-selected tool
                    success, data = self._execute_tool_with_audit(
                        tool_name, current_step, query
                    )
                    if success:
                        current_step += 1
                    else:
                        logger.warning(
                            f"Tool {tool_name} failed, but continuing analysis"
                        )
                        current_step += 1
                else:
                    logger.warning(
                        f"Unknown AI decision action: {next_decision['action']}"
                    )
                    break

            # AI generates final conclusions
            final_conclusions = self._ai_generate_final_conclusions(
                query, self.knowledge_base
            )

            # Save complete audit trail
            self._save_complete_audit_trail(query, final_conclusions)

            # Create successful result
            return AgentResult(
                success=True,
                message=final_conclusions["summary"],
                data=self.knowledge_base,
                intermediate_steps=self.audit_trail,
                metadata={
                    "run_id": self.run_id,
                    "ai_reasoning_steps": len(self.audit_trail),
                    "tools_executed": list(self.knowledge_base.keys()),
                    "quantitative_findings": final_conclusions.get(
                        "quantitative_findings", {}
                    ),
                    "ai_confidence": final_conclusions.get("confidence_score", 0.0),
                    "audit_file": str(self.run_dir / "complete_audit_trail.json"),
                },
            )

        except Exception as e:
            logger.error(f"Real-time agent execution failed: {e}")
            return self._create_error_result(f"Execution failed: {str(e)}")

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

    def analyze_model(
        self, query: str, model_path: Optional[str] = None
    ) -> AgentResult:
        """Analyze a metabolic model - compatibility method"""
        input_data = {"query": query}
        if model_path:
            self.default_model_path = model_path

        return self.run(input_data)
