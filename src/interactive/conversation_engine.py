"""
Dynamic AI Conversation Engine for Interactive Analysis

Replaces the old templated system with real AI-powered conversation
using the RealTimeMetabolicAgent for genuine dynamic decision-making.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from ..agents import create_real_time_agent
from ..llm.factory import LLMFactory
from ..tools import ToolRegistry
from .query_processor import QueryAnalysis, QueryProcessor, QueryType
from .session_manager import AnalysisSession, Interaction, InteractionType
from .streaming_interface import RealTimeStreamingInterface

console = Console()
logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """States of the conversation flow"""

    GREETING = "greeting"
    ACTIVE = "active"
    AI_THINKING = "ai_thinking"
    PROCESSING = "processing"
    STREAMING_RESULTS = "streaming_results"
    WAITING_INPUT = "waiting_input"
    ERROR_RECOVERY = "error_recovery"
    GOODBYE = "goodbye"


class ResponseType(Enum):
    """Types of assistant responses"""

    GREETING = "greeting"
    AI_ANALYSIS = "ai_analysis"
    STREAMING_THOUGHT = "streaming_thought"
    TOOL_EXECUTION = "tool_execution"
    FINAL_SYNTHESIS = "final_synthesis"
    ERROR_MESSAGE = "error_message"
    GOODBYE = "goodbye"


@dataclass
class ConversationContext:
    """Context maintained during conversation"""

    current_model: Optional[str] = None
    active_analysis: Optional[str] = None
    ai_agent: Optional[Any] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

    # User profiling
    expertise_level: str = "intermediate"
    interaction_count: int = 0
    successful_queries: int = 0


@dataclass
class ConversationResponse:
    """Structured response from the conversation engine"""

    content: str
    response_type: ResponseType
    suggested_actions: List[str] = field(default_factory=list)
    requires_input: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    ai_reasoning_steps: int = 0
    clarification_needed: bool = False


class DynamicAIConversationEngine:
    """Real AI-powered conversation engine using RealTimeMetabolicAgent"""

    def __init__(self, session: AnalysisSession):
        self.session = session
        self.context = ConversationContext()
        self.state = ConversationState.GREETING
        self.ai_agent = None
        self.streaming_interface = RealTimeStreamingInterface()
        self._initialize_ai_agent()

    def _load_cli_config(self) -> Dict[str, Any]:
        """Load CLI configuration from persistent storage"""
        cli_config_file = Path.home() / ".modelseed-agent-cli.json"

        if cli_config_file.exists():
            try:
                with open(cli_config_file, "r") as f:
                    config = json.load(f)
                    console.print(
                        f"📁 Loaded CLI config: {config.get('llm_backend', 'unknown')} backend with {config.get('llm_config', {}).get('model_name', 'unknown')} model",
                        style="dim",
                    )
                    return config
            except Exception as e:
                console.print(f"⚠️ Could not load CLI config: {e}", style="yellow")

        console.print(
            "⚠️ No CLI config found. Run 'modelseed-agent setup' first.", style="yellow"
        )
        return {}

    def _initialize_ai_agent(self):
        """Initialize the real AI agent using saved CLI configuration"""
        try:
            # Load saved CLI configuration
            cli_config = self._load_cli_config()

            # Use saved configuration if available, otherwise fallback to defaults
            if cli_config.get("llm_backend") and cli_config.get("llm_config"):
                llm_backend = cli_config["llm_backend"]
                llm_config = cli_config["llm_config"].copy()

                console.print(
                    f"🔧 Using saved config: {llm_backend} backend with {llm_config.get('model_name', 'unknown')} model",
                    style="green",
                )

                # Ensure system content is set for interactive mode
                if "system_content" not in llm_config:
                    llm_config["system_content"] = (
                        "You are an expert metabolic modeling AI agent that makes real-time decisions based on data analysis."
                    )

            else:
                # Fallback to default configuration
                console.print(
                    "⚠️ No saved configuration found, using defaults", style="yellow"
                )
                llm_backend = "argo"
                llm_config = {
                    "model_name": "gpt-4o-mini",
                    "system_content": "You are an expert metabolic modeling AI agent that makes real-time decisions based on data analysis.",
                    "temperature": 0.7,
                    "max_tokens": 4000,
                }

            try:
                llm = LLMFactory.create(llm_backend, llm_config)
                console.print(
                    f"🔗 Connected to {llm_backend.title()} LLM with model: {llm_config.get('model_name', 'unknown')}",
                    style="green",
                )
            except Exception as e:
                console.print(
                    f"❌ Failed to connect to {llm_backend}: {e}", style="red"
                )
                # Try fallback to argo with default config
                try:
                    fallback_config = {
                        "model_name": "gpt-4o-mini",
                        "system_content": "You are an expert metabolic modeling AI agent that makes real-time decisions based on data analysis.",
                        "temperature": 0.7,
                        "max_tokens": 4000,
                    }
                    llm = LLMFactory.create("argo", fallback_config)
                    console.print(
                        "🔗 Connected to Argo Gateway LLM (fallback)", style="yellow"
                    )
                except Exception:
                    console.print(
                        "⚠️ No LLM available - using fallback mode", style="yellow"
                    )
                    llm = None

            if llm:
                # Get available tools
                tool_names = ToolRegistry.list_tools()
                tools = []
                for tool_name in tool_names:
                    try:
                        tool = ToolRegistry.create_tool(tool_name, {})
                        tools.append(tool)
                        console.print(f"✅ Loaded tool: {tool_name}", style="dim")
                    except Exception as e:
                        console.print(
                            f"⚠️ Could not load tool {tool_name}: {e}", style="yellow"
                        )

                # Create the real AI agent
                # Check for LLM input logging via environment variable
                import os

                log_llm_inputs = os.getenv(
                    "MODELSEED_LOG_LLM_INPUTS", "false"
                ).lower() in ("true", "1", "yes")

                config = {
                    "max_iterations": 6,
                    "log_llm_inputs": log_llm_inputs,  # 🔍 Enable via env var
                }

                if log_llm_inputs:
                    console.print(
                        "🔍 LLM input logging enabled via MODELSEED_LOG_LLM_INPUTS",
                        style="yellow",
                    )

                self.ai_agent = create_real_time_agent(llm, tools, config)

                # DEBUG: Log what agent we actually created
                console.print(
                    f"🔍 DEBUG: Created agent type: {type(self.ai_agent).__name__}",
                    style="dim",
                )
                console.print(
                    f"🔍 DEBUG: Agent has _prepare_tool_input: {hasattr(self.ai_agent, '_prepare_tool_input')}",
                    style="dim",
                )
                self.context.ai_agent = self.ai_agent

                console.print(
                    f"🧠 Dynamic AI Agent initialized with {len(tools)} tools",
                    style="green",
                )
            else:
                console.print("⚠️ Running in demo mode without AI", style="yellow")

        except Exception as e:
            console.print(f"⚠️ AI initialization failed: {e}", style="red")
            self.ai_agent = None

    def start_conversation(self) -> ConversationResponse:
        """Start a new conversation with dynamic AI greeting"""
        self.state = ConversationState.GREETING

        # Determine if this is a first-time or returning user
        is_first_time = len(self.session.interactions) == 0

        if is_first_time:
            greeting = "👋 Welcome to ModelSEEDagent! I'm your AI-powered metabolic modeling assistant."
        else:
            greeting = (
                "🎉 Welcome back! Ready for more dynamic AI-driven metabolic analysis?"
            )

        greeting += "\n\n🧠 **Dynamic AI Features:**"
        greeting += "\n  • Real-time AI decision-making based on your data"
        greeting += "\n  • Adaptive tool selection that responds to results"
        greeting += "\n  • Complete reasoning transparency with audit trails"
        greeting += "\n  • Multi-step analysis with intelligent workflow adaptation"

        if self.ai_agent:
            greeting += (
                "\n\n✨ **AI Status:** Fully operational with real-time reasoning"
            )
            greeting += f"\n📊 **Available Tools:** {len(self.ai_agent._tools_dict)} specialized metabolic modeling tools"
        else:
            greeting += "\n\n⚠️ **AI Status:** Demo mode (LLM not available)"

        greeting += '\n\n💡 Try asking: "I need a comprehensive metabolic analysis of E. coli" and watch the AI think through the problem step by step!'

        self.state = ConversationState.WAITING_INPUT

        return ConversationResponse(
            content=greeting,
            response_type=ResponseType.GREETING,
            suggested_actions=[
                "I need a comprehensive metabolic characterization of the E. coli core model",
                "Analyze growth capabilities and nutritional requirements systematically",
                "Help me understand metabolic flexibility and essential components",
                "Show me how AI selects tools based on discovered data patterns",
            ],
            requires_input=True,
        )

    def process_user_input(self, user_input: str) -> ConversationResponse:
        """Process user input using real AI agent"""
        start_time = time.time()

        # Update conversation context
        self.context.interaction_count += 1

        if not self.ai_agent:
            return self._handle_no_ai_fallback(user_input)

        # Use real AI agent for processing - disable streaming for now to fix display issues
        return self._process_with_simple_ai(user_input, start_time)

    def _process_with_real_ai(
        self, user_input: str, start_time: float
    ) -> ConversationResponse:
        """Process query using the real AI agent with streaming display"""
        self.state = ConversationState.AI_THINKING

        # Show AI thinking indicator
        with console.status("🧠 AI analyzing your query...", spinner="dots"):
            try:
                # Run the dynamic AI agent
                result = self.ai_agent.run({"query": user_input})

                processing_time = time.time() - start_time

                if result.success:
                    # Create rich response showing AI reasoning
                    content = self._format_ai_response(result, user_input)

                    # Extract suggested follow-ups
                    suggested_actions = self._extract_follow_up_suggestions(result)

                    self.context.successful_queries += 1
                    self.state = ConversationState.WAITING_INPUT

                    # Log successful interaction
                    self._log_ai_interaction(user_input, result, processing_time, True)

                    return ConversationResponse(
                        content=content,
                        response_type=ResponseType.AI_ANALYSIS,
                        suggested_actions=suggested_actions,
                        requires_input=True,
                        metadata={
                            "ai_agent_result": True,
                            "tools_executed": result.metadata.get("tools_executed", []),
                            "ai_reasoning_steps": result.metadata.get(
                                "ai_reasoning_steps", 0
                            ),
                            "audit_file": result.metadata.get("audit_file"),
                            "quantitative_findings": result.metadata.get(
                                "quantitative_findings", {}
                            ),
                        },
                        processing_time=processing_time,
                        ai_reasoning_steps=result.metadata.get("ai_reasoning_steps", 0),
                    )

                else:
                    # Handle AI agent failure
                    return self._handle_ai_error(user_input, result, processing_time)

            except Exception as e:
                # Handle unexpected errors
                return self._handle_unexpected_error(
                    user_input, str(e), time.time() - start_time
                )

    def _process_with_streaming_ai(
        self, user_input: str, start_time: float
    ) -> ConversationResponse:
        """Process query using the real AI agent with real-time streaming display"""
        self.state = ConversationState.AI_THINKING

        try:
            # Start streaming interface
            self.streaming_interface.start_streaming(user_input)

            # Show initial AI thinking
            self.streaming_interface.show_ai_analysis(
                "Analyzing your query and planning the analysis approach..."
            )

            # Run the dynamic AI agent with streaming callbacks
            if hasattr(self.ai_agent, "run"):
                # Handle async run method
                import asyncio

                try:
                    # Check if run method is async
                    import inspect

                    if inspect.iscoroutinefunction(self.ai_agent.run):
                        result = asyncio.run(self.ai_agent.run({"query": user_input}))
                    else:
                        result = self.ai_agent.run({"query": user_input})
                except Exception as e:
                    # Fallback if asyncio.run fails
                    logger.error(f"Error running agent: {e}")
                    from ..agents.base import AgentResult

                    result = AgentResult(
                        success=False,
                        message=f"Agent execution failed: {str(e)}",
                        error=str(e),
                    )
            else:
                # Fallback for agents without run method
                result = self.ai_agent.process({"query": user_input})

            processing_time = time.time() - start_time

            if result.success:
                # Show completion
                self.streaming_interface.show_workflow_complete(
                    result.message, result.metadata
                )

                # Stop streaming after a brief pause
                time.sleep(1)
                self.streaming_interface.stop_streaming()

                # Create rich response showing AI reasoning
                content = self._format_ai_response(result, user_input)

                # Extract suggested follow-ups
                suggested_actions = self._extract_follow_up_suggestions(result)

                self.context.successful_queries += 1
                self.state = ConversationState.WAITING_INPUT

                # Log successful interaction
                self._log_ai_interaction(user_input, result, processing_time, True)

                return ConversationResponse(
                    content=content,
                    response_type=ResponseType.AI_ANALYSIS,
                    suggested_actions=suggested_actions,
                    requires_input=True,
                    metadata={
                        "ai_agent_result": True,
                        "tools_executed": result.metadata.get("tools_executed", []),
                        "ai_reasoning_steps": result.metadata.get(
                            "ai_reasoning_steps", 0
                        ),
                        "audit_file": result.metadata.get("audit_file"),
                        "quantitative_findings": result.metadata.get(
                            "quantitative_findings", {}
                        ),
                        "streaming_used": True,
                    },
                    processing_time=processing_time,
                    ai_reasoning_steps=result.metadata.get("ai_reasoning_steps", 0),
                )

            else:
                # Show error and stop streaming
                self.streaming_interface.show_error(result.error)
                time.sleep(1)
                self.streaming_interface.stop_streaming()

                # Handle AI agent failure
                return self._handle_ai_error(user_input, result, processing_time)

        except Exception as e:
            # Handle unexpected errors
            self.streaming_interface.show_error(
                str(e), "Unexpected error during AI processing"
            )
            time.sleep(1)
            self.streaming_interface.stop_streaming()
            return self._handle_unexpected_error(
                user_input, str(e), time.time() - start_time
            )

    def _run_agent_sync(self, user_input: str):
        """
        Synchronous wrapper for the async agent.run() method.
        Fallback to direct sync execution when in event loop context.
        """
        import asyncio

        try:
            # Try to use asyncio.run first
            return asyncio.run(self.ai_agent.run({"query": user_input}))
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                # We're in an event loop, so we need to call the agent differently
                # Let's try to call the agent's sync methods directly if possible
                logger.warning(
                    "Running in event loop context, using direct agent execution"
                )

                # Instead of trying complex async handling, let's use a simpler approach
                # Call the agent's standard analysis workflow directly
                try:
                    # Use the synchronous version if available, or fall back to a simple result
                    return self._run_agent_directly_sync(user_input)
                except Exception as direct_error:
                    logger.error(f"Direct agent execution failed: {direct_error}")
                    from ..agents.base import AgentResult

                    return AgentResult(
                        success=False,
                        message=f"Agent execution failed in event loop context: {str(direct_error)}",
                        error=str(direct_error),
                        data={},
                    )
            else:
                # Different RuntimeError, re-raise
                raise
        except Exception as e:
            logger.error(f"Error in agent execution: {e}")
            from ..agents.base import AgentResult

            return AgentResult(
                success=False,
                message=f"Agent execution failed: {str(e)}",
                error=str(e),
                data={},
            )

    def _run_agent_directly_sync(self, user_input: str):
        """
        Run a simplified version of the agent workflow synchronously.
        This bypasses the async complexity when in event loop context.
        """
        from ..agents.base import AgentResult

        try:
            # This is a simplified approach - just run the essential analysis tools directly
            logger.info("Running simplified synchronous agent workflow")

            # Use the agent's tool selection but skip the async workflow
            if hasattr(self.ai_agent, "_ai_analyze_query_for_first_tool"):
                first_tool, reasoning = self.ai_agent._ai_analyze_query_for_first_tool(
                    user_input
                )
                logger.info(f"Selected tool: {first_tool}")

                if (
                    first_tool
                    and hasattr(self.ai_agent, "_tools_dict")
                    and first_tool in self.ai_agent._tools_dict
                ):
                    # Prepare and execute the tool
                    tool_input = self.ai_agent._prepare_tool_input(
                        first_tool, user_input
                    )
                    tool = self.ai_agent._tools_dict[first_tool]
                    result = tool._run_tool(tool_input)

                    if result.success:
                        # Create a simplified success result
                        summary = f"Analysis completed using {first_tool}. Key findings: {str(result.data)[:200]}..."

                        return AgentResult(
                            success=True,
                            message=summary,
                            data={
                                "tool_executed": first_tool,
                                "tool_result": result.data,
                                "ai_reasoning": reasoning,
                                "simplified_execution": True,
                            },
                            metadata={
                                "tools_executed": [first_tool],
                                "ai_reasoning_steps": 1,
                                "execution_mode": "simplified_sync",
                            },
                        )
                    else:
                        return AgentResult(
                            success=False,
                            message=f"Tool execution failed: {result.error}",
                            error=result.error,
                            data={},
                        )
                else:
                    return AgentResult(
                        success=False,
                        message="Could not select appropriate tool for analysis",
                        error="Tool selection failed",
                        data={},
                    )
            else:
                return AgentResult(
                    success=False,
                    message="Agent does not support simplified execution",
                    error="Missing required agent methods",
                    data={},
                )

        except Exception as e:
            logger.error(f"Simplified agent execution failed: {e}")
            return AgentResult(
                success=False,
                message=f"Simplified agent execution failed: {str(e)}",
                error=str(e),
                data={},
            )

    def _process_with_simple_ai(
        self, user_input: str, start_time: float
    ) -> ConversationResponse:
        """Process query using the real AI agent with simple display (no streaming)"""
        self.state = ConversationState.AI_THINKING

        try:
            # Message already shown by CLI, no need to duplicate it

            # Simple approach: just run the agent directly with asyncio.run if needed
            import asyncio
            import inspect

            if hasattr(self.ai_agent, "run"):
                if inspect.iscoroutinefunction(self.ai_agent.run):
                    try:
                        # Try to get the current loop to see if we're in one
                        asyncio.get_running_loop()
                        # We're in a loop, so we need to handle this differently
                        # Use our synchronous wrapper
                        result = self._run_agent_sync(user_input)
                    except RuntimeError:
                        # No event loop running, safe to use asyncio.run
                        result = asyncio.run(self.ai_agent.run({"query": user_input}))
                else:
                    result = self.ai_agent.run({"query": user_input})
            else:
                # Fallback for agents without run method
                result = self.ai_agent.process({"query": user_input})

            processing_time = time.time() - start_time

            if result.success:
                # Create rich response showing AI reasoning
                content = self._format_ai_response(result, user_input)

                # Extract suggested follow-ups
                suggested_actions = self._extract_follow_up_suggestions(result)

                self.context.successful_queries += 1
                self.state = ConversationState.WAITING_INPUT

                # Log successful interaction
                self._log_ai_interaction(user_input, result, processing_time, True)

                return ConversationResponse(
                    content=content,
                    response_type=ResponseType.AI_ANALYSIS,
                    suggested_actions=suggested_actions,
                    requires_input=True,
                    metadata={
                        "ai_agent_result": True,
                        "tools_executed": result.metadata.get("tools_executed", []),
                        "ai_reasoning_steps": result.metadata.get(
                            "ai_reasoning_steps", 0
                        ),
                        "audit_file": result.metadata.get("audit_file"),
                        "quantitative_findings": result.metadata.get(
                            "quantitative_findings", {}
                        ),
                        "simple_display_used": True,
                    },
                    processing_time=processing_time,
                    ai_reasoning_steps=result.metadata.get("ai_reasoning_steps", 0),
                )

            else:
                # Handle AI agent failure
                return self._handle_ai_error(user_input, result, processing_time)

        except Exception as e:
            # Handle unexpected errors
            return self._handle_unexpected_error(
                user_input, str(e), time.time() - start_time
            )

    def _format_ai_response(self, result, user_input: str) -> str:
        """Format the AI agent response with rich formatting"""
        content = f"🧠 **AI Dynamic Analysis Complete**\n\n"

        # Show the AI's response
        content += f"**AI Response:**\n{result.message}\n\n"

        # Show tools that were executed
        tools_executed = result.metadata.get("tools_executed", [])
        if tools_executed:
            content += f"🔧 **Tools Executed ({len(tools_executed)}):**\n"
            for i, tool in enumerate(tools_executed, 1):
                content += f"  {i}. {tool}\n"
            content += "\n"

        # Show AI reasoning steps
        reasoning_steps = result.metadata.get("ai_reasoning_steps", 0)
        if reasoning_steps > 0:
            content += (
                f"🧠 **AI Reasoning Steps:** {reasoning_steps} decision points\n\n"
            )

        # Show quantitative findings if available
        findings = result.metadata.get("quantitative_findings", {})
        if findings:
            content += f"📊 **Key Quantitative Discoveries:**\n"
            for key, value in findings.items():
                content += f"  • {key.replace('_', ' ').title()}: {value}\n"
            content += "\n"

        # Show audit trail information
        audit_file = result.metadata.get("audit_file")
        if audit_file:
            content += f"🔍 **Complete Audit Trail:** {audit_file}\n"
            content += "  → Every AI decision and tool execution fully logged for verification\n\n"

        # Show AI confidence if available
        ai_confidence = result.metadata.get("ai_confidence")
        if ai_confidence:
            content += f"🎯 **AI Confidence:** {ai_confidence}\n\n"

        content += "✨ **Dynamic AI Features Demonstrated:**\n"
        content += "  • Real-time tool selection based on discovered data patterns\n"
        content += "  • Adaptive workflow that responds to actual results\n"
        content += "  • Complete reasoning transparency with audit trail\n"
        content += "  • Genuine AI decision-making (no templates!)\n"

        return content

    def _extract_follow_up_suggestions(self, result) -> List[str]:
        """Extract intelligent follow-up suggestions from AI analysis"""
        suggestions = []

        tools_executed = result.metadata.get("tools_executed", [])

        # Suggest complementary analyses based on what was done
        if "run_metabolic_fba" in tools_executed:
            suggestions.append(
                "Analyze flux variability to understand metabolic flexibility"
            )

        if "find_minimal_media" in tools_executed:
            suggestions.append(
                "Investigate auxotrophies to understand biosynthetic capabilities"
            )

        if "analyze_essentiality" in tools_executed:
            suggestions.append("Explore gene deletion effects on specific pathways")

        # Add general suggestions
        suggestions.extend(
            [
                "Ask for detailed explanation of any specific result",
                "Request deeper analysis of a particular metabolic aspect",
                "Compare with different growth conditions or models",
            ]
        )

        return suggestions[:4]  # Limit to 4 suggestions

    def _handle_ai_error(
        self, user_input: str, result, processing_time: float
    ) -> ConversationResponse:
        """Handle AI agent errors gracefully"""
        self.state = ConversationState.ERROR_RECOVERY

        content = f"🤖 **AI Analysis Encountered Issues**\n\n"
        content += f"**Error:** {result.error}\n\n"
        content += "**What happened:**\n"
        content += "The AI agent attempted to process your query but encountered technical difficulties.\n\n"
        content += "**Suggestions:**\n"
        content += "  • Try rephrasing your question more specifically\n"
        content += "  • Ask about a particular aspect of metabolic analysis\n"
        content += "  • Check if required data files are available\n\n"
        content += "**Example queries that work well:**\n"
        content += '  • "Analyze the growth rate of E. coli core model"\n'
        content += '  • "Find the minimal media requirements"\n'
        content += '  • "Identify essential genes in the model"\n'

        # Log failed interaction
        self._log_ai_interaction(user_input, result, processing_time, False)

        return ConversationResponse(
            content=content,
            response_type=ResponseType.ERROR_MESSAGE,
            suggested_actions=[
                "Analyze the growth rate of E. coli core model",
                "Find the minimal media requirements",
                "Identify essential genes in the model",
                "Help me understand FBA analysis",
            ],
            requires_input=True,
            metadata={"error": result.error, "ai_agent_failed": True},
            processing_time=processing_time,
        )

    def _handle_unexpected_error(
        self, user_input: str, error: str, processing_time: float
    ) -> ConversationResponse:
        """Handle unexpected errors"""
        content = f"💥 **Unexpected Error**\n\n"
        content += f"An unexpected error occurred: {error}\n\n"
        content += "The AI agent system encountered an unexpected issue. Please try again or contact support."

        return ConversationResponse(
            content=content,
            response_type=ResponseType.ERROR_MESSAGE,
            suggested_actions=["Try a simpler query", "Contact support"],
            requires_input=True,
            metadata={"unexpected_error": error},
            processing_time=processing_time,
        )

    def _handle_no_ai_fallback(self, user_input: str) -> ConversationResponse:
        """Handle queries when AI agent is not available"""
        content = f"🤖 **Demo Mode Response**\n\n"
        content += f"**Your Query:** {user_input}\n\n"
        content += "**Demo Mode Active:** The AI agent is not available (LLM not configured).\n\n"
        content += "**What would happen with the real AI:**\n"
        content += "  1. AI would analyze your query to understand intent\n"
        content += "  2. AI would select the best first tool to start analysis\n"
        content += (
            "  3. AI would examine tool results and choose next tools dynamically\n"
        )
        content += (
            "  4. AI would synthesize findings into comprehensive conclusions\n\n"
        )
        content += "**To enable real AI:**\n"
        content += "  • Configure Argo Gateway or OpenAI API credentials\n"
        content += "  • Restart the interactive interface\n"
        content += "  • Experience true dynamic AI decision-making\n"

        return ConversationResponse(
            content=content,
            response_type=ResponseType.AI_ANALYSIS,
            suggested_actions=[
                "Configure AI credentials to enable dynamic agent",
                "Learn more about the AI agent capabilities",
                "Try the test script: python test_dynamic_ai_agent.py",
            ],
            requires_input=True,
            metadata={"demo_mode": True},
        )

    def _log_ai_interaction(
        self, user_input: str, result, processing_time: float, success: bool
    ):
        """Log AI agent interaction to session"""
        interaction = Interaction(
            id=f"ai_{len(self.session.interactions)}",
            timestamp=datetime.now(),
            type=InteractionType.QUERY,
            input_data=user_input,
            output_data=result.message if success else f"Error: {result.error}",
            metadata={
                "ai_agent_used": True,
                "tools_executed": (
                    result.metadata.get("tools_executed", []) if success else []
                ),
                "ai_reasoning_steps": (
                    result.metadata.get("ai_reasoning_steps", 0) if success else 0
                ),
                "audit_file": result.metadata.get("audit_file") if success else None,
                "dynamic_decision_making": True,
                "success": success,
            },
            execution_time=processing_time,
            success=success,
        )

        self.session.add_interaction(interaction)

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation"""
        return {
            "session_id": self.session.id,
            "current_state": self.state.value,
            "interaction_count": self.context.interaction_count,
            "successful_queries": self.context.successful_queries,
            "success_rate": self.context.successful_queries
            / max(1, self.context.interaction_count),
            "ai_agent_available": self.ai_agent is not None,
            "dynamic_ai_enabled": True,
            "conversation_engine": "RealTimeMetabolicAgent",
        }

    def display_conversation_status(self) -> None:
        """Display current conversation status"""
        summary = self.get_conversation_summary()

        status_table = Table(show_header=False, box=box.SIMPLE)
        status_table.add_column("Aspect", style="bold cyan")
        status_table.add_column("Value", style="bold white")

        status_table.add_row("Session", summary["session_id"])
        status_table.add_row("AI Engine", "Dynamic Real-Time Agent")
        status_table.add_row(
            "AI Status", "Available" if summary["ai_agent_available"] else "Demo Mode"
        )
        status_table.add_row("Interactions", str(summary["interaction_count"]))
        status_table.add_row("Success Rate", f"{summary['success_rate']:.1%}")

        console.print(
            Panel(
                status_table,
                title="[bold blue]🧠 Dynamic AI Conversation Status[/bold blue]",
                border_style="blue",
            )
        )


# Backward compatibility alias
ConversationEngine = DynamicAIConversationEngine
