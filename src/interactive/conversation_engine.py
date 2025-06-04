"""
Conversation Engine for Interactive Analysis

Manages natural dialogue flow, maintains conversation context,
and provides intelligent responses for metabolic modeling queries.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .query_processor import QueryAnalysis, QueryProcessor, QueryType
from .session_manager import AnalysisSession, Interaction, InteractionType

console = Console()


class ConversationState(Enum):
    """States of the conversation flow"""

    GREETING = "greeting"
    ACTIVE = "active"
    CLARIFYING = "clarifying"
    PROCESSING = "processing"
    PRESENTING_RESULTS = "presenting_results"
    SUGGESTING_FOLLOWUP = "suggesting_followup"
    WAITING_INPUT = "waiting_input"
    ERROR_RECOVERY = "error_recovery"
    GOODBYE = "goodbye"


class ResponseType(Enum):
    """Types of assistant responses"""

    GREETING = "greeting"
    ACKNOWLEDGMENT = "acknowledgment"
    CLARIFICATION = "clarification"
    ANALYSIS_RESULT = "analysis_result"
    SUGGESTION = "suggestion"
    ERROR_MESSAGE = "error_message"
    HELP_RESPONSE = "help_response"
    GOODBYE = "goodbye"


@dataclass
class ConversationContext:
    """Context maintained during conversation"""

    current_model: Optional[str] = None
    active_analysis: Optional[str] = None
    last_query_type: Optional[QueryType] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    pending_clarifications: List[str] = field(default_factory=list)
    suggested_follow_ups: List[str] = field(default_factory=list)

    # User profiling
    expertise_level: str = "intermediate"  # beginner, intermediate, expert
    preferred_detail_level: str = "moderate"  # brief, moderate, detailed
    interaction_count: int = 0
    successful_queries: int = 0


@dataclass
class ConversationResponse:
    """Structured response from the conversation engine"""

    content: str
    response_type: ResponseType
    suggested_actions: List[str] = field(default_factory=list)
    clarification_needed: bool = False
    requires_input: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


class ConversationEngine:
    """Manages natural conversation flow for interactive analysis"""

    def __init__(self, session: AnalysisSession, query_processor: QueryProcessor):
        self.session = session
        self.query_processor = query_processor
        self.context = ConversationContext()
        self.state = ConversationState.GREETING
        self.response_templates = self._initialize_response_templates()
        self.conversation_hooks: Dict[str, Callable] = {}

    def _initialize_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize response templates for different scenarios"""
        return {
            "greetings": {
                "first_time": [
                    "👋 Welcome to ModelSEEDagent! I'm here to help you analyze metabolic models.",
                    "🧬 Hello! I'm your metabolic modeling assistant. What would you like to explore today?",
                    "🚀 Welcome! I can help you analyze metabolic networks, run flux analyses, and much more.",
                ],
                "returning": [
                    "🎉 Welcome back! Ready for more metabolic modeling analysis?",
                    "👋 Great to see you again! How can I help with your models today?",
                    "🔬 Hello again! What metabolic analysis shall we dive into?",
                ],
            },
            "acknowledgments": {
                "understanding": [
                    "I understand you want to {action}. Let me analyze that...",
                    "Got it! You're interested in {action}. Processing...",
                    "Perfect! I'll help you {action}. One moment...",
                ],
                "clarification": [
                    "I need a bit more information to help you better.",
                    "Let me ask a few questions to make sure I understand correctly.",
                    "To provide the best analysis, could you clarify a few things?",
                ],
            },
            "suggestions": {
                "follow_up": [
                    "Based on these results, you might want to {suggestion}.",
                    "Great results! Consider {suggestion} as a next step.",
                    "This analysis opens up possibilities for {suggestion}.",
                ],
                "alternative": [
                    "If you're interested, we could also {alternative}.",
                    "Another approach would be to {alternative}.",
                    "You might find it useful to {alternative}.",
                ],
            },
            "errors": {
                "understanding": [
                    "I'm not quite sure what you mean. Could you rephrase that?",
                    "I didn't fully understand. Could you be more specific?",
                    "Let me make sure I understand - are you asking about {guess}?",
                ],
                "technical": [
                    "I encountered an issue while processing. Let me try a different approach.",
                    "There was a technical problem. I'll attempt an alternative method.",
                    "Something went wrong, but I have some suggestions to try instead.",
                ],
            },
            "help": {
                "general": [
                    "I can help you with metabolic model analysis, pathway exploration, flux calculations, and more!",
                    "I'm equipped to analyze model structure, growth conditions, optimization, and visualizations.",
                    "Ask me about FBA, pathway analysis, model comparison, or any metabolic modeling question!",
                ],
                "specific": [
                    "For {topic}, you can ask me to {examples}.",
                    "Try questions like: {examples}",
                    "Here are some things you can explore: {examples}",
                ],
            },
        }

    def start_conversation(self) -> ConversationResponse:
        """Start a new conversation"""
        self.state = ConversationState.GREETING

        # Determine if this is a first-time or returning user
        is_first_time = len(self.session.interactions) == 0
        template_key = "first_time" if is_first_time else "returning"

        greeting = self._select_template("greetings", template_key)

        # Add contextual information
        context_info = []
        if self.session.model_files:
            context_info.append(
                f"I see you have {len(self.session.model_files)} model(s) loaded."
            )

        if not is_first_time:
            recent_activities = [
                i.input_data[:40] + "..." if len(i.input_data) > 40 else i.input_data
                for i in self.session.get_recent_interactions(2)
            ]
            if recent_activities:
                context_info.append(
                    f"Your recent activities: {', '.join(recent_activities)}"
                )

        full_greeting = greeting
        if context_info:
            full_greeting += "\n\n" + " ".join(context_info)

        full_greeting += "\n\n💡 You can ask me to analyze models, explore pathways, calculate fluxes, or get help with any metabolic modeling question!"

        self.state = ConversationState.WAITING_INPUT

        return ConversationResponse(
            content=full_greeting,
            response_type=ResponseType.GREETING,
            suggested_actions=[
                "Upload and analyze a metabolic model",
                "Explore pathway analysis capabilities",
                "Learn about flux balance analysis",
                "Get help with modeling questions",
            ],
            requires_input=True,
        )

    def process_user_input(self, user_input: str) -> ConversationResponse:
        """Process user input and generate appropriate response"""
        start_time = time.time()

        # Update conversation context
        self.context.interaction_count += 1

        # Analyze the query
        query_analysis = self.query_processor.analyze_query(
            user_input, context=self._get_query_context()
        )

        # Update context with query analysis
        self.context.last_query_type = query_analysis.query_type

        # Determine conversation flow based on query analysis
        response = self._generate_response(user_input, query_analysis)

        # Update processing time
        response.processing_time = time.time() - start_time

        # Log interaction
        self._log_interaction(user_input, response, query_analysis)

        return response

    def _generate_response(
        self, user_input: str, analysis: QueryAnalysis
    ) -> ConversationResponse:
        """Generate appropriate response based on query analysis"""

        # Handle help requests
        if analysis.query_type == QueryType.HELP_REQUEST:
            return self._generate_help_response(analysis)

        # Handle low confidence queries
        if analysis.confidence < 0.5:
            return self._generate_clarification_response(user_input, analysis)

        # Handle queries needing clarification
        if analysis.clarification_questions:
            return self._generate_clarification_response(user_input, analysis)

        # Handle normal analysis queries
        return self._generate_analysis_response(user_input, analysis)

    def _generate_help_response(self, analysis: QueryAnalysis) -> ConversationResponse:
        """Generate helpful response for help requests"""
        self.state = ConversationState.PRESENTING_RESULTS

        help_content = self._select_template("help", "general")

        # Add specific examples based on detected entities
        examples = []
        if analysis.model_references:
            examples.extend(
                [
                    "analyze the structure of your model",
                    "calculate growth rates and yields",
                    "explore metabolic pathways",
                ]
            )
        elif analysis.pathway_references:
            examples.extend(
                [
                    "analyze glycolysis pathway fluxes",
                    "compare pathway activities",
                    "visualize pathway networks",
                ]
            )
        else:
            examples.extend(
                [
                    "load and analyze SBML models",
                    "run flux balance analysis",
                    "explore growth conditions",
                    "visualize metabolic networks",
                ]
            )

        help_content += f"\n\n🔧 **What you can try:**\n"
        for i, example in enumerate(examples[:4], 1):
            help_content += f"  {i}. {example.capitalize()}\n"

        if analysis.follow_up_suggestions:
            help_content += f"\n💡 **Suggestions:**\n"
            for suggestion in analysis.follow_up_suggestions:
                help_content += f"  • {suggestion}\n"

        self.state = ConversationState.WAITING_INPUT

        return ConversationResponse(
            content=help_content,
            response_type=ResponseType.HELP_RESPONSE,
            suggested_actions=examples[:3],
            requires_input=True,
        )

    def _generate_clarification_response(
        self, user_input: str, analysis: QueryAnalysis
    ) -> ConversationResponse:
        """Generate response requesting clarification"""
        self.state = ConversationState.CLARIFYING

        clarification_content = self._select_template(
            "acknowledgments", "clarification"
        )

        # Add specific clarification questions
        if analysis.clarification_questions:
            clarification_content += (
                "\n\n❓ **Questions to help me assist you better:**\n"
            )
            for i, question in enumerate(analysis.clarification_questions, 1):
                clarification_content += f"  {i}. {question}\n"

        # Add alternative phrasings if available
        if analysis.alternative_phrasings:
            clarification_content += f"\n🔄 **Did you mean:**\n"
            for alt in analysis.alternative_phrasings:
                clarification_content += f"  • {alt}\n"

        # Store pending clarifications
        self.context.pending_clarifications = analysis.clarification_questions

        suggested_actions = (
            analysis.clarification_questions + analysis.alternative_phrasings
        )

        return ConversationResponse(
            content=clarification_content,
            response_type=ResponseType.CLARIFICATION,
            suggested_actions=suggested_actions[:5],
            clarification_needed=True,
            requires_input=True,
            metadata={"original_query": user_input, "analysis": analysis.to_dict()},
        )

    def _generate_analysis_response(
        self, user_input: str, analysis: QueryAnalysis
    ) -> ConversationResponse:
        """Generate response for analysis queries"""
        self.state = ConversationState.PROCESSING

        # Acknowledge the query
        action_description = self._describe_analysis_action(analysis)
        acknowledgment = self._select_template(
            "acknowledgments", "understanding"
        ).format(action=action_description)

        # Simulate analysis processing
        processing_content = f"{acknowledgment}\n\n🔬 **Analysis Plan:**\n"

        # Show suggested tools
        if analysis.suggested_tools:
            processing_content += (
                f"  • Using tools: {', '.join(analysis.suggested_tools)}\n"
            )

        # Show complexity and expected time
        complexity_time = {
            "simple": "a few seconds",
            "moderate": "10-30 seconds",
            "complex": "30-60 seconds",
            "expert": "1-2 minutes",
        }
        expected_time = complexity_time.get(analysis.complexity.value, "a moment")
        processing_content += f"  • Complexity: {analysis.complexity.value.title()} (estimated {expected_time})\n"

        # Show expected outputs
        if analysis.expected_outputs:
            processing_content += (
                f"  • Expected results: {', '.join(analysis.expected_outputs)}\n"
            )

        # Mock analysis results (in real implementation, this would call actual tools)
        mock_results = self._generate_mock_results(analysis)

        processing_content += f"\n✅ **Analysis Complete!**\n\n{mock_results}"

        # Add follow-up suggestions
        if analysis.follow_up_suggestions:
            processing_content += f"\n\n💡 **What's next?**\n"
            for i, suggestion in enumerate(analysis.follow_up_suggestions, 1):
                processing_content += f"  {i}. {suggestion}\n"

        self.state = ConversationState.SUGGESTING_FOLLOWUP
        self.context.successful_queries += 1
        self.context.suggested_follow_ups = analysis.follow_up_suggestions

        return ConversationResponse(
            content=processing_content,
            response_type=ResponseType.ANALYSIS_RESULT,
            suggested_actions=analysis.follow_up_suggestions,
            requires_input=True,
            metadata={"analysis": analysis.to_dict(), "mock_results": True},
        )

    def _describe_analysis_action(self, analysis: QueryAnalysis) -> str:
        """Describe what analysis action will be performed"""
        action_descriptions = {
            QueryType.STRUCTURAL_ANALYSIS: "analyze the model structure and components",
            QueryType.GROWTH_ANALYSIS: "calculate growth rates and biomass production",
            QueryType.PATHWAY_ANALYSIS: "examine metabolic pathway fluxes and activities",
            QueryType.FLUX_ANALYSIS: "perform flux balance analysis and optimization",
            QueryType.OPTIMIZATION: "optimize the metabolic model for your objectives",
            QueryType.COMPARISON: "compare different models or conditions",
            QueryType.VISUALIZATION: "create visualizations and plots",
            QueryType.MODEL_MODIFICATION: "modify the metabolic model as requested",
            QueryType.GENERAL_QUESTION: "provide information about your question",
        }

        return action_descriptions.get(analysis.query_type, "process your request")

    def _generate_mock_results(self, analysis: QueryAnalysis) -> str:
        """Generate mock analysis results for demonstration"""
        mock_results = {
            QueryType.STRUCTURAL_ANALYSIS: "📊 **Model Structure:**\n  • 95 reactions, 72 metabolites, 137 genes\n  • 15 pathways identified\n  • Network connectivity: 85% connected",
            QueryType.GROWTH_ANALYSIS: "📈 **Growth Analysis:**\n  • Predicted growth rate: 0.873 h⁻¹\n  • Biomass yield: 0.394 g/g glucose\n  • ATP yield: 32.1 mol/mol glucose",
            QueryType.PATHWAY_ANALYSIS: "🛣️ **Pathway Analysis:**\n  • Glycolysis flux: 8.2 mmol/gDW/h\n  • TCA cycle flux: 6.1 mmol/gDW/h\n  • Active pathways: 12 of 15 analyzed",
            QueryType.FLUX_ANALYSIS: "⚡ **Flux Analysis:**\n  • Optimal objective value: 0.873\n  • 45 active reactions\n  • Flux variability: 23% average range",
            QueryType.OPTIMIZATION: "🎯 **Optimization Results:**\n  • Target improved by 15.3%\n  • 3 gene modifications suggested\n  • New predicted yield: 0.452 g/g",
            QueryType.COMPARISON: "🔄 **Comparison Results:**\n  • Growth rate difference: +12.5%\n  • 23 reactions show significant changes\n  • Metabolic efficiency improved",
            QueryType.VISUALIZATION: "🎨 **Visualization Created:**\n  • Network diagram generated\n  • Flux heatmap created\n  • Interactive plot available",
            QueryType.MODEL_MODIFICATION: "🔧 **Model Modified:**\n  • 2 reactions added/removed\n  • Gene associations updated\n  • Model validation passed",
        }

        return mock_results.get(
            analysis.query_type,
            "📋 **Analysis completed successfully!**\nResults are ready for your review.",
        )

    def _select_template(self, category: str, subcategory: str, **kwargs) -> str:
        """Select and format a response template"""
        templates = self.response_templates.get(category, {}).get(
            subcategory, ["Response not available."]
        )

        # Simple selection (could be improved with ML/preference learning)
        template = templates[self.context.interaction_count % len(templates)]

        # Format with any provided kwargs
        try:
            return template.format(**kwargs)
        except KeyError:
            return template

    def _get_query_context(self) -> Dict[str, Any]:
        """Get context for query processing"""
        return {
            "current_model": self.context.current_model,
            "last_query_type": (
                self.context.last_query_type.value
                if self.context.last_query_type
                else None
            ),
            "interaction_count": self.context.interaction_count,
            "expertise_level": self.context.expertise_level,
            "recent_interactions": [
                i.input_data for i in self.session.get_recent_interactions(3)
            ],
        }

    def _log_interaction(
        self, user_input: str, response: ConversationResponse, analysis: QueryAnalysis
    ) -> None:
        """Log the interaction to the session"""
        interaction = Interaction(
            id=f"conv_{len(self.session.interactions)}",
            timestamp=datetime.now(),
            type=InteractionType.QUERY,
            input_data=user_input,
            output_data=response.content,
            metadata={
                "query_analysis": analysis.to_dict(),
                "response_type": response.response_type.value,
                "conversation_state": self.state.value,
                "suggested_actions": response.suggested_actions,
            },
            execution_time=response.processing_time,
            success=response.response_type != ResponseType.ERROR_MESSAGE,
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
            "current_context": {
                "model": self.context.current_model,
                "last_query_type": (
                    self.context.last_query_type.value
                    if self.context.last_query_type
                    else None
                ),
                "pending_clarifications": len(self.context.pending_clarifications),
                "suggested_follow_ups": len(self.context.suggested_follow_ups),
            },
            "user_profile": {
                "expertise_level": self.context.expertise_level,
                "preferred_detail_level": self.context.preferred_detail_level,
            },
        }

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a callback for conversation events"""
        self.conversation_hooks[event] = callback

    def display_conversation_status(self) -> None:
        """Display current conversation status"""
        summary = self.get_conversation_summary()

        status_table = Table(show_header=False, box=box.SIMPLE)
        status_table.add_column("Aspect", style="bold cyan")
        status_table.add_column("Value", style="bold white")

        status_table.add_row("Session", summary["session_id"])
        status_table.add_row(
            "State", summary["current_state"].replace("_", " ").title()
        )
        status_table.add_row("Interactions", str(summary["interaction_count"]))
        status_table.add_row("Success Rate", f"{summary['success_rate']:.1%}")

        if summary["current_context"]["model"]:
            status_table.add_row("Current Model", summary["current_context"]["model"])

        if summary["current_context"]["last_query_type"]:
            status_table.add_row(
                "Last Query",
                summary["current_context"]["last_query_type"].replace("_", " ").title(),
            )

        console.print(
            Panel(
                status_table,
                title="[bold blue]💬 Conversation Status[/bold blue]",
                border_style="blue",
            )
        )
