#!/usr/bin/env python3
"""
Collaborative Reasoning System - Phase 8.3

Implements AI-human collaborative reasoning capabilities where the AI can ask
for user guidance when uncertain, present options for user selection, and
incorporate human expertise into the analysis workflow.

Key Features:
- AI uncertainty detection and user consultation requests
- Interactive decision points with multiple options
- Human expertise integration into reasoning chains
- Collaborative hypothesis refinement
- Real-time guidance incorporation
"""

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import questionary
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class CollaborationType(Enum):
    """Types of collaboration requests"""

    UNCERTAINTY = "uncertainty"  # AI is uncertain about next step
    CHOICE = "choice"  # Multiple valid options available
    EXPERTISE = "expertise"  # Human expertise needed
    VALIDATION = "validation"  # Validate AI reasoning
    REFINEMENT = "refinement"  # Refine hypothesis or approach
    PRIORITIZATION = "prioritization"  # Prioritize analysis directions


class CollaborationRequest(BaseModel):
    """Request for human collaboration in reasoning"""

    request_id: str = Field(description="Unique request identifier")
    collaboration_type: CollaborationType = Field(
        description="Type of collaboration needed"
    )
    timestamp: str = Field(description="When request was made")

    # Request context
    context: str = Field(description="Current analysis context")
    ai_reasoning: str = Field(description="AI's current reasoning")
    uncertainty_description: str = Field(description="What the AI is uncertain about")

    # Options and choices
    options: List[Dict[str, Any]] = Field(
        default_factory=list, description="Available options"
    )
    ai_recommendation: Optional[str] = Field(
        default=None, description="AI's preferred option"
    )

    # User response
    user_response: Optional[str] = Field(default=None, description="User's guidance")
    user_choice: Optional[int] = Field(
        default=None, description="User's selected option index"
    )
    additional_context: Optional[str] = Field(
        default=None, description="Additional user context"
    )

    # Resolution
    resolved: bool = Field(default=False, description="Whether request was resolved")
    resolution_reasoning: Optional[str] = Field(
        default=None, description="How guidance was incorporated"
    )


class CollaborativeDecision(BaseModel):
    """Record of a collaborative decision made"""

    decision_id: str = Field(description="Unique decision identifier")
    original_request: CollaborationRequest = Field(
        description="Original collaboration request"
    )

    # Decision details
    final_choice: str = Field(description="Final decision made")
    decision_rationale: str = Field(description="Combined AI-human reasoning")
    confidence_score: float = Field(description="Confidence in collaborative decision")

    # Impact
    impact_on_analysis: str = Field(description="How decision affected analysis")
    follow_up_actions: List[str] = Field(
        default_factory=list, description="Actions taken based on decision"
    )

    timestamp: str = Field(description="When decision was finalized")


class UncertaintyDetector:
    """Detects when AI should request human collaboration"""

    def __init__(self, llm):
        """Initialize uncertainty detector"""
        self.llm = llm
        self.uncertainty_threshold = 0.7
        self.collaboration_patterns = self._load_collaboration_patterns()

    def should_request_collaboration(
        self, ai_reasoning: str, available_options: List[str], context: Dict[str, Any]
    ) -> Tuple[bool, Optional[CollaborationType]]:
        """Determine if AI should request human collaboration"""

        # Use AI to assess its own uncertainty
        uncertainty_prompt = f"""
        Analyze this reasoning situation and determine if human collaboration would be beneficial:

        Current Reasoning: {ai_reasoning}
        Available Options: {available_options}
        Context: {json.dumps(context, indent=2)}

        Assess:
        1. How confident are you in the best next step? (0.0 = very uncertain, 1.0 = very confident)
        2. Are there multiple equally valid approaches?
        3. Would domain expertise improve the decision?
        4. Is the analysis at a critical decision point?

        Respond with JSON:
        {{
            "confidence": 0.8,
            "multiple_valid_options": true/false,
            "expertise_needed": true/false,
            "critical_decision": true/false,
            "collaboration_type": "uncertainty" | "choice" | "expertise" | "validation",
            "reasoning": "Explanation of why collaboration would or wouldn't help"
        }}
        """

        response = self.llm._generate_response(uncertainty_prompt)

        try:
            assessment = json.loads(response.text)

            # Decision logic for requesting collaboration
            confidence = assessment.get("confidence", 1.0)

            if confidence < self.uncertainty_threshold:
                return True, CollaborationType.UNCERTAINTY
            elif assessment.get("multiple_valid_options", False):
                return True, CollaborationType.CHOICE
            elif assessment.get("expertise_needed", False):
                return True, CollaborationType.EXPERTISE
            elif assessment.get("critical_decision", False):
                return True, CollaborationType.VALIDATION
            else:
                return False, None

        except (json.JSONDecodeError, KeyError):
            # Conservative fallback - request collaboration if parsing fails
            return True, CollaborationType.UNCERTAINTY

    def _load_collaboration_patterns(self) -> Dict[str, Any]:
        """Load patterns that indicate when collaboration is beneficial"""

        patterns = {
            "uncertainty_indicators": [
                "multiple viable options",
                "conflicting evidence",
                "novel situation",
                "high-stakes decision",
            ],
            "expertise_domains": [
                "pathway interpretation",
                "biological significance",
                "experimental design",
                "model validation",
            ],
        }

        return patterns


class CollaborationInterface:
    """Interactive interface for AI-human collaboration"""

    def __init__(self):
        """Initialize collaboration interface"""
        self.console = Console()
        self.request_history = []

    async def request_collaboration(
        self, request: CollaborationRequest, interactive: bool = True
    ) -> CollaborationRequest:
        """Present collaboration request to user and get response"""

        self.request_history.append(request)

        if interactive:
            return await self._interactive_collaboration(request)
        else:
            return await self._automated_collaboration(request)

    async def _interactive_collaboration(
        self, request: CollaborationRequest
    ) -> CollaborationRequest:
        """Handle interactive collaboration with user"""

        # Display collaboration request
        self._display_collaboration_request(request)

        if request.collaboration_type == CollaborationType.CHOICE:
            # Present options for user selection
            user_choice = await self._get_user_choice(request.options)
            request.user_choice = user_choice

        elif request.collaboration_type == CollaborationType.UNCERTAINTY:
            # Get open-ended guidance
            user_guidance = await self._get_user_guidance(request)
            request.user_response = user_guidance

        elif request.collaboration_type == CollaborationType.EXPERTISE:
            # Request domain expertise
            expertise = await self._get_domain_expertise(request)
            request.additional_context = expertise

        elif request.collaboration_type == CollaborationType.VALIDATION:
            # Validate AI reasoning
            validation = await self._get_reasoning_validation(request)
            request.user_response = validation

        # Get any additional context
        additional = questionary.text(
            "Any additional context or considerations? (optional):", default=""
        ).ask()

        if additional:
            request.additional_context = (
                (request.additional_context or "") + "\n" + additional
            )

        request.resolved = True
        return request

    async def _automated_collaboration(
        self, request: CollaborationRequest
    ) -> CollaborationRequest:
        """Handle non-interactive collaboration (use AI defaults)"""

        # For automated mode, use AI's best judgment
        if (
            request.ai_recommendation
            and request.collaboration_type == CollaborationType.CHOICE
        ):
            # Use AI recommendation if available
            request.user_choice = 0  # Assume first option is AI recommendation
        else:
            # Use conservative default
            request.user_response = "Proceed with AI's best judgment"

        request.resolved = True
        return request

    def _display_collaboration_request(self, request: CollaborationRequest):
        """Display collaboration request to user"""

        # Create rich display
        title = (
            f"🤖 AI Collaboration Request - {request.collaboration_type.value.title()}"
        )

        content = f"""
[bold]Current Context:[/bold]
{request.context}

[bold]AI Reasoning:[/bold]
{request.ai_reasoning}

[bold]Uncertainty/Issue:[/bold]
{request.uncertainty_description}
"""

        if request.ai_recommendation:
            content += f"\n[bold]AI Recommendation:[/bold]\n{request.ai_recommendation}"

        panel = Panel(content, title=title, border_style="yellow")
        self.console.print(panel)

    async def _get_user_choice(self, options: List[Dict[str, Any]]) -> int:
        """Get user's choice from available options"""

        # Create choice table
        table = Table(title="Available Options")
        table.add_column("Option", style="bold")
        table.add_column("Description")
        table.add_column("AI Assessment")

        choice_list = []
        for i, option in enumerate(options):
            table.add_row(
                f"{i + 1}",
                option.get("description", "No description"),
                option.get("ai_assessment", "Not assessed"),
            )
            choice_list.append(f"{i + 1}. {option.get('name', f'Option {i + 1}')}")

        self.console.print(table)

        # Get user selection
        choice = questionary.select(
            "Which option do you prefer?", choices=choice_list
        ).ask()

        # Extract choice index
        return int(choice.split(".")[0]) - 1

    async def _get_user_guidance(self, request: CollaborationRequest) -> str:
        """Get open-ended guidance from user"""

        guidance = questionary.text(
            "What guidance can you provide for this situation?", multiline=True
        ).ask()

        return guidance

    async def _get_domain_expertise(self, request: CollaborationRequest) -> str:
        """Get domain-specific expertise from user"""

        expertise = questionary.text(
            "Please share your domain expertise relevant to this situation:",
            multiline=True,
        ).ask()

        return expertise

    async def _get_reasoning_validation(self, request: CollaborationRequest) -> str:
        """Get validation of AI reasoning"""

        validation = questionary.select(
            "How do you assess the AI's reasoning?",
            choices=[
                "Reasoning is sound - proceed as planned",
                "Reasoning is mostly correct but needs minor adjustments",
                "Reasoning has significant issues - major revision needed",
                "Reasoning is fundamentally flawed - completely different approach needed",
            ],
        ).ask()

        if (
            "adjustments" in validation
            or "issues" in validation
            or "flawed" in validation
        ):
            details = questionary.text(
                "Please explain what needs to be changed:", multiline=True
            ).ask()
            validation += f"\n\nDetails: {details}"

        return validation


class CollaborativeReasoner:
    """Manages collaborative reasoning workflows"""

    def __init__(self, llm, uncertainty_detector=None, collaboration_interface=None):
        """Initialize collaborative reasoner"""
        self.llm = llm
        self.uncertainty_detector = uncertainty_detector or UncertaintyDetector(llm)
        self.collaboration_interface = (
            collaboration_interface or CollaborationInterface()
        )
        self.collaborative_decisions = []

    async def collaborative_decision_point(
        self,
        reasoning_context: str,
        available_options: List[Dict[str, Any]],
        analysis_context: Dict[str, Any],
        interactive: bool = True,
    ) -> CollaborativeDecision:
        """Handle a collaborative decision point in analysis"""

        # Check if collaboration is needed
        should_collaborate, collaboration_type = (
            self.uncertainty_detector.should_request_collaboration(
                reasoning_context,
                [opt.get("name", "Option") for opt in available_options],
                analysis_context,
            )
        )

        if should_collaborate:
            # Create collaboration request
            request = CollaborationRequest(
                request_id=str(uuid.uuid4())[:8],
                collaboration_type=collaboration_type,
                timestamp=datetime.now().isoformat(),
                context=json.dumps(analysis_context, indent=2),
                ai_reasoning=reasoning_context,
                uncertainty_description=self._generate_uncertainty_description(
                    reasoning_context, available_options, collaboration_type
                ),
                options=available_options,
                ai_recommendation=self._generate_ai_recommendation(available_options),
            )

            # Get user collaboration
            resolved_request = await self.collaboration_interface.request_collaboration(
                request, interactive
            )

            # Process collaborative input
            decision = await self._process_collaborative_input(
                resolved_request, available_options
            )

        else:
            # Make decision autonomously
            decision = await self._make_autonomous_decision(
                reasoning_context, available_options, analysis_context
            )

        self.collaborative_decisions.append(decision)
        return decision

    def _generate_uncertainty_description(
        self,
        reasoning: str,
        options: List[Dict[str, Any]],
        collaboration_type: CollaborationType,
    ) -> str:
        """Generate description of why AI is requesting collaboration"""

        if collaboration_type == CollaborationType.UNCERTAINTY:
            return f"I'm uncertain about the best next step given the current evidence and {len(options)} available options."
        elif collaboration_type == CollaborationType.CHOICE:
            return f"Multiple valid approaches are available ({len(options)} options) and domain expertise would help select the most appropriate one."
        elif collaboration_type == CollaborationType.EXPERTISE:
            return "This situation requires domain-specific knowledge that would benefit from human expertise."
        elif collaboration_type == CollaborationType.VALIDATION:
            return "This is a critical decision point where validation of the reasoning approach would be valuable."
        else:
            return "Collaboration would improve the quality of this decision."

    def _generate_ai_recommendation(
        self, options: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Generate AI's recommended option"""

        if not options:
            return None

        # Use AI to assess options and make recommendation
        recommendation_prompt = f"""
        Analyze these options and provide your recommendation:

        Options: {json.dumps(options, indent=2)}

        Which option do you think is best and why? Provide a brief rationale.
        """

        response = self.llm._generate_response(recommendation_prompt)
        return response.text.strip()

    async def _process_collaborative_input(
        self, resolved_request: CollaborationRequest, options: List[Dict[str, Any]]
    ) -> CollaborativeDecision:
        """Process user input and create collaborative decision"""

        # Determine final choice based on user input
        if resolved_request.user_choice is not None:
            final_choice = options[resolved_request.user_choice].get(
                "name", f"Option {resolved_request.user_choice + 1}"
            )
            # choice_reasoning = options[resolved_request.user_choice].get(
            #     "description", "User selected option"
            # )
        else:
            final_choice = resolved_request.user_response or "AI recommendation"
            # choice_reasoning = (
            #     resolved_request.user_response or "Proceeding with AI judgment"
            # )

        # Incorporate additional context
        combined_reasoning = f"AI reasoning: {resolved_request.ai_reasoning}\n"
        if resolved_request.user_response:
            combined_reasoning += f"User guidance: {resolved_request.user_response}\n"
        if resolved_request.additional_context:
            combined_reasoning += (
                f"Additional context: {resolved_request.additional_context}\n"
            )

        decision = CollaborativeDecision(
            decision_id=str(uuid.uuid4())[:8],
            original_request=resolved_request,
            final_choice=final_choice,
            decision_rationale=combined_reasoning,
            confidence_score=0.9,  # Higher confidence due to human input
            impact_on_analysis="Decision incorporates human expertise and guidance",
            timestamp=datetime.now().isoformat(),
        )

        return decision

    async def _make_autonomous_decision(
        self,
        reasoning_context: str,
        available_options: List[Dict[str, Any]],
        analysis_context: Dict[str, Any],
    ) -> CollaborativeDecision:
        """Make decision autonomously when collaboration is not needed"""

        # Use AI to select best option
        selection_prompt = f"""
        Select the best option for this analysis situation:

        Context: {reasoning_context}
        Options: {json.dumps(available_options, indent=2)}

        Which option is most appropriate and why?
        """

        response = self.llm._generate_response(selection_prompt)

        decision = CollaborativeDecision(
            decision_id=str(uuid.uuid4())[:8],
            original_request=CollaborationRequest(
                request_id="autonomous",
                collaboration_type=CollaborationType.UNCERTAINTY,
                timestamp=datetime.now().isoformat(),
                context=reasoning_context,
                ai_reasoning=reasoning_context,
                uncertainty_description="Autonomous decision - no collaboration needed",
            ),
            final_choice=response.text.strip(),
            decision_rationale=f"Autonomous AI decision: {response.text}",
            confidence_score=0.8,
            impact_on_analysis="AI made autonomous decision with high confidence",
            timestamp=datetime.now().isoformat(),
        )

        return decision

    def get_collaboration_summary(self) -> Dict[str, Any]:
        """Get summary of collaborative decisions made"""

        total_decisions = len(self.collaborative_decisions)
        collaborative_decisions = len(
            [
                d
                for d in self.collaborative_decisions
                if d.original_request.collaboration_type
                != CollaborationType.UNCERTAINTY
                or d.original_request.resolved
            ]
        )

        collaboration_types = {}
        for decision in self.collaborative_decisions:
            collab_type = decision.original_request.collaboration_type.value
            collaboration_types[collab_type] = (
                collaboration_types.get(collab_type, 0) + 1
            )

        avg_confidence = (
            sum(d.confidence_score for d in self.collaborative_decisions)
            / total_decisions
            if total_decisions > 0
            else 0
        )

        return {
            "total_decisions": total_decisions,
            "collaborative_decisions": collaborative_decisions,
            "autonomous_decisions": total_decisions - collaborative_decisions,
            "collaboration_types": collaboration_types,
            "average_confidence": avg_confidence,
        }


def create_collaborative_reasoning_system(llm, interactive=True):
    """Factory function to create collaborative reasoning system"""

    uncertainty_detector = UncertaintyDetector(llm)
    collaboration_interface = CollaborationInterface() if interactive else None

    return CollaborativeReasoner(llm, uncertainty_detector, collaboration_interface)
